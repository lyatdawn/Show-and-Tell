#-*- coding: utf-8 -*-
import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle
import glob
import time
import random
from nltk.translate.bleu_score import *

# from tensorflow.models.rnn import rnn_cell # Error! use tf.nn.rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from collections import Counter
from cnn_util import *
from util import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', './models/tensorflow/model-72',
                           "Output folder where checkpoints are dumped.")
tf.app.flags.DEFINE_bool('use_flickr', 'False',
                           "Whether use Flickr dataset to test.")                           
tf.app.flags.DEFINE_string('phase', 'train',
                           "Which operation to run. [train|test|test_tf]")
tf.app.flags.DEFINE_integer('maxlen', 30,
                            "The max length of genereted sentence.")

class Caption_Generator():
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words, bias_init_vector=None):

        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.batch_size = np.int(batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -0.1, 0.1), name='Wemb')

        self.bemb = self.init_bias(dim_embed, name='bemb')

        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden)

        #self.encode_img_W = self.init_weight(dim_image, dim_hidden, name='encode_img_W')
        self.encode_img_W = tf.Variable(tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_img_W')
        self.encode_img_b = self.init_bias(dim_hidden, name='encode_img_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')

        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = self.init_bias(n_words, name='embed_word_b')

    def build_model(self):

        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b # (batch_size, dim_hidden)

        state = tf.zeros([self.batch_size, self.lstm.state_size])

        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): # maxlen + 1
                if i == 0:
                    current_emb = image_emb
                else:
                    with tf.device("/cpu:0"):
                        current_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i-1]) + self.bemb

                if i > 0 : tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(current_emb, state) # (batch_size, dim_hidden)

                if i > 0: # 이미지 다음 바로 나오는건 #START# 임. 이건 무시.
                    labels = tf.expand_dims(sentence[:, i], 1) # (batch_size)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat(1, [indices, labels])
                    onehot_labels = tf.sparse_to_dense(
                            concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0) # (batch_size, n_words)

                    logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                    cross_entropy = cross_entropy * mask[:,i]#tf.expand_dims(mask, 1)

                    current_loss = tf.reduce_sum(cross_entropy)
                    loss = loss + current_loss

            loss = loss / tf.reduce_sum(mask[:,1:])
            return loss, image, sentence, mask

    def build_generator(self, maxlen):
        image = tf.placeholder(tf.float32, [1, self.dim_image])
        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

        state = tf.zeros([1, self.lstm.state_size])
        #last_word = image_emb # 첫 단어 대신 이미지
        generated_words = []

        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_emb, state)
            last_word = tf.nn.embedding_lookup(self.Wemb, [0]) + self.bemb

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(last_word, state)

                logit_words = tf.matmul(output, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)

                with tf.device("/cpu:0"):
                    last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

                last_word += self.bemb

                generated_words.append(max_prob_word)

        return image, generated_words

def get_caption_data(annotation_path, feat_path):
    feats = np.load(feat_path)
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    captions = annotations['caption'].values

    return feats, captions

def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))
    # filtered words from 20326 to 2942.

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


################### 학습 관련 Parameters #####################

dim_embed = 256
dim_hidden = 256
dim_image = 4096
batch_size = 128

#learning_rate = 0.001
n_epochs = 100
###############################################################
#################### 잡다한 Parameters ########################
model_path = './models/tensorflow'
vgg_path = './data/vgg16.tfmodel' # test_tf need!
data_path = './ImageCaption/data'
feat_path = './data/feats.npy'
annotation_path = os.path.join(data_path, 'results_20130124.token')
################################################################


def train():
    learning_rate = 0.001
    momentum = 0.9
    feats, captions = get_caption_data(annotation_path, feat_path)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    np.save('data/ixtoword', ixtoword)

    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]

    sess = tf.Session()
    n_words = len(wordtoix)
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
    caption_generator = Caption_Generator(
            dim_image=dim_image,
            dim_hidden=dim_hidden,
            dim_embed=dim_embed,
            batch_size=batch_size,
            n_lstm_steps=maxlen+2,
            n_words=n_words,
            bias_init_vector=bias_init_vector)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=50)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()

    for epoch in range(n_epochs):
        #train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        for start, end in zip( \
                range(0, len(feats), batch_size),
                range(batch_size, len(feats), batch_size)
                ):

            current_feats = feats[start:end]
            current_captions = captions[start:end]

            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+2, current_caption_matrix ))
            #  +2 -> #START# and '.'

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats,
                sentence : current_caption_matrix,
                mask : current_mask_matrix
                })

            print "Current Cost: ", loss_value

        print "Epoch ", epoch, " is done. Saving the model ... "
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        # learning_rate *= 0.95 # lr do not change!

def read_image(path):
     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[:,:,:3]
     img = img[None, ...]
     return img

def test_multiple(test_file_path=None, model_path='./models/tensorflow/model-72', result_token='./results/results.token', use_flickr=False, maxlen=30): # Naive greedy search
    filepaths = glob.glob(test_file_path + '/*.*')

    result_file = open(result_token, "w")
    result_file.write('================ The predicted captions of test image (%s)  ================\n' % (time.strftime("%c")))

    with open(vgg_path) as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images":images})

    ixtoword = np.load('data/ixtoword.npy').tolist()
    n_words = len(ixtoword)

    sess_fc7 = tf.Session()
    sess = tf.Session() # Two sess. One for fc7, the other one for captions.
    caption_generator = Caption_Generator(
           dim_image=dim_image,
           dim_hidden=dim_hidden,
           dim_embed=dim_embed,
           batch_size=batch_size,
           n_lstm_steps=maxlen,
           n_words=n_words)
    fc7_tf_placeholder, generated_words_placeholder = caption_generator.build_generator(maxlen=maxlen)

    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    if use_flickr:
        index = np.arange(maxlen) # Default length is maxlen.
        np.random.shuffle(index) # Random list.
        sum_BLEU_1=0.0
        sum_BLEU_2=0.0
        sum_BLEU_3=0.0
        sum_BLEU_4=0.0
        for ix in index:
            test_image, reference1, reference2, reference3, reference4, reference5 = get_image_caption(ix)
            image_val = read_image(test_image)
            fc7 = sess_fc7.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={images:image_val})

            generated_word_index = sess.run(generated_words_placeholder, feed_dict={fc7_tf_placeholder:fc7})
            generated_word_index = np.hstack(generated_word_index)

            generated_words = [ixtoword[x] for x in generated_word_index]
            punctuation = np.argmax(np.array(generated_words) == '.')+1

            generated_words = generated_words[:punctuation]
            generated_sentence = ' '.join(generated_words)

            result_file.write("Image is: %s , its caption is: %s"%(test_image, generated_sentence))
            result_file.write('\n')

            print ("Image is: %s , its caption is: %s")%(test_image, generated_sentence)

            generated_sentence = generated_sentence.split()
            reference1 = reference1.split()
            reference2 = reference2.split()
            reference3 = reference3.split()
            reference4 = reference4.split()
            reference5 = reference5.split()

            chencherry = SmoothingFunction() # SmoothingFunction object.smoothing techniques for segment-level BLEU scores.
            # Use method7.
            BLEU_1 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                        generated_sentence, weights=[0.25], smoothing_function=chencherry.method7) # list(str).
            sum_BLEU_1+=BLEU_1
            print("%s, the BLEU-1 Score is: %f"%(test_image, BLEU_1))
            result_file.write("%s, the BLEU-1 Score is: %f"%(test_image, BLEU_1))
            result_file.write('\n')

            BLEU_2 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                        generated_sentence, weights=(0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
            sum_BLEU_2+=BLEU_2
            print("%s, the BLEU-2 Score is: %f"%(" ".rjust(len(test_image)), BLEU_2))
            result_file.write("%s  the BLEU-2 Score is: %f"%(" ".rjust(len(test_image)), BLEU_2))
            result_file.write('\n')

            BLEU_3 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                        generated_sentence, weights=(0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
            sum_BLEU_3+=BLEU_3
            print("%s, the BLEU-3 Score is: %f"%(" ".rjust(len(test_image)), BLEU_3))
            result_file.write("%s  the BLEU-3 Score is: %f"%(" ".rjust(len(test_image)), BLEU_3))
            result_file.write('\n')

            BLEU_4 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                        generated_sentence, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
            sum_BLEU_4+=BLEU_4
            print("%s, the BLEU-4 Score is: %f"%(" ".rjust(len(test_image)), BLEU_4))
            result_file.write("%s  the BLEU-4 Score is: %f"%(" ".rjust(len(test_image)), BLEU_4))
            result_file.write('\n')

        print('\n')
        result_file.write('\n')
        print("The average BLEU-1 Score of %d images is: %f"%(maxlen, sum_BLEU_1/maxlen))
        result_file.write("The average BLEU-1 Score of %d images is: %f"%(maxlen, sum_BLEU_1/maxlen))
        result_file.write('\n')

        print("The average BLEU-2 Score of %d images is: %f"%(maxlen, sum_BLEU_2/maxlen))
        result_file.write("The average BLEU-2 Score of %d images is: %f"%(maxlen, sum_BLEU_2/maxlen))
        result_file.write('\n')

        print("The average BLEU-3 Score of %d images is: %f"%(maxlen, sum_BLEU_3/maxlen))
        result_file.write("The average BLEU-3 Score of %d images is: %f"%(maxlen, sum_BLEU_3/maxlen))
        result_file.write('\n')

        print("The average BLEU-4 Score of %d images is: %f"%(maxlen, sum_BLEU_4/maxlen))
        result_file.write("The average BLEU-4 Score of %d images is: %f"%(maxlen, sum_BLEU_4/maxlen))
        result_file.write('\n')

    else:
        for test_image in filepaths:
            # print(test_image) './image_file/kb.png'
            image_val = read_image(test_image)
            fc7 = sess_fc7.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={images:image_val})

            generated_word_index = sess.run(generated_words_placeholder, feed_dict={fc7_tf_placeholder:fc7})
            generated_word_index = np.hstack(generated_word_index)

            generated_words = [ixtoword[x] for x in generated_word_index]
            punctuation = np.argmax(np.array(generated_words) == '.')+1

            generated_words = generated_words[:punctuation]
            generated_sentence = ' '.join(generated_words)

            result_file.write("Image is: %s , its caption is: %s"%(test_image, generated_sentence))
            result_file.write('\n')

            print ("Image is: %s , its caption is: %s")%(test_image, generated_sentence)

def test_single(test_image_path=None, model_path='./models/tensorflow/model-50', use_flickr=False, maxlen=30):
    # e.g.: ./ImageCaption/images/flickr30k-images/1000092795.jpg for calculating BLEU.
    # use_flickr: Whether use flickr dataset.
    with open(vgg_path) as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images":images})

    ixtoword = np.load('./data/ixtoword.npy').tolist()
    n_words = len(ixtoword)

    if use_flickr:
        index = random.randint(0, 158914) # random.randint(a, b).
        # print(index)
        test_image_path, reference1, reference2, reference3, reference4, reference5 = get_image_caption(index)

    image_val = read_image(test_image_path)
    sess = tf.Session()

    caption_generator = Caption_Generator(
           dim_image=dim_image,
           dim_hidden=dim_hidden,
           dim_embed=dim_embed,
           batch_size=batch_size,
           n_lstm_steps=maxlen,
           n_words=n_words)

    graph = tf.get_default_graph()
    # print(graph)
    fc7 = sess.run(graph.get_tensor_by_name("import/fc7_relu:0"), feed_dict={images:image_val})
    # print(fc7.shape) # ndarray, shape: (1, 4096).

    fc7_tf, generated_words = caption_generator.build_generator(maxlen=maxlen)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index= sess.run(generated_words, feed_dict={fc7_tf:fc7})
    generated_word_index = np.hstack(generated_word_index)

    generated_words = [ixtoword[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    print ("Image is: %s , its caption is: %s")%(test_image_path, generated_sentence)

    if use_flickr:
        generated_sentence = generated_sentence.split()
        reference1 = reference1.split()
        reference2 = reference2.split()
        reference3 = reference3.split()
        reference4 = reference4.split()
        reference5 = reference5.split()

        chencherry = SmoothingFunction() # SmoothingFunction object.smoothing techniques for segment-level BLEU scores.
        # Use method7.
        BLEU_1 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                    generated_sentence, weights=[0.25], smoothing_function=chencherry.method7) # list(str).
        print("%s, the BLEU-1 Score is: %f"%(test_image_path, BLEU_1))

        BLEU_2 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                    generated_sentence, weights=(0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
        print("%s, the BLEU-2 Score is: %f"%(test_image_path, BLEU_2))

        BLEU_3 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                    generated_sentence, weights=(0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
        print("%s, the BLEU-3 Score is: %f"%(test_image_path, BLEU_3))

        BLEU_4 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                    generated_sentence, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
        print("%s, the BLEU-4 Score is: %f"%(test_image_path, BLEU_4))

if __name__=="__main__":
    if FLAGS.phase == 'train':
        train() # Do not use pretrained model.
    elif FLAGS.phase == 'test_multiple':
        test_multiple(test_file_path='./image_file',  model_path=FLAGS.checkpoint_dir, use_flickr=FLAGS.use_flickr,
                      maxlen=FLAGS.maxlen) # Multiple image.
    elif FLAGS.phase == 'test_single':
        test_single(test_image_path="./ImageCaption/images/flickr30k-images/1000092795.jpg", use_flickr=FLAGS.use_flickr,  
                    model_path=FLAGS.checkpoint_dir, maxlen=FLAGS.maxlen) # Single image.
