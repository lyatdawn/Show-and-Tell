# Neural Caption Generator
* Tensorflow implementation of "Show and Tell"  in the paper: http://arxiv.org/abs/1411.4555. The Show and Tell model is a deep neural network that learns how to describe the content of images.
* Borrowed code and ideas from jazzsaxmafia's show_and_tell.tensorflow: https://github.com/jazzsaxmafia/show_and_tell.tensorflow.
There are some modifications in model.py, see Code for details.
* You need flickr30k data (images and annotations). You can put those in `ImageCaption/data` and `ImageCaption/images` folder respectively.

## Install Required Packages
First ensure that you have installed the following required packages:
* TensorFlow0.10.0rc0 ([instructions](https://www.tensorflow.org/install/))
* Caffe ([instructions](http://caffe.berkeleyvision.org/installation.html))
* Keras1.2.1 ([instructions](https://keras.io/))
* Natural Language Toolkit (NLTK):
    * First install NLTK ([instructions](http://www.nltk.org/install.html))
    * Then install the NLTK data ([instructions](http://www.nltk.org/data.html))

See requirements.txt for details.
 
## Code
* make_flickr_dataset.py : Extracting feats of flickr30k images, and save them in './data/feats.npy'.
  * First, you shoule download the caffemodel and deploy.prototxt of VGG19. You can download those from here.
* model.py : TensorFlow Version. **There are some modifications in model.py**:
  * Add some command arguments, run more convenient.
  * The **test_single()** in model.py is for a single image. If use_flickr=False, it just generate the caption of a image; If use_flickr=True, it will randomly pick a image and respective five reference captions from flickr30k dataset, generate the caption and calculate the BLEU Score.
  * The **test_multiple()** in model.py is for multiple images. If use_flickr=False, it just generate the captions of some images; If use_flickr=True, it will randomly pick some images and respective five reference captions from flickr30k dataset, generate the captions and calculate the BLEU Scores.
 
## Getting Started
* Training a Model
Run the training script.
```shell
python model.py --phase train
```
The checkpoint data will be stored in the model/tensorflow folder periodically.
* Generating Captions and/or not Calculate BLEU Scores
Your trained Show and Tell model can generate captions for any JPEG/PNG image! The following command line will generate captions for an image or some images.
```shell
python model.py --phase test_single --use_flickr False
python model.py --phase test_single --use_flickr True
# The script will generate the caption and/or not calculate the BLEU Score.
python model.py --phase test_multiple --use_flickr False
python model.py --phase test_multiple --use_flickr True
# The script will generate the captions and/or not calculate the BLEU Scores.
```

## Downloading data/trained model
* You might want to download flickr30k dataset(images and annotations) from [here](http://web.engr.illinois.edu/~bplumme2/Flickr30kEntities/).
* Extraced FC7 data: [download](https://drive.google.com/file/d/0B5o40yxdA9PqTnJuWGVkcFlqcG8/view?usp=sharing). 
This is used in train() function in model.py. You can skip feature extraction part by using this.
* Pretrained model: [download](https://drive.google.com/drive/folders/0Bzr5eCe3vusWUDIyTkdEc2I1R0k?usp=sharing). 
This is used in test_single() and test_multiple() in model.py. If you just want to check out captioning, download and test the model.
* Tensorflow VGG net: [download](https://drive.google.com/file/d/0B5o40yxdA9PqSGtVODN0UUlaWTg/view?usp=sharing). 
This file is used in test_single() and test_multiple() in model.py.

## License
* BSD license
