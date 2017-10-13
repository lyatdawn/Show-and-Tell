#coding=utf-8
import pandas as pd
import numpy as np
import os
import cPickle

class switch(object):
    def __init__(self, value):   # init value.
        self.value = value
        self.fall = False        # no break, then fall=False.  
    def __iter__(self):
        yield self.match         # match method to create.
        raise StopIteration      # exception to check loop.
    def match(self, *args):
        if self.fall or not args:  
            return True
        elif self.value in args: # successful.
            self.fall = True
            return True  
        else:                    # fail.
            return False

annotation_path = './ImageCaption/data/results_20130124.token'
flickr_image_path = './ImageCaption/images/flickr30k-images/'

def get_image_caption(index=None):
	annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])

	test_image_path = os.path.join(flickr_image_path, annotations['image'][index].split('#')[0])
	caption_index = annotations['image'][index].split('#')[1]
	# print(caption_index)

	if index==0:
		reference1 = annotations['caption'][index]
		reference2 = annotations['caption'][index + 1]
		reference3 = annotations['caption'][index + 2]
		reference4 = annotations['caption'][index + 3]
		reference5 = annotations['caption'][index + 4]
	elif index==1:
		reference1 = annotations['caption'][index - 1]
		reference2 = annotations['caption'][index]
		reference3 = annotations['caption'][index + 1]
		reference4 = annotations['caption'][index + 2]
		reference5 = annotations['caption'][index + 3]
	elif index==2:
		reference1 = annotations['caption'][index - 2]
		reference2 = annotations['caption'][index - 1]
		reference3 = annotations['caption'][index]
		reference4 = annotations['caption'][index + 1]
		reference5 = annotations['caption'][index + 2]
	elif index==3:
		reference1 = annotations['caption'][index - 3]
		reference2 = annotations['caption'][index - 2]
		reference3 = annotations['caption'][index - 1]
		reference4 = annotations['caption'][index]
		reference5 = annotations['caption'][index + 1]
	else: 
		for case in switch(str(caption_index)):
			if case('0'):  
				# print('caption_index==0')
				reference1 = annotations['caption'][index]
				reference2 = annotations['caption'][index + 1]
				reference3 = annotations['caption'][index + 2]
				reference4 = annotations['caption'][index + 3]
				reference5 = annotations['caption'][index + 4] 
				break  
			if case('1'):  
				# print('caption_index==1')
				reference1 = annotations['caption'][index - 1]
				reference2 = annotations['caption'][index]
				reference3 = annotations['caption'][index + 1]
				reference4 = annotations['caption'][index + 2]
				reference5 = annotations['caption'][index + 3]
				break  
			if case('2'):  
				# print('caption_index==2')
				reference1 = annotations['caption'][index - 2]
				reference2 = annotations['caption'][index - 1]
				reference3 = annotations['caption'][index]
				reference4 = annotations['caption'][index + 1]
				reference5 = annotations['caption'][index + 2]
				break  
			if case('3'):  
				# print('caption_index==3')
				reference1 = annotations['caption'][index - 3]
				reference2 = annotations['caption'][index - 2]
				reference3 = annotations['caption'][index - 1]
				reference4 = annotations['caption'][index]
				reference5 = annotations['caption'][index + 1]
				break  
			if case('4'):  
				# print('caption_index==4')
				reference1 = annotations['caption'][index - 4]
				reference2 = annotations['caption'][index - 3]
				reference3 = annotations['caption'][index - 2]
				reference4 = annotations['caption'][index - 1]
				reference5 = annotations['caption'][index]
	return test_image_path, reference1, reference2, reference3, reference4, reference5