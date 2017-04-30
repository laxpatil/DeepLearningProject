'''
Copyright 2017 Aarav Madan, Laxmikant Patil

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from __future__ import print_function

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization

import numpy as np
from skimage import color, exposure, transform
NUM_CLASSES = 2
IMG_SIZE = 227


##############################################################################################
def preprocess_img(img):


    """
    This function preprocesses image that is given as input
    Returns:
        image : Returns transformed image which is center cropped
                          
    """
    
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central square crop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img,0)

    return img

#################################################################################################


'''
Following part of code takes the input images and GENDER labels from original data folder and preprocesses those
images and stores in given output folder

'''
from skimage import io
import os
import glob
for i in range(1,5):
    save_path = "../processed_train_" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/gender_train.txt'
    actual_images='../aligned/aligned/'
    imgs=[]
    labels=[]
    all_img_paths = open(images_path_file)
    for img_path in all_img_paths:
        print ('Processing train image: ' + str(img_path))
        path = img_path.strip('\n').split(' ')[0]
        label = int(img_path.strip('\n').split(' ')[1])
        labels.append(label)
        img = preprocess_img(io.imread(actual_images+path))
        filename = save_path+path
        try:
 	    os.makedirs(os.path.dirname(filename))
        except:
	    pass
        io.imsave(filename,img)
    f = open(save_path+"labels_gender_train.txt", 'w+')
    f.write(str(labels))

    save_path = "../processed_test_" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/gender_test.txt'
    actual_images='../aligned/aligned/'
    imgs=[]
    labels=[]
    all_img_paths = open(images_path_file)
    for img_path in all_img_paths:
        print ('Processing test image: ' + str(img_path))
        path = img_path.strip('\n').split(' ')[0]
        label = int(img_path.strip('\n').split(' ')[1])
        labels.append(label)
        img = preprocess_img(io.imread(actual_images+path))
        filename = save_path+path
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            pass
        io.imsave(filename,img)
    f = open(save_path+"labels_gender_test.txt", 'w+')
    f.write(str(labels))

    save_path = "../processed_validation_" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/gender_val.txt'
    actual_images='../aligned/aligned/'
    imgs=[]
    labels=[]
    all_img_paths = open(images_path_file)
    for img_path in all_img_paths:
        print ('Processing validation image: ' + str(img_path))
        path = img_path.strip('\n').split(' ')[0]
        label = int(img_path.strip('\n').split(' ')[1])
        labels.append(label)
        img = preprocess_img(io.imread(actual_images+path))
        filename = save_path+path
        try:
            os.makedirs(os.path.dirname(filename))
        except:
            pass
        io.imsave(filename,img)
    f = open(save_path+"labels_gender_validation.txt", 'w+')
    f.write(str(labels))

