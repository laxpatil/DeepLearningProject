from __future__ import print_function

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization

import numpy as np
from skimage import color, exposure, transform
NUM_CLASSES = 2
IMG_SIZE = 227
def preprocess_img(img):
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


from skimage import io
import os
import glob
for i in range(0,1):
    save_path = "../processed_train_age" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/age_train.txt'
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
    f = open(save_path+"labels_age_train.txt", 'w+')
    f.write(str(labels))

    save_path = "../processed_test_age" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/age_test.txt'
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
    f = open(save_path+"labels_age_test.txt", 'w+')
    f.write(str(labels))

    save_path = "../processed_validation_age" + str(i) + "/"
    images_path_file = '../AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_' + str(i) + '/age_val.txt'
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
    f = open(save_path+"labels_age_validation.txt", 'w+')
    f.write(str(labels))

