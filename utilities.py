from __future__ import print_function

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal, Constant
from keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint

import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
import glob
import shutil

def remove_folder(path):
    if os.path.exists(path):
         print ("Deleting exisitng folder")
         shutil.rmtree(path)
    os.makedirs(path)
    print ("New folder created")

def save_model(model, model_file, weight_file):
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight_file)
    print("Saved model to disk")

def load_data(home_folder, source_folder, fold_number, data_type, image_type):
    images_path_file = '{}/{}{}/{}_{}.txt'.format(home_folder, source_folder, fold_number, image_type, data_type)
    actual_images = '{}/processed_{}_{}/'.format(home_folder, data_type, fold_number)
    load_message = 'Loading {} data\n#############\n'.format(data_type)

    data=[]
    labels=[]

    all_img_paths = open(images_path_file)
    print (load_message)
    for i, img_path in enumerate(all_img_paths):
        if(i%1000 == 0):
            print ("Loaded " + str(i) + " images")

        path = img_path.strip('\n').split(' ')[0]
        label = int(img_path.strip('\n').split(' ')[1])
        img = io.imread(actual_images+path)
        data.append(img)
        labels.append(label)

    X = np.array(data, dtype='float32')
    # Make one hot targets
    Y = np.eye(constants.NUM_CLASSES, dtype='uint8')[labels]
    return X, Y
