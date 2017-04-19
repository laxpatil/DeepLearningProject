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
import constants

def load_model():
    json_file = open('/home/ubuntu/CourseProject/notebook/rmsprop_fold_0_LR_0_0001/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('/home/ubuntu/CourseProject/notebook/rmsprop_fold_0_LR_0_0001/model.h5')
    print("Loaded model from disk")


train_X, train_Y =  load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD, 'train', constants.DATA_TYPE_AGE)
# test_X, test_Y =  load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'test', constants.DATA_TYPE_AGE)
# validation_X, validation_Y =  load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'validation', constants.DATA_TYPE_AGE)
