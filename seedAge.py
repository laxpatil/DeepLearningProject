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
import utilities
import layers

def load_model():
    json_file = open('/home/ubuntu/CourseProject/notebook/rmsprop_fold_0_LR_0_0001/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('/home/ubuntu/CourseProject/notebook/rmsprop_fold_0_LR_0_0001/model.h5')
    print("Loaded model from disk")
    return loaded_model

def build_age_model(loaded_model):
    loaded_model.pop()
    layers.add_objective_layer(loaded_model, constants.NUM_LABELS_AGE)

def train_age_model(loaded_model, X, Y, val_X, val_Y):
    utilities.remove_folder(constants.FOLDER_NAME_AGE)
    csv_logger = CSVLogger(constants.LOG_FILE_AGE)
    build_age_model(loaded_model)
    loaded_model.compile(loss='binary_crossentropy',
                  optimizer=constants.OPTIMIZER,
                  metrics=['accuracy'])
    loaded_model.fit(X, Y,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS,
              validation_split=0.0,
              validation_data=(val_X,val_Y),
              callbacks=[LearningRateScheduler(utilities.lr_schedule), csv_logger]
             )

train_X, train_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD, 'train', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
# test_X, test_Y =  load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'test', constants.DATA_TYPE_AGE)
validation_X, validation_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'validation', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)

loaded_model = load_model()
train_age_model(loaded_model, train_X, train_Y, validation_X, validation_Y)
# test_model(model)
utilities.save_model(model, constants.MODEL_FILE_AGE, constants.WEIGHT_FILE_AGE)
