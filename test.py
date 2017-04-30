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
import layers_updated as layers

def build_age_model(model, trainable):
    layers.add_layer_1(model, trainable)
    layers.add_layer_2(model, trainable)
    layers.add_layer_3(model, trainable)
    layers.add_dense_layers(model, trainable)
    layers.add_objective_layer(model, constants.NUM_LABELS_AGE, 'age')

def train_age_model(loaded_model, X, Y, val_X, val_Y):
    utilities.remove_folder(constants.FOLDER_NAME_AGE)
    csv_logger = CSVLogger(constants.LOG_FILE_AGE)
    build_age_model(loaded_model, True)
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer=constants.OPTIMIZER_AGE,
                  metrics=['accuracy'])
    loaded_model.fit(X, Y,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS_AGE,
              validation_split=0.0,
              validation_data=(val_X,val_Y),
              callbacks=[LearningRateScheduler(utilities.lr_schedule_age), csv_logger]
             )

train_X, train_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD_AGE, 'train', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
test_X, test_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'test', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
validation_X, validation_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'validation', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)

loaded_model = Sequential()
train_age_model(loaded_model, train_X, train_Y, validation_X, validation_Y)
utilities.test_model(loaded_model, test_X, test_Y)
utilities.save_model(loaded_model, constants.MODEL_FILE_AGE, constants.WEIGHT_FILE_AGE)
