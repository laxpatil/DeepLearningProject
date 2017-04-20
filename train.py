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
import layers

#######################################################################

train_X, train_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD, 'train', constants.DATA_TYPE_GENDER)
test_X, test_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'test', constants.DATA_TYPE_GENDER)
validation_X, validation_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD, 'validation', constants.DATA_TYPE_GENDER)

#######################################################################

def build_gender_model(model):
    layers.add_layer_1(model)
    layers.add_layer_2(model)
    layers.add_layer_3(model)
    layers.add_dense_layers(model)
    layers.add_objective_layer(model, constants.NUM_LABELS_GENDER)

def lr_schedule(epoch):
    return constants.LEARNING_RATE*(0.1**int(2*epoch/constants.EPOCHS))

def train_gender_model(model, X, Y, val_X, val_Y):
    utilities.remove_folder(constants.FOLDER_NAME_GENDER)
    csv_logger = CSVLogger(constants.LOG_FILE_GENDER)
    build_gender_model(model)
    model.compile(loss='binary_crossentropy',
                  optimizer=constants.OPTIMIZER,
                  metrics=['accuracy'])
    model.fit(X, Y,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS,
              validation_split=0.0,
              validation_data=(val_X,val_Y),
              callbacks=[LearningRateScheduler(lr_schedule), csv_logger]
             )

def test_model(model):
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model = Sequential()
train_gender_model(model, train_X, train_Y, validation_X, validation_Y)
test_model(model)
utilities.save_model(model, constants.MODEL_FILE, constants.WEIGHT_FILE)
