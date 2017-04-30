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
import utilities

def load_model():
    model_file = constants.MODEL_FILE_GENDER 
    weight_file = constants.WEIGHT_FILE_GENDER
       
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
   
    loaded_model.load_weights(weight_file)
    print("Loaded model from disk\n {}".format(constants.MODEL_FILE_GENDER))
    return loaded_model


def unfreeze_dense_layers(model):
    model.layers[13].trainable=True  #dense layer1
    model.layers[16].trainable=True  #dense layer2

def build_gender_model(model):
    for each_layer in model.layers:
        each_layer.trainable=False

    #unfreeze dense
    unfreeze_dense_layers(model)

    model.layers.pop()
    layers.add_objective_layer(model, constants.NUM_LABELS_AGE, 'age')

def train_age_model(model, X, Y, val_X, val_Y):
    utilities.remove_folder(constants.FOLDER_NAME_AGE)
    csv_logger = CSVLogger(constants.LOG_FILE_AGE)
    build_gender_model(model)
    model.compile(loss='categorical_crossentropy',
                  optimizer=constants.OPTIMIZER_AGE,
                  metrics=['accuracy'])
    model.fit(X, Y,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS_AGE,
              validation_split=0.0,
              validation_data=(val_X,val_Y),
              callbacks=[LearningRateScheduler(utilities.lr_schedule), csv_logger]
             )

def test_model(model):
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#######################################################################

model = load_model()

#######################################################################
train_X, train_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD_AGE, 'train', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
test_X, test_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'test', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
validation_X, validation_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'validation', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)

#######################################################################




train_age_model(model, train_X, train_Y, validation_X, validation_Y)
test_model(model)
utilities.save_model(model, constants.MODEL_FILE_AGE, constants.WEIGHT_FILE_AGE)
