'''
Copyright 2017 Aarav Madan, Laxmikant Patil

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
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

import train_base_gender

def load_model():
    """
    This function loads the gender model and loads it with the learnt weights
    Returns:
        Keras.model: A keras model object that has been initialized with weights.
                          
    """
    model_file = constants.MODEL_FILE_GENDER 
    weight_file = constants.WEIGHT_FILE_GENDER
       
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
   
    loaded_model.load_weights(weight_file)
    print("Loaded model from disk\n {}".format(constants.MODEL_FILE_GENDER))
    return loaded_model

def build_age_model(model):
    """
    This function replaces the gender softmax layer with age softmax layer
    Args:
        model: A keras model object to be fine tuned.
                          
    """    
    model.layers.pop()
    layers.add_objective_layer(model, constants.NUM_LABELS_AGE, 'age')

def train_age_model(model, X, Y, val_X, val_Y):
    """ 
    This function trains the age classification model
    Args:
        model: A keras model object to be trained.
        X: A numpy.ndarray of training data
        Y: A numpy.ndarray of training data labels
        val_X: A numpy.ndarray of validation data
        val_Y: A numpy.ndarray of validation data labels
                          
    """    
    utilities.remove_folder(constants.FOLDER_NAME_AGE)
    csv_logger = CSVLogger(constants.LOG_FILE_AGE)
    build_age_model(model)
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

def test_model(model, test_X, test_Y):
    """
    This function checks accuracy of the model on the supplied test data.
    Args:
        model: The Keras model object which is to be tested.
        test_X: Numpy array of test data 
        test_Y: Numpy array of test data ground truth        
                          
    """    
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def train_model():
    """ 
    This function does run time profiling and wraps around the function for model training. It loads the train, test and validation data, trains the age models and saves down parameters.
    
    Returns:
        Keras.model: Returns the learnt model                      
    
    """
    model= train_base_gender.train_model()
    
    #####################
    
    import time
    start = time.clock()

    ######################    
    
    #######################################################################
    
    train_X, train_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER , constants.FOLD_AGE, 'train', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
    test_X, test_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'test', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)
    validation_X, validation_Y =  utilities.load_data(constants.HOME_PATH, constants.SOURCE_FOLDER, constants.FOLD_AGE, 'validation', constants.DATA_TYPE_AGE, constants.NUM_LABELS_AGE)

    #######################################################################
    
    train_age_model(model, train_X, train_Y, validation_X, validation_Y)
    test_model(model, test_X, test_Y)
    utilities.save_model(model, constants.MODEL_FILE_AGE, constants.WEIGHT_FILE_AGE)
    
    #############################

    total_time = time.clock() - start

    print("\n##################################################################")
    print("\n\nTotal Exceution time for this section : "  + str(total_time) )
    print("\n##################################################################")
    
    #############################

    return model

if __name__ == "__main__":
    train_model()

