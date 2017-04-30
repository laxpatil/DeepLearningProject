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

def remove_folder(path):
    """
    This function removes the folder to store all the parmeters learnt if it exists already. It then creates the folder to store the learnt model, log and weights learnt.
    Code reference: Charles Chow, How do I remove/delete a folder that is not empty with Python?, http://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty-with-python
    Args:
        path: A system path of the folder to delete/create
                          
    """
    if os.path.exists(path):
         print ("Deleting exisitng folder")
         shutil.rmtree(path)
    os.makedirs(path)
    print ("New folder created")

def save_model(model, model_file, weight_file):
    """
    This function saves the model descrition as JSON and weights learnt as h5 file.
    Code reference: Jason Browniee, Save and Load Your Keras Deep Learning Models, http://machinelearningmastery.com/save-load-keras-deep-learning-models/
    Args:
        model: The Keras model object which is to be saved .
        model_file: A string containing the model file name and its path to store at.
        weight_file: A string containing the weight file name and its path to store at.
                          
    """
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(weight_file)
    print("Saved model to disk")

def load_data(home_folder, source_folder, fold_number, data_type, image_type, num_labels):
    """
    This fucntion loads the training, test and validation data and prepares it for processing.
    Args:
        home_folder: String to build load path for data
        source_folder: String specifying location of data
        fold_number: Integer stating the fold of data to load
        image_type: String stating if the data is for gender classification or age classification task.
        num_labels: Integer stating the number of labels for the classification task
                          
    """
    images_path_file = '{}/{}{}/{}_{}.txt'.format(home_folder, source_folder, fold_number, image_type, data_type)
    actual_images = '{}/processed_{}_{}_{}/'.format(home_folder, data_type, image_type, fold_number)
    load_message = '\n Loading {} data from \n image path: {} \n actual image: {}\n#############\n'.format(data_type,images_path_file, actual_images)

    data=[]
    labels=[]

    all_img_paths = open(images_path_file)
    print (load_message)
    for i, img_path in enumerate(all_img_paths):

        path = img_path.strip('\n').split(' ')[0]
        label = int(img_path.strip('\n').split(' ')[1])
        img = io.imread(actual_images+path)
        data.append(img)
        labels.append(label)

        if(i%1000 == 0):
            print ("Loaded " + str(i) + " images")

    X = np.array(data, dtype='float32')
    # Make one hot targets
    Y = np.eye(num_labels, dtype='uint8')[labels]
    return X, Y

def lr_schedule(epoch):
    """
    This function does the update of learning rate for gender model.
    Args:
        epoch: Integer stating the current epoch number
                          
    """
    return constants.LEARNING_RATE*(0.1**int(2*epoch/constants.EPOCHS_GENDER))

def lr_schedule_age(epoch):
    """
    This function does the update of learning rate for age model.
    Args:
        epoch: Integer stating the current epoch number
                          
    """
    return constants.LEARNING_RATE_AGE*(0.1**int(2*epoch/constants.EPOCHS_AGE))

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


