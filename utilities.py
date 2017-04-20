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
