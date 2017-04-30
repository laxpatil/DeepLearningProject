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
from keras.layers.normalization import BatchNormalization

import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
import glob
import shutil
import constants

def add_layer_1(model):
    """
    This function adds the first layer in the network. The intial parameters are obtained from the specified distribution. Comprises of ConvPool layer with RelU activation and BatchNorm
    Args:
        model: A keras model to which layers are to be added 
                          
    """    
    conv1_kernel_init= RandomNormal(stddev=0.01)
    conv1_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 96, kernel_size=(7,7), strides=(4,4), padding="valid",  kernel_initializer=conv1_kernel_init, bias_initializer=conv1_bias_init, input_shape=(227,227,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =(2,2)))
    print("conv1"+str(model.output_shape))

def add_layer_2(model):
    """
    This function adds the second layer in the network. The intial parameters are obtained from the specified distribution. Comprises of ConvPool layer with RelU activation and BatchNorm
    Args:
        model: A keras model to which layers are to be added 
                          
    """
    conv2_kernel_init= RandomNormal(stddev=0.01)
    conv2_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 256, kernel_size=(5,5), strides=(1,1), kernel_initializer=conv2_kernel_init, bias_initializer=conv2_bias_init,  padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =2))
    print("conv2"+str(model.output_shape))

def add_layer_3(model):
    """
   This function adds the third layer in the network. The intial parameters are obtained from the specified distribution. Comprises of ConvPool layer with RelU activation followed by Flattening of 2D layer to 1D layer.
    Args:
        model: A keras model to which layers are to be added 
                          
    """
    conv3_kernel_init= RandomNormal(stddev=0.01)
    conv3_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 384, kernel_size=(3,3),strides=(1,1),  kernel_initializer=conv3_kernel_init, bias_initializer=conv3_bias_init,  padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =2))
    print("conv3"+str(model.output_shape))
    model.add(Flatten())
    print("flatten"+str(model.output_shape))

def add_dense_layers(model):
    """
    This function adds the fourth and fifth layers in the network. The intial parameters are obtained from the specified distribution. Comprises of Dense layer with RelU activation and Dropout
    Args:
        model: A keras model to which layers are to be added 
                          
    """
    dense_kernel_init= RandomNormal(stddev=0.005)
    dense_bias_init = Constant(value=1)
    model.add(Dense(units= 512, kernel_initializer=dense_kernel_init, bias_initializer=dense_bias_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    print("dense1"+str(model.output_shape))

    model.add(Dense(units= 512, kernel_initializer=dense_kernel_init, bias_initializer=dense_bias_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    print("dens2"+str(model.output_shape))

def add_objective_layer(model, num_labels, name):
    """
    This function adds the final softmax layer in the network with number of nodes depending on the classification task being performed.
    Args:
        model: A keras model to which layers are to be added 
                          
    """
    model.add(Dense(num_labels, activation='softmax', name = name))
    print("softmax"+str(model.output_shape))
