from __future__ import print_function

from keras.models import Sequential
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
    if os.path.exists(path):
         print ("Deleting exisitng folder")
         shutil.rmtree(path)
    os.makedirs(path)
    print ("New folder created")

def load_data(source_path, fold_number, data_type, image_type):
    images_path_file = '{}{}/{}_{}.txt'.format(source_path, fold_number, image_type, data_type)
    actual_images = '../processed_{}_{}/'.format(data_type, fold_number)
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

#######################################################################

train_X, train_Y =  load_data(constants.SOURCE_PATH , constants.FOLD, 'train', constants.DATA_TYPE)
test_X, test_Y =  load_data(constants.SOURCE_PATH, constants.FOLD, 'test', constants.DATA_TYPE)
validation_X, validation_Y =  load_data(constants.SOURCE_PATH, constants.FOLD, 'validation', constants.DATA_TYPE)

#######################################################################

def add_layer_1(model):
    conv1_kernel_init= RandomNormal(stddev=0.01)
    conv1_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 96, kernel_size=(7,7), strides=(4,4), padding="valid",  kernel_initializer=conv1_kernel_init, bias_initializer=conv1_bias_init, input_shape=(227,227,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =(2,2)))
    print("conv1"+str(model.output_shape))

def add_layer_2(model):
    conv2_kernel_init= RandomNormal(stddev=0.01)
    conv2_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 256, kernel_size=(5,5), strides=(1,1), kernel_initializer=conv2_kernel_init, bias_initializer=conv2_bias_init,  padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =2))
    print("conv2"+str(model.output_shape))

def add_layer_3(model):
    conv3_kernel_init= RandomNormal(stddev=0.01)
    conv3_bias_init = Constant(value=0)
    model.add(Conv2D(filters = 384, kernel_size=(3,3),strides=(1,1),  kernel_initializer=conv3_kernel_init, bias_initializer=conv3_bias_init,  padding="valid"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides =2))
    print("conv3"+str(model.output_shape))
    model.add(Flatten())
    print("flatten"+str(model.output_shape))

def add_dense_layers(model):
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

def add_objective_layer(model):
    model.add(Dense(2, activation='softmax'))
    print("softmax"+str(model.output_shape))

def build_model(model):
    add_layer_1(model)
    add_layer_2(model)
    add_layer_3(model)
    add_dense_layers(model)
    add_objective_layer(model)

def lr_schedule(epoch):
    return constants.LEARNING_RATE*(0.1**int(2*epoch/constants.EPOCHS))

def train_model(model):
    csv_logger = CSVLogger(constants.LOG_FILE)
    remove_folder(constants.FOLDER_NAME)
    build_model(model)
    model.compile(loss='binary_crossentropy',
                  optimizer=constants.OPTIMIZER,
                  metrics=['accuracy'])
    model.fit(train_X, train_Y,
              batch_size=constants.BATCH_SIZE,
              epochs=constants.EPOCHS,
              validation_split=0.0,
              validation_data=(validation_X,validation_Y),
              callbacks=[LearningRateScheduler(lr_schedule), csv_logger]
             )

def save_model(model):
    model_json = model.to_json()
    with open(constants.MODEL_FILE, "w") as json_file:
        json_file.write(model_json)
    model.save_weights(constants.WEIGHT_FILE)
    print("Saved model to disk")

def test_model(model):
    scores = model.evaluate(test_X, test_Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

def load_model:
    json_file = open('rmsprop_fold_0_LR_0_0001/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('rmsprop_fold_0_LR_0_0001/model.h5')
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(test_X, test_Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


model = Sequential()
# train_model(model)
# test_model(model)
# save_model(model)
load_model()
