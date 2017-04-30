'''
Copyright 2017 Aarav Madan, Laxmikant Patil

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
#This file sets the configuration of the system.

IMG_SIZE = 227
FOLD_GENDER = "0"
FOLD_AGE = "0"
DATA_TYPE_GENDER = "gender"
DATA_TYPE_AGE = "age"
HOME_PATH = "../data"
SOURCE_FOLDER = "../data/train_test_fold_"
LEARNING_RATE=0.0001
LEARNING_RATE_AGE=0.00000001
BATCH_SIZE = 50
EPOCHS_AGE = 50
EPOCHS_GENDER = 25
OPTIMIZER_GENDER = 'rmsprop'
OPTIMIZER_AGE = 'rmsprop'


FOLDER_NAME_GENDER = 'TYPE_{}_LR_{}_OPTIMIZER_{}_FOLD_{}'.format(DATA_TYPE_GENDER, LEARNING_RATE, OPTIMIZER_GENDER, FOLD_GENDER)
LOG_FILE_GENDER = '{}/status.log'.format(FOLDER_NAME_GENDER)
MODEL_FILE_GENDER = '{}/model.json'.format(FOLDER_NAME_GENDER)
WEIGHT_FILE_GENDER = '{}/weights.h5'.format(FOLDER_NAME_GENDER)
NUM_LABELS_GENDER = 2

FOLDER_NAME_AGE = 'TYPE_{}_LR_{}_OPTIMIZER_{}_FOLD_{}'.format(DATA_TYPE_AGE, LEARNING_RATE, OPTIMIZER_AGE, FOLD_AGE)
LOG_FILE_AGE = '{}/status.log'.format(FOLDER_NAME_AGE)
MODEL_FILE_AGE = '{}/model.json'.format(FOLDER_NAME_AGE)
WEIGHT_FILE_AGE = '{}/weights.h5'.format(FOLDER_NAME_AGE)
NUM_LABELS_AGE = 8
