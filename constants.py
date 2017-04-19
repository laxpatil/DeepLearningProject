NUM_CLASSES = 2
IMG_SIZE = 227
FOLD = "0"
DATA_TYPE_GENDER = "gender"
DATA_TYPE_AGE = "age"
HOME_PATH = "/home/ubuntu/CourseProject"
SOURCE_FOLDER = "AgeGenderDeepLearning/Folds/train_val_txt_files_per_fold/test_fold_is_"
LEARNING_RATE=0.0001
EPOCHS = 2
BATCH_SIZE = 50
OPTIMIZER = 'rmsprop'
FOLDER_NAME = 'TYPE_{}_LR_{}_OPTIMIZER_{}_FOLD_{}'.format(DATA_TYPE, LEARNING_RATE, OPTIMIZER, FOLD)
LOG_FILE = '{}/status.log'.format(FOLDER_NAME)
MODEL_FILE = '{}/model.json'.format(FOLDER_NAME)
WEIGHT_FILE = '{}/weights.h5'.format(FOLDER_NAME)
