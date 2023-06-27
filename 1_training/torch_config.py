import torch
import os

DATASET_PATH = 'C:\\Users\\Пользователь\\Desktop\\SegRatBrainCells\\Init_All dataset\\StainData'

IMAGE_PATH = os.path.join(DATASET_PATH, 'PP_train_images_all')
MASK_PATH = os.path.join(DATASET_PATH, 'PP_train_masks_all')

TEST_SPLIT = 0.20

BATCH_SIZE = 32
NUM_EPOCHS = 300  # 300
EPOCHS_NO_IMPROVE = 10

SEED_COUNTER = 1

CNN_DEPTH = [4]  # [3, 4, 5, 6, 7, 8, 9]
FIRST_CHANNELS = [32]  # [32, 64]
BATCH_NORM = [True]
DROP_OUT = [True]

OPTIMIZER = 'Adam'  # SGD, RMSprop
AMSGRAD = [False]  # [False, True]
LR = [0.001]  # [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
MOMENT = [(0.9, 0.999)]  # [(0.9, 0.99), (0.9, 0.999), (0.9, 0.9999)]
WEIGHT_DECAY = [0]  # [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

IS_SAVE = True
BASE_OUTPUT = 'output'
MODEL_NAME = 'unet_Adam_all_data.pth'
PATH_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT)
MODEL_OUTPUT = os.path.join(DATASET_PATH, BASE_OUTPUT, MODEL_NAME)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

