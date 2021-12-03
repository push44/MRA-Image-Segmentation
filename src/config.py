import os

############## To create patches ##############
DIR_PATH = "../"
#DIR_PATH = "/media/push44/PushDrive/MRIData"

FILENAMES_PATH = f"{DIR_PATH}/MRIData"

INPUT_PATH = f"{DIR_PATH}/image"
MASK_PATH = f"{DIR_PATH}/mask"

MASK_PATCH_PATH = f"{DIR_PATH}/patch/mask"
IMAGE_PATCH_PATH = f"{DIR_PATH}/patch/image"
HIGH_RESOLUTION_PATCH_PATH = f"{DIR_PATH}/patch/high_resolution"
LOW_RESOLUTION_PATCH_PATH = f"{DIR_PATH}/patch/low_resolution"


MASK_PATCH_SIZE = 24
HIGH_RESOLUTION_PATCH_SIZE = 40
LOW_RESOLUTION_CROP_SIZE = 64
LOW_RESOLUTION_PATCH_SIZE = 22

# Number of patches per image
NUM_PATCHES = 1000
# Proportion of positve patches per image
BALANCE = 0.5

OUTPUT_SHAPE = 24
############## For training ##############

##### Train test split #####
TEST_PORTION = 0.3

TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1

EPOCHS = 5
MAX_WAITING = 10

MODEL_FILE = "../models/model.bin"
TRAIN_LOSS_FILE = "../models/train_loss.npy"
VALID_LOSS_FILE = "../models/valid_loss.npy"