import os

############## To create patches ##############
INPUT_PATH = "/media/push44/PushDrive/MRIData/image"
MASK_PATH = "/media/push44/PushDrive/MRIData/mask"

MASK_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/mask"
HIGH_RESOLUTION_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/high_resolution"
LOW_RESOLUTION_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/low_resolution"


"""INPUT_PATH = "../MRIData/image"
MASK_PATH = "../MRIData/mask"

MASK_PATCH_PATH = "../MRIData/patch/mask"
HIGH_RESOLUTION_PATCH_PATH = "../MRIData/patch/high_resolution"
LOW_RESOLUTION_PATCH_PATH = "../MRIData/patch/low_resolution" """

MASK_PATCH_SIZE = 24
HIGH_RESOLUTION_PATCH_SIZE = 40
LOW_RESOLUTION_CROP_SIZE = 64
LOW_RESOLUTION_PATCH_SIZE = 22

# Number of patches per image
NUM_PATCHES = 1000
# Proportion of positve patches per image
BALANCE = 0.5

############## For training ##############

##### Train test split #####
TEST_PORTION = 0.3

TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2

EPOCHS = 20
MAX_WAITING = 5

MODEL_FILE = "../models/model.bin"
TRAIN_LOSS_FILE = "../models/train_loss.npy"
VALID_LOSS_FILE = "../models/valid_loss.npy"