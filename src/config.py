############## To create patches ##############
INPUT_PATH = "/media/push44/PushDrive/MRIData/image/"
MASK_PATH = "/media/push44/PushDrive/MRIData/mask/"

MASK_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/mask/"
HIGH_RESOLUTION_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/high_resolution/"
LOW_RESOLUTION_PATCH_PATH = "/media/push44/PushDrive/MRIData/patch/low_resolution/"

MASK_PATCH_SIZE = 24
HIGH_RESOLUTION_PATCH_SIZE = 40
LOW_RESOLUTION_CROP_SIZE = 64
LOW_RESOLUTION_PATCH_SIZE = 22

SAMPLES_PER_IMAGE = 1000
BALANCE = 0.5

############## For training ##############
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4

EPOCHS = 1

MODEL_FILE = "../models/model.bin"