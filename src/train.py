import config
from dataset import MRIDataset
from model import Critic, DeepMedic
from engine import train_one_epoch
from engine import validate_one_epoch

############ REMOVE #############
#from engine import train_one_step
#################################

import os
import torch
import numpy as np

from math import floor
from torch.optim import lr_scheduler


def train_validation_split(file_list):
    split = 0.7
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

def train():
    ###################### STAGE 1 ######################
    # Fetch train image ids
    fnames = os.listdir(f"{config.MASK_PATCH_PATH}/train/")
    # Lists with image locations
    high_resolution_patch_paths = list(map(lambda fname: f"{config.HIGH_RESOLUTION_PATCH_PATH}/train/{fname}", fnames))
    low_resolution_patch_paths = list(map(lambda fname: f"{config.LOW_RESOLUTION_PATCH_PATH}/train/{fname}", fnames))
    mask_patch_paths = list(map(lambda fname: f"{config.MASK_PATCH_PATH}/train/{fname}", fnames))
    image_patch_paths = list(map(lambda fname: f"{config.IMAGE_PATCH_PATH}/train/{fname}", fnames))

    # Split train and validation images (filenames)
    # -------------------- Add index to path_zip_list for prototyping --------------------
    path_zip_list = list(zip(
        high_resolution_patch_paths,
        low_resolution_patch_paths,
        mask_patch_paths,
        image_patch_paths
    ))
    train_paths, valid_paths = train_validation_split(path_zip_list)
    
    # Create train and validation file locations
    high_resolution_patch_paths_train = []
    low_resolution_patch_paths_train = []
    mask_patch_paths_train = []
    image_patch_paths_train = []
    for file in train_paths:
        high_resolution_patch_paths_train.append(file[0])
        low_resolution_patch_paths_train.append(file[1])
        mask_patch_paths_train.append(file[2])
        image_patch_paths_train.append(file[3])

    high_resolution_patch_paths_valid = []
    low_resolution_patch_paths_valid = []
    mask_patch_paths_valid = []
    image_patch_paths_valid = []
    for file in valid_paths:
        high_resolution_patch_paths_valid.append(file[0])
        low_resolution_patch_paths_valid.append(file[1])
        mask_patch_paths_valid.append(file[2])
        image_patch_paths_valid.append(file[3])

    ###################### STAGE 2 ######################
    # Train data loader
    filepath_dict = {
        "high_resolution": high_resolution_patch_paths_train,
        "low_resolution": low_resolution_patch_paths_train,
        "mask": mask_patch_paths_train,
        "image": image_patch_paths_train
    }
    train_dataset = MRIDataset(filepath_dict)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE
    )

    # Validation data loader
    filepath_dict = {
        "high_resolution": high_resolution_patch_paths_valid,
        "low_resolution": low_resolution_patch_paths_valid,
        "mask": mask_patch_paths_valid,
        "image": image_patch_paths_valid
    }
    valid_dataset = MRIDataset(filepath_dict)

    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = config.VALID_BATCH_SIZE
    )
    ###################### STAGE 3 ######################
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_s = DeepMedic().to(device)
    optimizer_s = torch.optim.RMSprop(model_s.parameters(), lr=2e-4, weight_decay=0)
    scheduler_s = lr_scheduler.ReduceLROnPlateau(optimizer_s, mode="min", patience=1, verbose=True)

    model_c = Critic().to(device)
    optimizer_c = torch.optim.RMSprop(model_s.parameters(), lr=2e-4, weight_decay=0)

    ###################### STAGE 4 ######################
    directory = "../models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_loss = np.inf
    waiting = 0

    training_l1_loss = []
    training_dice_loss = []
    validation_l1_loss = []
    #validation_dice_loss = []
    for epoch in range(config.EPOCHS):
        
        train_l1_loss, train_dice_loss = train_one_epoch(model_s, optimizer_s, model_c, optimizer_c, train_data_loader, device)
        training_l1_loss.append(train_l1_loss)
        training_dice_loss.append(train_dice_loss)

        valid_l1_loss = validate_one_epoch(model_s, model_c, valid_data_loader, device)
        validation_l1_loss.append(valid_l1_loss)

        scheduler_s.step(validation_l1_loss[-1])

        print("======="*20)
        print(f"Epoch: {epoch+1}")
        print(f"\tTrain loss: {train_l1_loss}")
        print(f"\tValid loss: {valid_l1_loss}")
        print("======="*20)
        print("\n\n")

        if valid_l1_loss<best_loss:
            best_loss = valid_l1_loss
            torch.save(model_s.state_dict(), config.MODEL_FILE)
            waiting = 0
        
        else:
            waiting+=1
            if waiting==config.MAX_WAITING:
                with open(f"{config.TRAIN_LOSS_FILE}", "wb") as file:
                    np.save(file, training_l1_loss)

                with open(f"{config.VALID_LOSS_FILE}", "wb") as file:
                    np.save(file, validation_l1_loss)

                print("======"*20)
                print(f"Best Loss: {round(best_loss, 5)}")
                print("======"*20)
                break

    return None

if __name__=="__main__":
    train()