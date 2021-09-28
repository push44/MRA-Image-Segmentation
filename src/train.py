import config
from dataset import MRIDataset
from model import DeepMedic
from engine import train_one_epoch
from engine import validate_one_epoch

import os
import torch
import numpy as np

from math import floor


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
    high_res_paths = list(map(lambda fname: f"{config.HIGH_RESOLUTION_PATCH_PATH}/train/{fname}", fnames))
    low_res_paths = list(map(lambda fname: f"{config.LOW_RESOLUTION_PATCH_PATH}/train/{fname}", fnames))
    mask_paths = list(map(lambda fname: f"{config.MASK_PATCH_PATH}/train/{fname}", fnames))    


    # Split train and validation images (filenames)
    # -------------------- Add index to path_zip_list for prototyping --------------------
    path_zip_list = list(zip(high_res_paths, low_res_paths, mask_paths))
    train_paths, valid_paths = train_validation_split(path_zip_list)
    
    # Create train and validation file locations
    high_res_train_paths = []
    low_res_train_paths = []
    mask_train_paths = []
    for file in train_paths:
        high_res_train_paths.append(file[0])
        low_res_train_paths.append(file[1])
        mask_train_paths.append(file[2])

    high_res_valid_paths = []
    low_res_valid_paths = []
    mask_valid_paths = []
    for file in valid_paths:
        high_res_valid_paths.append(file[0])
        low_res_valid_paths.append(file[1])
        mask_valid_paths.append(file[2])

    ###################### STAGE 2 ######################
    # Train data loader
    train_dataset = MRIDataset(high_res_train_paths, low_res_train_paths, mask_train_paths)

    train_data_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = config.TRAIN_BATCH_SIZE
    )

    # Validation data loader
    valid_dataset = MRIDataset(high_res_valid_paths, low_res_valid_paths, mask_valid_paths)

    valid_data_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = config.VALID_BATCH_SIZE
    )
    ###################### STAGE 3 ######################

    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepMedic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    ###################### STAGE 4 ######################
    directory = "../models/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_loss = np.inf
    waiting = 0

    training_loss = []
    validation_loss = []
    for epoch in range(config.EPOCHS):
        
        train_loss = train_one_epoch(model, train_data_loader, optimizer, device)
        training_loss.append(train_loss)

        valid_loss = validate_one_epoch(model, valid_data_loader, device)
        validation_loss.append(valid_loss)

        print("======="*20)
        print(f"Epoch: {epoch}")
        print(f"\tTrain loss: {train_loss}")
        print(f"\tValid loss: {valid_loss}")
        print("======="*20)

        if valid_loss<best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), config.MODEL_FILE)
            waiting = 0
        
        else:
            waiting+=1
            if waiting==config.MAX_WAITING:
                with open(f"{config.TRAIN_LOSS_FILE}", "wb") as file:
                    np.save(file, training_loss)

                with open(f"{config.VALID_LOSS_FILE}", "wb") as file:
                    np.save(file, validation_loss)

                print("======"*20)
                print(f"Best Loss: {round(best_loss, 5)}")
                print("======"*20)
                break

    return None

if __name__=="__main__":
    train()