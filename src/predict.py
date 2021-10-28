import os
import torch

import config

import numpy as np

from dataset import MRIDataset
from model import DeepMedic
from engine import predict_one_epoch

def predict():
    ###################### STAGE 1 ######################
    # Lists with image locations
    high_res_paths = list(map(lambda fname: f"{config.HIGH_RESOLUTION_PATCH_PATH}/test/{fname}", os.listdir(f"{config.HIGH_RESOLUTION_PATCH_PATH}/test")))
    low_res_paths = list(map(lambda fname: f"{config.LOW_RESOLUTION_PATCH_PATH}/test/{fname}", os.listdir(f"{config.LOW_RESOLUTION_PATCH_PATH}/test")))
    mask_paths = []
    if len(os.listdir(f"{config.MASK_PATCH_PATH}/test"))>0:
        for file in os.listdir(f"{config.MASK_PATCH_PATH}/test"):
            os.remove(f"{config.MASK_PATCH_PATH}/test/{file}")

    filenames = list(map(lambda val: val.split("/")[-1].split(".")[0], high_res_paths))

    ###################### STAGE 2 ######################
    # Train data loader
    filepath_dict = {
        "high_resolution": high_res_paths,
        "low_resolution": low_res_paths,
        "mask": mask_paths,
        "filename": filenames
    }
    dataset = MRIDataset(filepath_dict)

    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = config.TRAIN_BATCH_SIZE
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepMedic().to(device)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=torch.device(device)))

    predictions, filenames = predict_one_epoch(model, data_loader, device)

    for pred, fname in zip(predictions, filenames):
        with open(f"{config.MASK_PATCH_PATH}/test/{fname}.npy", "wb") as file:
            np.save(file, pred)

    return None

if __name__ == "__main__":
    predict()