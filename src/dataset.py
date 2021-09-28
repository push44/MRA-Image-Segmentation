import torch
import os
import numpy as np

class MRIDataset:
    def __init__(self, high_res_path, low_res_path, mask_path, filenames=None):
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        self.mask_path = mask_path
        self.filenames = filenames

        num_files = len(self.mask_path)

        self.empty=False
        if num_files>0:
            pass
        else:
            self.empty=True

    def __len__(self):
        return len(self.high_res_path)

    def __getitem__(self, item):

        high_res_img = np.load(self.high_res_path[item])

        low_res_img = np.load(self.low_res_path[item])

        if self.empty==True:
            mask_img = np.zeros(shape=(24, 24, 24))
        else:
            mask_img = np.load(self.mask_path[item])

        return_dict = {
                "high_res_img": torch.unsqueeze(torch.tensor(high_res_img, dtype=torch.float), 0),
                "low_res_img": torch.unsqueeze(torch.tensor(low_res_img, dtype=torch.float), 0),
                "mask_img": torch.unsqueeze(torch.tensor(mask_img, dtype=torch.float), 0)
            }

        if self.filenames!=None:
            return_dict["filename"] = self.filenames[item]

        return return_dict
