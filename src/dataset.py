import torch
import os
import config
import numpy as np

class MRIDataset:
    def __init__(self, filepath_dict):
        # filepath_dict is a dictionary of lists containing filepaths of images.
        self.filepath_dict = filepath_dict

        # In case of test time there will not be mask files.
        self.empty=False
        if "mask" in self.filepath_dict.keys():
            self.num_files = len(self.filepath_dict["mask"])

            if self.num_files>0:
                pass
            else:
                self.empty=True

    def __len__(self):
        key = list(self.filepath_dict.keys())[0]
        return len(self.filepath_dict[key])

    def __getitem__(self, item):

        return_dict = {}
        for key in self.filepath_dict.keys():
            if key == "mask":
                if self.empty==True:
                    output_shape = config.OUTPUT_SHAPE
                    arr = np.zeros(shape=(output_shape, output_shape, output_shape))
                else:
                    arr = np.load(self.filepath_dict[key][item])
                tensor = torch.unsqueeze(torch.tensor(arr, dtype=torch.float), 0)

            elif key == "filename":
                tensor = self.filepath_dict[key][item]

            else:
                arr = np.load(self.filepath_dict[key][item])
                tensor = torch.unsqueeze(torch.tensor(arr, dtype=torch.float), 0)

            return_dict[key] = tensor

        return return_dict
