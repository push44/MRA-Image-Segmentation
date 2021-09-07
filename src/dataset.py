import torch
import numpy as np

from PIL import Image
from torchvision import transforms

def normalize_image(img, max_value, min_value):
    return ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

class MRIDataset:
    def __init__(self, high_res_path, low_res_path, mask_path):
        self.high_res_path = high_res_path
        self.low_res_path = low_res_path
        self.mask_path = mask_path

    def __len__(self):
        return len(self.high_res_path)

    def __getitem__(self, item):

        high_res_img = np.load(self.high_res_path[item])
        high_res_img = normalize_image(high_res_img, np.min(high_res_img), np.max(high_res_img))

        low_res_img = np.load(self.low_res_path[item])
        low_res_img = normalize_image(low_res_img, np.min(low_res_img), np.max(low_res_img))

        mask_img = np.load(self.mask_path[item])

        return {
            "high_res_img": torch.unsqueeze(torch.tensor(high_res_img, dtype=torch.float), 0),
            "low_res_img": torch.unsqueeze(torch.tensor(low_res_img, dtype=torch.float), 0),
            "mask_img": torch.unsqueeze(torch.tensor(mask_img, dtype=torch.float), 0)
        }

