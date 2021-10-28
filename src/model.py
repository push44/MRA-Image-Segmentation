import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config

def dice_loss(preds, targets):
    # credit: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    smooth = 1

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds*targets).sum()
    dice = (2*intersection+smooth)/(preds.sum()+targets.sum()+smooth)

    return 1-dice

class DeepMedic(nn.Module):
    def __init__(self):
        super(DeepMedic, self).__init__()
        # High resolution
        self.conv1h = nn.Conv3d(in_channels=1, out_channels=30, kernel_size=5, stride=1)
        self.bn1h = nn.BatchNorm3d(30)

        self.conv2h = nn.Conv3d(in_channels=30, out_channels=40, kernel_size=5, stride=1)
        self.bn2h = nn.BatchNorm3d(40)

        self.conv3h = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=5, stride=1)
        self.bn3h = nn.BatchNorm3d(40)

        self.conv4h = nn.Conv3d(in_channels=40, out_channels=50, kernel_size=5, stride=1)
        self.bn4h = nn.BatchNorm3d(50)

        # Low resolution
        self.conv1l = nn.Conv3d(in_channels=1, out_channels=30, kernel_size=5, stride=1)
        self.bn1l = nn.BatchNorm3d(30)

        self.conv2l = nn.Conv3d(in_channels=30, out_channels=40, kernel_size=5, stride=1)
        self.bn2l = nn.BatchNorm3d(40)

        self.conv3l = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=5, stride=1)
        self.bn3l = nn.BatchNorm3d(40)

        self.conv4l = nn.Conv3d(in_channels=40, out_channels=50, kernel_size=5, stride=1)
        self.bn4l = nn.BatchNorm3d(50)

        self.upsampling = nn.Upsample(scale_factor=(4, 4, 4))

        # Common layers
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        # Final convolutions
        self.convf1 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=1)
        self.bn1f = nn.BatchNorm3d(100)

        self.convf2 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=1)
        self.bn2f = nn.BatchNorm3d(100)

        self.convf3 = nn.Conv3d(in_channels=100, out_channels=1, kernel_size=1)


    def forward(self, high_resolution, low_resolution, mask):
        bs1, c1, d1, h1, w1 = high_resolution.size()
        bs2, c2, d2, h2, w2 = low_resolution.size()
        
        # High resolution
        x1 = F.relu(self.conv1h(high_resolution)) # size: (bs, 1, 40, 40, 40)
        x1 = self.bn1h(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv2h(x1)) # size: (bs, 30, 36, 36, 36)
        x1 = self.bn2h(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv3h(x1)) # size: (bs, 40, 32, 32, 32)
        x1 = self.bn3h(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv4h(x1)) # size: (bs, 40, 28, 28, 28)
        x1 = self.bn4h(x1)
        x1 = self.dropout1(x1)

        # Low resolution
        x2 = F.relu(self.conv1l(low_resolution)) # size: (bs, 1, 22, 22, 22)
        x2 = self.bn1l(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv2l(x2)) # size: (bs, 30, 18, 18, 18)
        x2 = self.bn2l(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv3l(x2)) # size: (bs, 40, 14, 14, 14)
        x2 = self.bn3l(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv4l(x2)) # size: (bs, 40, 10, 10, 10)
        x2 = self.bn4l(x2)
        x2 = self.dropout1(x2)

        x2 = self.upsampling(x2) # size: (bs, 50, 6, 6, 6)

        # Concatinate
        x3 = torch.cat(
            [x1, x2],
            axis = 1
        ) # size: (bs, 50, 24, 24, 24), size: (bs, 50, 24, 24, 24)

        # Final convolutions
        x3 = F.relu(self.convf1(x3)) # size: (bs, 100, 24, 24, 24)
        x3 = self.bn1f(x3)
        x3 = self.dropout2(x3)

        x3 = F.relu(self.convf2(x3)) # size: (bs, 100, 24, 24, 24)
        x3 = self.bn2f(x3)
        x3 = self.dropout2(x3)

        # Output (Dimmension: (bs, 1, 24, 24, 24))
        out_img = torch.sigmoid(self.convf3(x3)) # size: (bs, 100, 24, 24, 24)

        # Optimization loss
        loss = dice_loss(mask, out_img)
        return out_img, float(loss)


def l1_loss(x1_list, x2_list):
    l1_loss = nn.L1Loss()

    loss = 0
    for x1_item, x2_item in zip(x1_list, x2_list):
        loss+=l1_loss(x1_item, x2_item)
        loss/=x1_item.numel()
        loss*=24**3

    loss/=len(x1_list)
    return loss

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        s = 2
        k = 4
        n = config.OUTPUT_SHAPE
        p = ((n-1) - n + k)//2

        n1 = 64
        n2 = 128
        n3 = 256

        self.convblock1 = nn.Conv3d(in_channels=1, out_channels=n1, kernel_size=k, stride=s, padding=p)
        self.convblock2 = nn.Conv3d(in_channels=n1, out_channels=n2, kernel_size=k, stride=s, padding=p)
        self.convblock3 = nn.Conv3d(in_channels=n2, out_channels=n3, kernel_size=k, stride=s, padding=p)

        self.bn1 = nn.BatchNorm3d(n1)
        self.bn2 = nn.BatchNorm3d(n2)

    def forward(self, prediction_masked_image, gt_masked_image):

        # Prediction mask channel
        x1_concat = []
        x1_concat.append(prediction_masked_image)

        x1 = F.relu(self.convblock1(prediction_masked_image))
        #print(f"Layer 1 x1:{x1.shape}")

        x1_concat.append(x1)
        x1 = self.bn1(x1)

        x1 = F.relu(self.convblock2(x1))
        #print(f"Layer 2 x1:{x1.shape}")

        x1_concat.append(x1)
        x1 = self.bn2(x1)

        x1 = F.relu(self.convblock3(x1))
        #print(f"Layer 3 x1:{x1.shape}")

        x1_concat.append(x1)

        # Ground truth mask channel
        x2_concat = []
        x2_concat.append(gt_masked_image)

        x2 = F.relu(self.convblock1(gt_masked_image))
        #print(f"Layer x2 1:{x2.shape}")

        x2_concat.append(x2)
        x2 = self.bn1(x2)

        x2 = F.relu(self.convblock2(x2))
        #print(f"Layer x2 2:{x2.shape}")

        x2_concat.append(x2)
        x2 = self.bn2(x2)

        x2 = F.relu(self.convblock3(x2))
        #print(f"Layer x2 3:{x2.shape}")

        x2_concat.append(x2)

        return None, l1_loss(x1_concat, x2_concat)