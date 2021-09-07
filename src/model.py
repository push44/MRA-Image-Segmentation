import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv2h = nn.Conv3d(in_channels=30, out_channels=40, kernel_size=5, stride=1)
        self.conv3h = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=5, stride=1)
        self.conv4h = nn.Conv3d(in_channels=40, out_channels=50, kernel_size=5, stride=1)

        # Low resolution
        self.conv1l = nn.Conv3d(in_channels=1, out_channels=30, kernel_size=5, stride=1)
        self.conv2l = nn.Conv3d(in_channels=30, out_channels=40, kernel_size=5, stride=1)
        self.conv3l = nn.Conv3d(in_channels=40, out_channels=40, kernel_size=5, stride=1)
        self.conv4l = nn.Conv3d(in_channels=40, out_channels=50, kernel_size=5, stride=1)
        self.upsampling = nn.Upsample(scale_factor=(4, 4, 4))

        # Common layers
        self.dropout1 = nn.Dropout(0.02)
        self.dropout2 = nn.Dropout(0.5)

        # Final convolutions
        self.convf1 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=1)
        self.convf2 = nn.Conv3d(in_channels=100, out_channels=100, kernel_size=1)
        self.convf3 = nn.Conv3d(in_channels=100, out_channels=1, kernel_size=1)


    def forward(self, high_res_img, low_res_img, mask_img):
        bs1, c1, d1, h1, w1 = high_res_img.size()
        bs2, c2, d2, h2, w2 = low_res_img.size()
        
        # High resolution
        x1 = F.relu(self.conv1h(high_res_img)) # size: (bs, 1, 40, 40, 40)
        x1 = nn.BatchNorm3d(30)(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv2h(x1)) # size: (bs, 30, 36, 36, 36)
        x1 = nn.BatchNorm3d(40)(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv3h(x1)) # size: (bs, 40, 32, 32, 32)
        x1 = nn.BatchNorm3d(40)(x1)
        x1 = self.dropout1(x1)

        x1 = F.relu(self.conv4h(x1)) # size: (bs, 40, 28, 28, 28)
        x1 = nn.BatchNorm3d(50)(x1)
        x1 = self.dropout1(x1)

        # Low resolution
        x2 = F.relu(self.conv1l(low_res_img)) # size: (bs, 1, 22, 22, 22)
        x2 = nn.BatchNorm3d(30)(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv2l(x2)) # size: (bs, 30, 18, 18, 18)
        x2 = nn.BatchNorm3d(40)(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv3l(x2)) # size: (bs, 40, 14, 14, 14)
        x2 = nn.BatchNorm3d(40)(x2)
        x2 = self.dropout1(x2)

        x2 = F.relu(self.conv4l(x2)) # size: (bs, 40, 10, 10, 10)
        x2 = nn.BatchNorm3d(50)(x2)
        x2 = self.dropout1(x2)

        x2 = self.upsampling(x2) # size: (bs, 50, 6, 6, 6)

        # Concatinate
        x3 = torch.cat(
            [x1, x2],
            axis = 1
        ) # size: (bs, 50, 24, 24, 24), size: (bs, 50, 24, 24, 24)

        # Final convolutions
        x3 = F.relu(self.convf1(x3)) # size: (bs, 100, 24, 24, 24)
        x3 = nn.BatchNorm3d(100)(x3)
        x3 = self.dropout2(x3)

        x3 = F.relu(self.convf2(x3)) # size: (bs, 100, 24, 24, 24)
        x3 = nn.BatchNorm3d(100)(x3)
        x3 = self.dropout2(x3)

        # Output (Dimmension: (bs, 1, 24, 24, 24))
        out_img = torch.sigmoid(self.convf3(x3)) # size: (bs, 100, 24, 24, 24)

        # Optimization loss
        loss = dice_loss(mask_img, out_img)
        return out_img, loss

