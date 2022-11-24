# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # -----------------------ENCODER with VGG16 kinda architecture ------------------
        self.enc_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.enc_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.enc_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.enc_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.enc_5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.enc_6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.enc_7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.enc_8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.enc_9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.enc_10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.enc_11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.enc_12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.enc_13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        # The output after this layer should be 7x7x512. So we should perhaps not use the maxpooling layer.
        # Something to test at least. With and without it. 

        # ------------------------------------------------------------------------------------
        # -----------------------DECODER------------------------------------------------------
        self.dec_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 14x14x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),   # 14x14x512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.dec_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),   # 14x14x512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.dec_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 28x28x512
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),   # 28x28x512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.dec_4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),   # 28x28x512
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.dec_5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 56x56x512
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),   # 56x56x256
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.dec_6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),   # 56x56x256
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.dec_7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 112x112x256
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),   # 112x112x128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec_8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),   # 112x112x128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.dec_9 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 224x224x128
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),   # 224x224x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec_10 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),   # 224x224x3
            #nn.BatchNorm2d(3), don't know if this is a good idea to have or not??
            nn.ReLU()  # I think this is good since pixel values should be >=0
        )

        
        """
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        """
    """ 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    """
    def random_mask(imgs, masking_ratio):
        "Takes in the images and randomly masks them. Then it returns masked imgs"
        pass

    def forward_cnn(self, imgs, mask_ratio):
        masked_imgs = self.random_mask(imgs, mask_ratio) # we should randomly mask x and I believe that should be it. 

        # ---------------ENCODER SPECIFICS -----------------
        x = self.pool(F.relu(self.conv1(masked_imgs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # ----------------------------------------------------

        # Maybe something here in the middle

        # ---------------DECODER SPECIFICS -----------------
        # Something something 
        # The output should be the reconstructed image.

    def loss_forward(imgs, preds, mask):
        pass

    def forward(self, imgs, mask_ratio=0.75): # def forward(self, img, masking_ratio)
        preds, masks = self.forward_cnn(self, imgs, mask_ratio)
        loss = self.loss_forward(imgs, preds, masks)
        return loss, preds, masks # return loss






"""
class CNN(nn.Module):
    " Masked Autoencoder with CNN backbone
    "    
    def __init__(self, patch_size, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, ....)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPool2d(kernel_size, stride=None, padding=0, ...)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Linear(in_features, out_features, bias=True, ...) 
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def random_mask(imgs, masking_ratio):
        "Takes in the images and randomly masks them. Then it returns masked imgs"
        pass

    def forward_cnn(self, imgs, mask_ratio):
        masked_imgs = self.random_mask(imgs, mask_ratio) # we should randomly mask x and I believe that should be it. 

        # ---------------ENCODER SPECIFICS -----------------
        x = self.pool(F.relu(self.conv1(masked_imgs)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # ----------------------------------------------------

        # Maybe something here in the middle

        # ---------------DECODER SPECIFICS -----------------
        # Something something 
        # The output should be the reconstructed image.

    def loss_forward(imgs, preds, mask):
        pass

    def forward(self, imgs, mask_ratio=0.75): # def forward(self, img, masking_ratio)
        preds, masks = self.forward_cnn(self, imgs, mask_ratio)
        loss = self.loss_forward(imgs, preds, masks)
        return loss, preds, masks # return loss
"""

def cnn_model(**kwargs):
    model = CNN(
        **kwargs)
    return model

# set recommended archs
cnn = cnn_model

