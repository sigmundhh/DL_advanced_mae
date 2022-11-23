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
    """ Masked Autoencoder with CNN backbone
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.patch_size = 16
        # Conv2s(in_channels, out_channels, kernel_size, stride, padding, dilation, ....)
        self.conv1 = nn.Conv2d(3, 6, 5)
        # MaxPool2d(kernel_size, stride=None, padding=0, ...)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Linear(in_features, out_features, bias=True, ...) 
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, mask_ratio=0.75): # def forward(self, img, masking_ratio)
        # x = random_mask(x, masking_ratio) # we should randomly mask x and I believe that should be it. 

        # change the network so we get encoder/decoder arch. 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # loss = loss_forward(x, img)
        return torch.mean(x), 5, 5 # return loss

def cnn_model(**kwargs):
    model = CNN(
        **kwargs)
    return model

# set recommended archs
cnn = cnn_model

