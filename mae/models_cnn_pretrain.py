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


class small_CNN(nn.Module):
    def __init__(self):
        super(small_CNN, self).__init__()

        # -----------------------ENCODER with small-VGG16 kinda architecture ------------------
        self.enc_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #112
        self.enc_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #56
        self.enc_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #28
        self.enc_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #14
        self.enc_5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)) #7
        # The output after this layer should be 7x7x512. So we should perhaps not use the maxpooling layer.
        # Something to test at least. With and without it. 

        # ------------------------------------------------------------------------------------
        # -----------------------DECODER------------------------------------------------------
        self.dec_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 14x14x512 --> (H,W,C) format
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),   # 14x14x128
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.dec_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 28x28x128
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),   # 28x28x64
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.dec_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 56x56x64
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),   # 56x56x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dec_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 112x112x32
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),   # 112x112x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.dec_5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 224x224x128
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)   # 224x224x3
        )
        # -------------------------------------------------------------------------------------
        
    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def random_mask(self, imgs, mask_ratio, patch_size):
        "Takes in the images and randomly masks them. Then it returns masked imgs"
        # imgs = [BS, C, H, W] 
        # random_tensor = create a radnom tensor of shape (BS, H//patch_size, W//patch_size)
        # argsort = argsort the random vector in random_tensor [1, 2] dimensions (except Batch dimension)
        # keep the '1-masking_ratio' ratio of the smallest/biggest ones.  
        # make the random_tensor into T-F tensor
        # and apply that tensor to the imgs
        N, c, h, w = imgs.shape
        h_patch = h//patch_size   # 2
        w_patch = w//patch_size   # 2
        L = h_patch * w_patch     # total number of patches
        
        len_keep = int(L * (1 - mask_ratio))  # number of patches to keep of the total

        noise = torch.rand(N, L, device=imgs.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.zeros([N, L], device=imgs.device)
        mask[:, :len_keep] = 1.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)   # [bs, (h_patch x w_patch)]

        # resize mask
        mask_3d = torch.reshape(mask, shape = (N, h_patch, w_patch))
        
        # add the color channels to mask tensor
        mask_3d = mask_3d[:, None, :, :]
        mask_colored = mask_3d.repeat(1, 3, 1, 1)

        mask_colored_increased = torch.repeat_interleave(mask_colored, patch_size, dim=2)
        batch_mask = torch.repeat_interleave(mask_colored_increased, patch_size, dim=3)

        masked_imgs = imgs * batch_mask
        return masked_imgs, batch_mask

    def forward_encoder(self, masked_imgs):
        out = self.enc_1(masked_imgs)
        out = self.enc_2(out)
        out = self.enc_3(out)
        out = self.enc_4(out)
        latent = self.enc_5(out)
        return latent

    def forward_decoder(self, latent):
        out = self.dec_1(latent)
        out = self.dec_2(out)
        out = self.dec_3(out)
        out = self.dec_4(out)
        pred_imgs = self.dec_5(out)
        return pred_imgs

    def loss_forward(self, imgs, preds, masks):
        # masks 1 is keep 0 is remove
        square_differnece = (imgs - preds) ** 2  # size = (bs, c, h, w)
        simple_loss = torch.mean(square_differnece) 
        inverse_masks = 1 - masks # 0 is keep 1 is remove, size = (bs, c, h, w)
        removed_patch_loss = (square_differnece * inverse_masks).sum()/inverse_masks.sum()
        return removed_patch_loss, simple_loss

    def forward(self, imgs, mask_ratio=0.75, patch_size=16): # def forward(self, img, masking_ratio)
        masked_imgs, masks = self.random_mask(imgs, mask_ratio, patch_size) # we should randomly mask x and I believe that should be it. 
        latent = self.forward_encoder(masked_imgs)  
        pred_imgs = self.forward_decoder(latent)   # Should ideally reconstruct the images
        loss, simple_loss = self.loss_forward(imgs, pred_imgs, masks)
        return loss, simple_loss 


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
            nn.Upsample(scale_factor=2, mode='bilinear'),   # 14x14x512 --> (H,W,C) format
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
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)   # 224x224x3
        )
        # -------------------------------------------------------------------------------------

    def initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def random_mask(self, imgs, mask_ratio, patch_size):
        "Takes in the images and randomly masks them. Then it returns masked imgs as batches"
        N, c, h, w = imgs.shape
        h_patch = h//patch_size   # height and width must be a multiple of patch size
        w_patch = w//patch_size   
        L = h_patch * w_patch     # total number of patches
        
        num_keep = int(L * (1 - mask_ratio))  # number of patches to keep of the total

        noise = torch.rand(N, L, device=imgs.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # generate the binary mask: 1 is keep, 0 is remove
        mask = torch.zeros([N, L], device=imgs.device)
        mask[:, :num_keep] = 1.
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)   # [bs, (h_patch x w_patch)]

        # resize mask
        mask_3d = torch.reshape(mask, shape = (N, h_patch, w_patch))
        
        # add the color channels to mask tensor
        mask_3d = mask_3d[:, None, :, :]
        mask_colored = mask_3d.repeat(1, 3, 1, 1)

        mask_colored_increased = torch.repeat_interleave(mask_colored, patch_size, dim=2)
        batch_mask = torch.repeat_interleave(mask_colored_increased, patch_size, dim=3)

        masked_imgs = imgs * batch_mask
        return masked_imgs, batch_mask

    def forward_encoder(self, masked_imgs):
        out = self.enc_1(masked_imgs)
        out = self.enc_2(out)
        out = self.enc_3(out)
        out = self.enc_4(out)
        out = self.enc_5(out)
        out = self.enc_6(out)
        out = self.enc_7(out)
        out = self.enc_8(out)
        out = self.enc_9(out)
        out = self.enc_10(out)
        out = self.enc_11(out)
        out = self.enc_12(out)
        latent = self.enc_13(out)   # Should be of shape BSx512x7x7
        return latent

    def forward_decoder(self, latent):
        out = self.dec_1(latent)
        out = self.dec_2(out)
        out = self.dec_3(out)
        out = self.dec_4(out)
        out = self.dec_5(out)
        out = self.dec_6(out)
        out = self.dec_7(out)
        out = self.dec_8(out)
        out = self.dec_9(out)
        pred_imgs = self.dec_10(out)
        return pred_imgs

    def loss_forward(self, imgs, preds, masks):
        # masks 1 is keep 0 is remove
        square_differnece = (imgs - preds) ** 2  # size = (bs, c, h, w)
        simple_loss = torch.mean(square_differnece) 
        inverse_masks = 1 - masks # 0 is keep 1 is remove, size = (bs, c, h, w)
        removed_patch_loss = (square_differnece * inverse_masks).sum()/inverse_masks.sum()
        return removed_patch_loss, simple_loss

    def forward(self, imgs, mask_ratio=0.75, patch_size=16): # def forward(self, img, masking_ratio)
        masked_imgs, masks = self.random_mask(imgs, mask_ratio, patch_size) # we should randomly mask x and I believe that should be it. 
        latent = self.forward_encoder(masked_imgs)  
        pred_imgs = self.forward_decoder(latent)   # Should ideally reconstruct the images
        loss, simple_loss = self.loss_forward(imgs, pred_imgs, masks)
        return loss, simple_loss 

def cnn_model(**kwargs):
    model = CNN(
        **kwargs)
    model = model.apply(model.initialize_weights)  # initialize the weights
    return model

def cnn_model_small(**kwargs):
    model = small_CNN(
        **kwargs
    )
    model = model.apply(model.initialize_weights)  # initialize the weights
    return model

cnn = cnn_model
cnn_small = cnn_model_small
