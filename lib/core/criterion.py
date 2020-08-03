# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
#import models_lpf
import torch.nn.functional as F
import torchvision.models.vgg as vgg
from collections import namedtuple

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()

LossOutput = namedtuple(
    "LossOutput", ["relu1","relu2"])

class LossNetwork(nn.Module):
    """Reference:
        https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    """

    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features

        '''
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }
        '''



        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2"

        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
            if name > '8':
                break
        return LossOutput(**output)


class Perceptual_loss(nn.Module):
    def __init__(self, args, device):
        super(Perceptual_loss, self).__init__()

        self.model = LossNetwork()
        self.model.cuda()
        self.model.eval()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.mask_loss = torch.nn.CrossEntropyLoss()
        self.loss = torch.nn.L1Loss(reduction='mean')
        # self.loss = torch.nn.MSELoss(reduction='mean')

    # # Minye
    # def forward(self, x, target):
    #     ph, pw = x.size(2), x.size(3)
    #     h, w = target.size(2), target.size(3)
    #     if ph != h or pw != w:
    #         x = F.upsample(input=x, size=(h, w), mode='bilinear')
    #     x_feature = self.model(x)
    #     target_feature = self.model(target)
    #
    #     feature_loss = self.loss(x_feature.relu1,target_feature.relu1)+self.loss(x_feature.relu2,target_feature.relu2)
    #
    #     return feature_loss

    # L1
    def forward(self, x, target):
        ph, pw = x.size(2), x.size(3)
        h, w = target.size(2), target.size(3)
        if ph != h or pw != w:
            x = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.loss(x, target)

