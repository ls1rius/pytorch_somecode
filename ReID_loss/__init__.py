# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
import torch.nn as nn
import torch


def make_loss_with_center(num_classes):    # modified by gu

    triplet = TripletLoss(1)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=64, use_gpu=True)  # center loss
    def loss_func(score, feat, target):
        return xent(score, target) + \
               triplet(feat, target)[0] + \
               5e-4 * center_criterion(feat, target)
    return loss_func, center_criterion


def make_optimizer_with_center(model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": 3.5e-4, "weight_decay": 5e-4}]
    optimizer = getattr(torch.optim, 'Adam')(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)
    return optimizer, optimizer_center