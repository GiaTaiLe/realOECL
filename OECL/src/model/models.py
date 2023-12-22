"""
Author: Le Gia Tai
"""

import torch.nn as nn
from model.resnet import resnet18
from model.resnet_cifar10 import resnet18_cifar10


class BaseModel(nn.Module):
    """
    ResNetBW, WRN
    """

    def __init__(self, backbone, final_dim):
        """
        Parameters
        ----------
        backbone: WideResNet or ResNet
        """
        super(BaseModel, self).__init__()
        if backbone == "resnet18":
            backbone = resnet18()

        elif backbone == "resnet18_cifar10":
            backbone = resnet18_cifar10()

        else:
            raise ValueError("The model {} is not available".format(backbone))
        self.backbone = backbone["backbone"]
        self.backbone_dim = backbone["dim"]
        self.final_dim = final_dim
        self.head = nn.Sequential(nn.Linear(self.backbone_dim, self.backbone_dim),
                                  nn.BatchNorm1d(self.backbone_dim),
                                  nn.ReLU(),
                                  nn.Linear(self.backbone_dim, self.final_dim))

    def forward(self, sample):
        """
        @param sample: outputs
        @return: out and previous layer (out_penultimate)
        """
        f_f = self.backbone(sample)
        f_fh = self.head(f_f)

        return f_fh, f_f
