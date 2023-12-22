
"""
Author: Le Gia Tai
"""

import torch
from optimizer.lars import LARS


def sgd(model_params, lr, weight_decay, momentum, nesterov=False):
    """
    @param model_params:
    @param lr:
    @param weight_decay:
    @param momentum:
    @param nesterov:
    @return:
    """
    return torch.optim.SGD(params=model_params, lr=lr, weight_decay=weight_decay,
                           momentum=momentum, nesterov=nesterov)


def lars(model_params, lr, weight_decay, momentum, nesterov=False):
    """
    :param model_params:
    :param lr:
    :param weight_decay:
    :param momentum:
    :param nesterov:
    :return:
    """
    opt = torch.optim.SGD(params=model_params, lr=lr, weight_decay=weight_decay,
                           momentum=momentum, nesterov=nesterov)

    return LARS(optimizer=opt, eps=0.0)


def adam(model_params, lr, weight_decay):
    """
    @param model_params:
    @param lr:
    @param weight_decay:
    @return:
    """
    return torch.optim.Adam(params=model_params, lr=lr, weight_decay=weight_decay)


def constant_lr(optimizer, factor=0.5):
    """
    @param optimizer: sgd, adam,...
    @param factor: The number we multiply learning rate until the milestone
    @return:
    """
    return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=factor)


def multistep_lr(optimizer, steps=None):
    """
    :param optimizer:
    :param steps:
    :return:
    """
    if steps is None:
        steps = [100, 150]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=0.1)


def cosine_lr(optimizer, T_max):
    """
    @param optimizer: sgd, adam
    @param T_max: a cycle in cosine
    @return:
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)