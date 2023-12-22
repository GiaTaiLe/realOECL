"""
Author: LGT
"""
import torch
from termcolor import colored

from data.data_formatter import MultiTransforms
from data.data_transforms import data_trn

NUM_OF_CLASSES = {
    "imagenet30": 30,
    "cifar10": 10,
    "dior": 19,
    "wbc": 5
}


def get_trn(params, trn):
    """
    :param params:
    :param trn: transforms
    standard, simclr for CLR training, val for testing.
    :return:
    """
    return data_trn(params["dataset"], trn)


def get_data(params, train, category, trn, formatter="aug"):
    """
    :param formatter:
    :param params:
    :param train:
    :param category: id / ood/ oe
    :param trn: transforms
    :return:
    """

    if category.lower() == "id":
        dataset_type = "dataset"
    elif category.lower() == "ood":
        dataset_type = "dataset_ood"
    elif category.lower() == "oe":
        dataset_type = "dataset_oe"

    else:
        raise ValueError("Invalid dataset category {}! Unable to load dataset".format(category))

    dataset_name = params[dataset_type]
    sub_classes = params["class"]

    if isinstance(trn, str):
        f_trn = get_trn(params, trn=trn)
    else:
        f_trn = trn

    if formatter == "aug":  # if trn is a dict, use Aug_model
        f_trn = MultiTransforms(f_trn)

    if params[dataset_type] == "imagenet1k":
        from data.datasets import ImageNet1k
        dataset = _get_data(ImageNet1k, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "imagenet30":
        from data.datasets import ImageNet30
        dataset = _get_data(ImageNet30, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "dior":
        from data.datasets import DIOR
        dataset = _get_data(DIOR, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "wbc":
        from data.datasets import WBC
        dataset = _get_data(WBC, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "cifar10":
        from data.datasets import CIFAR10
        dataset = _get_data(CIFAR10, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "cifar100":
        from data.datasets import CIFAR100
        dataset = _get_data(CIFAR100, dataset_type, dataset_name, sub_classes, train, f_trn)

    elif params[dataset_type] == "tiny80m":
        from data.datasets import TinyImages
        from util.utility import get_subset_with_len
        # in tiny80m there is no options for train/test. Load all

        dataset = _get_data(TinyImages, dataset_type, dataset_name, sub_classes, train, f_trn)
        # dataset = get_subset_with_len(TinyImages(transform=f_trn), length=10000000)

    else:
        raise ValueError("Dataset {} currently unavailable".format(params[dataset_type]))

    return dataset


def _get_data(data, dataset_type, dataset_name, sub_classes, train, f_trn):
    if dataset_type == "dataset":
        classes = [i for i in range(NUM_OF_CLASSES[dataset_name]) if i not in [sub_classes]]
        dataset = data(train=train, chosen_classes=classes, transform=f_trn)

    elif dataset_type == "dataset_ood":
        dataset = data(train=train, transform=f_trn, chosen_classes=sub_classes)

    elif dataset_type == "dataset_oe":
        dataset = data(transform=f_trn)

    else:
        raise ValueError("Check again get_data in wrapper.py")

    return dataset


def get_dataloader(params, dataset, drop_last, shuffle):
    """
    dataloader
    :param params:
    :param dataset:
    :param drop_last:
    :param shuffle:
    :return:
    """
    return torch.utils.data.DataLoader(dataset, num_workers=params['num_workers'],
                                       batch_size=params['batch_size'], pin_memory=True,
                                       drop_last=drop_last, shuffle=shuffle)


def get_model(params, final_dim):
    """
    @param final_dim:
    @param params: yaml
    @return:
    """
    # get backbone
    from model.models import BaseModel

    model = BaseModel(backbone=params["backbone"], final_dim=final_dim)
    print(">>Model with neural network {} and {} chosen_classes".format(
        colored(params["backbone"], "green"), colored(str(final_dim), "green")))

    return model


def get_optimizer(params, model):
    """
    @param params:
    @param model:
    @return:
    """
    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if params["optimizer"] == "sgd":
        from optimizer.optim import sgd
        optimizer = sgd(model_params, **params["optim_kwargs"])

    elif params["optimizer"] == "adam":
        from optimizer.optim import adam
        optimizer = adam(model_params, **params["optim_kwargs"])

    elif params["optimizer"] == "lars":
        from optimizer.optim import lars
        optimizer = lars(model_params, **params["optim_kwargs"])

    else:
        raise ValueError("Invalid optimizer: {}".format(params["optimizer"]))

    return optimizer


def get_scheduler(params, optimizer):
    """
    @param params: yaml
    @param optimizer: optimizer outputs
    @return:
    """

    if params["scheduler"] == "cosine":  # cosine scheduler for learning rate
        from optimizer.optim import cosine_lr
        scheduler_lr = cosine_lr(
            optimizer=optimizer,
            T_max=params["scheduler_kwargs"]["t_max"]
        )

    elif params["scheduler"] == "constant":
        from optimizer.optim import constant_lr
        scheduler_lr = constant_lr(optimizer=optimizer)

    elif params["scheduler"] == "multi-steps":
        from optimizer.optim import multistep_lr
        scheduler_lr = multistep_lr(optimizer=optimizer)

    else:
        raise ValueError("Invalid learning rate schedule {}".format(params["scheduler"]))

    return scheduler_lr


def get_criterion(params):
    """
    @param params:
    @return:
    """
    if params["method"] in ["simclr"]:
        from optimizer.loss_f import NTXent
        loss_func = NTXent()

    elif params["method"] in ["simclr_oe"]:
        from optimizer.loss_f import NTXentOE
        loss_func = NTXentOE()

    else:
        raise ValueError("Invalid criterion {}".format(params["criterion"]))

    return loss_func
