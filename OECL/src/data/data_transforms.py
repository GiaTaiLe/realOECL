"""
Author: Le Gia Tai
"""
import torchvision.transforms as trn


def data_trn(dataset, trn_type=None):
    """
    :param dataset:
    :param trn_type: for augment transforms
    :return:
    """

    if dataset in ["imagenet30", "imagenet1k", "dior"]:
        kwargs_trn = dict(resize=256, resize_crop=224, color_jitter=(0.4, 0.4, 0.4, 0.1), scale=(0.08, 1.))

        if trn_type == "val":
            transform = _test_trn(**kwargs_trn)

        elif trn_type == "simclr":
            transform = _simclr_trn(**kwargs_trn)

        else:
            raise ValueError("Invalid trn_type: {}".format(trn_type))

    elif dataset == "wbc":
        kwargs_trn = dict(resize=256, resize_crop=224, color_jitter=(0.4, 0.4, 0.4, 0.1), scale=(0.5, 1.))

        if trn_type == "val":
            transform = _test_trn(**kwargs_trn)

        elif trn_type == "simclr":
            transform = _simclr_trn(**kwargs_trn)

        else:
            raise ValueError("Invalid trn_type: {}".format(trn_type))

    elif dataset in ["cifar10", "tiny80m"]:
        kwargs_trn = dict(resize_crop=32, color_jitter=(0.4, 0.4, 0.4, 0.1), scale=(0.08, 1.),
                          mean=[x / 255 for x in [125.3, 123.0, 113.9]], std=[x / 255 for x in [63.0, 62.1, 66.7]])

        if trn_type == "val":
            transform = _test_trn_cifar10(**kwargs_trn)

        elif trn_type == "simclr":
            transform = _simclr_trn_cifar10(**kwargs_trn)

        else:
            raise ValueError("Invalid trn_type: {}".format(trn_type))

    else:
        raise ValueError("Invalid dataset: {}".format(dataset))

    return transform


def _simclr_trn(resize, resize_crop, color_jitter, scale, **kwargs):
    return trn.Compose([
        trn.Resize(resize),
        trn.RandomResizedCrop(resize_crop, scale=scale),
        trn.RandomHorizontalFlip(),
        trn.RandomApply([trn.ColorJitter(*color_jitter)], p=0.8),
        trn.RandomGrayscale(p=0.2),
        trn.ToTensor()
    ])


def _test_trn(resize, resize_crop, **kwargs):
    return trn.Compose([
        trn.Resize(resize),
        trn.CenterCrop(resize_crop),
        trn.ToTensor()])


def _simclr_trn_cifar10(resize_crop, color_jitter, mean, std, scale, **kwargs):
    """
    for cifar10 only
    """
    return trn.Compose([
        trn.RandomResizedCrop(resize_crop, scale=scale),
        trn.RandomHorizontalFlip(),
        trn.RandomApply([trn.ColorJitter(*color_jitter)], p=0.8),
        trn.RandomGrayscale(p=0.2),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std),
    ])


def _test_trn_cifar10(resize_crop, mean, std, **kwargs):
    """
    for cifar10 only
    """
    return trn.Compose([
        trn.Resize(resize_crop),
        trn.CenterCrop(resize_crop),
        trn.ToTensor(),
        trn.Normalize(mean=mean, std=std)])
