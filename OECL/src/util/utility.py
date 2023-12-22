"""
Author: Le Gia Tai
"""

import errno
import math
import os
import random
import sys

import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn.functional as F
from termcolor import colored
from torch.utils.data.dataset import Subset


def mkdir_if_missing(directory):
    """
    make dir if dir in the current path is missing.
    :param directory: path/to/dir
    :return: create path/to/dir if missing
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=":.2%"):
        self.name = name
        self.fmt = fmt
        self.acc = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, count=1):
        self.acc = val / count
        self.sum += val
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {acc' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MovingAverageMeter(object):
    def __init__(self, name, fmt=':ff', moving=0.9):
        self.name = name
        self.fmt = fmt
        self.moving = moving
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.avg = self.moving * self.avg + (1 - self.moving) * self.val

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} (avg: {avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressBar(object):
    def __init__(self, loader_len, infor, epoch, bar_len=10):
        self.bar_len = int(bar_len)
        self.infor = infor
        self.loader_len = loader_len
        self.epoch = epoch

    def display(self, index, time_index):
        if index > self.loader_len:
            index = self.loader_len
        index_len = math.ceil((index / self.loader_len) * self.bar_len)
        sys.stdout.write("\r")
        sys.stdout.write("[")
        sys.stdout.write(colored("=" * index_len, "magenta"))
        sys.stdout.write(colored(">" * (self.bar_len - index_len), "cyan") + "]")
        print_elements = ["[Epoch: {:d}] [{:2d}|{:2d}]".format(self.epoch, index, self.loader_len)]
        print_elements += [str(infor) for infor in self.infor]
        print_elements += ["Time: {:.2f}s".format(time_index)]
        sys.stdout.write(" | ".join(print_elements))
        sys.stdout.flush()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def auroc(pos, neg):
    """
    :param pos:
    :param neg:
    :return:
    """
    pos = np.array(pos).reshape((-1, 1))
    neg = np.array(neg).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc_score = sk.roc_auc_score(labels, examples)

    return auroc_score


def get_subset_with_len(dataset, length, shuffle=False):
    set_random_seed(0)
    dataset_size = len(dataset)

    index = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(index)

    index = torch.from_numpy(index[0:length])
    subset = Subset(dataset, index)

    assert len(subset) == length

    return subset


class AdvancedStatUpdate(object):
    """
    Create an online mean and cov computing module.
    Each time this module is given a seq of aug view's features = [[],...,[]] with shape = [N, Features_dim]
    :return: mean and cov for Features along augmented views of all N samples
    """

    def __init__(self):
        self.t = 1
        self.num_of_samples = 0

    def __call__(self, seq):
        # make sure outputs seq is numpy array
        assert type(seq) == np.ndarray

        if len(seq.shape) <= 2:
            seq = np.expand_dims(seq, axis=1)

        assert seq.shape[1] == 1

        if self.t == 1:
            # ~initialize mean = x1 and s = 0
            self.mean = seq  # E(x_i)
            self.s = np.zeros_like(seq)  # Cov(X_i)
            self.l2_var = 0.  # E(||x-Ex||_2^2) = trace(Cov) = sum of eigenvalues of Cov
            self.l2_sum = np.sum(np.square(seq), axis=-1)  # \Sigma_i ||x_i||_2^2

        else:
            new_mean = self.mean + (seq - self.mean) / self.t
            if len(seq.shape) == 2:
                self.s = self.s + (seq - new_mean) * (seq - self.mean)

            else:
                self.s = self.s + np.transpose(seq - new_mean, (0, 2, 1)) * (seq - self.mean)

            self.mean = new_mean
            self.l2_var = np.array([np.sum(s.diagonal()) / self.t for s in self.s])
            self.l2_sum += np.sum(np.square(seq), axis=-1)
            self.cov = self.s / self.t

        self.t += 1
        self.num_of_samples = self.t - 1
        self.l2_mean = self.l2_sum / self.num_of_samples
        self.total = self.mean * self.num_of_samples


def get_features(loader, model, normalized=False, ff=False, device="cuda"):
    features = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            img = batch["image"].to(device)

            if ff:
                _, rep = model(img)
            else:
                rep, _ = model(img)

            if normalized:
                features.append(F.normalize(rep, dim=-1).detach().cpu())

            else:
                features.append(rep.detach().cpu())

        features = torch.cat(features, dim=0)

        return features


class Rot90(object):
    def __init__(self, dims, idx=None):
        self.dims = dims
        self.idx = idx

    def __call__(self, data):
        if self.idx is None:
            k = np.random.randint(2, 4)

        elif isinstance(self.idx, tuple):
            k = int(np.random.randint(self.idx[0], self.idx[1]))

        elif isinstance(self.idx, int):
            k = int(self.idx % 4)

        else:
            k = 0

        return torch.rot90(data, k, dims=self.dims)
