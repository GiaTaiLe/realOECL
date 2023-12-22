import torch.nn as nn


class NormalizeLayer(nn.Module):
    """
    source: CSI original paper.
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, inputs):
        return (inputs - 0.5) / 0.5