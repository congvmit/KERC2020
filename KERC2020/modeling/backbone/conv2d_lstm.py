import torch
import torch.nn as nn


class Conv2DLSTM(nn.Module):
    def __init__(self, backbone):
        super(Conv2DLSTM, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return x
