from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from einops import rearrange

from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from .layers.mlp import MLP

class MLPs(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args 
        self.layers = MLP(args.dim_hidden, args.tf_emb_size * 3, args.dim_hidden, num_layer=args.TF_depth,
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.layers(x)

        x = rearrange(x, 'x d -> 1 d x')
        x = self.avgpool(x)
        x = rearrange(x, '1 d 1 -> 1 d')
        return x