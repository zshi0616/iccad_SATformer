'''
    Merge clause (group) embedding into group embedding
    Author: ICCAD Anonymous author
    Date: 05/07/2022
    Ref. (MLP-Mixer) https://blog.csdn.net/weixin_44855366/article/details/120801252
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

class FeedForward(nn.Module):
    def __init__(self,d_in,hidden_dim,d_out,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_in,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,d_out),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        x=self.net(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, dim, window_size, hidden_dim, dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.token_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(window_size, hidden_dim, window_size, dropout),
            Rearrange('b d n -> b n d')
 
         )
        self.channel_mixer=nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, hidden_dim, dim, dropout)
        )
        self.linear = nn.Linear(dim * window_size, dim)
    def forward(self,x):
        x = x+self.token_mixer(x)
        x = x+self.channel_mixer(x)
        x = rearrange(x, 'n l d -> n (l d)', l = self.window_size)
        x = self.linear(x)
        return x

class LinearReduction(nn.Module):
    def __init__(self, dim_in, dim_out, hidden_dim, layers, dropout) -> None:
        super(LinearReduction, self).__init__()
        fc = []
        fc.append(nn.Linear(dim_in, hidden_dim))
        for layer in range(layers - 1):
            fc.append(nn.Linear(hidden_dim, hidden_dim))
            fc.append(nn.Dropout(dropout))
        fc.append(nn.Linear(hidden_dim, dim_out))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        out = self.fc(x)
        return out

class clauseMerge(nn.Module):
    def __init__(self, args) -> None:
        super(clauseMerge, self).__init__()
        self.args = args
        if self.args.window_pooling == 'linear':
            dim_in = args.windows_size * args.tf_emb_size
            dim_out = args.tf_emb_size
            self.reduction = nn.Linear(dim_in, dim_out)
            # self.reduction = LinearReduction(dim_in, dim_out, args.tf_emb_size, 3, args.dropout)
        elif self.args.window_pooling == 'mlp':
            self.reduction = MixerBlock(args.tf_emb_size, args.windows_size, args.tf_emb_size, args.dropout)
        elif self.args.window_pooling == 'max':
            self.reduction = nn.AdaptiveMaxPool1d(1)
        elif self.args.window_pooling == 'avg':
            self.reduction = nn.AdaptiveAvgPool1d(1)
        else:
            raise('Unsupport window pooling type: {}'.format(self.args.window_pooling))
    
    def forward(self, x):
        windows_size = self.args.windows_size
        if self.args.window_pooling == 'linear':
            windows_x = rearrange(x, '(n l) d -> n (l d)', l=windows_size)
            res_x = self.reduction(windows_x)
        elif self.args.window_pooling == 'mlp':
            windows_x = rearrange(x, '(n l) d -> n l d', l=windows_size)
            res_x = self.reduction(windows_x)
        else:
            windows_x = rearrange(x, '(n l) d -> n d l', l = windows_size)
            res_x = self.reduction(windows_x)
            res_x = res_x.squeeze(2)
        return res_x
