'''
    SA2T Transformer Model 
    Author: ICCAD Anonymous author
    Date: 24/05/2022
    Ref. https://github.com/FrancescoSaverioZuppichini/ViT
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
import utils.share as share
from utils.debug_utils import stat_attn

class ClauseEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args 
        self.projection = nn.Sequential( 
            nn.Linear(args.node_size, args.clause_size)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.emb_size = args.tf_emb_size
        self.num_heads = args.head_num
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(args.clause_size, self.emb_size * 3 * self.num_heads)
        self.att_drop = nn.Dropout(args.dropout)
        self.projection = nn.Linear(self.emb_size * self.num_heads, self.emb_size)
        
    def forward(self, x : Tensor) -> Tensor:
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "n (h d qkv) -> (qkv) h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('hqd, hkd -> hqk', queries, keys) # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('hal, hlv -> hav ', att, values)
        out = rearrange(out, "h n d -> n (h d)")
        out = self.projection(out)

        # Debug
        # a = energy[0]
        # plt.subplots(figsize=(len(a), len(a)))
        # sns.heatmap(a, annot=True, vmax=1, square=True, cmap="Reds")
        # plt.savefig('./fig/att.png')
        # plt.clf()
        
        # if share.first_round:
        #     stat_attn(share.core_res, share.core_gt, att[0])
        #     share.first_round = False
        
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            nn.Linear(args.tf_emb_size, args.MLP_expansion * args.tf_emb_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.MLP_expansion * args.tf_emb_size, args.tf_emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.clause_size),
                MultiHeadAttention(args),
                nn.Dropout(args.dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.tf_emb_size),
                FeedForwardBlock(args),
                nn.Dropout(args.dropout)
            )
            ))

class NodeTransformer(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            # ClauseEmbedding(args),
            # *[TransformerEncoderBlock(args) for _ in range(args.TF_depth)], 
        )
        self.args = args
        # self.clause_emb_layer = ClauseEmbedding(args)
        self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(args.TF_depth)]
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x = self.clause_emb_layer(x)
        for tf in self.tf_encoder_layers:
            x = tf(x)
        
        if self.args.readout == 'glonode':
            x = x[0, :].unsqueeze(0)
        elif self.args.readout == 'average':
            x = rearrange(x, 'x d -> 1 d x')
            x = self.avgpool(x)
            x = rearrange(x, '1 d 1 -> 1 d')
        else:
            raise ('Unsupport Readout: {}'.format(self.args.readout))

        return x
