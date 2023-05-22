'''
    SA2T Transformer Model 
    Author: ICCAD Anonymous author
    Date: 16/06/2022
    Ref. https://blog.csdn.net/weixin_44422920/article/details/123398874
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from einops import rearrange

from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor

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
        # Windows
        windows_size = self.args.windows_size
        windows_x = rearrange(x, '(n l) d -> n l d', l=windows_size)

        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(windows_x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        x = rearrange(out, 'n l d -> (n l) d')

        return x
    
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

class clauseMerge(nn.Module):
    def __init__(self, args) -> None:
        super(clauseMerge, self).__init__()
        self.args = args
        self.dim_in = args.windows_size * args.tf_emb_size
        self.dim_out = args.tf_emb_size
        self.reduction = nn.Linear(self.dim_in, self.dim_out)
    
    def forward(self, x):
        windows_size = self.args.windows_size
        windows_x = rearrange(x, '(n l) d -> n (l d)', l=windows_size)
        # res_x = self.reduction(windows_x)
        return windows_x

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
            )), 
            clauseMerge(args)
        )

class NodeTransformer_Hierarchical(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            # ClauseEmbedding(args),
            # *[TransformerEncoderBlock(args) for _ in range(args.TF_depth)], 
        )
        self.args = args
        self.clause_emb_layer = ClauseEmbedding(args)
        # self.tf_encoder_layers = [TransformerEncoderBlock(args).to(self.args.device) for _ in range(args.TF_depth)]

        self.tf_encoder_layers = []
        first_emb_size = args.tf_emb_size
        for layer_idx in range(args.TF_depth):
            args.tf_emb_size = first_emb_size * (layer_idx * args.windows_size)
            self.tf_encoder_layers.append(TransformerEncoderBlock(args).to(self.args.device))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
    
    def padding(self, x):
        origin_length = len(x)
        windows_size = self.args.windows_size
        # Random padding
        if len(x) % windows_size != 0:
            pad_size = windows_size - (len(x) % windows_size)
        else:
            pad_size = 0
        pad_x = x[torch.randint(len(x), (pad_size, ))]
        res_x = torch.cat([x, pad_x], dim=0)
        return res_x, origin_length

    def forward(self, x):
        x = self.clause_emb_layer(x)
        for tf in self.tf_encoder_layers:
            x, _ = self.padding(x)
            x = tf(x)
        
        x = rearrange(x, 'x d -> 1 d x')
        x = self.avgpool(x)
        x = rearrange(x, '1 d 1 -> 1 d')
        return x


