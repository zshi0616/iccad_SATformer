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

from .layers.mlp import MLP
from .layers.clause_merge import clauseMerge

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

class MultiHeadAttention_Windows(nn.Module):
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

class TransformerEncoderBlock_Windows(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.clause_size),
                MultiHeadAttention_Windows(args),
                nn.Dropout(args.dropout)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(args.tf_emb_size),
                FeedForwardBlock(args),
                nn.Dropout(args.dropout)
            )), 
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
            )), 
        )

class Transformer_Block(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            # ClauseEmbedding(args),
            # *[TransformerEncoderBlock(args) for _ in range(args.TF_depth)], 
        )
        self.args = args
        # self.clause_emb_layer = ClauseEmbedding(args)
        self.block = TransformerEncoderBlock_Windows(args).to(self.args.device)
        self.mlp = MLP(args.dim_hidden, args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=None, act_layer=args.activation_layer, sigmoid=True, tanh=False)
        self.hf_pool = nn.AdaptiveAvgPool1d(1)
        if self.args.level_pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif self.args.level_pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        self.clause_merge = clauseMerge(args)
        

    def forward(self, x):
        # x = self.clause_emb_layer(x)
        y_pred = None

        x_fp = rearrange(x, 'x d -> 1 d x')
        x_fp = self.pool(x_fp)
        x_fp = rearrange(x_fp, '1 d 1 -> 1 d')
        hf_emb = x_fp
        
        while len(x) > 1:
            x = self.block(x)

            # Level embedding
            x_fp = rearrange(x, 'x d -> 1 d x')
            x_fp = self.pool(x_fp)
            x_fp = rearrange(x_fp, '1 d 1 -> 1 d')
            hf_emb = torch.cat([hf_emb, x_fp], dim=0)

            # Group embedding
            x = self.clause_merge(x)

            if y_pred == None:
                y_pred = self.mlp(x)
            else:
                y_pred = torch.cat([y_pred, self.mlp(x)], dim=0)

        # for level in range(1):
        #     x = self.block(x)

        #     # Level embedding
        #     x_fp = rearrange(x, 'x d -> 1 d x')
        #     x_fp = self.pool(x_fp)
        #     x_fp = rearrange(x_fp, '1 d 1 -> 1 d')
        #     hf_emb = torch.cat([hf_emb, x_fp], dim=0)

        #     # Group embedding
        #     x = self.clause_merge(x)
        # y_pred = torch.zeros([5, 1])
            
        hf_emb = rearrange(hf_emb, 'x d -> 1 d x')
        hf_emb = self.hf_pool(hf_emb)
        hf_emb = rearrange(hf_emb, '1 d 1 -> 1 d')

        return hf_emb, y_pred


