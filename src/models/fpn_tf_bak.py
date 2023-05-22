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
        self.qkv = nn.Linear(args.clause_size, self.emb_size * 3)
        self.att_drop = nn.Dropout(args.dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)

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
        self.qkv = nn.Linear(args.clause_size, self.emb_size * 3)
        self.att_drop = nn.Dropout(args.dropout)
        self.projection = nn.Linear(self.emb_size, self.emb_size)
        
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

class NodeTransformer_Hierarchical(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            # ClauseEmbedding(args),
            # *[TransformerEncoderBlock(args) for _ in range(args.TF_depth)], 
        )
        self.args = args
        self.clause_emb_layer = ClauseEmbedding(args)
        self.tf_encoder_layers = TransformerEncoderBlock(args).to(self.args.device)

        tf_fp_layers = [TransformerEncoderBlock_Windows(args).to(self.args.device) for _ in range(args.TF_depth)]
        self.tf_fp_layers = nn.Sequential(*tf_fp_layers)
        
        if self.args.level_pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif self.args.level_pooling == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        self.clause_merge = clauseMerge(args)
    
    def padding(self, x):
        origin_length = len(x)
        windows_size = self.args.windows_size
        # Random padding
        if len(x) % windows_size != 0:
            pad_size = windows_size - (len(x) % windows_size)
        else:
            pad_size = 0
        # Note 2020.07.05: Try change to determinstic padding, instead of random
        # pad_x = x[torch.randint(len(x), (pad_size, ))]
        if x.shape[0] < pad_size: 
            tmp_x = x.repeat(int(pad_size/x.shape[0]) + 1, 1)
            pad_x = tmp_x[0: pad_size]
        else:
            pad_x = x[0: pad_size]
        res_x = torch.cat([x, pad_x], dim=0)
        return res_x, origin_length

    def forward(self, x):
        x = self.clause_emb_layer(x)
        x = self.tf_encoder_layers(x)
        
        # Hier Feature vector 
        if self.args.readout == 'glonode':
            hf_emb = x[0, :]
            x = x[1:, :]
        elif self.args.readout == 'average':
            hf_emb = rearrange(x, 'x d -> 1 d x')
            hf_emb = self.pool(hf_emb)
            hf_emb = rearrange(hf_emb, '1 d 1 -> d')
        else:
            raise ('Unsupport Readout: {}'.format(self.args.readout))

        # Feature Pyramid 
        for tf in self.tf_fp_layers:
            x, _ = self.padding(x)
            x = tf(x)

            # Level embedding
            x_fp = rearrange(x, 'x d -> 1 d x')
            x_fp = self.pool(x_fp)
            x_fp = rearrange(x_fp, '1 d 1 -> d')
            hf_emb = torch.cat([hf_emb, x_fp], dim=0)

            # Group embedding
            x = self.clause_merge(x)

        hf_emb = hf_emb.unsqueeze(0)
        return hf_emb


