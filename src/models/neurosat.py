from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
from torch.nn import LSTM, GRU
import random
from einops import repeat
import numpy as np

from .layers.gat_conv import AGNNConv
from .layers.gcn_conv import AggConv
from .layers.deepset_conv import DeepSetConv
from .layers.gated_sum_conv import GatedSumConv
from .layers.mlp import MLP
from .layers.layernorm_gru import LayerNormGRU
from .layers.layernorm_lstm import LayerNormLSTM
from utils.dag_utils import subgraph

_aggr_function_factory = {
    'agnnconv': AGNNConv,
    'deepset': DeepSetConv,
    'gated_sum': GatedSumConv,
    'conv_sum': AggConv,
}

_update_function_factory = {
    'lstm': LSTM,
    'gru': GRU,
    'layernorm_lstm': LayerNormLSTM,
    'layernorm_gru': LayerNormGRU,
}

class NeuronSAT(nn.Module):
    '''
    Recurrent Graph Neural Networks of NeuroSAT.
    The structure follows the pytorch version: https://github.com/ryanzhangfan/NeuroSAT
    '''
    def __init__(self, args):
        super(NeuronSAT, self).__init__()
        
        self.args = args

         # configuration
        self.num_rounds = self.args.num_rounds
        # assert self.num_rounds == 26, '# rounds in NeuroSAT is 26'
        self.device = self.args.device

        # dimensions
        self.dim_hidden = args.dim_hidden
        # assert self.dim_hidden == 128, 'Size of hidden state in NeuroSAT is 128'
        self.dim_mlp = args.dim_mlp
        self.num_fc = args.num_fc
        # assert self.num_fc == 3, '# layers in NeuroSAT is 3'

        # 1. message/aggr-related
        # assert self.args.aggr_function == 'deepset', 'The aggregation function used in NeuroSAT is deepset.'
        L_msg = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=self.num_fc, p_drop=0.)
        C_msg = MLP(self.dim_hidden, self.dim_hidden, self.dim_hidden, num_layer=self.num_fc, p_drop=0.)

        self.aggr_forward = _aggr_function_factory[self.args.aggr_function](self.dim_hidden, mlp=L_msg)
        self.aggr_backward = _aggr_function_factory[self.args.aggr_function](self.dim_hidden, mlp=C_msg, reverse=True)
        

        # 2. update-related
        assert self.args.update_function == 'lstm', 'The update function used in NeuroSAT is LSTM'
        self.L_update = _update_function_factory[self.args.update_function](self.dim_hidden*2, self.dim_hidden)
        self.C_update = _update_function_factory[self.args.update_function](self.args.dim_hidden, self.dim_hidden)

        # consider the embedding for the LSTM/GRU model initialized by non-zeros
        self.one = torch.ones(1).to(self.device)
        self.L_init = nn.Linear(1, self.dim_hidden)
        self.C_init = nn.Linear(1, self.dim_hidden)
        self.one.requires_grad = False

    def forward(self, G):
        num_nodes = G.num_nodes
        node_embedding = self._lstm_forward(G, num_nodes)
        
        return node_embedding
            
    
    def _lstm_forward(self, G, num_nodes):
        edge_index = G.edge_index
        gate_type = G.gate_type.unsqueeze(1)

        var_mask = (gate_type == self.args.gate2index['VAR'])
        negvar_mask = (gate_type == self.args.gate2index['NEGVAR'])
        l_mask = var_mask | negvar_mask
        c_mask = (gate_type == self.args.gate2index['CLAUSE'])
        l_init = self.L_init(self.one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        l_init = l_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        c_init = self.C_init(self.one).view(1, 1, -1) # (1 x 1 x dim_hidden)
        c_init = c_init.repeat(1, num_nodes, 1) # (1 x num_nodes x dim_hidden)
        h_init = l_init * l_mask + c_init * c_mask

        if self.args.readout == 'glonode':
            po_mask = (gate_type == self.args.gate2index['PO'])
            po_init = self.C_init(torch.zeros(1).to(self.device)).view(1, 1, -1)
            po_init = po_init.repeat(1, num_nodes, 1)
            h_init += po_init * po_mask
            c_mask = c_mask | (gate_type == self.args.gate2index['PO'])

        c_index = torch.arange(0, G.x.size(0)).to(self.device)[c_mask.squeeze(1)]
        l_index = torch.arange(0, G.x.size(0)).to(self.device)[l_mask.squeeze(1)]

        node_state = (h_init, torch.zeros(1, num_nodes, self.dim_hidden).to(self.device)) # (h_0, c_0). here we only initialize h_0.
        
        for no_rounds in range(self.num_rounds):
            # forward layer
            c_state = (torch.index_select(node_state[0], dim=1, index=c_index), 
                        torch.index_select(node_state[1], dim=1, index=c_index))
                            
            msg = self.aggr_forward(node_state[0].squeeze(0), edge_index)
            c_msg = torch.index_select(msg, dim=0, index=c_index)
            
            _, c_state = self.C_update(c_msg.unsqueeze(0), c_state)

            node_state[0][:, c_index, :] = c_state[0]
            node_state[1][:, c_index, :] = c_state[1]

            # backward layer
            l_state = (torch.index_select(node_state[0], dim=1, index=l_index), 
                        torch.index_select(node_state[1], dim=1, index=l_index))
            msg = self.aggr_backward(node_state[0].squeeze(0), edge_index)
            l_msg = torch.index_select(msg, dim=0, index=l_index)
            
            l_neg = self.flip(G, node_state[0].squeeze(0))
                
            _, l_state = self.L_update(torch.cat([l_msg, l_neg], dim=1).unsqueeze(0), l_state)
            
            node_state[0][:, l_index, :] = l_state[0]
            node_state[1][:, l_index, :] = l_state[1]

        node_embedding = node_state[0].squeeze(0)

        # logits = torch.index_select(node_state[0].squeeze(0), dim=0, index=l_index)  
        # vote = torch.zeros((num_nodes, 1)).to(self.device)
        # vote[l_index, :] = self.L_vote(logits)
        # vote_mean = scatter_sum(vote, G.batch, dim=0).squeeze(1) / (G.n_vars * 2)

        return node_embedding
    
    def flip(self, G, state):
        offset = 0
        select_index = []
        if G.num_graphs == 1 and isinstance(G.n_vars, int):
            n_vars = G.n_vars
            n_nodes = G.n_nodes
            select_index.append(torch.arange(offset+n_vars, offset+2*n_vars, dtype=torch.long))
            select_index.append(torch.arange(offset, offset+n_vars, dtype=torch.long))
            offset += n_nodes
        else:
            for idx_g in range(G.num_graphs):
                n_vars = G.n_vars[idx_g]
                n_nodes = G.n_nodes[idx_g]
                select_index.append(torch.arange(offset+n_vars, offset+2*n_vars, dtype=torch.long))
                select_index.append(torch.arange(offset, offset+n_vars, dtype=torch.long))
                offset += n_nodes
        select_index = torch.cat(select_index, dim=0).to(self.device)

        flip_state = torch.index_select(state, dim=0, index=select_index)

        return flip_state
