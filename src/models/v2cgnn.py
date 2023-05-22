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

class V2CGNN(nn.Module):
    def __init__(self, args):
        super(V2CGNN, self).__init__()
        
        self.args = args

        # configuration
        self.device = args.device
        self.wx_update = args.wx_update

        # dimensions
        self.dim_hidden = args.dim_hidden
        self.dim_mlp = args.dim_mlp
        self.num_fc = args.num_fc
        self.node_size = args.node_size

        # 1. message/aggr-related
        if self.args.aggr_function in _aggr_function_factory.keys():
            self.aggr_reverse_var = _aggr_function_factory[self.args.aggr_function](args.pos_size, self.dim_hidden)
            self.aggr_var_to_cla = _aggr_function_factory[self.args.aggr_function](self.dim_hidden, self.dim_hidden)
        else:
            raise KeyError('no support {} aggr function.'.format(self.args.aggr_function))


        # 2. update-related
        if self.args.update_function in _update_function_factory.keys():
            # Here only consider the inputs as the concatenated vector from embedding and feature vector.
            if self.wx_update:
                self.update = MLP(args.dim_hidden+args.dim_node_feature, args.dim_mlp, args.dim_hidden, 
                              num_layer=args.num_fc, norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)
            else:
                self.update = MLP(args.dim_hidden, args.dim_mlp, args.dim_hidden, 
                              num_layer=args.num_fc, norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)
        else:
            raise KeyError('no support {} update function.'.format(self.args.update_function))

        # 3. predictor-related
        self.predictor = MLP(args.dim_hidden, args.dim_mlp, args.node_size, 
                             num_layer=args.num_fc, norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=False, tanh=False)

    def forward(self, G):
        # Dataset requirement:
        # G: {gate_type: list, n_vars: int, n_clauses: int, edge_index: list, forward_index: list}
        h_init = torch.zeros([G.num_nodes, self.dim_hidden])

        if 'gru' in self.args.update_function:
            node_embedding = self._gru_forward(G, h_init)
        else:
            raise NotImplementedError('The update function should be specified as gru.')
        
        return node_embedding
    
    def _gru_forward(self, G, h_init):
        x, edge_index = G.x, G.edge_index
        edge_attr = None
        node_state = h_init # (h_0). here we initialize h_0. TODO: option of not initializing the hidden state of GRU.
        # Position encoding
        # pos_state = torch.randn(G.num_nodes, self.args.pos_size)
        var_mask = G.gate_type == self.args.gate2index['VAR']
        var_node = G.forward_index[var_mask]
        negvar_mask = G.gate_type == self.args.gate2index['NEGVAR']
        negvar_node = G.forward_index[negvar_mask]
        cla_mask = G.gate_type == self.args.gate2index['CLAUSE']
        cla_node = G.forward_index[cla_mask]
        
        pos_state = torch.zeros([G.num_nodes, self.args.pos_size])
        pos_random = torch.randn(len(var_node), self.args.pos_size)
        pos_state[var_node, :] = pos_random
        pos_state[negvar_node, :] = -1 * pos_random
        
        # l_idx = 0 (var) / 1 (neg var)
        var_edge_index, var_edge_attr = subgraph(var_node, edge_index, edge_attr, dim=1)
        msg = self.aggr_reverse_var(pos_state, var_edge_index, var_edge_attr)
        var_msg = torch.index_select(msg, dim=0, index=var_node)
        var_x = torch.index_select(x, dim=0, index=var_node)
        if self.wx_update:
            var_state = self.update(torch.cat([var_msg, var_x], dim=1))
        else:
            var_state = self.update(var_msg)
        
        negvar_edge_index, negvar_edge_attr = subgraph(negvar_node, edge_index, edge_attr, dim=1)
        msg = self.aggr_reverse_var(pos_state, negvar_edge_index, negvar_edge_attr)
        negvar_msg = torch.index_select(msg, dim=0, index=negvar_node)
        negvar_x = torch.index_select(x, dim=0, index=negvar_node)
        if self.wx_update:
            negvar_state = self.update(torch.cat([negvar_msg, negvar_x], dim=1))
        else:
            negvar_state = self.update(negvar_msg)
        
        node_state[var_node, :] = var_state
        node_state[negvar_node, :] = negvar_state

        # Var to clause 
        cla_edge_index, cla_edge_attr = subgraph(cla_node, edge_index, edge_attr, dim=1)
        msg = self.aggr_var_to_cla(node_state, cla_edge_index, cla_edge_attr)
        cla_msg = torch.index_select(msg, dim=0, index=cla_node)
        cla_x = torch.index_select(x, dim=0, index=cla_node)
        if self.wx_update:
            cla_state = self.update(torch.cat([cla_msg, cla_x], dim=1))
        else:
            cla_state = self.update(cla_msg)
        node_state[cla_node, :] = cla_state

        node_embedding = self.predictor(node_state)
        return node_embedding
