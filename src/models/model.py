from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from platform import node
from turtle import forward
import xdrlib

import torchvision.models as models
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_sum
import sys
from einops import repeat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
from sklearn.cluster import KMeans

from .vit import NodeTransformer
from .fpn_tf import NodeTransformer_Hierarchical
from .hie_block_tf import Transformer_Block
from .mlps import MLPs
from .fpn_mlp import NodeMLP_Hierarchical
from .deepgate import V2CGNN, _aggr_function_factory
from .neurosat import NeuronSAT
from .layers.mlp import MLP
from utils.dag_utils import subgraph
import utils.share as share
from utils.sat_utils import cnf_simulation

def draw_pca(clause_emb, label, filename):
    pca = PCA(n_components=2)
    x = pca.fit_transform(clause_emb.detach().numpy())
    bx = []
    by = []
    rx = []
    ry = []
    for clause_idx in range(len(x)):
        if label[clause_idx]:
            rx.append(x[clause_idx][0])
            ry.append(x[clause_idx][1])
        else:
            bx.append(x[clause_idx][0])
            by.append(x[clause_idx][1])
    plt.scatter(bx, by, color='b', s=8)
    plt.scatter(rx, ry, color='r', s=8)
    plt.savefig(filename)
    plt.clf()

class SA2T(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # Select Model 
        if self.args.gnn == 'neurosat':
            self.gnn_model = NeuronSAT(args)
        else:
            raise('Unsupport GNN model {:}'.format(self.args.gnn))

        # Transformer and Readout
        if self.args.transformer_type == 'fpn':
            self.tf_clause_model = NodeTransformer_Hierarchical(args)
            self.readout = MLP(args.dim_hidden * (args.TF_depth + 1), args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=True, tanh=False)
        elif self.args.transformer_type == 'vit':
            self.tf_clause_model = NodeTransformer(args)
            self.readout = MLP(args.dim_hidden, args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=True, tanh=False)
        elif self.args.transformer_type == 'mlp':
            self.tf_clause_model = NodeMLP_Hierarchical(self.args)
            self.readout = MLP(args.dim_hidden * (args.TF_depth + 1), args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=True, tanh=False)
        elif self.args.transformer_type == 'block':
            self.tf_clause_model = Transformer_Block(self.args)
            self.readout = MLP(args.dim_hidden, args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=True, tanh=False)
        else:
            raise('Unsupport {}'.format(self.args.transformer_type))
        
        # Supervision
        if self.args.spc:
            self.clause_classifier = MLP(args.dim_hidden, args.dim_rd, 1, num_layer=args.num_rd, 
                norm_layer=args.norm_layer, act_layer=args.activation_layer, sigmoid=True, tanh=False)
            

    def change_glonode(self, G):
        # cut edge connect to literal / clause 
        edge_nopo_mask = G.gate_type[G.edge_index[1]] != self.args.gate2index['PO']
        edge_index = torch.arange(0, G.edge_index.size(1)).to(self.args.device)[edge_nopo_mask]
        G.edge_index = torch.index_select(G.edge_index, dim=1, index=edge_index)

        # add edge connect literal and PO
        for batch_idx in range(int(G.batch.max()) + 1):
            batch_mask = G.batch == batch_idx
            po_mask = batch_mask & (G.gate_type == self.args.gate2index['PO'])
            po_index = torch.arange(0, G.x.size(0)).to(self.args.device)[po_mask]
            var_mask = batch_mask & ((G.gate_type == self.args.gate2index['VAR']) | 
                                     (G.gate_type == self.args.gate2index['NEGVAR']))
            var_index = torch.arange(0, G.x.size(0)).to(self.args.device)[var_mask]
            src_list = var_index.unsqueeze(0)
            dst_list = po_index.repeat(len(var_index)).unsqueeze(0)
            new_edge = torch.cat([src_list, dst_list], dim=0)
            G.edge_index = torch.cat([G.edge_index, new_edge], dim=1)

        return G
            
    def forward(self, G):
        G = G.to(self.args.device)
        offset = 0

        if self.args.readout == 'glonode':
            G = self.change_glonode(G)
        elif self.args.readout == 'block':
            G = G
            group_res = torch.tensor([]).to(self.args.device)
        elif self.args.readout == 'average':
            G = G
        else:
            raise('Only support glonode readout')

        #######################
        # GNN
        #######################
        node_emb = self.gnn_model(G)
        # Supervised unsat core 
        if self.args.spc:
            clause_node = G.forward_index[(G.gate_type == self.args.gate2index['CLAUSE'])]
            clause_emb = torch.index_select(node_emb, dim=0, index=clause_node)
            core_res = self.clause_classifier(clause_emb)
            core_res = core_res.squeeze(1)
        else:
            core_res = []

        #######################
        # Transformer
        #######################
        cnf_emb = torch.tensor([]).to(self.args.device)
        for batch_idx in range(int(G.batch.max()) + 1):
            batch_mask = G.batch == batch_idx
            if self.args.rd_type == 'clause':
                batch_clause_node = G.forward_index[batch_mask & (G.gate_type == self.args.gate2index['CLAUSE'])]
                batch_clause_emb = torch.index_select(node_emb, dim=0, index=batch_clause_node)
                # start_index = int(G.n_clauses[:batch_idx].sum())
                # share.core_res = core_res[start_index: start_index+int(G.n_clauses[batch_idx])]
                # share.core_gt = G.unsat_core[start_index: start_index+int(G.n_clauses[batch_idx])]
                # share.first_round = True
                # share.is_sat = G.y[batch_idx]

                if self.args.readout == 'glonode':
                    batch_po_node = G.forward_index[batch_mask & (G.gate_type == self.args.gate2index['PO'])]
                    batch_po_emb = node_emb[batch_po_node]
                    batch_clause_emb = torch.cat([batch_po_emb, batch_clause_emb], dim=0)
                
                if self.args.transformer_type == 'block':
                    batch_cnf_emb, one_group_res = self.tf_clause_model(batch_clause_emb)
                    group_res = torch.cat([group_res, one_group_res], dim=0)
                else:
                    batch_cnf_emb = self.tf_clause_model(batch_clause_emb)

            if self.args.rd_type == 'literal' or self.args.rd_type == 'both':
                print('Unsupport readout Literal')

            batch_var_node = G.forward_index[batch_mask & (G.gate_type == self.args.gate2index['VAR'])]
            batch_var_emb = torch.index_select(node_emb, dim=0, index=batch_var_node)
            
            # TODO: 0801
            if G.y[batch_idx] and self.args.decode_sol:
                sat, sol = self.get_solution(self.args, G, batch_mask, batch_var_emb)
                share.tot_sat_cnt += 1
                if sat:
                    share.solve_sat_cnt += 1
                # print('Solved: {:}'.format(sat))

            cnf_emb = torch.cat([cnf_emb, batch_cnf_emb], dim=0)
                
        #######################
        # Readout
        #######################
        if self.args.readout == 'block':
            po_res = self.readout(cnf_emb).squeeze(1)
            group_res = group_res.squeeze(1)
        else:
            po_res = self.readout(cnf_emb).squeeze(1)
            group_res = []

        return po_res, group_res, core_res
        
    def get_solution(self, args, G, batch_mask, l_emb):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(l_emb)
        sol = kmeans.predict(l_emb)

        sat = cnf_simulation(args, G, batch_mask, sol)
        if not sat:
            sol[sol == 0] = -1
            sol[sol == 1] = 0
            sol[sol == -1] = 1
            sat = cnf_simulation(args, G, batch_mask, sol)

        return sat, sol

def create_model(args):
    model = SA2T(args)
    return model

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None, val_loss=None):
    start_epoch = 0
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
        if 'val_loss' in checkpoint.keys():
            val_loss = checkpoint['val_loss']
            print('The best validation loss: ', val_loss)
    if optimizer is not None:
        return model, optimizer, start_epoch, val_loss
    else:
        return model


def save_model(path, epoch, model, optimizer=None, val_loss=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    if not (val_loss is None):
        data['val_loss'] = val_loss
    torch.save(data, path)

