'''
Implement two datasets for SAT problems. One for NeuronSAT and the other for CircuitSAT
'''

from typing import Optional, Callable, List
import os.path as osp
import random
import numpy as np
import time
import os
import subprocess

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.sat_utils import gen_iclause_pair
from utils.dataset_utils import cnf_parse_pyg
import utils.aiger_utils as aiger_utils
import utils.cnf_utils as cnf_utils

def convert_cnf_aig2aig(args, cnf, no_var):
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    cnf_filename = os.path.join(args.tmp_dir, '{:}_{:}.cnf'.format(int(time.time()), random.randint(0, 100)))
    cnf_utils.save_cnf(cnf, no_var, cnf_filename)
    aig_filename = cnf_filename.replace('.cnf', '.aig')
    aag_filename = cnf_filename.replace('.cnf', '.aag')
    x_data, edge_index = aiger_utils.cnf_to_xdata(cnf_filename, aig_filename, aag_filename)
    os.remove(cnf_filename)
    os.remove(aig_filename)
    os.remove(aag_filename)
    
    new_cnf = aiger_utils.aig_to_cnf(x_data, edge_index)
    new_var = len(x_data)
    return new_cnf, new_var

def convert_cnf_abc(args, cnf, no_var):
    if not os.path.exists(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    cnf_filename = os.path.join(args.tmp_dir, '{:}_{:}.cnf'.format(int(time.time()), random.randint(0, 100)))
    cnf_utils.save_cnf(cnf, no_var, cnf_filename)
    aig_filename = cnf_filename.replace('.cnf', '.aig')
    cnf2aig_cmd = 'cnf2aig {} {}'.format(cnf_filename, aig_filename)
    info = os.popen(cnf2aig_cmd).readlines()
    new_cnf_filename = cnf_filename.replace('.cnf', '_new.cnf')
    subprocess.call(["abc", "-c", "read %s; \
                     balance; rewrite -lz; balance; rewrite -lz; \
                     balance; rewrite -lz; balance; cec; write_cnf %s" \
            % (aig_filename, new_cnf_filename)])

    new_cnf, new_var = cnf_utils.read_cnf(new_cnf_filename)
    os.remove(cnf_filename)
    os.remove(aig_filename)
    os.remove(new_cnf_filename)
    
    return new_cnf, new_var


class RandomCktDataset(InMemoryDataset):
    r"""
    The PyG dataset for NeuroSATDataset.

    Args:
        root (string): Root directory where the dataset should be saved.
        args (object): The arguments specified by the main program.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.name = args.dataset
        self.args = args
        self.min_n = args.min_n
        self.max_n = args.max_n

        print('cnf format SR{} to SR{} problems.'.format(self.min_n, self.max_n))
        print('total # of problems: ', self.args.n_pairs)

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."
        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        if self.args.random_augment > 1:
            name = "ckt_sr{:}to{:}_p{:}x{:}".format(self.args.min_n, self.args.max_n, 
                        self.args.n_pairs, self.args.random_augment)
        else:
            name = "ckt_sr{:}to{:}_p{:}".format(self.args.min_n, self.args.max_n, 
                        self.args.n_pairs)
        if self.args.cv_ratio != -1:
            name += '_cv{:2f}'.format(self.args.cv_ratio)
        return osp.join(self.root, name)

    @property
    def raw_file_names(self) -> List[str]:
        # since the data is generated on the fly, we don't need the raw files here.
        return ['INFO']

    @property
    def processed_file_names(self) -> str:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        '''
        The cnf dataset generation proecess followed by https://github.com/ryanzhangfan/NeuroSAT/blob/master/src/data_maker.py
        Here we ignore the constraint of `max_nodes_per_batch`.
        '''
        data_list = []

        n_cnt = self.args.max_n - self.args.min_n + 1
        problems_per_n = self.args.n_pairs * 1.0 / n_cnt

        for n_var in range(self.min_n, self.max_n+1):
            lower_bound = int((n_var - self.min_n) * problems_per_n)
            upper_bound = int((n_var - self.min_n + 1) * problems_per_n)
            for problems_idx in range(lower_bound, upper_bound):
                if (problems_idx % 1000) == 0:
                    print('generate {}/{} sat problems...'.format(problems_idx, self.args.n_pairs * 2))
                
                n_vars, iclauses_sat, iclauses_unsat = gen_iclause_pair(self.args, n_var)

                # Convert to circuit and convert back 
                sat_cnf, sat_n_vars = convert_cnf_abc(self.args, iclauses_sat, n_vars)
                unsat_cnf, unsat_n_vars = convert_cnf_abc(self.args, iclauses_unsat, n_vars)

                if len(sat_cnf) <= sat_n_vars or len(unsat_cnf) <= unsat_n_vars:
                    continue

                for aug_idx in range(self.args.random_augment):
                    if aug_idx > 0:
                        random.shuffle(sat_cnf)
                        random.shuffle(unsat_cnf)

                    graph_sat = cnf_parse_pyg(self.args, sat_cnf, 1, sat_n_vars, len(sat_cnf))
                    graph_unsat = cnf_parse_pyg(self.args, unsat_cnf, 0, unsat_n_vars, len(unsat_cnf))

                    data_list.append(graph_sat)
                    data_list.append(graph_unsat)
                print('[INFO] Generate {:} / {:} sat problems...'.format(
                    len(data_list), self.args.n_pairs * self.args.random_augment
                ))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(len(data_list))

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'