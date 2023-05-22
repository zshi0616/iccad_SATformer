'''
Implement two datasets for SAT problems. One for NeuronSAT and the other for CircuitSAT
'''

from typing import Optional, Callable, List
import os.path as osp
import random
import os
import subprocess
import time

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.sat_utils import gen_iclause_pair
from utils.dataset_utils import cnf_parse_pyg, get_unsat_core, community, reorder_cnf
import utils.cnfgen_utils as cnfgen_utils
import utils.cnf_utils as cnf_utils

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


class CnfgenDataset(InMemoryDataset):
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
            name = "{}_n{:}to{:}_k{:}to{:}_x{:}".format(self.args.problem_type, 
                self.args.cnfgen_n[0], self.args.cnfgen_n[1], self.args.cnfgen_k[0], self.args.cnfgen_k[1], self.args.random_augment)
        else:
            name = "{}_n{:}to{:}_k{:}".format(self.args.problem_type, 
                self.args.cnfgen_n[0], self.args.cnfgen_n[1], self.args.cnfgen_k[0], self.args.cnfgen_k[1])
        if self.args.community: 
            name += '_com'
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

        sat_cnfs = []
        sat_vars = []
        unsat_cnfs = []
        unsat_vars = []
        
        for cnf_idx in range(self.args.n_pairs):
            n = random.randint(self.args.cnfgen_n[0], self.args.cnfgen_n[1])
            k = random.randint(self.args.cnfgen_k[0], self.args.cnfgen_k[1])
            p = self.args.cnfgen_p 
            n_var, iclauses, is_sat = cnfgen_utils.create_problem(self.args, n, p, k)
            iclauses, is_sat = convert_cnf_abc(self.args, iclauses, n_var)
            graph = cnf_parse_pyg(self.args, iclauses, 1, n_var, len(iclauses))
            data_list.append(graph)
            
            print('[INFO] Generate {:} / {:} sat problems...'.format(
                len(data_list), self.args.n_pairs
            ))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'