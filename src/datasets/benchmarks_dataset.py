'''
Implement two datasets for SAT problems. One for NeuronSAT and the other for CircuitSAT
'''

from tokenize import Triple
from typing import Optional, Callable, List
import os.path as osp
import random
import os
import glob

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from utils.sat_utils import gen_iclause_pair, solve_sat
from utils.dataset_utils import cnf_parse_pyg


def parse_cnf_file(cnf_path):
    f = open(cnf_path, 'r')
    lines = f.readlines()
    f.close()

    n_vars = -1
    n_clauses = -1
    begin_parse_cnf = False
    iclauses = []
    for line in lines:
        if begin_parse_cnf:
            arr = line.replace('\n', '').split(' ')
            clause = []
            for ele in arr:
                if ele.replace('-', '').isdigit() and ele != '0':
                    clause.append(int(ele))
            if len(clause) > 0:
                iclauses.append(clause)
                
        elif line.replace(' ', '')[0] == 'c':
            continue
        elif line.replace(' ', '')[0] == 'p': 
            arr = line.replace('\n', '').split(' ')
            get_cnt = 0
            for ele in arr:
                if ele == 'p':
                    get_cnt += 1
                elif ele == 'cnf':
                    get_cnt += 1
                elif ele != '':
                    if get_cnt == 2:
                        n_vars = int(ele)
                        get_cnt += 1
                    else: 
                        n_clauses = int(ele)
                        break
            assert n_vars != -1
            assert n_clauses != -1
            begin_parse_cnf = True
        
    
    return iclauses, n_vars, n_clauses

class BenchmarkDataset(InMemoryDataset):
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

        assert (transform == None) and (pre_transform == None) and (pre_filter == None), "Cannot accept the transform, pre_transfrom and pre_filter args now."

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        name = 'benchmark_{}'.format(self.args.benchmark_name)
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
        data_list = []
        
        # TODO: parse benchmark
        for file in glob.glob(os.path.join(self.args.rawdata_dir, '*.cnf')):
            cnf_name = file.split('/')[-1].split('.')[0]
            iclauses, n_vars, n_clauses = parse_cnf_file(file)
            print('[INFO] Parsing {}, # Vars: {:} # Clauses: {:}'.format(file, n_vars, n_clauses))
            is_sat, sol = solve_sat(n_vars, iclauses)
            if is_sat:
                graph = cnf_parse_pyg(self.args, iclauses, 1, n_vars, n_clauses)
            else:
                graph = cnf_parse_pyg(self.args, iclauses, 0, n_vars, n_clauses)
            data_list.append(graph)
            print('{} SAT/UNSAT: {:}'.format(cnf_name, int(is_sat)))
            print()
            
        for file in glob.glob(os.path.join(self.args.rawdata_dir, '*.txt')):
            cnf_name = file.split('/')[-1].split('.')[0]
            iclauses, n_vars, n_clauses = parse_cnf_file(file)
            print('[INFO] Parsing {}, # Vars: {:} # Clauses: {:}'.format(file, n_vars, n_clauses))
            is_sat, sol = solve_sat(n_vars, iclauses)
            if is_sat:
                graph = cnf_parse_pyg(self.args, iclauses, 1, n_vars, n_clauses)
            else:
                graph = cnf_parse_pyg(self.args, iclauses, 0, n_vars, n_clauses)
            data_list.append(graph)
            print('{} SAT/UNSAT: {:}'.format(cnf_name, int(is_sat)))
            print()

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'