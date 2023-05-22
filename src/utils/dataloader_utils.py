import os 
import torch
import numpy as np
import random
import copy

from torch_geometric.loader import DataLoader
from typing import Union, List
from collections.abc import Mapping, Sequence
import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, HeteroData, Dataset, Batch

class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data) or isinstance(elem, HeteroData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)
 
class SA2T_DataLoader(DataLoader):
    def __init__(
        self,
        args, 
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: List[str] = [],
        exclude_keys: List[str] = [],
        **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
        self.args = args

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(follow_batch,
                                             exclude_keys), **kwargs)

    def shuffle_dataloader(self, args, data):
        tmp_unsatcore = copy.deepcopy(data.unsat_core)

        # Generate shuffle index
        clause_random_list = []
        for clause_idx in range(data.n_clauses[0]):
            clause_random_list.append(clause_idx)
        random.shuffle(clause_random_list)
        clause_start_pos = data.n_vars[0] * 2

        # Reorder
        for idx in range(data.n_clauses[0]):
            data.unsat_core[idx] = tmp_unsatcore[clause_random_list[idx]]
        for idx in range(len(data.edge_index[0])):
            if data.gate_type[data.edge_index[0][idx]] == args.gate2index['CLAUSE']:
                data.edge_index[0][idx] = clause_random_list[data.edge_index[0][idx] - clause_start_pos] + clause_start_pos
            if data.gate_type[data.edge_index[1][idx]] == args.gate2index['CLAUSE']:
                data.edge_index[1][idx] = clause_random_list[data.edge_index[1][idx] - clause_start_pos] + clause_start_pos
        
        return data


    def shuffle(self): 
        for idx in range(len(self.dataset)):
            self.shuffle_dataloader(self.args, self.dataset[idx])
