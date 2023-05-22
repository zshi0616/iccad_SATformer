from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .benchmarks_dataset import BenchmarkDataset
from .random_dataset import RandomDataset
from .randomckt_dataset import RandomCktDataset
from .block_dataset import BlockDataset
from .cnfgen_dataset import CnfgenDataset

dataset_factory = {
  'benchmark': BenchmarkDataset, 
  'random': RandomCktDataset, 
  'block': BlockDataset, 
  'cnfgen': CnfgenDataset, 
  'sr': RandomDataset
}