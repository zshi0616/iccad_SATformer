import os 
import torch
import numpy as np
from torchsummary import summary
from torch_geometric.loader import DataLoader, DataListLoader

from config import get_parse_args
from utils.logger import Logger
from datasets.dataset_factory import dataset_factory
from models.model import create_model, load_model, save_model
from trains.train import SA2T_Trainer

if __name__ == '__main__':
    args = get_parse_args()
    print('==> Using settings {}'.format(args))
    logger = Logger(args)

    # Dataset
    dataset = dataset_factory[args.dataset](args.data_dir, args)
