import os
import argparse
import torch

def get_parse_args():
    parser = argparse.ArgumentParser(description='SATisfiability Transformer (SA2T)')
    # Top
    parser.add_argument('--exp_id', default='default')
    parser.add_argument('--gnn', default='neurosat', type=str, choices=['deepgate', 'neurosat'])
    parser.add_argument('--resume', action='store_true', default=False)
    
    # Experiment
    parser.add_argument('--transformer_type', type=str, default='fpn', choices=['fpn', 'vit', 'mlp', 'block'])

    # Transformer Configuration
    parser.add_argument('--TF_depth', type=int, default=4, help='The depth of TF blocks')
    parser.add_argument('--clause_size', type=int, default=128, help='The dimension of clause embedding in TF')
    parser.add_argument('--tf_emb_size', type=int, default=128, help='The dimension of TF embedding, Q K V')
    parser.add_argument('--head_num', type=int, default=8, help='The number of attention heads')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--MLP_expansion', type=int, default=4, help='The O/I ratio of MLP in TF')

    # Swin Transformer Configuration
    parser.add_argument('--windows_size', type=int, default=4)
    parser.add_argument('--window_pooling', type=str, default='linear', choices=['avg', 'max', 'linear', 'mlp'])
    parser.add_argument('--level_pooling', type=str, default='max', choices=['avg', 'max'])

    # DeepGate
    parser.add_argument('--node_size', type=int, default=128, help='The dimension of node embedding in GNN')
    parser.add_argument('--pos_size', type=int, default=128, help='The dimension of position encoding')
    parser.add_argument('--activation_layer', default='relu', type=str, choices=['relu', 'relu6', 'sigmoid'],
                             help='The activation function to use in the FC layers.')  
    parser.add_argument('--norm_layer', default='batchnorm', type=str,
                             help='The normalization function to use in the FC layers.')
    parser.add_argument('--num_fc', default=3, type=int,
                             help='The number of FC layers')                          
    parser.add_argument('--aggr_function', default='deepset', type=str, choices=['deepset', 'agnnconv', 'gated_sum', 'conv_sum'],
                             help='the aggregation function to use.')
    parser.add_argument('--update_function', default='lstm', type=str, choices=['gru', 'lstm', 'layernorm_lstm', 'layernorm_gru'],
                             help='the update function to use.')
    parser.add_argument('--wx_update', action='store_true', default=False,
                            help='The inputs for the update function considers the node feature of mlp.')
    parser.add_argument('--dim_hidden', type=int, default=128, metavar='N',
                             help='hidden size of recurrent unit.')
    parser.add_argument('--dim_mlp', type=int, default=32, metavar='N',
                             help='hidden size of readout layers') 
    parser.add_argument('--wx_mlp', action='store_true', default=False,
                             help='The inputs for the mlp considers the node feature of mlp.')    

    # NeuroSAT
    parser.add_argument('--num_rounds', type=int, default=5)
    parser.add_argument('--decode_sol', action='store_true', default=False)

    # Readout Configuration
    parser.add_argument('--dim_rd', type=int, default=128, help='The number of Readout hidden dimension')
    parser.add_argument('--num_rd', default=3, type=int, help='The number of Readout FC layers')
    parser.add_argument('--rd_type', type=str, default='clause', choices=['clause', 'literal', 'both'])
    parser.add_argument('--readout', type=str, default='average', choices=['average', 'max', 'clstoken', 'glonode', 'block'])


    # Training
    parser.add_argument('--gpu', action='store_true', default=False)  
    parser.add_argument('--batch_size', type=int, default=32,
                             help='batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                             help='dataloader threads. 0 for single-thread.')
    parser.add_argument('--lr', type=float, default=1.0e-4, 
                             help='learning rate for batch size 32.')
    parser.add_argument('--weight_decay', type=float, default=1e-10, 
                             help='weight decay (default: 1e-10)')
    parser.add_argument('--lr_step', type=str, default='',
                             help='drop learning rate by 10.')
    parser.add_argument('--grad_clip', type=float, default=0.,
                        help='gradiant clipping')
    parser.add_argument('--num_epochs', type=int, default=200,
                             help='total training epochs.')
    parser.add_argument('--loss', default='BCE', choices=['CrossEntropy', 'BCE'])
    parser.add_argument('--load_model', default='')
    parser.add_argument('--num_iters', type=int, default=-1,
                             help='default: #samples / batch_size.')
    parser.add_argument('--save_all', action='store_true',
                        help='save model to disk every 5 epochs.')
    parser.add_argument('--save_intervals', type=int, default=5,
                             help='number of epochs to run validation.')
    # parser.add_argument('--spc', action='store_true', default=False, help='supervise clause')
    parser.add_argument('--spc_kl', action='store_true', default=False)
    parser.add_argument('--spc', action='store_true', default=False)
    parser.add_argument('--spg', action='store_true', default=False, help='Supervise group by y_window')
    parser.add_argument('--spc_weight', type=float, default=9.0, help='In dataset neg:pos = 9:1')
    parser.add_argument('--spg_weight', type=float, default=100.0, help='In dataset neg:pos = 100:1')
    
    parser.add_argument('--binary_loss_weight', type=float, default=5)
    parser.add_argument('--clause_loss_weight', type=float, default=1)
    parser.add_argument('--group_loss_weight', type=float, default=1)

    # Dataset
    parser.add_argument('--reverse_label', action='store_true', default=False)
    parser.add_argument('--dataset', default='random', type=str, choices=['benchmark', 'random', 'block', 'cnfgen', 'sr'],
                             metavar='NAME', help='target dataset')
    parser.add_argument('--dim_node_feature', type=int, default=4)
    parser.add_argument('--trainval_split', default=0.9, type=float,
                             help='the splitting setting for training dataset and validation dataset.')
    parser.add_argument('--random_augment', type=int, default=1)
    parser.add_argument('--community', action='store_true', default=False)
    parser.add_argument('--shuffle', action='store_true', default=False)
    parser.add_argument('--disable_core', default=False, action='store_true')
    
    # Dataset - Benchmark
    parser.add_argument('--benchmark_name', type=str, default='100-430', help='The folder where download benchmark')
    parser.add_argument('--rawdata_dir', type=str)
    
    # Dataset - Random
    parser.add_argument('--min_n', type=int, default=3)
    parser.add_argument('--max_n', type=int, default=10)
    parser.add_argument('--n_pairs', type=int, default=1000)
    parser.add_argument('--p_k_2', type=float, default=0.3)
    parser.add_argument('--p_geo', type=float, default=0.4)
    parser.add_argument('--cv_ratio', type=float, default=-1)

    # Dataset - cnfgen problem 
    parser.add_argument('--problem_type', type=str, default='kcolor', choices=['kcolor', 'kclique', 'domset'])
    parser.add_argument('--cnfgen_n', type=str, default='6,12')
    parser.add_argument('--cnfgen_k', type=str, default='3,5')
    parser.add_argument('--cnfgen_p', type=float, default=0.37)

    # Test
    parser.add_argument('--thro', type=float, default=0.5)

    args = parser.parse_args()

    if args.dataset == 'benchmark':
        args.dataset_name = '{}_{}'.format(args.dataset, args.benchmark_name)
    elif args.dataset == 'sr' or args.dataset == 'random':
        args.dataset_name = '{}_sr{:}_{:}_{:}'.format('random', args.min_n, args.max_n, args.n_pairs)
    else:
        args.dataset_name = '{}_sr{:}_{:}_{:}'.format(args.dataset, args.min_n, args.max_n, args.n_pairs)
    if args.community:
        args.random_augment = 1
    args.cnfgen_n = args.cnfgen_n.split(',')
    args.cnfgen_n[0] = int(args.cnfgen_n[0])
    args.cnfgen_n[1] = int(args.cnfgen_n[1])
    args.cnfgen_k = args.cnfgen_k.split(',')
    args.cnfgen_k[0] = int(args.cnfgen_k[0])
    args.cnfgen_k[1] = int(args.cnfgen_k[1])

    # Output 
    args.root_dir = os.path.join(os.path.dirname(__file__), '..')
    args.save_dir = os.path.join(args.root_dir, 'exp', args.exp_id)
    args.debug_dir = os.path.join(args.save_dir, 'debug')
    args.data_dir = os.path.join(args.root_dir, 'data', args.dataset_name)
    args.tmp_dir = os.path.join(args.save_dir, 'tmp')
    
    # Model
    if args.resume:
        model_path = args.save_dir
        if args.load_model == '':
            args.load_model = os.path.join(model_path, 'model_last.pth')
        else:
            args.load_model = os.path.join(model_path, args.load_model)

    if args.gpu:
        if torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'
    else:
        args.device = 'cpu'

    args.gate2index = {'VAR': 0, 'NEGVAR': 1, 'CLAUSE': 2, 'PO': 3}
    if args.lr_step == '':
        args.lr_step = []
    else:
        args.lr_step = [int(i) for i in args.lr_step.split(',')]

    # if args.batch_size > 1:
    #     print('[Warning] No support batch yet')
    #     args.batch_size = 1

    return args
