import enum
import os 
import torch
import numpy as np
from torchsummary import summary
from torch_geometric.loader import DataLoader, DataListLoader
from progress.bar import Bar
from sklearn.metrics import roc_curve, auc

from config import get_parse_args
from utils.logger import Logger
from datasets.dataset_factory import dataset_factory
from models.model import create_model, load_model, save_model
from detectors.detector import SA2T_Detector

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = get_parse_args()
    print('==> Using settings {}'.format(args))

    logger = Logger(args)
    print('Using device: ', args.device)

    # Dataset
    dataset = dataset_factory[args.dataset](args.data_dir, args)
    dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Detector
    detector = SA2T_Detector(args)
    
    # Inference
    num_iters = len(dataset)
    bar = Bar('{}'.format(args.exp_id), max=num_iters)
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    g_cnt = 0
    output_list = []
    y_list = []

    # Stat Dataset
    num_clauses = 0
    num_vars = 0
    for idx, g in enumerate(dataset):
        num_vars += int(g.n_vars.sum())
        num_clauses += int(g.n_clauses.sum())
    print('[INFO] Dataset C/V: {:.4f}'.format(num_clauses*1.0 / num_vars))

    for idx, g in enumerate(dataset):
        # if idx != 1 and idx != 3:
        #     continue

        ret = detector.run(g)
        
        binary_res = ret['results'][0].to('cpu')
        pred = (binary_res > args.thro).int()
        y = (g.y > args.thro).int()
        output_list += binary_res.tolist()
        y_list += y.tolist()
        g_cnt += len(y)

        if args.spc:
            core_res = ret['results'][2].to('cpu')
            core_acc = ((core_res > 0.1288) & (g.unsat_core > 0.5)).sum() / len(core_res)
            # output_list += core_res.tolist()
            # y_list += (g.unsat_core > 0.5).int().tolist()


        tp += ((pred == 1) & (y == 1)).sum()
        tn += ((pred == 0) & (y == 0)).sum()
        fp += ((pred == 1) & (y == 0)).sum()
        fn += ((pred == 0) & (y == 1)).sum()
        
        if args.spc:
            print('[INFO] Batch {:} in {:.2f}ms, Acc: {:.4f}, Core Acc: {:.4f}'.format(
                idx, ret['net_time']*1000, (pred == y).sum()*1.0/len(y), core_acc))
        else:
            print('[INFO] Batch {:} in {:.2f}ms, Acc: {:.4f}'.format(
                idx, ret['net_time']*1000, (pred == y).sum()*1.0/len(y)))

    preds_val = np.array(output_list)
    y_val = np.array(y_list)
    fpr, tpr, thro = roc_curve(y_val, preds_val, pos_label=1)
    roc_auc = auc(fpr, tpr)

    print('*'*20)
    print('TP: {:.2f}%'.format(tp*100.0 / g_cnt))
    print('TN: {:.2f}%'.format(tn*100.0 / g_cnt))
    print('FP: {:.2f}%'.format(fp*100.0 / g_cnt))
    print('FN: {:.2f}%'.format(fn*100.0 / g_cnt))
    print('*'*10)
    print('ACC: {:.2f}%'.format((tp+tn)*100.0 / g_cnt))
    print('AUC: {:.2f}'.format(roc_auc))
    print('*'*20)

    # Adjust threshold by AUC
    opt_thro = thro[np.argmax(tpr - fpr)]
    pred = (preds_val > opt_thro)
    print()
    print('='*20)
    print('Threshold: {:.4f}'.format(opt_thro))
    print('TP: {:.2f}%'.format(((pred == 1) & (y_val == 1)).sum()*100.0 / g_cnt))
    print('TN: {:.2f}%'.format(((pred == 0) & (y_val == 0)).sum()*100.0 / g_cnt))
    print('FP: {:.2f}%'.format(((pred == 1) & (y_val == 0)).sum()*100.0 / g_cnt))
    print('FN: {:.2f}%'.format(((pred == 0) & (y_val == 1)).sum()*100.0 / g_cnt))
    print('ACC: {:.2f}%'.format((pred == y_val).sum()*100.0 / g_cnt))
    print('='*20)

    