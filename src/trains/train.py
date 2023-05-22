from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn 
import numpy as np
from torch_geometric.nn import DataParallel
from .loss import SmoothStep
from progress.bar import Bar
from utils.train_utils import AverageMeter
import torch.nn.functional as F
# from thop import profile, clever_format

_loss_factory = {
    'BCE': nn.BCEWithLogitsLoss
}

class ModelWithLoss(torch.nn.Module):
    def __init__(self, args, model, loss):
        super(ModelWithLoss, self).__init__()
        self.args = args
        self.model = model.to(self.args.device)
        self.loss = loss

    def forward(self, batch):

        # flops, params = profile(self.model, inputs=(batch,))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops)
        # print(params)

        po_outputs, group_outputs, clause_outputs = self.model(batch)
        # Binary output loss
        bl = self.loss[0](po_outputs, batch.y.float())
        
        # Clause loss
        if self.args.spc:
            if not self.args.spc_kl:
                cl = self.loss[1](clause_outputs, batch.unsat_core.float())
            else:
                cl = 0
                offset = 0
                unsat_batch_cnt = 0
                for batch_idx in range(int(batch.batch.max()) + 1):
                    if (batch.y[batch_idx] == 1 and self.args.reverse_label == False) or \
                        (batch.y[batch_idx] == 0 and self.args.reverse_label == True): 
                        continue
                    unsat_batch_cnt += 1
                    batch_out = clause_outputs[offset: offset + batch.n_clauses[batch_idx]]
                    batch_y = batch.unsat_core[offset: offset + batch.n_clauses[batch_idx]]
                    offset += batch.n_clauses[batch_idx]
                    cl += F.kl_div(batch_out.softmax(dim=-1).log(), batch_y.float().softmax(dim=-1), reduction='sum')
                if unsat_batch_cnt > 0:
                    cl /= unsat_batch_cnt
                else:
                    cl = torch.tensor(0, dtype=torch.float)

            loss = (bl * self.args.binary_loss_weight + cl * self.args.clause_loss_weight) \
                / (self.args.binary_loss_weight + self.args.clause_loss_weight)
            loss_stats = {'loss': loss, 'bl': bl, 'cl': cl}
            task2_outputs = clause_outputs
        elif self.args.spg:
            gl = self.loss[2](group_outputs, batch.y_window.float())
            loss = (bl * self.args.binary_loss_weight + gl * self.args.group_loss_weight) \
                / (self.args.binary_loss_weight + self.args.group_loss_weight)
            loss_stats = {'loss': loss, 'bl': bl, 'gl': gl}
            task2_outputs = group_outputs
        else:
            cl = 0
            task2_outputs = batch.unsat_core
            loss = bl
            loss_stats = {'loss': loss}

        if self.args.transformer_type == 'block':
            outputs = po_outputs
            
        return po_outputs, task2_outputs, loss, loss_stats


class SA2T_Trainer(object):
    def __init__(
            self, args, model, optimizer=None):
        self.args = args
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(args)
        self.model_with_loss = ModelWithLoss(args, model, self.loss)
        self.avg_prec_stats = ['ACC']

    def set_device(self, device):
        self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, dataset):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            model_with_loss.model.soft = True
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
            model_with_loss.model.soft = False

        args = self.args
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = int(len(dataset)) if args.num_iters < 0 else args.num_iters
        bar = Bar('{}'.format(args.exp_id), max=num_iters)
        end = time.time()

        if self.args.spc:
            acc_stats = ['acc', 'musacc', 'tp', 'tn', 'fp', 'fn']
        elif self.args.spg:
            acc_stats = ['acc', 'gacc', 'tp', 'tn', 'fp', 'fn']
        else:
            acc_stats = ['acc']
        avg_acc_stats = {l: AverageMeter() for l in acc_stats}

        if args.shuffle and epoch % 10 == 0 and epoch != 0:
            dataset.shuffle()
        for iter_id, batch in enumerate(dataset):
            if iter_id >= num_iters:
                break 
            # Reverse 
            if args.reverse_label:
                batch.y = 1 - batch.y
            # batch = batch.to(self.args.device)
            data_time.update(time.time() - end)
            batch = batch.to(self.args.device)

            output, task2_outputs, loss, loss_stats = model_with_loss(batch)
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model_with_loss.parameters(), args.grad_clip)
                self.optimizer.step()
            
            preds = (output > 0.5).int()
            y = (batch.y > 0.5).int()
            acc = float((preds == y).sum()) / int(len(preds))
            avg_acc_stats['acc'].update(acc, batch.num_graphs)

            if self.args.spc: 
                clause_preds = (task2_outputs > 0.5).int()
                clause_y = (batch.unsat_core > 0.5).int()
                clause_acc = float((clause_preds == clause_y).sum()) / int(len(clause_preds))
                clause_tp = float(((clause_preds == 1) & (clause_y == 1)).sum()) / int(len(clause_preds))
                clause_tn = float(((clause_preds == 0) & (clause_y == 0)).sum()) / int(len(clause_preds))
                clause_fp = float(((clause_preds == 1) & (clause_y == 0)).sum()) / int(len(clause_preds))
                clause_fn = float(((clause_preds == 0) & (clause_y == 1)).sum()) / int(len(clause_preds))
                avg_acc_stats['musacc'].update(clause_acc, batch.num_graphs)
                avg_acc_stats['tp'].update(clause_tp, batch.num_graphs)
                avg_acc_stats['tn'].update(clause_tn, batch.num_graphs)
                avg_acc_stats['fp'].update(clause_fp, batch.num_graphs)
                avg_acc_stats['fn'].update(clause_fn, batch.num_graphs)
            if self.args.spg:
                group_preds = (task2_outputs > 0.5).int()
                group_y = (batch.y_window > 0.5).int()
                group_acc = float((group_preds == group_y).sum()) / int(len(group_preds))
                group_tp = float(((group_preds == 1) & (group_y == 1)).sum()) / int(len(group_preds))
                group_tn = float(((group_preds == 0) & (group_y == 0)).sum()) / int(len(group_preds))
                group_fp = float(((group_preds == 1) & (group_y == 0)).sum()) / int(len(group_preds))
                group_fn = float(((group_preds == 0) & (group_y == 1)).sum()) / int(len(group_preds))
                avg_acc_stats['gacc'].update(group_acc, batch.num_graphs)
                avg_acc_stats['tp'].update(group_tp, batch.num_graphs)
                avg_acc_stats['tn'].update(group_tn, batch.num_graphs)
                avg_acc_stats['fp'].update(group_fp, batch.num_graphs)
                avg_acc_stats['fn'].update(group_fn, batch.num_graphs)

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|{total:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch.num_graphs)
                Bar.suffix = Bar.suffix + \
                    '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            Bar.suffix = Bar.suffix + '|ACC {:.2f} '.format(acc)
            if self.args.spc:
                Bar.suffix = Bar.suffix + '|MUSACC {:.2f} '.format(clause_acc)
            if self.args.spg:
                Bar.suffix = Bar.suffix + '|GACC {:.2f} '.format(group_acc)
            
            Bar.suffix = Bar.suffix + '|Net {bt.avg:.3f}s'.format(bt=batch_time)
            bar.next()
            del output, loss, loss_stats

        # Final Display
        Bar.suffix = '{phase}: [{:}]| '.format(epoch, phase=phase)
        for l in avg_loss_stats:
            Bar.suffix += '{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        Bar.suffix += ' | '
        for l in avg_acc_stats:
            Bar.suffix += '{} {:.4f} '.format(l, avg_acc_stats[l].avg)
        bar.next()
        
        # update the temperature
        bar.finish()
        ret = {}
        for l in avg_loss_stats:
            ret[l] = avg_loss_stats[l].avg
        for l in avg_acc_stats:
            ret[l] = avg_acc_stats[l].avg
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret, []

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    def _get_losses(self, args):
        if args.loss in _loss_factory.keys():
            loss = _loss_factory[args.loss]()
            if args.spc:
                cl = _loss_factory[args.loss](pos_weight=torch.tensor(args.spc_weight))
                gl = None
                loss_states = ['loss', 'bl', 'cl']
            elif args.spg:
                cl = None
                gl = _loss_factory[args.loss](pos_weight=torch.tensor(args.spg_weight))
                loss_states = ['loss', 'bl', 'gl']
            else:
                cl = None
                gl = None
                loss_states = ['loss']
        else:
            raise KeyError
        return loss_states, [loss, cl, gl]

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
