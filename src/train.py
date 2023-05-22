import os 
import torch
import numpy as np
from torchsummary import summary

from utils.dataloader_utils import SA2T_DataLoader
from config import get_parse_args
from utils.logger import Logger
from datasets.dataset_factory import dataset_factory
from models.model import create_model, load_model, save_model
from trains.train import SA2T_Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args = get_parse_args()
    print('==> Using settings {}'.format(args))

    logger = Logger(args)
    print('Using device: ', args.device)

    # Dataset
    dataset = dataset_factory[args.dataset](args.data_dir, args)
    perm = torch.randperm(len(dataset))
    dataset = dataset[perm]
    print("Statistics: ")
    data_len = len(dataset)
    print("Size: ", len(dataset))
    print('Splitting the dataset into training and validation sets..')
    training_cutoff = int(data_len * args.trainval_split)
    print('# training circuits: ', training_cutoff)
    print('# validation circuits: ', data_len - training_cutoff)
    train_dataset = dataset[:training_cutoff]
    val_dataset = dataset[training_cutoff:] 

    # use PyG dataloader
    train_dataset = SA2T_DataLoader(args, train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dataset = SA2T_DataLoader(args, val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print('==> Creating model...')
    model = create_model(args)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best = 1e10
    if args.resume:
        if args.load_model != '':
            model, optimizer, start_epoch, best = load_model(
            model, args.load_model, optimizer, args.resume, args.lr, args.lr_step, best)
        else:
            raise "No trained model"
        
    trainer = SA2T_Trainer(args, model, optimizer)
    trainer.set_device(args.device)

    print('==> Starting training...')
    # best = 1e10
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        mark = epoch if args.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_dataset)
        logger.write('epoch: {} | Train | '.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        logger.write('\n')
        if epoch % args.save_intervals == 0:
            save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)), 
                    epoch, model, optimizer)
        with torch.no_grad():
            log_dict_val, _ = trainer.val(epoch, val_dataset)
        logger.write('epoch: {} | Val | '.format(epoch))
        for k, v in log_dict_val.items():
            logger.scalar_summary('val_{}'.format(k), v, epoch)
            logger.write('{} {:8f} | '.format(k, v))
        logger.write('\n')
        if epoch in args.lr_step:
            # save_model(os.path.join(args.save_dir, 'model_{}.pth'.format(epoch)), 
            #         epoch, model, optimizer)
            lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()