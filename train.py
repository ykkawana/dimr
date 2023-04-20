import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import open3d as o3d
import dotenv
dotenv.load_dotenv()
DEBUG = os.getenv('DEBUG', '0') == '1'

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np
import wandb

from util.config import cfg
# from util import config
# cfg = config.get_parser()
# cfg.exp_path =  os.path.join(f'{cfg.exp_root}/exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5])
from util.log import logger
import util.utils as utils
# import wandb


def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)
    wandb.init(config=cfg)

    # wandb.init(sync_tensorboard=True)

def train_epoch(train_loader, model, model_fn, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        ##### prepare input and forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if epoch <= cfg.prepare_epochs:
            sys.stdout.write(
                    "{} / {} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} ang:{:.4f}/{:.4f} | T: {remain_time}\n".format
                (epoch, i + 1, 
                am_dict['loss'].avg, 
                optimizer.param_groups[0]['lr'],
                am_dict['semantic_loss'].avg,
                am_dict['offset_norm_loss'].avg,
                am_dict['offset_dir_loss'].avg,
                am_dict['angle_label_loss'].avg,
                am_dict['angle_residual_loss'].avg,
                remain_time=remain_time)
            )
            if (current_iter + 1) % 10 == 0:
                log = {
                    'loss': am_dict['loss'].val,
                    'semantic_loss': am_dict['semantic_loss'].val,
                    'offset_norm_loss': am_dict['offset_norm_loss'].val,
                    'offset_dir_loss': am_dict['offset_dir_loss'].val,
                    'angle_label_loss': am_dict['angle_label_loss'].val,
                    'angle_residual_loss': am_dict['angle_residual_loss'].val,
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log = {f'train/{k}': v for k, v in log.items()}
                wandb.log(log)
        elif epoch <= cfg.prepare_epochs_2:
            sys.stdout.write(
                "{} / {} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} ang:{:.4f}/{:.4f} | score: {:.4f} | T: {remain_time}\n".format
                (epoch, i + 1,
                am_dict['loss'].avg, 
                optimizer.param_groups[0]['lr'],
                am_dict['semantic_loss'].avg,
                am_dict['offset_norm_loss'].avg,
                am_dict['offset_dir_loss'].avg,
                am_dict['angle_label_loss'].avg,
                am_dict['angle_residual_loss'].avg,
                am_dict['score_loss'].avg,
                remain_time=remain_time)
            )
            if (current_iter + 1) % 10 == 0:
                log = {
                    'loss': am_dict['loss'].val,
                    'semantic_loss': am_dict['semantic_loss'].val,
                    'offset_norm_loss': am_dict['offset_norm_loss'].val,
                    'offset_dir_loss': am_dict['offset_dir_loss'].val,
                    'angle_label_loss': am_dict['angle_label_loss'].val,
                    'angle_residual_loss': am_dict['angle_residual_loss'].val,
                    'score_loss': am_dict['score_loss'].val,
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log = {f'train/{k}': v for k, v in log.items()}
                wandb.log(log)
        else:
            sys.stdout.write(
                "{} / {} | L: {:.4f} lr:{:.6f} | sem: {:.4f} off: {:.4f}/{:.4f} ang:{:.4f}/{:.4f} | score: {:.4f} | z: {:.4f} center: {:.4f} scale: {:.4f} | T: {remain_time}\n".format
                (epoch, i + 1,
                am_dict['loss'].avg, 
                optimizer.param_groups[0]['lr'],
                am_dict['semantic_loss'].avg,
                am_dict['offset_norm_loss'].avg,
                am_dict['offset_dir_loss'].avg,
                am_dict['angle_label_loss'].avg,
                am_dict['angle_residual_loss'].avg,
                am_dict['score_loss'].avg,
                am_dict['z_loss'].avg,
                am_dict['center_loss'].avg,
                am_dict['scale_loss'].avg,
                remain_time=remain_time)
            )
            if (current_iter + 1) % 10 == 0:
                log = {
                    'loss': am_dict['loss'].val,
                    'semantic_loss': am_dict['semantic_loss'].val,
                    'offset_norm_loss': am_dict['offset_norm_loss'].val,
                    'offset_dir_loss': am_dict['offset_dir_loss'].val,
                    'angle_label_loss': am_dict['angle_label_loss'].val,
                    'angle_residual_loss': am_dict['angle_residual_loss'].val,
                    'score_loss': am_dict['score_loss'].val,
                    'z_loss': am_dict['z_loss'].val,
                    'center_loss': am_dict['center_loss'].val,
                    'scale_loss': am_dict['scale_loss'].val,
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                }
                log = {f'train/{k}': v for k, v in log.items()}
                wandb.log(log)
        if (i == len(train_loader) - 1): print()
        if DEBUG:
            if i > 100:
                break


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, model_fn, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):

            ##### prepare input and forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            ##### meter_dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                am_dict[k].update(v[0], v[1])

            ##### print
            sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
            if (i == len(val_loader) - 1): print()
        log = {f'test/{k}': v.avg for k, v in am_dict.items() if k.endswith('loss')}
        log['epoch'] = epoch
        wandb.log(log)

        logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


if __name__ == '__main__':
    ##### init
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')

    from model.rfs import RfSNet as Network
    from model.rfs import model_fn_decorator

    model = Network(cfg)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### model_fn (criterion)
    model_fn = model_fn_decorator()

    ##### dataset
    if cfg.dataset == 'scannetv2':
        if data_name == 'scannet':
            import data.scannetv2_inst
            dataset = data.scannetv2_inst.Dataset()
            dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda)      # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, epoch)

        if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch):
            eval_epoch(dataset.val_data_loader, model, model_fn, epoch)
