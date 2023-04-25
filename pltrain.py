import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import pytorch_lightning as pl
# import open3d as o3d
import dotenv
import glob
dotenv.load_dotenv()
DEBUG = os.getenv('DEBUG', '0') == '1'
import datetime
import uuid

import torch
import torch.optim as optim
import time, sys, os, random
# from tensorboardX import SummaryWriter
import numpy as np
# import wandb
from util.config import cfg
# from util import config
# cfg = config.get_parser()
# cfg.exp_path =  os.path.join(f'{cfg.exp_root}/exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5])
import util.utils as utils
# import wandb
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import data.scannetv2_inst
from model.rfs import RfSNet as Network
from model.rfs import model_fn_decorator

from pytorch_lightning.strategies.launchers.multiprocessing import _is_forking_disabled
print(torch.cuda.device_count(), torch.multiprocessing.get_all_start_methods(), _is_forking_disabled())
if os.getenv('NO_WANDB', '0') == '0' and cfg.run_id is not None:
    import wandb

# pl.seed_everything(12345)


class PLModelWrapper(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters("cfg")
        self.model = Network(cfg)
        self.model_fn = model_fn_decorator(cfg)

    def step_learning_rate(self, optimizer, base_lr, epoch, step_epoch, multiplier=0.1, clip=1e-6):
        """Sets the learning rate to the base LR decayed by 10 every step epochs"""
        lr = max(base_lr * (multiplier ** (epoch / step_epoch)), clip)
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr
        self.learning_rate = lr
        return lr

    def configure_optimizers(self):
        if cfg.optim == 'Adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
        elif cfg.optim == 'SGD':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = cfg.lr 
        # return [optimizer], []
        return [optimizer], {
            'scheduler': LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: self.step_learning_rate(
                    optimizer,
                    cfg.lr,
                    epoch,
                    cfg.step_epoch,
                    cfg.multiplier)),
            'interval': 'epoch',
            'frequency': 1}

    def _common_step(self, batch, batch_idx, stage):
        loss, _, visual_dict, meter_dict = self.model_fn(batch, self.model, self.current_epoch)

        ##### meter_dict
        new_loss_dict = meter_dict
        # for k, v in new_loss_dict.items():
        #     print(k, v)
            # if isinstance(v, torch.Tensor):
            #     print(k, v.shape)
            # elif isinstance(v, (list, tuple)):
            #     print(k, len(v), type(v[0]))
        new_loss_dict = {f'{stage}/{k}': v[0] for k, v in new_loss_dict.items()}
        if stage == 'train':
            new_loss_dict['lr'] = self.learning_rate
        new_loss_dict['Step'] = self.global_step
        self.log_dict(new_loss_dict)
        return loss

    def training_step(self, batch, batch_idx):
        # torch.cuda.empty_cache()
        return self._common_step(batch, batch_idx, "train")

    # def training_epoch_end(self, train_step_outputs):
    #     torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

class PLDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.dataset = data.scannetv2_inst.Dataset(cfg)

    def train_dataloader(self):
        self.dataset.trainLoader()
        return self.dataset.train_data_loader

    def val_dataloader(self):
        self.dataset.valLoader()
        return self.dataset.val_data_loader

    def test_dataloader(self):
        self.dataset.testLoader()
        return self.dataset.test_data_loader



if __name__ == '__main__':
    ##### init
    # init()
    checkpoint_path = None
    # cfg.batch_size = 16
    if cfg.run_id is not None:
        assert hasattr(cfg, 'default_root_dir')
        original_ch_dir =  os.path.join(cfg.default_root_dir, 'checkpoints')
        checkpoints = sorted(glob.glob(os.path.join(original_ch_dir, 'latest-*.ckpt')))
        if len(checkpoints) > 0:
            print("Resume pl from", checkpoints[-1])
            checkpoint_path = checkpoints[-1]
    pl.seed_everything(cfg.manual_seed)
    dirname = str(datetime.datetime.now()).split(".")[0].replace(" ", "-").replace(":", "-")+"-"+str(uuid.uuid4())[:8]
    default_root_dir = os.path.join(cfg.exp_root, dirname)
    cfg.default_root_dir = default_root_dir
    ch_dir =  os.path.join(default_root_dir, 'checkpoints')
    os.makedirs(ch_dir, exist_ok=True)
    # cfg.batch_size = 2


    # val_metric_checkpoint = ModelCheckpoint(
    #     monitor='val/loss',
    #     dirpath=ch_dir,
    #     filename="best-{epoch}-{step}",
    #     save_last=True,
    #     save_top_k=1,
    #     auto_insert_metric_name=False,
    #     mode="min",
    # )
    # latest_checkpoint = ModelCheckpoint(
    #     filename="latest-{epoch}-{Step}",
    #     dirpath=ch_dir,
    #     monitor='Step',
    #     save_top_k=10,
    #     # every_n_epochs=10,
    #     every_n_val_epochs=1,
    #     # every_n_train_steps=3,
    #     auto_insert_metric_name=False,
    #     save_last=True,
    #     mode="max",
    # )
    val_metric_checkpoint = ModelCheckpoint(
        # monitor=os.path.join(*cfg['test_cfg']['best_metric']),
        monitor='val/loss',
        dirpath=ch_dir,
        filename="best-{epoch}-{step}",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
        mode="min",
    )
    latest_checkpoint = ModelCheckpoint(
        # filename="latest-{epoch}-{step}",
        filename="latest-{epoch}-{Step}",
        dirpath=ch_dir,
        # monitor='step',
        monitor='Step',
        # save_top_k=cfg['train_cfg'].get('checkpoint_save_top_k', 1),
        save_top_k=10,
        every_n_epochs=1,
        # every_n_epochs=cfg['train_cfg'].get('checkpoint_save_every_n_epochs', 1),
        # every_n_train_steps=3,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        mode="max",
    )
    if os.getenv("NO_WANDB", "0") == "1":
        assert cfg.run_id is None
        wandb_logger = None
    else:
        if cfg.resume_with_wandb_run_id:
            run_id = cfg.run_id.split('/')[-1]
            resume = 'must'
        else:
            run_id = None
            resume = None

        wandb_logger = WandbLogger(
            project=os.environ['WANDB_PROJECT'],
            id=run_id,
            resume=resume,
        )
    ngpus = len(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(','))
    strategy = "ddp" if ngpus > 1 else None
    # strategy = "deepspeed_stage_2"
    # strategy = "fsdp"
    if hasattr(cfg, 'accumulate_grad_batches'):
        accumulate_grad_batches = cfg.accumulate_grad_batches
    else:
        accumulate_grad_batches = None
    # assert accumulate_grad_batches == 4
    # trainer = pl.Trainer(
    #     log_every_n_steps=10,
    #     logger=wandb_logger,
    #     default_root_dir=default_root_dir,
    #     # gradient_clip_val=args.clip_gradient,
    #     # precision=16 if cfg['optimizer'].get('use_amp', False) else 32,
    #     benchmark=True,
    #     callbacks=[latest_checkpoint, val_metric_checkpoint],
    #     max_epochs=cfg.epochs,
    #     check_val_every_n_epoch=10,
    #     # resume_from_checkpoint=resume_ckpt,
    #     # num_sanity_val_steps=0,
    #     # accelerator="gpu",
    #     gpus=ngpus,
    #     # strategy='fsdp')
    #     accelerator=strategy)
    trainer = pl.Trainer(
        log_every_n_steps=10,
        logger=wandb_logger,
        default_root_dir=default_root_dir,
        gradient_clip_val=0.5,#args.clip_gradient,
        precision=16,# if cfg['optimizer'].get('use_amp', False) else 32,
        benchmark=True,
        callbacks=[latest_checkpoint, val_metric_checkpoint],
        max_epochs=cfg.epochs,
        # max_epochs=args.max_epoch,
        check_val_every_n_epoch=10,
        # check_val_every_n_epoch=args.eval_every_epoch,
        # resume_from_checkpoint=resume_ckpt,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=ngpus,
        accumulate_grad_batches=accumulate_grad_batches,
        # strategy='fsdp')
        # num_nodes=2,
        strategy=strategy)

    model = PLModelWrapper(
        cfg)
    if wandb_logger is not None:
        wandb_logger.watch(model, log="all")

    datamodule = PLDataModule(cfg)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint_path)
