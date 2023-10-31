""" Training routine for GraspNet baseline model. """

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

import sys
# from typing import Any
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import MinkowskiEngine as ME

# from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR

# import resource
# # RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# hard_limit = rlimit[1]
# soft_limit = min(500000, hard_limit)
# print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

seed_everything(42, workers=True)

# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

# import random
# def setup_seed(seed):
#      torch.manual_seed(seed)
#      torch.cuda.manual_seed_all(seed)
#      np.random.seed(seed)
#      random.seed(seed)
#      torch.backends.cudnn.deterministic = True
# # 设置随机数种子
# setup_seed(0)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

# from graspnet import GraspNet, get_loss
from GSNet import IGNet
# from IGNet_loss import get_loss
# from pytorch_utils import BNMomentumScheduler
# from graspnet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
from ignet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
from label_generation import process_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/8TB/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log/ignet_v0.3.5.1', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=61, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
# LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
# LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
# assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
# DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
# CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
#     else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

# LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
# LOG_FOUT.write(str(cfgs)+'\n')
# def log_string(out_str):
#     LOG_FOUT.write(out_str+'\n')
#     LOG_FOUT.flush()
#     print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

# Create Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', 
                                num_points=cfgs.num_point, remove_outlier=True, augment=True, real_data=True, 
                                syn_data=True)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', 
                               num_points=cfgs.num_point, remove_outlier=True, augment=False, real_data=True, 
                               syn_data=True)

# net.to(device)

class IGNetModule(LightningModule):
    def __init__(
        self,
        model,
        cfgs,
        # optimizer_name="SGD",
        # lr=1e-3,
        # weight_decay=1e-5,
        # voxel_size=0.05,
        # batch_size=12,
        # val_batch_size=6,
        # train_num_workers=4,
        # val_num_workers=2,
    ):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.model = model
        self.learning_rate = cfgs.learning_rate
        self.weight_decay = cfgs.weight_decay
        self.batch_size = cfgs.batch_size
        self.score_criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        self.width_criterion = nn.SmoothL1Loss(reduction='none')
        
    # def train_dataloader(self):
    #     return DataLoader(
    #         DummyDataset("train", voxel_size=self.voxel_size),
    #         batch_size=self.batch_size,
    #         collate_fn=minkowski_collate_fn,
    #         shuffle=True,
    #     )
    
    # def val_dataloader(self):
    #     return DataLoader(
    #         DummyDataset("val", voxel_size=self.voxel_size),
    #         batch_size=self.val_batch_size,
    #         collate_fn=minkowski_collate_fn,
    #     )
    
    def compute_score_loss(self, end_points):
        # criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        # criterion = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        grasp_score_pred = end_points['grasp_score_pred'].permute((0, 3, 1, 2))
        grasp_score_pred = torch.clamp(grasp_score_pred, min=0., max=1.0)
        
        grasp_score_label = end_points['batch_grasp_score_ids']
        loss = self.score_criterion(grasp_score_pred, grasp_score_label)
        return loss

    def compute_width_loss(self, end_points):
        # criterion = nn.SmoothL1Loss(reduction='none')
        grasp_width_pred = end_points['grasp_width_pred']
        grasp_width_label = end_points['batch_grasp_width'] * 10
        loss = self.width_criterion(grasp_width_pred, grasp_width_label)
        grasp_score_label = end_points['batch_grasp_score']
        loss_mask = grasp_score_label > 0
        loss = loss[loss_mask].mean()
        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        end_points = self(train_batch)
        # loss, end_points = get_loss(end_points)
        score_loss = self.compute_score_loss(end_points)
        wdith_loss = self.compute_width_loss(end_points)
        loss = score_loss + wdith_loss
        loss_dict = {'train_loss':loss, 
                     'score_loss':score_loss, 
                     'width_loss':wdith_loss}
        self.log_dict(loss_dict, batch_size=self.batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        end_points = self(val_batch)
        # loss, end_points = get_loss(end_points)
        score_loss = self.compute_score_loss(end_points)
        wdith_loss = self.compute_width_loss(end_points)
        loss = score_loss + wdith_loss
        loss_dict = {'val_loss':loss, 
                     'score_loss':score_loss, 
                     'width_loss':wdith_loss}
        self.log_dict(loss_dict, batch_size=self.batch_size)
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        optim_dict = {        
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingLR(optimizer, T_max=16, eta_min=0.0),
                    # "monitor": "metric_to_track",
                    # "frequency": 1,
                    # If "monitor" references validation metrics, then "frequency" should be set to a
                    # multiple of "trainer.check_val_every_n_epoch".
                },
        }
        return optim_dict
    # def configure_optimizers(self):
    #     return Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

# Load checkpoint if there is any
# it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
# start_epoch = 0
# if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     net.load_state_dict(checkpoint['model_state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     start_epoch = checkpoint['epoch']
#     log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
# BN_MOMENTUM_INIT = 0.5
# BN_MOMENTUM_MAX = 0.001
# bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
# bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

# def get_current_lr(epoch):
#     lr = cfgs.learning_rate
#     for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
#         if epoch >= lr_decay_epoch:
#             lr *= LR_DECAY_RATES[i]
#     return lr

# def adjust_learning_rate(optimizer, epoch):
#     lr = get_current_lr(epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

# TensorBoard Visualizers
# TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
# TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

# def train_one_epoch():
#     stat_dict = {} # collect statistics
#     # adjust_learning_rate(optimizer, EPOCH_CNT)
#     # bnm_scheduler.step() # decay BN momentum
#     # set model to training mode
#     net.train()
#     for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
#         for key in batch_data_label:
#             if 'list' in key:
#                 for i in range(len(batch_data_label[key])):
#                     for j in range(len(batch_data_label[key][i])):
#                         batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda()
#             else:
#                 batch_data_label[key] = batch_data_label[key].cuda()

#         # Forward pass
#         end_points = net(batch_data_label)

#         # Compute loss and gradients, update parameters.
#         loss, end_points = get_loss(end_points)
#         loss.backward()
#         if (batch_idx+1) % 1 == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         # Accumulate statistics and print out
#         for key in end_points:
#             if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
#                 if key not in stat_dict: stat_dict[key] = 0
#                 stat_dict[key] += end_points[key].item()

#         batch_interval = 10
#         if (batch_idx+1) % batch_interval == 0:
#             log_string(' ---- batch: %03d ----' % (batch_idx+1))
#             for key in sorted(stat_dict.keys()):
#                 TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
#                 log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
#                 stat_dict[key] = 0

# def evaluate_one_epoch():
#     stat_dict = {} # collect statistics
#     # set model to eval mode (for bn and dp)
#     net.eval()
#     for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
#         if batch_idx % 10 == 0:
#             print('Eval batch: %d'%(batch_idx))
#         for key in batch_data_label:
#             if 'list' in key:
#                 for i in range(len(batch_data_label[key])):
#                     for j in range(len(batch_data_label[key][i])):
#                         batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda()
#             else:
#                 batch_data_label[key] = batch_data_label[key].cuda()
        
#         # Forward pass
#         with torch.no_grad():
#             end_points = net(batch_data_label)

#         # Compute loss
#         loss, end_points = get_loss(end_points)

#         # Accumulate statistics and print out
#         for key in end_points:
#             if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
#                 if key not in stat_dict: stat_dict[key] = 0
#                 stat_dict[key] += end_points[key].item()

#     for key in sorted(stat_dict.keys()):
#         TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
#         log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

#     mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
#     return mean_loss


# def train(start_epoch):
#     global EPOCH_CNT 
#     min_loss = 1e10
#     loss = 0
#     for epoch in range(start_epoch, cfgs.max_epoch):
#         EPOCH_CNT = epoch
#         log_string('**** EPOCH %03d ****' % (epoch))
#         log_string('Current learning rate: %f' % (lr_scheduler.get_last_lr()[0]))
#         # log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
#         log_string(str(datetime.now()))
#         # Reset numpy seed.
#         # REF: https://github.com/pytorch/pytorch/issues/5059
#         np.random.seed()
#         train_one_epoch()
#         lr_scheduler.step()
        
#         loss = evaluate_one_epoch()
#         # Save checkpoint
#         save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': loss,
#                     }
#         try: # with nn.DataParallel() the net is added as a submodule of DataParallel
#             save_dict['model_state_dict'] = net.module.state_dict()
#         except:
#             save_dict['model_state_dict'] = net.state_dict()
#         # if epoch in LR_DECAY_STEPS:
#         #     torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(epoch)))
#         if not EPOCH_CNT % 5:
#             torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT)))
#         torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=8, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=8, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

model = IGNet(num_view=cfgs.num_view, seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
net = IGNetModule(model=model, cfgs=cfgs)
logger = TensorBoardLogger(os.path.join(cfgs.log_dir), name="ignet", version=0)

# trainer = Trainer(max_epochs=cfgs.max_epoch, logger=logger, accelerator='gpu', strategy='ddp', 
                #   devices=[0, 1], deterministic=True)
trainer = Trainer(max_epochs=cfgs.max_epoch, logger=logger, devices=2, accelerator='gpu', 
                  strategy='ddp', deterministic=False)
# trainer = Trainer(max_epochs=cfgs.max_epoch, logger=logger, accelerator='gpu', deterministic=True)
trainer.fit(net, TRAIN_DATALOADER, TEST_DATALOADER)
# def main():

# if __name__=='__main__':
#     train(start_epoch)