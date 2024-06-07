""" Training routine for GraspNet baseline model. """

import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import resource
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

# from graspnet import GraspNet, get_loss
from models.GSNet import GraspNet
from models.GSNet_loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log/gsnet', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Create Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=True)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False)

print(len(TRAIN_DATASET), len(TEST_DATASET))
# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
# TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
#     num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=8, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=8, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
# net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#                         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])

net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
net.to(device)
# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate)
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, start_epoch))
# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    lr = lr * (0.95 ** epoch)
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        end_points = net(batch_data_label)
        loss, end_points = get_loss(end_points)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0


def train(start_epoch):
    global EPOCH_CNT
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_one_epoch()

        save_dict = {'epoch': epoch + 1, 'optimizer_state_dict': optimizer.state_dict(),
                     'model_state_dict': net.state_dict()}
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'epoch' + str(epoch + 1).zfill(2) + '.tar'))


if __name__ == '__main__':
    train(start_epoch)