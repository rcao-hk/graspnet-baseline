""" Training routine for GraspNet baseline model. """

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR


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
from models.GSNet import GraspNet, GraspNet_multimodal
from models.GSNet_loss import get_loss
from dataset.graspnet_dataset import GraspNetDataset, GraspNetMultiDataset, collate_fn, minkowski_collate_fn, load_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/data/jhpan/dataset/graspnet', help='Dataset root')
parser.add_argument('--big_file_root', default=None, help='Big file root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--resume_checkpoint', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--ckpt_root', default='log', help='Checkpoint dir to save model [default: log]')
parser.add_argument('--method_id', default='gsnet_virtual', help='Method version')
parser.add_argument('--log_root', default='log', help='Log dir to save log [default: log]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')
parser.add_argument('--lr_sched', default=False, action='store_true')
parser.add_argument('--lr_sched_period', type=int, default=16, help='T_max of cosine learing rate scheduler [default: 16]')
parser.add_argument('--ckpt_save_interval', type=int, default=5, help='Number for save checkpoint[default: 5]')
parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--worker_num', type=int, default=12, help='Worker number for dataloader [default: 4]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--eval_start_epoch', type=int, default=6, help='Number of epoch starting ckpt saving')
parser.add_argument('--pin_memory', action='store_true', help='Set pin_memory for faster training [default: False]')
parser.add_argument('--multi_modal', action='store_true', default=False, help='Use multi-modal gsnet[default: False]')
parser.add_argument('--fusion_type', default='early', choices=['early', 'concat', 'intermediate'])
parser.add_argument('--virtual_depth', action='store_true', default=False, help='Use virtual depth for training [default: False]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
print(cfgs)

cfgs.ckpt_dir = os.path.join(cfgs.ckpt_root, cfgs.method_id, cfgs.camera)
cfgs.log_dir = os.path.join(cfgs.log_root, cfgs.method_id, cfgs.camera)
os.makedirs(cfgs.ckpt_dir, exist_ok=True)
os.makedirs(cfgs.log_dir, exist_ok=True)

EPOCH_CNT = 0
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.ckpt_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.resume_checkpoint if cfgs.resume_checkpoint is not None \
    else DEFAULT_CHECKPOINT_PATH

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
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.big_file_root if cfgs.big_file_root is not None else cfgs.dataset_root)
if cfgs.multi_modal:
    TRAIN_DATASET = GraspNetMultiDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False)
    TEST_DATASET = GraspNetMultiDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False)
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
        num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
        num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
else:
    TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, cfgs.big_file_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, load_label=True, depth_type='virtual' if cfgs.virtual_depth else 'real')
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, cfgs.big_file_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, voxel_size=cfgs.voxel_size, remove_outlier=True, augment=False, depth_type='virtual' if cfgs.virtual_depth else 'real')
    TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
        num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn, pin_memory=cfgs.pin_memory)
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
        num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn, pin_memory=cfgs.pin_memory)


print(len(TRAIN_DATASET), len(TEST_DATASET))
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
# net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#                         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])

if cfgs.multi_modal:
    net = GraspNet_multimodal(seed_feat_dim=cfgs.seed_feat_dim, img_feat_dim=64, is_training=True, fuse_type=cfgs.fusion_type)
else:
    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
net.to(device)

# Load the Adam optimizer
optimizer = optim.AdamW(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
if cfgs.lr_sched:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfgs.lr_sched_period, eta_min=1e-4)

start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if cfgs.lr_sched:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
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
    # adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    batch_interval = 20
    overall_loss = 0
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

        overall_loss += stat_dict['loss/overall_loss']
        if (batch_idx + 1) % batch_interval == 0:
            log_string(' ----epoch: %03d  ---- batch: %03d ----' % (EPOCH_CNT, batch_idx + 1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key] / batch_interval,
                                        (EPOCH_CNT * len(TRAIN_DATALOADER) + batch_idx) * cfgs.batch_size)
                log_string('mean %s: %f' % (key, stat_dict[key] / batch_interval))
                stat_dict[key] = 0

    overall_loss = overall_loss/float(cfgs.batch_size)
    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    overall_loss = 0
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            log_string('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].cuda(non_blocking=cfgs.pin_memory)
            else:
                batch_data_label[key] = batch_data_label[key].cuda(non_blocking=cfgs.pin_memory)
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
    
        overall_loss += stat_dict['loss/overall_loss']
        # overall_loss += (stat_dict['loss/score_loss'] + stat_dict['loss/width_loss'] + stat_dict['loss/rot_graspness_loss'])
    for key in sorted(stat_dict.keys()):
        TRAIN_WRITER.add_scalar('test_' + key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    overall_loss = overall_loss/float(cfgs.batch_size)
    log_string('overall loss:{}, batch num:{}'.format(overall_loss, batch_idx+1))
    mean_loss = overall_loss/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT
    min_loss = np.inf
    best_epoch = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % epoch)
        log_string('Current learning rate: %f' % (get_current_lr(epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        train_loss = train_one_epoch()

        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict()}
        
        if cfgs.lr_sched:
            lr_scheduler.step()
            save_dict['lr_scheduler'] = lr_scheduler.state_dict()
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
            
        # torch.save(save_dict, os.path.join(cfgs.log_dir, 'epoch' + str(epoch + 1).zfill(2) + '.tar'))
        if epoch >= cfgs.eval_start_epoch:
            eval_loss = evaluate_one_epoch()
            if eval_loss < min_loss:
                min_loss = eval_loss
                best_epoch = epoch
                ckpt_name = "epoch_" + str(best_epoch) \
                            + "_train_" + str(train_loss) \
                            + "_val_" + str(eval_loss)
                torch.save(save_dict['model_state_dict'], os.path.join(cfgs.ckpt_dir, ckpt_name + '.tar'))
            elif not EPOCH_CNT % cfgs.ckpt_save_interval:
                torch.save(save_dict, os.path.join(cfgs.ckpt_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT)))
            log_string("best_epoch:{}".format(best_epoch))
            # if epoch in LR_DECAY_STEPS:
            #     torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(epoch)))
        torch.save(save_dict, os.path.join(cfgs.ckpt_dir, 'checkpoint.tar'))

if __name__ == '__main__':
    train(start_epoch)