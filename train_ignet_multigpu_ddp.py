import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

import resource
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import sys
from datetime import datetime
import argparse
import numpy as np

# import cv2
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)

import torch
# torch.set_num_threads(1)

import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch.multiprocessing as mp
import torch.distributed as dist

import MinkowskiEngine as ME

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, CosineAnnealingLR

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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

# from graspnet import GraspNet, get_loss
from GSNet import IGNet
from IGNet_loss import get_loss
# from pytorch_utils import BNMomentumScheduler
# from graspnet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
from ignet_dataset import GraspNetDataset, collate_fn, minkowski_collate_fn, load_grasp_labels
from label_generation import process_grasp_labels


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/8TB/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log/ignet_v0.3.5.3', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--num_point', type=int, default=256, help='Point Number [default: 20000]')
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=61, help='Epoch to run [default: 18]')
parser.add_argument('--batch_size', type=int, default=18, help='Batch Size during training [default: 2]')
parser.add_argument('--worker_num', type=int, default=3, help='Worker number for dataloader [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
# parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()


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

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


def main_worker(gpu, ngpus_per_node, args):
    global min_time
    args.gpu = gpu
    if args.gpu is not None:
        log_string("Use GPU: {} for training".format(args.gpu))
    args.rank = 0 * ngpus_per_node + gpu
    now = datetime.now()
    port_id = int(now.minute)*1000 + int(now.second)
    log_string("Port ID: {}".format(port_id))
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:" + str(port_id),
        world_size=args.world_size,
        rank=args.rank,
    )

    def train_one_epoch():
        stat_dict = {} # collect statistics
        # adjust_learning_rate(optimizer, EPOCH_CNT)
        # bnm_scheduler.step() # decay BN momentum
        # set model to training mode
        model.train()
        for batch_idx, batch_data_label in enumerate(train_dataloader):
            optimizer.zero_grad()

            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(args.gpu)
                else:
                    batch_data_label[key] = batch_data_label[key].to(args.gpu)
            # Forward pass
            end_points = model(batch_data_label)

            # Compute loss and gradients, update parameters.
            loss, end_points = get_loss(end_points, args.gpu)
            loss.backward()
            optimizer.step()

            # Must clear cache at regular interval
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

            batch_interval = 10
            if (batch_idx+1) % batch_interval == 0 and dist.get_rank() == 0:
                log_string(' ---- batch: %03d ----' % (batch_idx+1))
                TRAIN_WRITER.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], (EPOCH_CNT*len(train_dataloader)+batch_idx)*cfgs.batch_size)
                for key in sorted(stat_dict.keys()):
                    TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(train_dataloader)+batch_idx)*cfgs.batch_size)
                    log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                    stat_dict[key] = 0
            
    def evaluate_one_epoch():
        stat_dict = {} # collect statistics
        # set model to eval mode (for bn and dp)
        model.eval()
        for batch_idx, batch_data_label in enumerate(test_dataloader):
            if batch_idx % 10 == 0:
                log_string('Eval batch: %d'%(batch_idx))
            for key in batch_data_label:
                if 'list' in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(args.gpu)
                else:
                    batch_data_label[key] = batch_data_label[key].to(args.gpu)
            # Forward pass
            with torch.no_grad():
                end_points = model(batch_data_label)

            # Compute loss
            loss, end_points = get_loss(end_points, args.gpu)

            # Accumulate statistics and print out
            for key in end_points:
                if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                    if key not in stat_dict: stat_dict[key] = 0
                    stat_dict[key] += end_points[key].item()

        if dist.get_rank() == 0:
            TEST_WRITER.add_scalar('learning_rate', lr_scheduler.get_last_lr()[0], (EPOCH_CNT+1)*len(test_dataloader)*cfgs.batch_size)
            for key in sorted(stat_dict.keys()):
                TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(test_dataloader)*cfgs.batch_size)
                log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

        mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
        return mean_loss

    # Create Dataset and Dataloader
    valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
    train_dataset = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', 
                                    num_points=cfgs.num_point, remove_outlier=True, augment=True, real_data=True, 
                                    syn_data=True)
    test_dataset = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', 
                                num_points=cfgs.num_point, remove_outlier=True, augment=False, real_data=True, 
                                syn_data=False)

    log_string("{}, {}".format(len(train_dataset), len(test_dataset)))
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # real batch_size = batch_size * world_size
    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, num_workers=cfgs.worker_num, 
                                  worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn, sampler=train_sampler,
                                  pin_memory=False, persistent_workers=False)
    test_dataloader = DataLoader(test_dataset, batch_size=cfgs.batch_size, num_workers=cfgs.worker_num, 
                                 worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn, sampler=test_sampler,
                                 pin_memory=False, persistent_workers=False)
    log_string("{}, {}".format(len(train_dataloader), len(test_dataloader)))

    # create model
    model = IGNet(num_view=cfgs.num_view, seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # Synchronized batch norm
    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    optimizer = AdamW(model.parameters(), lr=cfgs.learning_rate)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=16, eta_min=1e-5)

    start_epoch = 0
    
    if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']
        log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f' % (lr_scheduler.get_last_lr()[0]))
        # log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))

        # Reset numpy seed.
        # # REF: https://github.com/pytorch/pytorch/issues/5059
        # np.random.seed()

        train_one_epoch()
        lr_scheduler.step()
        
        loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'lr_scheduler':lr_scheduler.state_dict(),
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = model.module.state_dict()
        except:
            save_dict['model_state_dict'] = model.state_dict()

        if dist.get_rank() == 0:
            if not EPOCH_CNT % 5:
                torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint_{}.tar'.format(EPOCH_CNT)))
            torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))


def main():
    num_devices = torch.cuda.device_count()
    # num_devices = min(2, num_devices)
    log_string("Testing {} GPUs.".format(num_devices))
    log_string("Total batch size: {}".format(num_devices * cfgs.batch_size))

    cfgs.world_size = num_devices
    mp.spawn(main_worker, nprocs=num_devices, args=(num_devices, cfgs))


if __name__ == "__main__":
    main()