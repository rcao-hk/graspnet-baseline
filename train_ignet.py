""" Training routine for GraspNet baseline model. """

import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# os.environ['OMP_NUM_THREADS'] = '18'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
# sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

import numpy as np
from datetime import datetime, timedelta
import time
import argparse
import torch.profiler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
    torch.backends.cudnn.benchmark = False

# from graspnet import GraspNet, get_loss
# from models.GSNet import IGNet
# from models.GSNet_v0_5 import IGNet
# from models.GSNet_v0_4 import IGNet
# from models.IGNet_loss import get_loss

# from models.IGNet_v0_6 import IGNet
# from models.IGNet_loss_v0_6 import get_loss
# from dataset.ignet_dataset import GraspNetDataset, minkowski_collate_fn, collate_fn, load_grasp_labels

# from models.IGNet_v0_7 import IGNet
# from models.IGNet_loss_v0_7 import get_loss
# from dataset.ignet_dataset import GraspNetDataset, minkowski_collate_fn, collate_fn, load_grasp_labels

# from models.IGNet_v0_7 import IGNet
# from models.IGNet_loss_v0_7 import get_loss

# from models.IGNet_v0_8 import IGNet
# from models.IGNet_loss_v0_8 import get_loss
from models.IGNet_v0_9 import IGNet
# from models.IGNet_v0_10 import IGNet
from models.IGNet_loss_v0_9 import get_loss
from dataset.ignet_multi_dataset import GraspNetDataset, GraspNetMultiDataset, minkowski_collate_fn, collate_fn, load_grasp_labels

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Dataset root')
parser.add_argument('--big_file_root', default=None, help='Big file root')
parser.add_argument('--camera', default='realsense', help='Camera split [realsense/kinect]')
parser.add_argument('--resume_checkpoint', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--ckpt_root', default='/media/gpuadmin/rcao/result/ignet', help='Checkpoint dir to save model [default: log]')
parser.add_argument('--method_id', default='ignet_v0.9', help='Method/version identifier used for logs and checkpoints')
parser.add_argument('--log_root', default='log', help='Log dir to save log [default: log]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--m_point', type=int, default=1024, help='Number of sampled points for grasp prediction [default: 1024]')
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--img_feat_dim', default=64, type=int, help='Image feature dim')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size for Quantize [default: 0.005]')
parser.add_argument('--visib_threshold', type=float, default=0.5, help='Visibility Threshold [default: 0.5]')
parser.add_argument('--match_point_num', type=int, default=350, help='Grasp Label Point Number [default: 350]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--max_epoch', type=int, default=61, help='Epoch to run [default: 61]')
parser.add_argument('--eval_start_epoch', type=int, default=0, help='Epoch to start evaluation [default: 0]')
parser.add_argument('--lr_sched', default=False, action='store_true')
parser.add_argument('--lr_sched_period', type=int, default=16, help='T_max of CosineAnnealingLR; set >= max_epoch for one-way decay')
parser.add_argument('--batch_size', type=int, default=20, help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate [default: 0.002]')
parser.add_argument('--worker_num', type=int, default=18, help='Worker number for dataloader [default: 4]')
parser.add_argument('--ckpt_save_interval', type=int, default=5, help='Number for save checkpoint[default: 5]')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--inst_denoise', default=False, action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--pin_memory', action='store_true', help='Set pin_memory for faster training [default: False]')
parser.add_argument('--seed', type=int, default=0, help='Random seed [default: 0]')
parser.add_argument('--log_interval', type=int, default=10, help='Batches between progress/ETA logs [default: 10]')
# parser.add_argument('--multi_modal_pose_augment', action='store_true', help='Set multi_modal_pose_augment for multi-modal consistent pose augmentation [default: False]')
# parser.add_argument('--pose_augment', action='store_true', help='Set pose_augment for pose augmentation [default: False]')
parser.add_argument('--augment', action='store_true', help='Set point_augment for point cloud augmentation [default: False]')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--fuse_type', default='intermediate', choices=['none', 'concat', 'add', 'gate', 'early', 'direct', 'intermediate'], help='Fusion type')
parser.add_argument('--grouping_type', default='rectangular', choices=['rectangular', 'cylinder'], help='Grouping type')
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()
if cfgs.log_interval <= 0:
    parser.error('--log_interval must be positive')
if cfgs.ckpt_save_interval <= 0:
    parser.error('--ckpt_save_interval must be positive')
if cfgs.grouping_type == 'cylinder' and cfgs.multi_scale_grouping:
    print('[WARNING] cylinder grouping ignores crop_size, so multi-scale crop sizes do not change the queried region.')
setup_seed(cfgs.seed)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

cfgs.ckpt_dir = os.path.join(cfgs.ckpt_root, cfgs.method_id, cfgs.camera)
cfgs.log_dir = os.path.join(cfgs.log_root, cfgs.method_id, cfgs.camera)
os.makedirs(cfgs.ckpt_dir, exist_ok=True)
os.makedirs(cfgs.log_dir, exist_ok=True)

EPOCH_CNT = 0
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.ckpt_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.resume_checkpoint if cfgs.resume_checkpoint is not None \
    else DEFAULT_CHECKPOINT_PATH

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def format_duration(seconds):
    """Format a duration for compact progress logs."""
    if seconds is None or not np.isfinite(seconds):
        return 'unknown'
    return str(timedelta(seconds=max(0, int(round(seconds)))))


def format_eta(seconds):
    """Return remaining duration and estimated wall-clock finish time."""
    if seconds is None or not np.isfinite(seconds):
        return 'ETA unknown'
    finish_time = datetime.now() + timedelta(seconds=max(0.0, float(seconds)))
    return '{} (finish {})'.format(
        format_duration(seconds), finish_time.strftime('%Y-%m-%d %H:%M:%S')
    )


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    # DataLoader assigns each worker a distinct torch seed. Reuse it for
    # numpy/python RNGs so augmentations are reproducible and non-identical.
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def to_device(x, device, non_blocking=False):
    """Recursively move tensors to device. Keep non-tensors unchanged."""
    if torch.is_tensor(x):
        return x.to(device=device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: to_device(v, device, non_blocking) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device, non_blocking) for v in x)
    # str / int / float / None / np scalar ... keep as is
    return x

if not torch.cuda.is_available():
    raise RuntimeError('IGNet training requires a CUDA-capable GPU.')
device = torch.device('cuda:0')
torch.cuda.set_device(0)

# Create Dataset and Dataloader
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.big_file_root if cfgs.big_file_root is not None else cfgs.dataset_root)
# TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, cfgs.big_file_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=False, multi_modal_pose_augment=cfgs.multi_modal_pose_augment, point_augment=cfgs.point_augment, denoise=cfgs.inst_denoise, real_data=True, syn_data=True, visib_threshold=cfgs.visib_threshold, voxel_size=cfgs.voxel_size)
# TEST_DATASET = GraspNetDataset(cfgs.dataset_root, cfgs.big_file_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=False, multi_modal_pose_augment=False, point_augment=False, denoise=cfgs.inst_denoise, real_data=True, syn_data=False, visib_threshold=cfgs.visib_threshold, voxel_size=cfgs.voxel_size)
TRAIN_DATASET = GraspNetMultiDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=True, augment=cfgs.augment, voxel_size=cfgs.voxel_size)
TEST_DATASET = GraspNetMultiDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False, voxel_size=cfgs.voxel_size)
print(len(TRAIN_DATASET), len(TEST_DATASET))
# TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
#     num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)
# TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
#     num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=minkowski_collate_fn)

train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(cfgs.seed)

TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn, pin_memory=cfgs.pin_memory,
    generator=train_loader_generator,
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn, pin_memory=cfgs.pin_memory,
)

# debug_target = "scene_0036_98"   # or None
# if debug_target is not None:
#     # parse "scene_0036_98" -> scene="scene_0036", frameid=98
#     scene_name, frame_str = debug_target.rsplit("_", 1)
#     frame_id = int(frame_str)

#     debug_idx = TRAIN_DATASET.get_index_by_scene_frame(scene_name, frame_id)
#     print(f"[DEBUG] Restrict training to {scene_name}, frame {frame_id}, dataset idx={debug_idx}")

#     TRAIN_DATASET = Subset(TRAIN_DATASET, [debug_idx])

#     TRAIN_DATALOADER = DataLoader(
#         TRAIN_DATASET,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,   # debug 时建议 0，最稳
#         collate_fn=collate_fn,
#         pin_memory=cfgs.pin_memory
#     )
#     TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False,
#     num_workers=0, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
# else:
#     TRAIN_DATALOADER = DataLoader(
#         TRAIN_DATASET,
#         batch_size=cfgs.batch_size,
#         shuffle=True,
#         num_workers=cfgs.worker_num,
#         worker_init_fn=my_worker_init_fn,
#         collate_fn=collate_fn,
#         pin_memory=cfgs.pin_memory
#     )
#     TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
#     num_workers=cfgs.worker_num, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=cfgs.pin_memory)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
# net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
#                         cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])

# instance-level baseline (v0.6)
# net = IGNet(num_view=cfgs.num_view, seed_feat_dim=cfgs.seed_feat_dim, is_training=True)
# net.to(device)

# v0.8
net = IGNet(
    m_point=cfgs.m_point,
    num_view=cfgs.num_view,
    seed_feat_dim=cfgs.seed_feat_dim,
    img_feat_dim=cfgs.img_feat_dim,
    is_training=True,
    multi_scale_grouping=cfgs.multi_scale_grouping,
    fuse_type=cfgs.fuse_type,
    grouping_type=cfgs.grouping_type,
)
net.to(device)
if hasattr(net, 'enable_vis'):
    net.enable_vis(f"vis/dbg/{cfgs.method_id}/{cfgs.camera}", vis_every=1000)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

# for param in net.img_backbone.dino.parameters():
#     param.requires_grad = False
    
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load the Adam optimizer
optimizer = optim.AdamW(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
if cfgs.lr_sched:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfgs.lr_sched_period, eta_min=1e-4)

# Load checkpoint if there is any
start_epoch = 0
resume_best_loss = np.inf
resume_best_epoch = -1

if cfgs.resume_checkpoint is not None and not os.path.isfile(cfgs.resume_checkpoint):
    raise FileNotFoundError('Requested checkpoint does not exist: {}'.format(cfgs.resume_checkpoint))

if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if cfgs.lr_sched and 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = int(checkpoint.get('epoch', 0))
        resume_best_loss = float(checkpoint.get('best_eval_loss', np.inf))
        resume_best_epoch = int(checkpoint.get('best_epoch', -1))
        log_string('-> loaded full checkpoint {} (next epoch: {})'.format(
            CHECKPOINT_PATH, start_epoch + 1
        ))
    else:
        # Backward compatibility with the old best-checkpoint format, which
        # stored only model_state_dict. Optimizer/epoch cannot be recovered.
        net.load_state_dict(checkpoint)
        log_string('-> loaded model weights only from {}; optimizer and epoch were not restored'.format(
            CHECKPOINT_PATH
        ))

if cfgs.lr_sched and cfgs.lr_sched_period < cfgs.max_epoch:
    log_string(
        '[WARNING] CosineAnnealingLR with T_max={} and max_epoch={} will increase '
        'again after epoch {}. Set --lr_sched_period >= --max_epoch for a '
        'single monotonic cosine decay.'.format(
            cfgs.lr_sched_period, cfgs.max_epoch, cfgs.lr_sched_period
        )
    )

# TensorBoard Visualizers
log_writer = SummaryWriter(os.path.join(cfgs.log_dir))
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {}
    stat_batch_count = 0
    net.train()
    loss_sum = 0.0
    batch_count = 0
    total_batches = len(TRAIN_DATALOADER)
    phase_start = time.perf_counter()
    last_batch_end = phase_start

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        optimizer.zero_grad(set_to_none=True)
        batch_data_label = to_device(
            batch_data_label, device, non_blocking=cfgs.pin_memory
        )

        end_points = net(batch_data_label)
        loss, end_points = get_loss(end_points, device)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().item())
        loss_sum += current_loss
        batch_count += 1
        stat_batch_count += 1

        for key, value in end_points.items():
            if ('loss' in key or 'acc' in key or 'prec' in key
                    or 'recall' in key or 'count' in key):
                if key not in stat_dict:
                    stat_dict[key] = 0.0
                stat_dict[key] += float(value.detach().item())

        now = time.perf_counter()
        batch_time = now - last_batch_end  # includes data loading + computation
        last_batch_end = now
        avg_batch_time = (now - phase_start) / batch_count
        batches_left = total_batches - batch_count
        phase_eta = avg_batch_time * batches_left

        should_log = (
            batch_count % cfgs.log_interval == 0
            or batch_count == total_batches
        )
        if should_log:
            log_string(
                ' ---- train batch: {:04d}/{:04d} | batch {:.2f}s | '
                'elapsed {} | train ETA {} ----'.format(
                    batch_count, total_batches, batch_time,
                    format_duration(now - phase_start), format_eta(phase_eta)
                )
            )
            global_step = EPOCH_CNT * total_batches + batch_count
            for key in sorted(stat_dict.keys()):
                mean_value = stat_dict[key] / float(stat_batch_count)
                log_writer.add_scalar('train_' + key, mean_value, global_step)
                log_string('mean {}: {:.6f}'.format(key, mean_value))
            stat_dict.clear()
            stat_batch_count = 0

    elapsed = time.perf_counter() - phase_start
    mean_loss = loss_sum / float(max(batch_count, 1))
    log_string(
        'train mean loss: {:.6f}, batch num: {}, elapsed: {}'.format(
            mean_loss, batch_count, format_duration(elapsed)
        )
    )
    return mean_loss, elapsed


def evaluate_one_epoch():
    stat_dict = {}
    net.eval()
    loss_sum = 0.0
    batch_count = 0
    total_batches = len(TEST_DATALOADER)
    phase_start = time.perf_counter()
    last_batch_end = phase_start

    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
            batch_data_label = to_device(
                batch_data_label, device, non_blocking=cfgs.pin_memory
            )
            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points, device)

            loss_sum += float(loss.detach().item())
            batch_count += 1

            for key, value in end_points.items():
                if ('loss' in key or 'acc' in key or 'prec' in key
                        or 'recall' in key or 'count' in key):
                    if key not in stat_dict:
                        stat_dict[key] = 0.0
                    stat_dict[key] += float(value.detach().item())

            now = time.perf_counter()
            batch_time = now - last_batch_end
            last_batch_end = now
            avg_batch_time = (now - phase_start) / batch_count
            phase_eta = avg_batch_time * (total_batches - batch_count)

            if (batch_count % cfgs.log_interval == 0
                    or batch_count == total_batches):
                log_string(
                    'Eval batch: {:04d}/{:04d} | batch {:.2f}s | '
                    'elapsed {} | eval ETA {}'.format(
                        batch_count, total_batches, batch_time,
                        format_duration(now - phase_start), format_eta(phase_eta)
                    )
                )

    for key in sorted(stat_dict.keys()):
        mean_value = stat_dict[key] / float(max(batch_count, 1))
        global_step = (EPOCH_CNT + 1) * len(TRAIN_DATALOADER)
        log_writer.add_scalar('test_' + key, mean_value, global_step)
        log_string('eval mean {}: {:.6f}'.format(key, mean_value))

    elapsed = time.perf_counter() - phase_start
    mean_loss = loss_sum / float(max(batch_count, 1))
    log_string(
        'eval mean loss: {:.6f}, batch num: {}, elapsed: {}'.format(
            mean_loss, batch_count, format_duration(elapsed)
        )
    )
    return mean_loss, elapsed


def train(start_epoch, min_loss=np.inf, best_epoch=-1):
    global EPOCH_CNT

    epoch_times = []
    run_start = time.perf_counter()

    for epoch in range(start_epoch, cfgs.max_epoch):
        epoch_start = time.perf_counter()
        EPOCH_CNT = epoch
        log_string('**** EPOCH {:03d}/{:03d} ****'.format(epoch + 1, cfgs.max_epoch))
        current_lr = optimizer.param_groups[0]['lr']
        log_string('Current learning rate: {:.8f}'.format(current_lr))
        log_string(str(datetime.now()))

        train_loss, train_elapsed = train_one_epoch()
        log_writer.add_scalar('training/learning_rate', current_lr, epoch + 1)

        eval_loss = None
        eval_elapsed = 0.0
        improved = False
        if epoch >= cfgs.eval_start_epoch:
            eval_loss, eval_elapsed = evaluate_one_epoch()
            if eval_loss < min_loss:
                min_loss = eval_loss
                best_epoch = epoch
                improved = True
            log_string(
                'best epoch: {}, best eval loss: {:.6f}'.format(
                    best_epoch + 1 if best_epoch >= 0 else 'N/A', min_loss
                )
            )

        # Step once after completing this epoch. The saved scheduler state is
        # therefore ready for the next epoch after resume.
        if cfgs.lr_sched:
            lr_scheduler.step()

        model_state_dict = unwrap_model(net).state_dict()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_eval_loss': min_loss,
            'best_epoch': best_epoch,
            'config': vars(cfgs),
        }
        if cfgs.lr_sched:
            save_dict['lr_scheduler'] = lr_scheduler.state_dict()

        if improved:
            # Preserve the old model-only file convention for inference, while
            # also writing a full checkpoint that can truly resume training.
            ckpt_name = 'epoch_{}_train_{:.6f}_val_{:.6f}'.format(
                epoch + 1, train_loss, eval_loss
            )
            torch.save(
                model_state_dict,
                os.path.join(cfgs.ckpt_dir, ckpt_name + '.tar')
            )
            torch.save(
                save_dict,
                os.path.join(cfgs.ckpt_dir, 'best_checkpoint.tar')
            )

        if (epoch + 1) % cfgs.ckpt_save_interval == 0:
            torch.save(
                save_dict,
                os.path.join(cfgs.ckpt_dir, 'checkpoint_{}.tar'.format(epoch + 1))
            )

        torch.save(save_dict, os.path.join(cfgs.ckpt_dir, 'checkpoint.tar'))

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)
        avg_epoch_time = float(np.mean(epoch_times[-3:]))
        remaining_epochs = cfgs.max_epoch - (epoch + 1)
        total_eta = avg_epoch_time * remaining_epochs
        run_elapsed = time.perf_counter() - run_start

        log_string(
            'Epoch {:03d} time: {} (train {}, eval {})'.format(
                epoch + 1,
                format_duration(epoch_elapsed),
                format_duration(train_elapsed),
                format_duration(eval_elapsed),
            )
        )
        log_string(
            'Run elapsed: {} | remaining epochs: {} | total ETA: {}'.format(
                format_duration(run_elapsed), remaining_epochs, format_eta(total_eta)
            )
        )

    log_string('Training finished. Total elapsed: {}'.format(
        format_duration(time.perf_counter() - run_start)
    ))


if __name__ == '__main__':
    try:
        train(start_epoch, min_loss=resume_best_loss, best_epoch=resume_best_epoch)
    finally:
        log_writer.close()
        LOG_FOUT.close()
