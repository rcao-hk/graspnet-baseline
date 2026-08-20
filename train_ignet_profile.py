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
import json

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
from fusion_profile_utils import (
    PROFILE_SCHEMA_VERSION,
    RuntimeMACProfiler,
    collect_architecture_metadata,
    collect_environment,
    collect_parameter_profile,
    common_variant_identity,
    parameter_group_rows,
    protocol_fingerprint,
    save_complexity_profile,
    summarize,
    write_json,
    write_rows_csv,
)

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
parser.add_argument('--enable_training_visualization', action='store_true',
                    help='Enable debug visualization during training; disabled for clean cost profiling')
image_pretrain_group = parser.add_mutually_exclusive_group()
image_pretrain_group.add_argument(
    '--image_backbone_pretrained', dest='image_backbone_pretrained',
    action='store_true', help='Use ImageNet initialization for the ResNet34 encoder')
image_pretrain_group.add_argument(
    '--no_image_backbone_pretrained', dest='image_backbone_pretrained',
    action='store_false', help='Randomly initialize the image backbone')
parser.set_defaults(image_backbone_pretrained=True)
parser.add_argument('--image_pretraining_source', default='ImageNet-1K',
                    help='External pretraining dataset/weight source recorded in the profile')
parser.add_argument('--freeze_image_backbone', action='store_true',
                    help='Freeze all image-backbone parameters during training')
preserve_group = parser.add_mutually_exclusive_group()
preserve_group.add_argument(
    '--preserve_pretrained_image_weights', dest='preserve_pretrained_image_weights',
    action='store_true', help='Do not overwrite the pretrained image branch in IGNet._init_weights')
preserve_group.add_argument(
    '--allow_reinit_pretrained_image_weights', dest='preserve_pretrained_image_weights',
    action='store_false', help='Legacy behavior: allow IGNet._init_weights to overwrite image weights')
parser.set_defaults(preserve_pretrained_image_weights=True)
parser.add_argument('--cost_num_gpus', type=int, default=1,
                    help='Number of GPUs used for GPU-hour/day accounting [default: 1]')
parser.add_argument('--cost_output_dir', default=None,
                    help='Training-cost CSV/JSON output directory; default: training log directory')
parser.add_argument('--profile_run_id', default=None,
                    help='Stable identifier shared by all fusion variants in one cost study')
parser.add_argument('--cost_warmup_iterations', type=int, default=20,
                    help='Optimizer steps excluded before latency/throughput aggregation')
cost_sync_group = parser.add_mutually_exclusive_group()
cost_sync_group.add_argument('--cost_sync_cuda', dest='cost_sync_cuda', action='store_true',
                             help='Synchronize CUDA at profiled optimizer-step boundaries')
cost_sync_group.add_argument('--no_cost_sync_cuda', dest='cost_sync_cuda', action='store_false',
                             help='Do not synchronize CUDA; timings become CPU launch time and are not comparable')
parser.set_defaults(cost_sync_cuda=True)
parser.add_argument('--cost_profile_complexity', action='store_true',
                    help='Profile input-dependent Conv/Linear/sparse-conv MACs on one post-warmup forward')
parser.add_argument(
    '--max_train_iterations', '--stop_after_iterations',
    dest='max_train_iterations',
    type=int,
    default=-1,
    help=(
        'Stop cleanly after this many optimizer steps in the current run and '
        'write the cost report; -1 means no iteration limit [default: -1]'
    ),
)
# parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
# parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
# parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')
# parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
cfgs = parser.parse_args()
if cfgs.log_interval <= 0:
    parser.error('--log_interval must be positive')
if cfgs.ckpt_save_interval <= 0:
    parser.error('--ckpt_save_interval must be positive')
if cfgs.cost_num_gpus <= 0:
    parser.error('--cost_num_gpus must be positive')
if cfgs.max_train_iterations == 0 or cfgs.max_train_iterations < -1:
    parser.error('--max_train_iterations must be -1 or a positive integer')
if cfgs.cost_warmup_iterations < 0:
    parser.error('--cost_warmup_iterations must be non-negative')
if (cfgs.max_train_iterations > 0
        and cfgs.cost_warmup_iterations >= cfgs.max_train_iterations):
    parser.error('--cost_warmup_iterations must be smaller than --max_train_iterations')
if cfgs.image_backbone_pretrained and not cfgs.preserve_pretrained_image_weights:
    print('[WARNING] pretrained image weights are requested but may be overwritten by IGNet initialization.')
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


_MIB = 1024.0 ** 2


def tensor_tree_nbytes(obj):
    """Count tensor bytes in a nested state dictionary."""
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(tensor_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(v) for v in obj)
    return 0


def cpu_peak_rss_mib():
    """Process peak RSS in MiB; ru_maxrss uses KiB on Linux."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / _MIB if sys.platform == 'darwin' else value / 1024.0


def infer_batch_size(batch):
    """Infer actual batch size from common batched tensor fields."""
    for key in ('point_clouds', 'img'):
        value = batch.get(key) if isinstance(batch, dict) else None
        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
    raise RuntimeError('Cannot infer batch size from batch_data_label.')


class TrainingCostTracker:
    """Track a controlled fusion variant with one shared reporting schema."""

    def __init__(self, model, optimizer, device, output_dir, num_gpus,
                 prior_state=None, history_complete=True,
                 warmup_iterations=20, sync_cuda=True,
                 profile_complexity=False, run_id=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.output_dir = os.path.abspath(output_dir)
        self.num_gpus = int(num_gpus)
        self.prior = prior_state or {}
        self.history_complete = bool(history_complete)
        self.warmup_iterations = max(0, int(warmup_iterations))
        self.sync_cuda = bool(sync_cuda)
        self.profile_complexity = bool(profile_complexity)
        self.run_id = run_id or '{}-{}-seed{}'.format(
            cfgs.method_id, cfgs.camera, cfgs.seed
        )
        self.run_start = time.perf_counter()
        self.rows = []
        self.iteration_rows = []
        self.iteration_seen = 0
        self.complexity = None
        self.complexity_profiled = False
        self.complexity_profile_error = None
        self.epoch_peak_allocated_mib = 0.0
        self.epoch_peak_reserved_mib = 0.0
        self.run_peak_allocated_mib = 0.0
        self.run_peak_reserved_mib = 0.0
        os.makedirs(self.output_dir, exist_ok=True)

        self._sync()
        torch.cuda.empty_cache()
        self.baseline_allocated_mib = torch.cuda.memory_allocated(self.device) / _MIB
        self.baseline_reserved_mib = torch.cuda.memory_reserved(self.device) / _MIB
        self.run_peak_allocated_mib = self.baseline_allocated_mib
        self.run_peak_reserved_mib = self.baseline_reserved_mib
        torch.cuda.reset_peak_memory_stats(self.device)

        model_ = unwrap_model(self.model)
        self.architecture = collect_architecture_metadata(model_)
        # The data loader provides N scene points even though the model only knows M.
        self.architecture.setdefault('sampling', {})['scene_points'] = cfgs.num_point
        self.parameters = collect_parameter_profile(model_)
        self.total_params = int(self.parameters['registered_total_params'])
        self.active_params = int(self.parameters['active_total_params'])
        self.trainable_params = int(self.parameters['trainable_total_params'])
        self.model_state_size_mib = tensor_tree_nbytes(model_.state_dict()) / _MIB
        self.controlled_protocol = self._build_controlled_protocol()
        self.protocol_hash = protocol_fingerprint(self.controlled_protocol)

    def _sync(self):
        if self.sync_cuda:
            torch.cuda.synchronize(self.device)

    def _build_controlled_protocol(self):
        """Protocol fields that must remain identical across fusion variants."""
        return {
            'dataset': {
                'name': 'GraspNet-1Billion',
                'dataset_class': 'GraspNetMultiDataset',
                'dataset_root': os.path.abspath(cfgs.dataset_root),
                'big_file_root': (
                    os.path.abspath(cfgs.big_file_root)
                    if cfgs.big_file_root is not None else None
                ),
                'camera': cfgs.camera,
                'train_split': 'train',
                'validation_split': 'test_seen',
                'num_point': cfgs.num_point,
                'voxel_size': cfgs.voxel_size,
                'remove_outlier': True,
                'augmentation': bool(cfgs.augment),
            },
            'shared_model': {
                'm_point': cfgs.m_point,
                'num_view': cfgs.num_view,
                'num_angle': 12,
                'num_depth': 4,
                'seed_feature_dim': cfgs.seed_feat_dim,
                'image_feature_dim': cfgs.img_feat_dim,
                'grouping_type': cfgs.grouping_type,
                'multi_scale_grouping': bool(cfgs.multi_scale_grouping),
                'image_backbone_architecture': 'ResNet34 + PSPNet',
                'image_backbone_pretrained': bool(cfgs.image_backbone_pretrained),
                'image_pretraining_source': cfgs.image_pretraining_source,
                'freeze_image_backbone': bool(cfgs.freeze_image_backbone),
                'preserve_pretrained_image_weights': bool(
                    cfgs.preserve_pretrained_image_weights
                ),
            },
            'optimization': {
                'optimizer': 'AdamW',
                'learning_rate': cfgs.learning_rate,
                'weight_decay': cfgs.weight_decay,
                'lr_scheduler': 'CosineAnnealingLR' if cfgs.lr_sched else None,
                'lr_scheduler_period': cfgs.lr_sched_period if cfgs.lr_sched else None,
                'batch_size': cfgs.batch_size,
                'worker_num': cfgs.worker_num,
                'pin_memory': bool(cfgs.pin_memory),
                'max_epoch': cfgs.max_epoch,
                'planned_train_batches_per_epoch': int(len(TRAIN_DATALOADER)),
                'planned_optimizer_steps': int(cfgs.max_epoch * len(TRAIN_DATALOADER)),
                'checkpoint_selection': 'minimum validation loss',
            },
            'reproducibility': {
                'seed': cfgs.seed,
                'cudnn_deterministic': True,
                'cudnn_benchmark': False,
            },
            'cost_measurement': {
                'warmup_iterations': self.warmup_iterations,
                'sync_cuda': self.sync_cuda,
                'max_train_iterations': cfgs.max_train_iterations,
                'complexity_profile_enabled': self.profile_complexity,
                'num_gpus': self.num_gpus,
                'precision': 'fp32',
                'amp_enabled': False,
                'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
                'tf32_cudnn': bool(torch.backends.cudnn.allow_tf32),
                'training_visualization_enabled': bool(cfgs.enable_training_visualization),
                'complexity_scope': (
                    'one post-warmup training forward; input-dependent Conv/Linear/'
                    'MinkowskiConvolution MACs'
                ),
            },
        }

    def begin_epoch(self):
        self._sync()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.epoch_peak_allocated_mib = self.baseline_allocated_mib
        self.epoch_peak_reserved_mib = self.baseline_reserved_mib

    def begin_iteration(self, *, data_wait_ms, epoch, batch_in_epoch):
        """Start one optimizer-step measurement after the DataLoader yielded a batch."""
        is_warmup = self.iteration_seen < self.warmup_iterations
        profile_complexity = (
            self.profile_complexity
            and not self.complexity_profiled
            and not is_warmup
        )
        self._sync()
        allocated_before = torch.cuda.memory_allocated(self.device) / _MIB
        reserved_before = torch.cuda.memory_reserved(self.device) / _MIB
        torch.cuda.reset_peak_memory_stats(self.device)
        return {
            'start_time': time.perf_counter(),
            'data_wait_ms': float(max(0.0, data_wait_ms)),
            'epoch': int(epoch),
            'batch_in_epoch': int(batch_in_epoch),
            'is_warmup': bool(is_warmup),
            'profile_complexity': bool(profile_complexity),
            'gpu_allocated_before_mib': allocated_before,
            'gpu_reserved_before_mib': reserved_before,
        }

    def end_iteration(self, state, *, batch_size, loss_value):
        self._sync()
        compute_ms = (time.perf_counter() - state['start_time']) * 1000.0
        peak_allocated = torch.cuda.max_memory_allocated(self.device) / _MIB
        peak_reserved = torch.cuda.max_memory_reserved(self.device) / _MIB
        end_allocated = torch.cuda.memory_allocated(self.device) / _MIB
        end_reserved = torch.cuda.memory_reserved(self.device) / _MIB
        if not state['profile_complexity']:
            # Kernel-map inspection used by the optional complexity hook may
            # allocate temporary tensors. Exclude that diagnostic-only step from
            # the reported deployment/training memory budget.
            self.epoch_peak_allocated_mib = max(
                self.epoch_peak_allocated_mib, peak_allocated
            )
            self.epoch_peak_reserved_mib = max(
                self.epoch_peak_reserved_mib, peak_reserved
            )
            self.run_peak_allocated_mib = max(self.run_peak_allocated_mib, peak_allocated)
            self.run_peak_reserved_mib = max(self.run_peak_reserved_mib, peak_reserved)

        self.iteration_seen += 1
        row = {
            'iteration': int(self.iteration_seen),
            'epoch': state['epoch'],
            'batch_in_epoch': state['batch_in_epoch'],
            'is_warmup': bool(state['is_warmup']),
            # Runtime complexity hooks introduce overhead and are excluded from the
            # steady-state time distribution even though the optimizer step is valid.
            'is_complexity_profile': bool(state['profile_complexity']),
            'included_in_steady_state': bool(
                not state['is_warmup'] and not state['profile_complexity']
            ),
            'batch_size': int(batch_size),
            'loss': float(loss_value),
            'data_wait_ms': float(state['data_wait_ms']),
            'optimizer_step_ms': float(compute_ms),
            'iteration_with_data_ms': float(state['data_wait_ms'] + compute_ms),
            'compute_samples_per_s': (
                1000.0 * batch_size / compute_ms if compute_ms > 0 else 0.0
            ),
            'wall_samples_per_s': (
                1000.0 * batch_size / (state['data_wait_ms'] + compute_ms)
                if state['data_wait_ms'] + compute_ms > 0 else 0.0
            ),
            'gpu_allocated_before_mib': state['gpu_allocated_before_mib'],
            'gpu_reserved_before_mib': state['gpu_reserved_before_mib'],
            'peak_gpu_allocated_mib': peak_allocated,
            'peak_gpu_reserved_mib': peak_reserved,
            'incremental_peak_gpu_allocated_mib': max(
                0.0, peak_allocated - state['gpu_allocated_before_mib']
            ),
            'incremental_peak_gpu_reserved_mib': max(
                0.0, peak_reserved - state['gpu_reserved_before_mib']
            ),
            'end_gpu_allocated_mib': end_allocated,
            'end_gpu_reserved_mib': end_reserved,
            'peak_cpu_rss_mib': cpu_peak_rss_mib(),
        }
        self.iteration_rows.append(row)
        return row

    @staticmethod
    def should_profile_complexity(iteration_state):
        return bool(iteration_state.get('profile_complexity', False))

    def set_complexity(self, complexity):
        self.complexity = complexity
        self.complexity_profiled = True

    def set_complexity_error(self, exc):
        self.complexity_profile_error = '{}: {}'.format(type(exc).__name__, exc)
        self.complexity_profiled = True

    def record_epoch(self, epoch, train_seconds, eval_seconds, epoch_seconds,
                     train_batches, eval_batches, train_samples, eval_samples,
                     train_loss, eval_loss, completed_epoch=True):
        self._sync()
        row = {
            'epoch': int(epoch),
            'completed_epoch': bool(completed_epoch),
            'train_seconds': float(train_seconds),
            'eval_seconds': float(eval_seconds),
            'epoch_seconds_before_checkpoint': float(epoch_seconds),
            'train_batches': int(train_batches),
            'eval_batches': int(eval_batches),
            'train_samples': int(train_samples),
            'eval_samples': int(eval_samples),
            'train_samples_per_s': float(train_samples / train_seconds) if train_seconds > 0 else 0.0,
            'eval_samples_per_s': float(eval_samples / eval_seconds) if eval_seconds > 0 else 0.0,
            'train_loss': float(train_loss),
            'eval_loss': None if eval_loss is None else float(eval_loss),
            'peak_gpu_allocated_mib': self.epoch_peak_allocated_mib,
            'peak_gpu_reserved_mib': self.epoch_peak_reserved_mib,
            'end_gpu_allocated_mib': torch.cuda.memory_allocated(self.device) / _MIB,
            'end_gpu_reserved_mib': torch.cuda.memory_reserved(self.device) / _MIB,
            'peak_cpu_rss_mib': cpu_peak_rss_mib(),
        }
        self.rows.append(row)
        self.write_epoch_csv()
        return row

    def _steady_iteration_rows(self):
        return [r for r in self.iteration_rows if r['included_in_steady_state']]

    def _iteration_summary(self):
        rows = self._steady_iteration_rows()
        metric_names = (
            'data_wait_ms',
            'optimizer_step_ms',
            'iteration_with_data_ms',
            'compute_samples_per_s',
            'wall_samples_per_s',
            'peak_gpu_allocated_mib',
            'peak_gpu_reserved_mib',
            'incremental_peak_gpu_allocated_mib',
            'incremental_peak_gpu_reserved_mib',
            'peak_cpu_rss_mib',
        )
        metrics = {
            name: summarize(float(r[name]) for r in rows)
            for name in metric_names
        }
        return {
            'warmup_iterations': min(self.iteration_seen, self.warmup_iterations),
            'profiled_iterations': len(rows),
            'complexity_profile_iterations': sum(
                int(r['is_complexity_profile']) for r in self.iteration_rows
            ),
            'sync_cuda': self.sync_cuda,
            'metrics': metrics,
        }

    def aggregate_state(self):
        run_wall = time.perf_counter() - self.run_start
        run_train = sum(r['train_seconds'] for r in self.rows)
        run_eval = sum(r['eval_seconds'] for r in self.rows)
        train_samples = sum(r['train_samples'] for r in self.rows)
        eval_samples = sum(r['eval_samples'] for r in self.rows)
        optimizer_steps = sum(r['train_batches'] for r in self.rows)

        cumulative_wall = float(self.prior.get('cumulative_wall_seconds', 0.0)) + run_wall
        cumulative_train = float(self.prior.get('cumulative_train_seconds', 0.0)) + run_train
        cumulative_eval = float(self.prior.get('cumulative_eval_seconds', 0.0)) + run_eval
        cumulative_train_samples = int(self.prior.get('cumulative_train_samples', 0)) + train_samples
        cumulative_eval_samples = int(self.prior.get('cumulative_eval_samples', 0)) + eval_samples
        cumulative_steps = int(self.prior.get('cumulative_optimizer_steps', 0)) + optimizer_steps
        cumulative_epochs = int(self.prior.get('cumulative_epochs', 0)) + sum(
            int(row.get('completed_epoch', True)) for row in self.rows
        )

        iteration_summary = self._iteration_summary()
        profiled_mean_ms = iteration_summary['metrics']['iteration_with_data_ms']['mean']
        if profiled_mean_ms > 0:
            mean_train_iteration_seconds = profiled_mean_ms / 1000.0
            timing_source = 'post-warmup synchronized iteration_with_data_ms'
        else:
            mean_train_iteration_seconds = (
                run_train / optimizer_steps if optimizer_steps > 0 else 0.0
            )
            timing_source = 'phase wall time / optimizer steps fallback'

        train_batches_per_epoch = int(len(TRAIN_DATALOADER))
        planned_full_optimizer_steps = int(cfgs.max_epoch * train_batches_per_epoch)
        projected_full_train_seconds = (
            mean_train_iteration_seconds * planned_full_optimizer_steps
        )
        projected_remaining_optimizer_steps = max(
            0, planned_full_optimizer_steps - cumulative_steps
        )
        projected_remaining_train_seconds = (
            mean_train_iteration_seconds * projected_remaining_optimizer_steps
        )

        return {
            'cumulative_complete': self.history_complete,
            'num_gpus': self.num_gpus,
            'cumulative_epochs': cumulative_epochs,
            'cumulative_optimizer_steps': cumulative_steps,
            'cumulative_train_samples': cumulative_train_samples,
            'cumulative_eval_samples': cumulative_eval_samples,
            'cumulative_wall_seconds': cumulative_wall,
            'cumulative_train_seconds': cumulative_train,
            'cumulative_eval_seconds': cumulative_eval,
            'cumulative_gpu_hours': cumulative_wall * self.num_gpus / 3600.0,
            'cumulative_gpu_days': cumulative_wall * self.num_gpus / 86400.0,
            'cumulative_optimization_gpu_days': cumulative_train * self.num_gpus / 86400.0,
            'cumulative_train_samples_per_s': (
                cumulative_train_samples / cumulative_train
                if cumulative_train > 0 else 0.0
            ),
            'run_wall_seconds': run_wall,
            'run_train_seconds': run_train,
            'run_eval_seconds': run_eval,
            'run_train_samples': train_samples,
            'run_eval_samples': eval_samples,
            'run_optimizer_steps': optimizer_steps,
            'run_gpu_hours': run_wall * self.num_gpus / 3600.0,
            'run_gpu_days': run_wall * self.num_gpus / 86400.0,
            'run_train_samples_per_s': train_samples / run_train if run_train > 0 else 0.0,
            'mean_train_iteration_seconds': mean_train_iteration_seconds,
            'mean_train_iteration_timing_source': timing_source,
            'train_batches_per_epoch': train_batches_per_epoch,
            'planned_full_optimizer_steps': planned_full_optimizer_steps,
            'projected_full_train_seconds': projected_full_train_seconds,
            'projected_full_train_wall_hours': projected_full_train_seconds / 3600.0,
            'projected_full_train_gpu_hours': projected_full_train_seconds * self.num_gpus / 3600.0,
            'projected_full_train_gpu_days': projected_full_train_seconds * self.num_gpus / 86400.0,
            'projected_full_training_gpu_days': projected_full_train_seconds * self.num_gpus / 86400.0,
            'actual_completed_training_gpu_hours': (
                cumulative_wall * self.num_gpus / 3600.0
                if self.history_complete and cumulative_steps >= planned_full_optimizer_steps
                else None
            ),
            'actual_completed_training_gpu_days': (
                cumulative_wall * self.num_gpus / 86400.0
                if self.history_complete and cumulative_steps >= planned_full_optimizer_steps
                else None
            ),
            'training_cost_source': (
                'actual_completed_run'
                if self.history_complete and cumulative_steps >= planned_full_optimizer_steps
                else 'projected_from_post_warmup_optimizer_steps'
            ),
            'projected_remaining_optimizer_steps': projected_remaining_optimizer_steps,
            'projected_remaining_train_seconds': projected_remaining_train_seconds,
            'projected_remaining_train_gpu_days': projected_remaining_train_seconds * self.num_gpus / 86400.0,
            'projection_scope': (
                'post-warmup optimizer-step wall time including DataLoader wait and '
                'host-to-device transfer; validation, checkpoint I/O and initialization excluded'
            ),
            'peak_gpu_allocated_mib': self.run_peak_allocated_mib,
            'peak_gpu_reserved_mib': self.run_peak_reserved_mib,
            'baseline_gpu_allocated_mib': self.baseline_allocated_mib,
            'baseline_gpu_reserved_mib': self.baseline_reserved_mib,
            'incremental_peak_gpu_allocated_mib': max(
                0.0, self.run_peak_allocated_mib - self.baseline_allocated_mib
            ),
            'peak_cpu_rss_mib': max(
                [cpu_peak_rss_mib()] + [r['peak_cpu_rss_mib'] for r in self.iteration_rows]
            ),
            'iteration_profile': iteration_summary,
        }

    def write_epoch_csv(self):
        path = os.path.join(self.output_dir, 'training_cost_epoch_rows.csv')
        fields = [
            'epoch', 'completed_epoch', 'train_seconds', 'eval_seconds',
            'epoch_seconds_before_checkpoint', 'train_batches', 'eval_batches',
            'train_samples', 'eval_samples', 'train_samples_per_s',
            'eval_samples_per_s', 'train_loss', 'eval_loss',
            'peak_gpu_allocated_mib', 'peak_gpu_reserved_mib',
            'end_gpu_allocated_mib', 'end_gpu_reserved_mib',
            'peak_cpu_rss_mib',
        ]
        write_rows_csv(path, self.rows, fieldnames=fields)

    def write_iteration_csv(self):
        return write_rows_csv(
            os.path.join(self.output_dir, 'training_cost_iteration_rows.csv'),
            self.iteration_rows,
        )

    @staticmethod
    def _nested(payload, *keys, default=None):
        value = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def _variant_row(self, summary):
        cost = summary['cost']
        timing = cost['iteration_profile']['metrics']
        architecture = summary['architecture']
        parameters = summary['parameters']
        complexity = summary.get('complexity') or {}
        row = common_variant_identity(
            architecture, phase='training', run_id=self.run_id
        )
        row.update({
            'protocol_fingerprint': self.protocol_hash,
            'status': summary['status'],
            'camera': cfgs.camera,
            'seed': cfgs.seed,
            'batch_size': cfgs.batch_size,
            'num_point': cfgs.num_point,
            'm_point': cfgs.m_point,
            'image_pretrained': self._nested(
                architecture, 'image_backbone', 'encoder_pretrained', default=False
            ),
            'image_pretraining_source': self._nested(
                architecture, 'image_backbone', 'encoder_pretraining_source'
            ),
            'image_backbone_frozen': self._nested(
                architecture, 'image_backbone', 'frozen', default=False
            ),
            'image_feature_dim': self._nested(
                architecture, 'feature_channels', 'image_feature_dim', default=0
            ),
            'point_feature_dim': self._nested(
                architecture, 'feature_channels', 'point_backbone_output_dim'
            ),
            'fused_feature_dim': self._nested(
                architecture, 'feature_channels', 'fused_feature_dim'
            ),
            'num_injections': self._nested(
                architecture, 'fusion', 'num_injections', default=0
            ),
            'injection_stages': json.dumps(
                self._nested(architecture, 'fusion', 'injection_stages', default=[])
            ),
            'registered_params_m': parameters['registered_total_params_m'],
            'active_params_m': parameters['active_total_params_m'],
            'trainable_params_m': parameters['trainable_total_params_m'],
            'image_params_m': parameters['groups']['image_backbone']['active_params_m'],
            'point_backbone_params_m': parameters['groups']['point_backbone']['active_params_m'],
            'fusion_projection_params_m': parameters['groups']['fusion_projection']['active_params_m'],
            'prediction_head_params_m': parameters['groups']['prediction_heads']['active_params_m'],
            'grouping_params_m': parameters['groups']['local_grouping']['active_params_m'],
            'profiled_iterations': cost['iteration_profile']['profiled_iterations'],
            'warmup_iterations': cost['iteration_profile']['warmup_iterations'],
            'train_iteration_mean_ms': timing['iteration_with_data_ms']['mean'],
            'train_iteration_p95_ms': timing['iteration_with_data_ms']['p95'],
            'optimizer_step_mean_ms': timing['optimizer_step_ms']['mean'],
            'data_wait_mean_ms': timing['data_wait_ms']['mean'],
            'train_wall_samples_per_s': timing['wall_samples_per_s']['mean'],
            'training_peak_allocated_mib': cost['peak_gpu_allocated_mib'],
            'training_peak_reserved_mib': cost['peak_gpu_reserved_mib'],
            'training_incremental_peak_mib': cost['incremental_peak_gpu_allocated_mib'],
            'projected_full_train_gpu_hours': cost['projected_full_train_gpu_hours'],
            'projected_full_train_gpu_days': cost['projected_full_train_gpu_days'],
            'actual_completed_training_gpu_hours': cost.get('actual_completed_training_gpu_hours'),
            'actual_completed_training_gpu_days': cost.get('actual_completed_training_gpu_days'),
            'training_cost_source': cost.get('training_cost_source'),
            'gpu_name': torch.cuda.get_device_name(self.device),
            'pytorch_version': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'precision': 'fp32',
            'forward_gmacs_per_batch': complexity.get('total_gmacs'),
            'forward_gflops_per_batch_2xmac': complexity.get('total_gflops'),
            'forward_gmacs_per_sample': complexity.get('gmacs_per_sample'),
            'forward_gflops_per_sample_2xmac': complexity.get('gflops_per_sample'),
            'sparse_count_method': complexity.get('sparse_count_method'),
            'complexity_scope': complexity.get('scope'),
            'architecture_warnings': json.dumps(architecture.get('warnings', [])),
        })
        return row

    def finalize(self, checkpoint_path, status):
        self._sync()
        state = self.aggregate_state()
        optimizer_size_mib = tensor_tree_nbytes(self.optimizer.state_dict()) / _MIB
        checkpoint_size_mib = (
            os.path.getsize(checkpoint_path) / _MIB
            if checkpoint_path and os.path.isfile(checkpoint_path) else None
        )
        summary = {
            'schema_version': PROFILE_SCHEMA_VERSION,
            'phase': 'training',
            'run_id': self.run_id,
            'status': status,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'variant': common_variant_identity(
                self.architecture, phase='training', run_id=self.run_id
            ),
            'environment': collect_environment(self.device),
            'controlled_protocol': self.controlled_protocol,
            'protocol_fingerprint': self.protocol_hash,
            'architecture': self.architecture,
            'parameters': self.parameters,
            'complexity': self.complexity,
            'complexity_profile_error': self.complexity_profile_error,
            'artifact_sizes': {
                'model_state_size_mib': self.model_state_size_mib,
                'optimizer_state_size_mib': optimizer_size_mib,
                'full_checkpoint_size_mib': checkpoint_size_mib,
            },
            # Keep a compact legacy-compatible config block for existing analysis.
            'config': {
                'method_id': cfgs.method_id,
                'camera': cfgs.camera,
                'batch_size': cfgs.batch_size,
                'num_point': cfgs.num_point,
                'm_point': cfgs.m_point,
                'start_epoch': start_epoch,
                'max_epoch': cfgs.max_epoch,
                'worker_num': cfgs.worker_num,
                'fuse_type': cfgs.fuse_type,
                'grouping_type': cfgs.grouping_type,
                'num_gpus_for_cost': self.num_gpus,
                'max_train_iterations': cfgs.max_train_iterations,
                'cost_warmup_iterations': cfgs.cost_warmup_iterations,
                'image_backbone_pretrained': cfgs.image_backbone_pretrained,
                'image_pretraining_source': cfgs.image_pretraining_source,
                'freeze_image_backbone': cfgs.freeze_image_backbone,
            },
            'model': {
                'total_params': self.total_params,
                'active_params': self.active_params,
                'trainable_params': self.trainable_params,
                'model_state_size_mib': self.model_state_size_mib,
                'optimizer_state_size_mib': optimizer_size_mib,
                'full_checkpoint_size_mib': checkpoint_size_mib,
            },
            'cost': state,
        }
        summary_path = write_json(
            os.path.join(self.output_dir, 'training_cost_summary.json'), summary
        )
        self.write_epoch_csv()
        self.write_iteration_csv()
        identity = common_variant_identity(
            self.architecture, phase='training', run_id=self.run_id
        )
        write_rows_csv(
            os.path.join(self.output_dir, 'training_parameter_groups.csv'),
            parameter_group_rows(self.parameters, common=identity),
        )
        if self.complexity is not None:
            save_complexity_profile(
                self.output_dir,
                self.complexity,
                basename='training_forward_complexity',
                common=identity,
            )
        variant_row_path = write_rows_csv(
            os.path.join(self.output_dir, 'training_variant_row.csv'),
            [self._variant_row(summary)],
        )

        log_string(
            '[TRAIN-COST] Params: {:.2f}M registered / {:.2f}M active / '
            '{:.2f}M trainable | image {:.2f}M | point {:.2f}M | fusion {:.3f}M'.format(
                self.total_params / 1e6,
                self.active_params / 1e6,
                self.trainable_params / 1e6,
                self.parameters['groups']['image_backbone']['active_params_m'],
                self.parameters['groups']['point_backbone']['active_params_m'],
                self.parameters['groups']['fusion_projection']['active_params_m'],
            )
        )
        log_string(
            '[TRAIN-COST] Peak VRAM: {:.2f} MiB allocated / {:.2f} MiB reserved | '
            'incremental {:.2f} MiB | peak CPU RSS: {:.2f} MiB'.format(
                state['peak_gpu_allocated_mib'],
                state['peak_gpu_reserved_mib'],
                state['incremental_peak_gpu_allocated_mib'],
                state['peak_cpu_rss_mib'],
            )
        )
        timing = state['iteration_profile']['metrics']['iteration_with_data_ms']
        log_string(
            '[TRAIN-COST] Steady-state iterations: {} after {} warmup | '
            'mean {:.3f} ms | p95 {:.3f} ms.'.format(
                state['iteration_profile']['profiled_iterations'],
                state['iteration_profile']['warmup_iterations'],
                timing['mean'], timing['p95'],
            )
        )
        log_string(
            '[TRAIN-COST] Run wall {} | train {} | eval {} | {:.2f} train samples/s'.format(
                format_duration(state['run_wall_seconds']),
                format_duration(state['run_train_seconds']),
                format_duration(state['run_eval_seconds']),
                state['run_train_samples_per_s'],
            )
        )
        suffix = '' if state['cumulative_complete'] else ' [incomplete history: old checkpoint]'
        log_string(
            '[TRAIN-COST] Run {:.4f} GPU-days | cumulative {:.4f} GPU-days{}.'.format(
                state['run_gpu_days'], state['cumulative_gpu_days'], suffix
            )
        )
        if state['run_optimizer_steps'] > 0:
            log_string(
                '[TRAIN-COST] Estimated FULL TRAINING cost: {} wall time | '
                '{:.3f} GPU-hours | {:.4f} GPU-days ({} epochs, {} GPU).'.format(
                    format_duration(state['projected_full_train_seconds']),
                    state['projected_full_train_gpu_hours'],
                    state['projected_full_train_gpu_days'],
                    cfgs.max_epoch,
                    self.num_gpus,
                )
            )
        if self.complexity is not None:
            log_string(
                '[TRAIN-COST] Forward complexity: {:.3f} GMACs / {:.3f} GFLOPs '
                '(2 FLOPs/MAC; sparse method={}).'.format(
                    self.complexity['total_gmacs'],
                    self.complexity['total_gflops'],
                    self.complexity['sparse_count_method'],
                )
            )
        elif self.complexity_profile_error:
            log_string('[TRAIN-COST] Complexity profiling failed: {}'.format(
                self.complexity_profile_error
            ))
        if checkpoint_size_mib is not None:
            log_string('[TRAIN-COST] Full resume checkpoint: {:.2f} MiB'.format(checkpoint_size_mib))
        log_string('[TRAIN-COST] Summary saved to {}'.format(summary_path))
        log_string('[TRAIN-COST] Unified variant row saved to {}'.format(variant_row_path))
        return summary

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
    image_backbone_pretrained=cfgs.image_backbone_pretrained,
    image_pretraining_source=cfgs.image_pretraining_source,
    freeze_image_backbone=cfgs.freeze_image_backbone,
    preserve_pretrained_image_weights=cfgs.preserve_pretrained_image_weights,
)
net.to(device)
if cfgs.enable_training_visualization and hasattr(net, 'enable_vis'):
    net.enable_vis(f"vis/dbg/{cfgs.method_id}/{cfgs.camera}", vis_every=1000)


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

# for param in net.img_backbone.dino.parameters():
#     param.requires_grad = False
    
# optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load the Adam optimizer
optimizer = optim.AdamW(
    (p for p in net.parameters() if p.requires_grad),
    lr=cfgs.learning_rate,
    weight_decay=cfgs.weight_decay,
)
if cfgs.lr_sched:
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfgs.lr_sched_period, eta_min=1e-4)

# Load checkpoint if there is any
start_epoch = 0
resume_best_loss = np.inf
resume_best_epoch = -1
resume_training_cost_state = None

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
        resume_training_cost_state = checkpoint.get('training_cost_state')
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

# Release checkpoint tensors loaded onto GPU before measuring training peak memory.
if 'checkpoint' in locals():
    del checkpoint
torch.cuda.empty_cache()

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

cost_output_dir = cfgs.cost_output_dir or cfgs.log_dir
cost_history_complete = not (start_epoch > 0 and resume_training_cost_state is None)
TRAINING_COST = TrainingCostTracker(
    model=net,
    optimizer=optimizer,
    device=device,
    output_dir=cost_output_dir,
    num_gpus=cfgs.cost_num_gpus,
    prior_state=resume_training_cost_state,
    history_complete=cost_history_complete,
    warmup_iterations=cfgs.cost_warmup_iterations,
    sync_cuda=cfgs.cost_sync_cuda,
    profile_complexity=cfgs.cost_profile_complexity,
    run_id=cfgs.profile_run_id,
)
log_string('[TRAIN-COST] Tracking enabled: {} GPU, output={}'.format(
    cfgs.cost_num_gpus, cost_output_dir
))
# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch(max_iterations=-1):
    stat_dict = {}
    stat_batch_count = 0
    net.train()
    loss_sum = 0.0
    batch_count = 0
    sample_count = 0
    total_batches = len(TRAIN_DATALOADER)
    phase_start = time.perf_counter()
    last_batch_end = phase_start
    iteration_limit_reached = False

    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        iteration_enter = time.perf_counter()
        data_wait_ms = (iteration_enter - last_batch_end) * 1000.0
        actual_batch_size = infer_batch_size(batch_data_label)
        iteration_state = TRAINING_COST.begin_iteration(
            data_wait_ms=data_wait_ms,
            epoch=EPOCH_CNT + 1,
            batch_in_epoch=batch_idx + 1,
        )

        optimizer.zero_grad(set_to_none=True)
        batch_data_label = to_device(
            batch_data_label, device, non_blocking=cfgs.pin_memory
        )

        if TRAINING_COST.should_profile_complexity(iteration_state):
            try:
                mac_profiler = RuntimeMACProfiler(
                    unwrap_model(net),
                    label='training_forward_after_warmup',
                )
                with mac_profiler:
                    end_points = net(batch_data_label)
                complexity_summary = mac_profiler.summary()
                complexity_summary['batch_size'] = actual_batch_size
                complexity_summary['gmacs_per_sample'] = (
                    complexity_summary['total_gmacs'] / actual_batch_size
                    if actual_batch_size > 0 else 0.0
                )
                complexity_summary['gflops_per_sample'] = (
                    complexity_summary['total_gflops'] / actual_batch_size
                    if actual_batch_size > 0 else 0.0
                )
                TRAINING_COST.set_complexity(complexity_summary)
            except Exception as exc:
                # Cost profiling must never invalidate the actual optimizer run.
                # Re-run the forward without hooks and record the diagnostic error.
                TRAINING_COST.set_complexity_error(exc)
                optimizer.zero_grad(set_to_none=True)
                end_points = net(batch_data_label)
        else:
            end_points = net(batch_data_label)

        loss, end_points = get_loss(end_points, device)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().item())
        iteration_row = TRAINING_COST.end_iteration(
            iteration_state,
            batch_size=actual_batch_size,
            loss_value=current_loss,
        )
        loss_sum += current_loss
        batch_count += 1
        sample_count += actual_batch_size
        stat_batch_count += 1

        for key, value in end_points.items():
            if ('loss' in key or 'acc' in key or 'prec' in key
                    or 'recall' in key or 'count' in key):
                if key not in stat_dict:
                    stat_dict[key] = 0.0
                stat_dict[key] += float(value.detach().item())

        now = time.perf_counter()
        batch_time = iteration_row['iteration_with_data_ms'] / 1000.0
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

        # Set this after logging so the next data_wait_ms contains only the
        # DataLoader wait, not logging/profiling bookkeeping from this step.
        last_batch_end = time.perf_counter()

        if max_iterations > 0 and batch_count >= max_iterations:
            iteration_limit_reached = True
            log_string(
                '[TRAIN-COST] Iteration limit reached after {} optimizer '
                'steps in this epoch.'.format(batch_count)
            )
            break

    elapsed = time.perf_counter() - phase_start
    mean_loss = loss_sum / float(max(batch_count, 1))
    log_string(
        'train mean loss: {:.6f}, batch num: {}, elapsed: {}'.format(
            mean_loss, batch_count, format_duration(elapsed)
        )
    )
    return (
        mean_loss,
        elapsed,
        batch_count,
        sample_count,
        iteration_limit_reached,
    )


def evaluate_one_epoch():
    stat_dict = {}
    net.eval()
    loss_sum = 0.0
    batch_count = 0
    sample_count = 0
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
            sample_count += infer_batch_size(batch_data_label)

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
    return mean_loss, elapsed, batch_count, sample_count


def train(start_epoch, min_loss=np.inf, best_epoch=-1):
    global EPOCH_CNT

    epoch_times = []
    run_start = time.perf_counter()
    run_train_iterations = 0

    if cfgs.max_train_iterations > 0:
        log_string(
            '[TRAIN-COST] Cost-probe mode: stop after {} optimizer steps '
            'and extrapolate all training iterations over {} epochs; '
            'validation is excluded.'.format(
                cfgs.max_train_iterations,
                cfgs.max_epoch,
            )
        )

    for epoch in range(start_epoch, cfgs.max_epoch):
        epoch_start = time.perf_counter()
        TRAINING_COST.begin_epoch()
        EPOCH_CNT = epoch
        log_string('**** EPOCH {:03d}/{:03d} ****'.format(epoch + 1, cfgs.max_epoch))
        current_lr = optimizer.param_groups[0]['lr']
        log_string('Current learning rate: {:.8f}'.format(current_lr))
        log_string(str(datetime.now()))

        remaining_iteration_budget = -1
        if cfgs.max_train_iterations > 0:
            remaining_iteration_budget = (
                cfgs.max_train_iterations - run_train_iterations
            )
            if remaining_iteration_budget <= 0:
                return 'iteration_limit_reached'

        (
            train_loss,
            train_elapsed,
            train_batches,
            train_samples,
            iteration_limit_reached,
        ) = train_one_epoch(max_iterations=remaining_iteration_budget)
        run_train_iterations += train_batches
        stop_after_train = (
            cfgs.max_train_iterations > 0
            and run_train_iterations >= cfgs.max_train_iterations
        )

        log_writer.add_scalar('training/learning_rate', current_lr, epoch + 1)

        eval_loss = None
        eval_elapsed = 0.0
        eval_batches = 0
        eval_samples = 0
        improved = False
        if (not stop_after_train) and epoch >= cfgs.eval_start_epoch:
            eval_loss, eval_elapsed, eval_batches, eval_samples = evaluate_one_epoch()
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
        if cfgs.lr_sched and not stop_after_train:
            lr_scheduler.step()

        epoch_elapsed_before_checkpoint = time.perf_counter() - epoch_start
        cost_row = TRAINING_COST.record_epoch(
            epoch=epoch + 1,
            train_seconds=train_elapsed,
            eval_seconds=eval_elapsed,
            epoch_seconds=epoch_elapsed_before_checkpoint,
            train_batches=train_batches,
            eval_batches=eval_batches,
            train_samples=train_samples,
            eval_samples=eval_samples,
            train_loss=train_loss,
            eval_loss=eval_loss,
            completed_epoch=(train_batches == len(TRAIN_DATALOADER)),
        )
        log_string(
            '[TRAIN-COST] Epoch {:03d}: peak VRAM {:.2f}/{:.2f} MiB '
            '(allocated/reserved), train throughput {:.2f} samples/s'.format(
                epoch + 1,
                cost_row['peak_gpu_allocated_mib'],
                cost_row['peak_gpu_reserved_mib'],
                cost_row['train_samples_per_s'],
            )
        )

        if stop_after_train:
            log_string(
                '[TRAIN-COST] Stopping at run iteration {}/{}. Training-only '
                'cost projection will now be written; no validation or training '
                'checkpoint is produced for this probe run.'.format(
                    run_train_iterations, cfgs.max_train_iterations
                )
            )
            return 'iteration_limit_reached'

        model_state_dict = unwrap_model(net).state_dict()
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_eval_loss': min_loss,
            'best_epoch': best_epoch,
            'config': vars(cfgs),
            'training_cost_state': TRAINING_COST.aggregate_state(),
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
    return 'completed'


if __name__ == '__main__':
    training_status = 'completed'
    try:
        training_status = train(
            start_epoch,
            min_loss=resume_best_loss,
            best_epoch=resume_best_epoch,
        )
    except BaseException:
        training_status = 'interrupted'
        raise
    finally:
        try:
            TRAINING_COST.finalize(
                checkpoint_path=(
                    DEFAULT_CHECKPOINT_PATH
                    if training_status == 'completed'
                    else None
                ),
                status=training_status,
            )
        finally:
            log_writer.close()
            LOG_FOUT.close()
