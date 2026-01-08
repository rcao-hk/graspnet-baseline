import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

import cv2
import time
import re
import glob
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME

from graspnetAPI import GraspGroup, GraspNetEval
from models.GSNet import GraspNet, pred_decode

from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.data_utils import (
    CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask,
    sample_points, apply_smoothing, add_gaussian_noise_depth_map,
    find_large_missing_regions, apply_dropout_to_regions
)

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

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Where dataset is')
parser.add_argument('--checkpoint_path', help='Model checkpoint path', default=None, required=True)
parser.add_argument('--dump_dir', help='Dump dir to save outputs', default=None, required=True)
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--data_type', type=str, default='real', choices=['real', 'syn', 'noise'],
                    help='Type of input data: real|syn|noise')

parser.add_argument('--smooth_size', type=int, default=1,
                    help='Box smoothing kernel size on depth (<=1 means off)')
parser.add_argument('--gaussian_noise_level', type=float, default=0.0,
                    help='Gaussian noise std in meters (0 means off)')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Depth-guided dropout: fraction of missing regions to DROP (0 means off)')
parser.add_argument('--dropout_min_size', type=int, default=200,
                    help='Min connected component size for missing regions (on FG mask)')

cfgs = parser.parse_args()


width = 1280
height = 720
# voxel_size = 0.005
# TOP_K = 300

def _disable_corruptions(cfgs):
    cfgs.smooth_size = 1
    cfgs.gaussian_noise_level = 0.0
    cfgs.dropout_rate = 0.0
    cfgs.dropout_min_size = 0
    cfgs.rgb_noise = 'none'
    cfgs.rgb_severity = 0
    
data_type = cfgs.data_type
split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
voxel_size = cfgs.voxel_size
dump_dir = os.path.join(cfgs.dump_dir)

if data_type != 'noise':
    _disable_corruptions(cfgs)

print(cfgs)

device = torch.device("cuda:"+cfgs.gpu_id if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
net.to(device)
net.eval()
checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)

net.load_state_dict(checkpoint['model_state_dict'])
eps = 1e-8

def inference(scene_idx):
    for anno_idx in range(256):

        depth_raw_path = None  # only used in noise mode

        if data_type == 'real':
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root,
                                      'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root,
                                     'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))

        elif data_type == 'syn':
            # use synthetic depth/label
            rgb_path = os.path.join(dataset_root,
                                    'virtual_scenes/scene_{:04d}/{}/{:04d}_rgb.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root,
                                      'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root,
                                     'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))

        elif data_type == 'noise':
            # follow script1: real RGB + synthetic clear depth + synthetic label + real depth for missing-region guidance
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root,
                                      'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            depth_raw_path = os.path.join(dataset_root,
                                          'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root,
                                     'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        meta_path = os.path.join(dataset_root,
                                 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(width, height,
                                 intrinsics[0][0], intrinsics[1][1],
                                 intrinsics[0][2], intrinsics[1][2],
                                 factor_depth)

        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        depth_mask = (depth > 0)
        camera_poses = np.load(os.path.join(dataset_root,
                                            'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(os.path.join(dataset_root,
                                         'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        # ---------------- Apply point corruptions in depth domain (from script1) ----------------
        depth_used = depth.copy()
        dropout_mask = None
        noisy_cloud = None

        # (A) smoothing
        if cfgs.smooth_size is not None and int(cfgs.smooth_size) > 1:
            depth_used = apply_smoothing(depth_used, size=int(cfgs.smooth_size))
            noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)

        # (B) gaussian noise (meters)
        if cfgs.gaussian_noise_level is not None and float(cfgs.gaussian_noise_level) > 0:
            depth_noisy = add_gaussian_noise_depth_map(
                depth_used.astype(np.float32),
                scale=factor_depth,
                level=float(cfgs.gaussian_noise_level),
                valid_min_depth=0.1
            )
            depth_used = np.clip(depth_noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)

        # (C) depth-guided dropout
        if cfgs.dropout_rate is not None and float(cfgs.dropout_rate) > 0:
            foreground_mask = (seg > 0)

            if depth_raw_path is not None and os.path.exists(depth_raw_path):
                real_depth = np.array(Image.open(depth_raw_path))
            else:
                # if not in noise mode, fall back to current depth map
                real_depth = depth

            large_missing_regions, labeled, filtered_labels = find_large_missing_regions(
                real_depth, foreground_mask, min_size=int(cfgs.dropout_min_size)
            )
            dropout_regions = apply_dropout_to_regions(
                large_missing_regions, labeled, filtered_labels, float(cfgs.dropout_rate)
            )
            dropout_mask = (dropout_regions > 0)

        if dropout_mask is not None:
            mask = mask & (~dropout_mask)

        # use noisy_cloud if generated, otherwise original cloud
        if noisy_cloud is not None:
            cloud_masked = noisy_cloud[mask]
            scene_cloud_for_collision = noisy_cloud.reshape(-1, 3)
        else:
            cloud_masked = cloud[mask]
            scene_cloud_for_collision = cloud.reshape(-1, 3)

        color_masked = color[mask]

        # sampling (use same helper as script1)
        idxs = sample_points(len(cloud_masked), cfgs.num_point)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        cloud_tensor = torch.tensor(cloud_sampled, dtype=torch.float32, device=device)
        color_tensor = torch.tensor(color_sampled, dtype=torch.float32, device=device)
        coors_tensor = torch.tensor(cloud_sampled / cfgs.voxel_size, dtype=torch.int32, device=device)
        feats_tensor = torch.ones_like(cloud_tensor).float().to(device)

        coordinates_batch, features_batch = ME.utils.sparse_collate([coors_tensor], [feats_tensor],
                                                                    dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

        batch_data_label = {"point_clouds": cloud_tensor.unsqueeze(0),
                            "cloud_colors": color_tensor.unsqueeze(0),
                            "coors": coordinates_batch,
                            "feats": features_batch,
                            "quantize2original": quantize2original}

        with torch.no_grad():
            end_points = net(batch_data_label)
            grasp_preds = pred_decode(end_points)
            preds = grasp_preds[0]
            gg = GraspGroup(preds)

        # collision detection (use same cloud domain as input)
        if cfgs.collision_thresh > 0:
            mfcdetector = ModelFreeCollisionDetectorTorch(scene_cloud_for_collision, voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # save grasps
        save_dir = os.path.join(dump_dir, 'scene_%04d' % scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '%04d' % anno_idx + '.npy')
        gg.save_npy(save_path)
        print('Saving {}, {}'.format(scene_idx, anno_idx))


scene_list = []
if split == 'test':
    for i in range(100, 190):
        scene_list.append(i)
elif split == 'test_seen':
    for i in range(100, 130):
        scene_list.append(i)
elif split == 'test_similar':
    for i in range(130, 160):
        scene_list.append(i)
elif split == 'test_novel':
    for i in range(160, 190):
        scene_list.append(i)
else:
    print('invalid split')

# scene_list = [100]
# res = []
for scene_idx in scene_list:
    inference(scene_idx)
    # res.append(results)
