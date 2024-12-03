import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import resource
# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit: ", soft_limit, "hard limit: ", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

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

from graspnetAPI import GraspGroup

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_point_cloud
from torchvision import transforms

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
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--img_feat_dim', default=256, type=int, help='Image feature dim')
parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Where dataset is')
parser.add_argument('--network_ver', type=str, default='v0.8.0', help='Network version')
parser.add_argument('--dump_dir', type=str, default='ignet_v0.8.0', help='Dump dir to save outputs')
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
parser.add_argument('--inst_denoise', action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--noise_level', type=float, default=0.0, help='Collision Threshold in collision detection [default: 0.01]')
cfgs = parser.parse_args()

print(cfgs)
minimum_num_pt = 50
img_width = 720
img_length = 1280

resize_shape = (224, 224)
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
        
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280, 1320]
def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_resized_idxs(idxs, orig_shape, resize_shape):
    orig_width, orig_length = orig_shape
    scale_x = resize_shape[1] / orig_length
    scale_y = resize_shape[0] / orig_width
    coords = np.unravel_index(idxs, (orig_width, orig_length))
    new_coords_y = np.clip((coords[0] * scale_y).astype(int), 0, resize_shape[0]-1)
    new_coords_x = np.clip((coords[1] * scale_x).astype(int), 0, resize_shape[1]-1)
    new_idxs = np.ravel_multi_index((new_coords_y, new_coords_x), resize_shape)
    return new_idxs

inst_denoise = cfgs.inst_denoise
    
num_pt = cfgs.inst_pt_num
denoise_pre_sample_num = int(num_pt * 1.5)

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

eps = 1e-8

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# from scipy.ndimage import uniform_filter
# filter_size = 5
# def apply_smoothing(depth_map, size=3):
#     # smoothed_depth = uniform_filter(depth_map, size=size)
#     # smoothed_depth = cv2.GaussianBlur(depth_map, (size, size), 0)
#     # smoothed_depth = cv2.medianBlur(depth_map, size)
#     smoothed_depth = cv2.blur(depth_map, (size, size))
#     return smoothed_depth

def random_point_dropout(point_cloud, min_num=50, num_points_to_drop=3, radius_percent=0.01):
    """
    Randomly selects a few center points in the point cloud and removes all points 
    within a spherical region centered on each selected point.

    Parameters:
    - point_cloud: numpy array of shape (N, 3), representing the point cloud data.
    - min_num: minimum acceptable number of points to retain.
    - num_points_to_drop: number of random center points to select.
    - radius_percent: percentage of the objects size to determine the radius of the spherical region (relative to the bounding box diagonal).

    Returns:
    - retained_point_cloud: the point cloud data with retained points.
    - retained_indices: indices of the retained points in the original point cloud.
    """
    num_points = point_cloud.shape[0]

    # Calculate object size using the bounding box diagonal length
    min_coords = np.min(point_cloud, axis=0)
    max_coords = np.max(point_cloud, axis=0)
    bbox_diagonal = np.linalg.norm(max_coords - min_coords)
    
    # Compute the radius based on the given percentage
    radius = radius_percent * bbox_diagonal

    # Initialize a mask to keep all points initially
    mask = np.ones(num_points, dtype=bool)

    # Randomly select `num_points_to_drop` points as the center points
    center_indices = np.random.choice(num_points, num_points_to_drop, replace=False)

    # For each selected center point, remove points within the radius
    for center_idx in center_indices:
        center = point_cloud[center_idx]

        # Calculate the distance from each point to the center
        distances = np.linalg.norm(point_cloud - center, axis=1)

        # Update the mask to set points within the radius to False
        mask &= distances > radius

    # Use the mask to get retained points and their indices
    retained_point_cloud = point_cloud[mask]
    
    # Ensure the retained points meet the minimum number requirement
    if len(retained_point_cloud) < min_num:
        return point_cloud, np.arange(num_points)
    
    retained_indices = np.where(mask)[0]  # Indices of retained points
    return retained_point_cloud, retained_indices


scene_idx = 0
# elapsed_time_list = []
for anno_idx in range(256):
    rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
    depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

    depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
    mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
        
    meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
    
    color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(mask_path))
    # normal = np.load(normal_path)['normals']

    meta = scio.loadmat(meta_path)

    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    intrinsics = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

    # depth = apply_smoothing(depth.astype(np.uint16), size=filter_size)

    cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
    scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    o3d.visualization.draw_geometries([scene])
    
    depth_mask = (depth > 0)
    camera_poses = np.load(
        os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
    align_mat = np.load(
        os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
    trans = np.dot(align_mat, camera_poses[anno_idx])
    workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    mask = (depth_mask & workspace_mask)

    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # normal_masked = normal

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(cloud_masked)
    scene.colors = o3d.utility.Vector3dVector(color_masked)
    # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
    # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    # normal_masked = np.asarray(scene.normals)
