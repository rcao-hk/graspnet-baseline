import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import sys

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

# from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_point_cloud
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
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
parser.add_argument('--inst_denoise', action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel Size to quantize point cloud [default: 0.005]')
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
def apply_smoothing(depth_map, size=3):
    # smoothed_depth = uniform_filter(depth_map, size=size)
    # smoothed_depth = cv2.GaussianBlur(depth_map, (size, size), 0)
    # smoothed_depth = cv2.medianBlur(depth_map, size)
    smoothed_depth = cv2.blur(depth_map, (size, size))
    return smoothed_depth

from scipy import ndimage
def find_large_missing_regions(depth, foreground_mask, min_size=50):
    """
    通过连通组件标记找到成块缺失的部分，滤除小的缺失区域，并仅考虑foreground mask上的缺失。

    输入:
    - depth: (H, W) 形状的 numpy 数组，表示深度图。
    - foreground_mask: (H, W) 形状的 numpy 数组，表示前景 mask（1 表示前景，0 表示背景）。
    - min_size: 连通区域的最小大小，滤除小于该大小的缺失区域。

    输出:
    - large_missing_regions: (H, W) 形状的 numpy 数组，标记出大的缺失区域。
    """
    
    # 1. 仅考虑前景区域的缺失点
    depth_mask = (depth == 0)  # 假设缺失点的深度值为 0
    valid_mask = depth_mask & foreground_mask  # 在前景区域内的缺失点

    # 2. 找到连通区域
    labeled, num_labels = ndimage.label(valid_mask)
    
    # 3. 获取各个区域的大小
    region_sizes = np.bincount(labeled.ravel())
    
    # 4. 创建一个新的 mask，标记大的缺失区域
    large_missing_regions = np.zeros_like(depth, dtype=np.int32)
    
    filtered_labels = []
    for label in range(1, num_labels + 1):
        if region_sizes[label] >= min_size:  # 如果区域的大小大于 min_size
            large_missing_regions[labeled == label] = label
            filtered_labels.append(label)
            
    return large_missing_regions, labeled, filtered_labels

def apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, dropout_rate):
    """
    根据 dropout_rate 随机选择部分区域，生成 dropout mask。

    输入:
    - large_missing_regions: (H, W) 形状的 numpy 数组，表示大缺失区域的标记。
    - labeled: (H, W) 形状的 numpy 数组，表示每个连通区域的标签。
    - filtered_labels: 连通区域标签的列表。
    - dropout_rate: 需要保留的区域比例。

    输出:
    - dropout_regions: (H, W) 形状的 numpy 数组，标记选择的 dropout 区域。
    """
    # 创建一个新的 dropout mask
    dropout_regions = np.zeros_like(large_missing_regions, dtype=np.int32)
    
    # 根据 dropout_rate 随机选择区域
    num_regions_to_keep = max(0, int(len(filtered_labels) * dropout_rate))  # 计算保留区域的数量
    if num_regions_to_keep == 0:
        return dropout_regions
    
    selected_labels = np.random.choice(filtered_labels, num_regions_to_keep, replace=False)  # 随机选择区域
    for label in selected_labels:
        dropout_regions[labeled == label] = label  # 标记选择的区域为 dropout 区域
    
    return dropout_regions


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


scene_idx = 105  # 105 40
# elapsed_time_list = []
filter_size = 5
anno_list = range(40, 41)
min_size = 200
dropout_rate = 1.0
for anno_idx in anno_list:
    print(anno_idx)
    rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
    real_depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

    clear_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
    mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))

    match_depth_path = os.path.join('/data/jhpan/dataset/graspnet_sim/rendered_output_raw', '{:05d}/{:04d}_depth_sim.png'.format(scene_idx, anno_idx))
        
    meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
    
    color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    real_depth = np.array(Image.open(real_depth_path))
    clear_depth = np.array(Image.open(clear_depth_path))
    # match_depth = np.array(Image.open(match_depth_path))
    seg = np.array(Image.open(mask_path))
    # normal = np.load(normal_path)['normals']

    meta = scio.loadmat(meta_path)

    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    intrinsics = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']
    camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

    smooth_depth = apply_smoothing(clear_depth.astype(np.uint16), size=filter_size)

    real_cloud = create_point_cloud_from_depth_image(real_depth, camera_info, organized=True)
    # match_cloud = create_point_cloud_from_depth_image(match_depth, camera_info, organized=True)
    smooth_cloud = create_point_cloud_from_depth_image(smooth_depth, camera_info, organized=True)
    clear_cloud = create_point_cloud_from_depth_image(clear_depth, camera_info, organized=True)
    
    # foreground_mask = (seg > 0)  # 假设前景 mask（1 表示前景，0 表示背景）
    # large_missing_regions, labeled, filtered_labels = find_large_missing_regions(real_depth, foreground_mask, min_size)

    # # 根据 dropout_rate 随机选择区域
    # dropout_regions = apply_dropout_to_regions(large_missing_regions, labeled, filtered_labels, dropout_rate)
    # dropout_mask = dropout_regions > 0

    # depth_mask = (clear_depth > 0)
    camera_poses = np.load(
        os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
    align_mat = np.load(
        os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
    trans = np.dot(align_mat, camera_poses[anno_idx])
    workspace_mask = get_workspace_mask(clear_cloud, seg, trans=trans, organized=True, outlier=0.02)
    # mask = (depth_mask & workspace_mask)

    real_mask = (real_depth > 0) & workspace_mask
    clear_mask = (clear_depth > 0) & workspace_mask
    # clear_mask = (clear_depth > 0) & workspace_mask & (~dropout_mask)
    
    cloud_masked = real_cloud[real_mask]
    color_masked = color[real_mask]
    clear_cloud_masked = clear_cloud[clear_mask]
    clear_color_masked = color[clear_mask]
    smooth_cloud_masked = smooth_cloud[clear_mask]
    
    smooth_vis_mask = np.abs(smooth_cloud_masked - clear_cloud_masked) > 0.001
    # dropout_cloud = clear_cloud[dropout_mask]
    # dropout_scene = o3d.geometry.PointCloud()
    # dropout_scene.points = o3d.utility.Vector3dVector(dropout_cloud)
    # dropout_scene.paint_uniform_color([1, 0, 0])
    # dropout_scene = dropout_scene.voxel_down_sample(voxel_size=cfgs.voxel_size)
    
    # match_cloud = match_cloud[mask]
    # normal_masked = normal

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(cloud_masked)
    scene.colors = o3d.utility.Vector3dVector(color_masked)
    scene = scene.voxel_down_sample(voxel_size=cfgs.voxel_size)
    
    clear_scene = o3d.geometry.PointCloud()
    clear_scene.points = o3d.utility.Vector3dVector(clear_cloud_masked)
    clear_scene.colors = o3d.utility.Vector3dVector(clear_color_masked)
    clear_scene = clear_scene.voxel_down_sample(voxel_size=cfgs.voxel_size)
    
    smooth_scene = o3d.geometry.PointCloud()
    smooth_scene.points = o3d.utility.Vector3dVector(smooth_cloud_masked)
    smooth_scene.colors = o3d.utility.Vector3dVector(clear_color_masked)
    smooth_scene_vis = smooth_scene.select_by_index(np.where(smooth_vis_mask)[0])
    smooth_scene_vis.paint_uniform_color([0, 0, 1])
    
    smooth_scene = smooth_scene.voxel_down_sample(voxel_size=cfgs.voxel_size)
    smooth_scene_vis = smooth_scene_vis.voxel_down_sample(voxel_size=cfgs.voxel_size)
    
    # match_scene = o3d.geometry.PointCloud()
    # match_scene.points = o3d.utility.Vector3dVector(match_cloud.reshape(-1, 3))
    # match_scene.paint_uniform_color([1, 0, 0])
    # match_scene = match_scene.voxel_down_sample(voxel_size=cfgs.voxel_size)
    
    o3d.visualization.draw_geometries([clear_scene])
    
    # scene_vis = scene + match_scene
    # o3d.io.write_point_cloud('scene_vis.ply', scene_vis)
    
    # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
    # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    # normal_masked = np.asarray(scene.normals)
