import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import cv2
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME
import multiprocessing
from kornia.filters import median_blur, bilateral_blur
import torch.nn.functional as F
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, random_point_dropout, transform_point_cloud
# from data_utils import apply_smoothing
from pytorch3d.loss import chamfer_distance

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# def add_smoothing(depth_map, size=3):
#     # smoothed_depth = cv2.blur(depth_map.astype(np.uint16), (size, size))
#     smoothed_depth = cv2.bilateralFilter(depth_map.astype(np.float32), size, 50, 10)
#     return smoothed_depth


def add_gaussian_noise(depth_map, depth_scale, level=0.005):
    """
    Adds Gaussian noise to a depth map in a differentiable way using PyTorch.

    Args:
        depth_map (torch.Tensor): Tensor of shape (H, W), representing the depth map.
        depth_mask (torch.Tensor, optional): Boolean tensor of shape (H, W), 
            where True indicates valid pixels. If None, pixels with depth > 0 are valid.
        level (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Depth map with added Gaussian noise.
    """
    # Ensure the input is a float tensor
    depth_map = depth_map.float() / depth_scale
    # level = torch.clamp(level, min=0.0)

    # Create a mask for valid depth pixels
    # if depth_mask is None:
    #     depth_mask = (depth_map > 0).float()  # Convert to float for differentiability

    # Generate Gaussian noise
    noise = torch.randn_like(depth_map) * level

    # Add noise only to valid pixels
    # noisy_depth_map = depth_map + noise * depth_mask
    noisy_depth_map = depth_map + noise
    noisy_depth_map = noisy_depth_map * depth_scale
    
    return noisy_depth_map


def GetGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (torch.arange(ksize, dtype=torch.double, device=sigma.device) - center)
    kernel1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = kernel / kernel.sum()
    return kernel


def BilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

    # patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim()  # 6
    
    # calculate normalized weight matrix
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

    # obtain the gaussian kernel
    weights_space = GetGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

    # caluculate the final weight
    weights = weights_space * weights_color
    weights_sum = weights.sum(dim=(-1, -2))
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


def create_point_cloud_from_depth_image_torch(depth, depth_scale, intrinics, organized=True):
    """
    Generate point cloud from a batch of depth images in a differentiable way using PyTorch.

    Args:
        depth (torch.Tensor): Depth image tensor of shape (B, H, W), where B is the batch size.
        camera (CameraInfo): Camera intrinsics containing fx, fy, cx, cy, and scale.
        organized (bool): If True, the output cloud shape will be (B, H, W, 3);
                          otherwise, it will be reshaped to (B, H*W, 3).

    Returns:
        torch.Tensor: Point cloud tensor of shape (B, H, W, 3) if organized=True, 
                      otherwise (B, H*W, 3).
    """
    # Ensure depth tensor is float
    depth = depth.float()  # Shape: (B, H, W)

    # Get depth dimensions
    B, H, W = depth.shape

    # Create coordinate grids
    xmap = torch.arange(W, device=depth.device).float().unsqueeze(0).repeat(H, 1)  # (H, W)
    ymap = torch.arange(H, device=depth.device).float().unsqueeze(1).repeat(1, W)  # (H, W)
    xmap = xmap.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
    ymap = ymap.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)

    # Compute z-coordinates (depth values)
    points_z = depth / depth_scale # Shape: (B, H, W)

    # Compute x and y coordinates
    points_x = (xmap - intrinics[0][2]) * points_z / intrinics[0][0]  # Shape: (B, H, W)
    points_y = (ymap - intrinics[1][2]) * points_z / intrinics[1][1] # Shape: (B, H, W)

    # Stack the coordinates to create the point cloud
    cloud = torch.stack([points_x, points_y, points_z], dim=-1)  # Shape: (B, H, W, 3)

    # Reshape to (B, H*W, 3) if not organized
    if not organized:
        cloud = cloud.view(B, -1, 3)

    return cloud


def select_fixed_points_and_generate_mask(depth_mask, point_num):
    """
    Select a fixed number of points (e.g., 20000) from each mask in depth_mask and 
    generate a new mask with only those points set to True.

    Args:
        depth_mask (torch.Tensor): Boolean mask tensor of shape (batch_size, 720, 1280).
        point_num (int): Number of points to select from each mask.

    Returns:
        torch.Tensor: New mask tensor of shape (batch_size, 720, 1280).
    """
    batch_size, h, w = depth_mask.shape
    depth_mask_flat = depth_mask.view(batch_size, -1)  # Shape: (batch_size, 720*1280)

    # Create an output mask initialized with False
    new_mask_flat = torch.zeros_like(depth_mask_flat, dtype=torch.bool)  # Shape: (batch_size, 720*1280)

    for batch_idx in range(batch_size):
        # Get valid indices for the current mask
        valid_indices = depth_mask_flat[batch_idx].nonzero(as_tuple=True)[0]  # Shape: (num_valid_points,)
        num_valid_points = valid_indices.size(0)

        # Enough valid points, randomly select point_num points
        selected_indices = torch.randperm(num_valid_points, device=depth_mask.device)[:point_num]

        # Mark the selected indices as True in the new mask
        new_mask_flat[batch_idx, valid_indices[selected_indices]] = True

    # Reshape the new mask back to (batch_size, 720, 1280)
    new_mask = new_mask_flat.view(batch_size, h, w)

    return new_mask


# scene_idx = 101
# color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# elapsed_time_list = []
# for anno_order_idx, anno_idx in enumerate([16, 96, 176]):

# sigmaColor = 0.05
batch_size = 32
iter_total_num = (256 * 90) // batch_size
# sigmaSpace = torch.tensor([5], dtype=torch.float32, device=device)
blur_kernel_size = 5
loss = torch.nn.L1Loss()
# gaussian_level = nn.Parameter(torch.tensor(0.001, dtype=torch.float32, device=device))
gaussian_level = torch.tensor(0.001, dtype=torch.float32, device=device)
# sigma_color = nn.Parameter(torch.tensor(5, dtype=torch.float32, device=device))
sigma_color = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=device))
sigma_space = nn.Parameter(torch.tensor(5.0, dtype=torch.float32, device=device))
# sigma_space = torch.tensor(5.0, dtype=torch.float32, device=device)
optimizer = torch.optim.AdamW([sigma_space, sigma_color], lr=1e-2)
factor_depth = torch.tensor([1000], dtype=torch.float32, device=device)

img_width = 720
img_length = 1280
intrinsics = np.array([[927.17, 0., 651.32],
                       [  0., 927.37, 349.62],
                       [  0., 0., 1.  ]])
camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 1000)
            
def distance_compute(cfgs):
    dataset_root = cfgs.dataset_root
    camera = cfgs.camera_type
    scene_count = 0
    iter_count = 0
    
    # 生成 scene_idx 和 anno_idx 组合的列表
    scene_idx_range = range(100, 190)
    anno_idx_range = range(256)

    # 使用列表推导式生成所有组合
    scene_anno_list = [(scene_idx, anno_idx) for scene_idx in scene_idx_range for anno_idx in anno_idx_range]

    # 打乱列表
    np.random.shuffle(scene_anno_list)
    for scene_idx, anno_idx in scene_anno_list:
        clear_depth_list = []
        real_depth_list = []
        depth_mask_list = []
        
        real_depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

        clear_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
        mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
                
        # color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        clear_depth = np.array(Image.open(clear_depth_path))
        real_depth = np.array(Image.open(real_depth_path))
        seg = np.array(Image.open(mask_path))
        # normal = np.load(normal_path)['normals']

        # cloud = create_point_cloud_from_depth_image(clear_depth, camera_info, organized=True)
        # depth_mask = (real_depth > 0)
        # camera_poses = np.load(
        #     os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        # align_mat = np.load(
        #     os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        # trans = np.dot(align_mat, camera_poses[anno_idx])
        # workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        # mask = (depth_mask & workspace_mask)
        
        depth_mask = (real_depth > 0)
        seg_mask = (seg > 0)
        mask = depth_mask & seg_mask
    
        clear_depth_list.append(clear_depth)
        real_depth_list.append(real_depth)
        depth_mask_list.append(mask)

        scene_count += 1

        if scene_count % batch_size == 0:
        
            optimizer.zero_grad()
            clear_depth = torch.tensor(np.array(clear_depth_list), dtype=torch.float32, device=device)
            real_depth = torch.tensor(np.array(real_depth_list), dtype=torch.float32, device=device)
            depth_mask = torch.tensor(np.array(depth_mask_list), dtype=torch.bool, device=device)
            
            D = torch.amax(clear_depth, dim=(1, 2), keepdim=False) - torch.amin(clear_depth, dim=(1, 2), keepdim=False)
            # noise_depth = apply_smoothing(clear_depth, size=cfgs.smooth_size)
            # clear_depth = median_blur(clear_depth, (5, 5))
            # blur_kernel_size = torch.tensor(3, dtype=torch.int32, device=device)
            noisy_depth = bilateral_blur(clear_depth.unsqueeze(1), (blur_kernel_size, blur_kernel_size), D * sigma_color, (sigma_space, sigma_space))
            
            # noisy_depth = median_blur(clear_depth.unsqueeze(1), (5, 5))
            # noisy_depth = BilateralFilter(noisy_depth, blur_kernel_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            
            # gaussian_level = torch.clamp(gaussian_level, 0.0)
            noisy_depth = noisy_depth.squeeze(1)
            noisy_depth = add_gaussian_noise(noisy_depth, factor_depth, level=gaussian_level)
            
            noisy_cloud = create_point_cloud_from_depth_image_torch(noisy_depth, factor_depth, intrinsics,
                                                                    organized=True)
            real_cloud = create_point_cloud_from_depth_image_torch(real_depth, factor_depth, intrinsics, 
                                                                    organized=True)
            
            depth_mask = select_fixed_points_and_generate_mask(depth_mask, 20000)
            noisy_cloud = noisy_cloud[depth_mask]
            real_cloud = real_cloud[depth_mask]

            noisy_cloud = noisy_cloud.view(batch_size, -1, 3)
            real_cloud = real_cloud.view(batch_size, -1, 3)
            
            point_loss, _ = chamfer_distance(noisy_cloud, real_cloud)
            # depth_loss = loss(noisy_depth[real_mask], real_depth[real_mask])

            point_loss.backward()
            optimizer.step()
            
            # print("Step: {}/{}, Loss: {}, Guassian level: {}, Sigma Color: {}".format(iter_count, iter_total_num,point_loss.item(), torch.clamp(gaussian_level, min=0).item(), sigma_color.item()))
            # print("Step: {}/{}, Loss: {}, Guassian level: {}, Sigma Color: {}, Sigma Space: {}".format(iter_count, iter_total_num, point_loss.item(), gaussian_level.item(), sigma_color.item(), sigma_space.item()))
            
            print("Step: {}/{}, Loss: {}, Sigma Color: {}, Sigma Space: {}".format(iter_count, iter_total_num, point_loss.item(), sigma_color.item(), sigma_space.item()))
            
            clear_depth_list = []
            real_depth_list  = []
            depth_mask_list  = []
            iter_count += 1

    # def parallel(scene_ids, cfgs, proc = 2):
    #     # from multiprocessing import Pool
    #     ctx_in_main = multiprocessing.get_context('forkserver')
    #     p = ctx_in_main.Pool(processes = proc)
    #     result_list = []
    #     for scene_id in scene_ids:
    #         scene_result = p.apply_async(distance_compute, (scene_id, cfgs))
    #         result_list.append(scene_result)
    #     p.close()
    #     p.join()
    #     return result_list


parser = argparse.ArgumentParser()
parser.add_argument('--camera_type', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--dataset_root', default='/data/jhpan/dataset/graspnet', help='Where dataset is')
parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel Size to quantize point cloud [default: 0.005]')
# parser.add_argument('--gaussian_noise_level', type=float, default=0.005, help='Collision Threshold in collision detection [default: 0.0]')
# parser.add_argument('--smooth_size', type=int, default=15, help='Blur size used for depth smoothing [default: 1]')
# parser.add_argument('--dropout_num', type=int, default=0, help=' [default: 0]')
cfgs = parser.parse_args()

distance_compute(cfgs)
# scene_list = list(range(100, 190))
# scene_list = [100, 101, 102, 143, 144, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188]
# scene_list = list(range(100, 130))
# scene_list = list(range(130, 160))
# scene_list = list(range(160, 190))
# result_list = parallel(scene_list, cfgs=cfgs, proc=10)

# np.save('g0.005_bs15_depth_distance.npy', results)