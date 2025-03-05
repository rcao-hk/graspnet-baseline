import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2
import argparse
import numpy as np
import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
import multiprocessing
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_point_cloud, random_point_dropout, transform_point_cloud, add_gaussian_noise_depth_map
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

img_width = 720
img_length = 1280

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


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

from kornia.filters import median_blur, bilateral_blur


def apply_smoothing(depth_map, size=3):
    # smoothed_depth = cv2.Blur(depth_map.astype(np.int16), (size, size))
    smoothed_depth = cv2.medianBlur(depth_map.astype(np.int16), size)
    return smoothed_depth

img_width = 720
img_length = 1280
intrinsics = np.array([[927.17, 0., 651.32],
                       [  0., 927.37, 349.62],
                       [  0., 0., 1.  ]])
camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], 1000)
print_interval = 10
# scene_idx = 101
# color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# elapsed_time_list = []
# for anno_order_idx, anno_idx in enumerate([16, 96, 176]):

def distance_compute(scene_idx, cfgs):
    result = np.zeros((256, 1))
    dataset_root = cfgs.dataset_root
    camera = cfgs.camera_type
    gaussian_level = torch.tensor(cfgs.gaussian_noise_level, dtype=torch.float32, device=device)

    blur_kernel_size = min(5 + cfgs.smooth_noise_level // 5, 25)  # Linearly increase kernel size
    blur_kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1  # Ensure odd kernel size
    sigma_color = torch.tensor(cfgs.smooth_noise_level * 0.1, dtype=torch.float32, device=device)
    sigma_space = torch.tensor(cfgs.smooth_noise_level * 0.5, dtype=torch.float32, device=device)
    
    factor_depth = torch.tensor([1000], dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for anno_idx in range(256):
            rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            real_depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

            clear_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
            
            meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
            
            # match_depth_path = os.path.join('/data/jhpan/dataset/graspnet_sim/rendered_output_raw', '{:05d}/{:04d}_depth_sim.png'.format(scene_idx, anno_idx))
            
            # color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
            clear_depth = np.array(Image.open(clear_depth_path))
            real_depth = np.array(Image.open(real_depth_path))
            # match_depth = np.array(Image.open(match_depth_path))
            seg = np.array(Image.open(mask_path))
            # normal = np.load(normal_path)['normals']

            depth_mask = (real_depth > 0)
            seg_mask = (seg > 0)
            mask = depth_mask & seg_mask

            clear_depth = torch.tensor(np.array([clear_depth]), dtype=torch.float32, device=device)
            real_depth = torch.tensor(np.array([real_depth]), dtype=torch.float32, device=device)
            # match_depth = torch.tensor(np.array([match_depth]), dtype=torch.float32, device=device)
            depth_mask = torch.tensor(np.array([mask]), dtype=torch.bool, device=device)

            # D = torch.amax(clear_depth, dim=(1, 2), keepdim=False) - torch.amin(clear_depth, dim=(1, 2), keepdim=False)
            # # noise_depth = apply_smoothing(clear_depth, size=cfgs.smooth_size)
            # # clear_depth = median_blur(clear_depth, (5, 5))
            # noisy_depth = bilateral_blur(clear_depth.unsqueeze(1), (blur_kernel_size, blur_kernel_size), D * sigma_color, (sigma_space, sigma_space))
            
            # if cfgs.smooth_noise_level > 0:
                # noisy_depth = median_blur(clear_depth.unsqueeze(1), (5, 5))
                # noisy_depth = BilateralFilter(noisy_depth, blur_kernel_size, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                # noisy_depth = noisy_depth.squeeze(1)
            if cfgs.smooth_noise_level > 1:
                noisy_depth = median_blur(clear_depth.unsqueeze(1), (cfgs.smooth_noise_level, cfgs.smooth_noise_level))
            #     noise_depth = torch.tensor(np.array([noise_depth]), dtype=torch.float32, device=device)
                noisy_depth = noisy_depth.squeeze(1)
                # noisy_depth = apply_smoothing(clear_depth, size=cfgs.smooth_noise_level)
            else:
                noisy_depth = clear_depth
            
            # noisy_depth = torch.tensor(np.array([noisy_depth]), dtype=torch.float32, device=device)
            if cfgs.gaussian_noise_level > 0:
                noisy_depth = add_gaussian_noise(noisy_depth, factor_depth, level=gaussian_level)
                
            # clear_cloud = create_point_cloud_from_depth_image_torch(clear_depth, factor_depth, intrinsics,
                                                                    # organized=True)
            noisy_cloud = create_point_cloud_from_depth_image_torch(noisy_depth, factor_depth, intrinsics,
                                                                    organized=True)
            real_cloud = create_point_cloud_from_depth_image_torch(real_depth, factor_depth, intrinsics, 
                                                                    organized=True)
            # match_cloud = create_point_cloud_from_depth_image_torch(match_depth, factor_depth, intrinsics,
                                                                    # organized=True)
            if cfgs.dropout_noise_level > 0:
                clear_depth = np.array(Image.open(clear_depth_path))
                real_depth = np.array(Image.open(real_depth_path))
                real_depth_mask = (real_depth > 0) & (seg > 0)
                clear_depth_mask = (clear_depth > 0) & (seg > 0)
                real_cloud = create_point_cloud_from_depth_image(real_depth, camera_info, organized=True)
                clear_cloud = create_point_cloud_from_depth_image(clear_depth, camera_info, organized=True)
                cloud_masked = clear_cloud[clear_depth_mask]
                seg_masked = seg[clear_depth_mask]
                real_cloud = real_cloud[real_depth_mask]
                
                meta = scio.loadmat(meta_path)
                obj_idxs = meta['cls_indexes'].flatten()
                inst_cloud = []
                for obj_idx in obj_idxs:
                    if (seg_masked == obj_idx).sum() < 50:
                        continue
                    inst_mask = (seg_masked == obj_idx)
                    inst_cloud_select, select_idx, dropout_idx = random_point_dropout(cloud_masked[inst_mask], min_num=50, num_points_to_drop=cfgs.dropout_noise_level, radius_percent=0.1)
                    inst_cloud.append(inst_cloud_select)
                noisy_cloud = np.vstack(inst_cloud)
                noisy_idxs = sample_points(len(noisy_cloud), cfgs.match_num)
                noisy_cloud = noisy_cloud[noisy_idxs]
                real_idxs = sample_points(len(real_cloud), cfgs.match_num)
                real_cloud = real_cloud[real_idxs]
                noisy_cloud = torch.tensor(noisy_cloud, dtype=torch.float32, device=device)
                real_cloud = torch.tensor(real_cloud, dtype=torch.float32, device=device)
            else:
                depth_mask = select_fixed_points_and_generate_mask(depth_mask, cfgs.match_num)
                
                # clear_cloud = clear_cloud[depth_mask]
                noisy_cloud = noisy_cloud[depth_mask]
                real_cloud = real_cloud[depth_mask]
                # match_cloud = match_cloud[depth_mask]
                
            # clear_cloud = clear_cloud.view(1, cfgs.match_num, 3)
            noisy_cloud = noisy_cloud.view(1, cfgs.match_num, 3)
            real_cloud = real_cloud.view(1, cfgs.match_num, 3)
            # match_cloud = match_cloud.view(1, cfgs.match_num, 3)
            
            noise_dis, _ = chamfer_distance(noisy_cloud, real_cloud)
                        
            # result[anno_idx, 0] = clear_dis.item()
            # result[anno_idx, 1] = noise_dis.item()
            # print(scene_idx, anno_idx, clear_dis.item(), noise_dis.item())
            
            result[anno_idx, 0] = noise_dis.item()
            if anno_idx % print_interval == 0:
                print(scene_idx, anno_idx, noise_dis.item())
            # if noise_dis.item() < clear_dis.item():
            #     print("Noise is better!")
            
            # noise_scene = o3d.geometry.PointCloud()
            # noise_scene.points = o3d.utility.Vector3dVector(cloud_masked)
            # # noise_scene.colors = o3d.utility.Vector3dVector(color_masked)
            # noise_scene.paint_uniform_color([1.0, 0.0, 0.0])
            # noise_scene = noise_scene.voxel_down_sample(voxel_size=0.005)
            # o3d.visualization.draw_geometries([clear_scene, noise_scene])
            
            # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
            # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
            # normal_masked = np.asarray(scene.normals)
    
    
    return result
        

def parallel(scene_ids, cfgs, proc = 2):
    # from multiprocessing import Pool
    ctx_in_main = multiprocessing.get_context('forkserver')
    p = ctx_in_main.Pool(processes = proc)
    result_list = []
    for scene_id in scene_ids:
        scene_result = p.apply_async(distance_compute, (scene_id, cfgs))
        result_list.append(scene_result)
    p.close()
    p.join()
    return result_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_type', default='realsense', help='Camera to use [kinect | realsense]')
    parser.add_argument('--dataset_root', default='/data/jhpan/dataset/graspnet', help='Where dataset is')
    parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel Size to quantize point cloud [default: 0.005]')
    parser.add_argument('--proc_num', type=int, default=10, help='Number of processes [default: 10]')
    parser.add_argument('--gaussian_noise_level', type=float, default=0.0, help='Collision Threshold in collision detection [default: 0.0]')
    # parser.add_argument('--blur_kernel_size', type=int, default=7, help='Kernel size for bilateral filter [default: 7]')
    # parser.add_argument('--blur_sigma_space', type=float, default=5.0, help='Sigma space for bilateral filter [default: 5.0]')
    # parser.add_argument('--blur_sigma_color', type=float, default=0.8, help='Sigma color for bilateral filter [default: 0.05]')
    parser.add_argument('--smooth_noise_level', type=int, default=1, help='Noise level for smoothing [default: 1]')
    parser.add_argument('--dropout_noise_level', type=int, default=0, help=' [default: 0]')
    parser.add_argument('--match_num', type=int, default=20000, help='Number of points to match [default: 20000]')

    cfgs = parser.parse_args()

    print(cfgs)
    # scene_list = list(range(100, 190))
    # scene_list = [100, 101, 102, 143, 144, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188]
    scene_list = list(range(100, 190))
    # scene_list = list(range(130, 160))
    # scene_list = list(range(160, 190))
    result_list = parallel(scene_list, cfgs=cfgs, proc=cfgs.proc_num)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    
    save_root = 'depth_distance'
    os.makedirs(save_root, exist_ok=True)
    np.save(os.path.join(save_root, 'g{}s{}d{}_depth_distance.npy'.format(cfgs.gaussian_noise_level, cfgs.smooth_noise_level, cfgs.dropout_noise_level)), results)
    # np.save(os.path.join(save_root, 'clear_depth_distance.npy'.format(cfgs.gaussian_noise_level, cfgs.blur_sigma_space, cfgs.blur_sigma_color)), results)
    # np.save(os.path.join(save_root, 'match_depth_distance.npy'.format(cfgs.gaussian_noise_level, cfgs.blur_sigma_space, cfgs.blur_sigma_color)), results)