import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

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

from graspnetAPI import GraspGroup
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_point_cloud, apply_smoothing, random_point_dropout, transform_point_cloud, add_gaussian_noise_depth_map

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

# from scipy.ndimage import uniform_filter
# filter_size = 5
# def apply_smoothing(depth_map, size=3):
#     # smoothed_depth = uniform_filter(depth_map, size=size)
#     # smoothed_depth = cv2.GaussianBlur(depth_map, (size, size), 0)
#     # smoothed_depth = cv2.medianBlur(depth_map, size)
#     smoothed_depth = cv2.blur(depth_map, (size, size))
#     return smoothed_depth

# scene_idx = 101
# color_list = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
# elapsed_time_list = []
# for anno_order_idx, anno_idx in enumerate([16, 96, 176]):

def distance_compute(scene_idx, cfgs):
    result = np.zeros((256, 2))
    dataset_root = cfgs.dataset_root
    camera = cfgs.camera_type
    for anno_idx in range(256):
        rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
        real_depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

        noisy_depth_path = os.path.join('/media/user/data1/rcao/DREDS-CatNovel', '{:05d}'.format(scene_idx), '{:04d}_depth.png'.format(anno_idx))
        
        clear_depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
        mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
        
        meta_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))
        
        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        noisy_depth = np.array(Image.open(noisy_depth_path))
        clear_depth = np.array(Image.open(clear_depth_path))
        real_depth = np.array(Image.open(real_depth_path))
        seg = np.array(Image.open(mask_path))
        # normal = np.load(normal_path)['normals']
        
        noisy_depth = cv2.resize(noisy_depth, (clear_depth.shape[1], clear_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

        noisy_cloud = create_point_cloud_from_depth_image(noisy_depth, camera_info, organized=True)
        clear_cloud = create_point_cloud_from_depth_image(clear_depth, camera_info, organized=True)   
        real_cloud = create_point_cloud_from_depth_image(real_depth, camera_info, organized=True)
        
        # clear_scene = o3d.geometry.PointCloud()
        # clear_scene.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
        # clear_scene.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
        # clear_scene = clear_scene.voxel_down_sample(voxel_size=0.005)
        # o3d.visualization.draw_geometries([scene])
    
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(clear_cloud, seg, trans=trans, organized=True, outlier=0.02)

        depth_mask = (noisy_depth > 0) & (noisy_depth < 1000)
        noisy_mask = (depth_mask & workspace_mask)
        noisy_cloud_masked = noisy_cloud[noisy_mask]
        
        real_mask = ((real_depth > 0) & workspace_mask)
        real_cloud_masked = real_cloud[real_mask]
        
        color_masked = color[real_mask]
        # normal_masked = normal

        noise_cloud = o3d.geometry.PointCloud()
        noise_cloud.points = o3d.utility.Vector3dVector(noisy_cloud_masked)
        noise_cloud = noise_cloud.voxel_down_sample(voxel_size=cfgs.voxel_size)
        
        real_cloud = o3d.geometry.PointCloud()
        real_cloud.points = o3d.utility.Vector3dVector(real_cloud_masked)
        real_cloud.colors = o3d.utility.Vector3dVector(color_masked)
        real_cloud = real_cloud.voxel_down_sample(voxel_size=cfgs.voxel_size)
        
        noise_cloud_tensor = torch.tensor(np.asarray(noise_cloud.points), dtype=torch.float32, device=device)
        real_cloud_tensor = torch.tensor(np.asarray(real_cloud.points), dtype=torch.float32, device=device)

        noise_dis, _ = chamfer_distance(noise_cloud_tensor.unsqueeze(0), real_cloud_tensor.unsqueeze(0))
        
        result[anno_idx, 1] = noise_dis.item()
        print(scene_idx, anno_idx,  noise_dis.item())
        # if noise_dis.item() < clear_dis.item():
        #     print("Noise is better!")

        # o3d.visualization.draw_geometries([noise_cloud, real_cloud])
        
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
    parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Where dataset is')
    parser.add_argument('--voxel_size', type=float, default=0.001, help='Voxel Size to quantize point cloud [default: 0.005]')
    parser.add_argument('--gaussian_noise_level', type=float, default=0.005, help='Collision Threshold in collision detection [default: 0.0]')
    parser.add_argument('--smooth_size', type=int, default=9, help='Blur size used for depth smoothing [default: 1]')
    parser.add_argument('--dropout_num', type=int, default=0, help=' [default: 0]')
    cfgs = parser.parse_args()

    scene_list = [100, 101, 102, 143, 144, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187, 188]
    # scene_list = list(range(100, 130))
    # scene_list = list(range(130, 160))
    # scene_list = list(range(160, 190))
    result_list = parallel(scene_list, cfgs=cfgs, proc=20)
    results = [result.get() for result in result_list]
    results = np.stack(results, axis=0)
    np.save('dreds_depth_distance.npy', results)