""" GraspNet dataset for multi-modal setting.
    Author: Rui Cao
"""

import os
import sys
import numpy as np
import scipy.io as scio
import cv2
import h5py
import open3d as o3d
from PIL import Image

import torch
import collections.abc as container_abcs
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.data_utils import CameraInfo, transform_point_cloud, create_point_cloud_from_depth_image,\
                            get_workspace_mask, remove_invisible_grasp_points, points_denoise, sample_points

img_width = 720
img_length = 1280
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800, 840, 880, 920, 960, 1000, 1040, 1080, 1120, 1160, 1200, 1240]
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


def get_bbox_center(label, obj_t, intrinsic):

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    center_2d = [
        round((fx * obj_t[0] / obj_t[2]) + cx),  # Projected x
        round((fy * obj_t[1] / obj_t[2]) + cy)   # Projected y
    ]
    
    # Compute mask's maximum width and height
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1

    # Adjust bounding box to center around `center_2d` while ensuring it covers the mask
    cmin = center_2d[0] - (center_2d[0] - cmin)
    cmax = center_2d[0] + (cmax - center_2d[0])
    rmin = center_2d[1] - (center_2d[1] - rmin)
    rmax = center_2d[1] + (rmax - center_2d[1])

    # Clip the bbox to fit within the image
    rmin = max(0, rmin)
    rmax = min(img_width, rmax)
    cmin = max(0, cmin)
    cmax = min(img_length, cmax)

    return rmin, rmax, cmin, cmax, center_2d


def add_noise_point_cloud(point_cloud, level=0.005, valid_min_z=0):
    """
    向点云数据添加高斯噪声，适用于 (N, 3) 形状的点云，每个点包含 (x, y, z) 坐标。

    参数：
    - point_cloud: numpy 数组，形状为 (N, 3)，表示点云数据。
    - level: 噪声强度的上限。
    - valid_min_z: 有效的最小深度值（z 轴），仅对满足条件的点添加噪声。

    返回：
    - noisy_point_cloud: 添加了噪声的点云数据。
    """
    # 确定有效的点，仅对深度 z 大于 valid_min_z 的点添加噪声
    mask = point_cloud[:, 2] > valid_min_z
    noisy_point_cloud = point_cloud.copy()

    # 随机生成噪声级别
    noise_level = np.random.uniform(0, level)

    # 生成高斯噪声，并应用于 (x, y, z) 三个通道
    noise = noise_level * np.random.randn(*point_cloud.shape)  # (N, 3) 形状
    noisy_point_cloud[mask] += noise[mask]

    return noisy_point_cloud


# def get_patch_point_cloud(patch_depth, intrinsics, bbox, choose, cam_scale):
#     """
#     Get the 3D point cloud from a patch depth map, dynamically creating xmap and ymap based on crop size.

#     Parameters:
#     - patch_depth: np.ndarray, shape (H, W), depth map of the patch (in meters).
#     - intrinsics: np.ndarray, shape (3, 3), camera intrinsic matrix.
#     - rmin, rmax, cmin, cmax: int, patch boundary in the full image.
#     - choose: np.ndarray, indices of valid points in the patch.
#     - cam_scale: float, depth scale factor.

#     Returns:
#     - point_cloud: np.ndarray, shape (N, 3), 3D point cloud.
#     """
#     rmin, rmax, cmin, cmax = bbox
#     # Intrinsic parameters
#     fx, fy = intrinsics[0, 0], intrinsics[1, 1]
#     cx, cy = intrinsics[0, 2], intrinsics[1, 2]

#     # Get the size of the cropped patch
#     patch_height = rmax - rmin
#     patch_width = cmax - cmin

#     # Dynamically generate xmap and ymap for the cropped patch
#     xmap = np.tile(np.arange(cmin, cmax), (patch_height, 1))  # X coordinates for crop
#     ymap = np.tile(np.arange(rmin, rmax).reshape(-1, 1), (1, patch_width))  # Y coordinates for crop

#     # Masked depth and pixel coordinates
#     depth_masked = patch_depth.flatten()[choose][:, np.newaxis].astype(np.float32)
#     xmap_masked = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
#     ymap_masked = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

#     # Scale depth
#     pt2 = depth_masked / cam_scale

#     # Compute 3D coordinates
#     pt0 = (xmap_masked - cx) * pt2 / fx
#     pt1 = (ymap_masked - cy) * pt2 / fy
#     cloud = np.concatenate((pt0, pt1, pt2), axis=1)

#     return cloud


def get_patch_point_cloud(patch_depth, intrinsics, bbox, cam_scale):
    """
    Get the 3D point cloud from a patch depth map, dynamically creating xmap and ymap based on crop size.

    Parameters:
    - patch_depth: np.ndarray, shape (H, W), depth map of the patch (in meters).
    - intrinsics: np.ndarray, shape (3, 3), camera intrinsic matrix.
    - rmin, rmax, cmin, cmax: int, patch boundary in the full image.
    - choose: np.ndarray, indices of valid points in the patch.
    - cam_scale: float, depth scale factor.

    Returns:
    - point_cloud: np.ndarray, shape (N, 3), 3D point cloud.
    """
    rmin, rmax, cmin, cmax = bbox
    # Intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Get the size of the cropped patch
    patch_height = rmax - rmin
    patch_width = cmax - cmin

    # Dynamically generate xmap and ymap for the cropped patch
    xmap = np.tile(np.arange(cmin, cmax), (patch_height, 1))  # X coordinates for crop
    ymap = np.tile(np.arange(rmin, rmax).reshape(-1, 1), (1, patch_width))  # Y coordinates for crop

    # Masked depth and pixel coordinates
    depth_masked = patch_depth.flatten()[:, np.newaxis].astype(np.float32)
    xmap_masked = xmap.flatten()[:, np.newaxis].astype(np.float32)
    ymap_masked = ymap.flatten()[:, np.newaxis].astype(np.float32)

    # Scale depth
    pt2 = depth_masked / cam_scale

    # Compute 3D coordinates
    pt0 = (xmap_masked - cx) * pt2 / fx
    pt1 = (ymap_masked - cy) * pt2 / fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    return cloud


def axangle2mat(axis, angle, is_normalized=False):
    """
    Convert axis-angle representation to rotation matrix (NumPy).
    
    Parameters:
    - axis: np.ndarray, shape (B, 3), rotation axes.
    - angle: np.ndarray, shape (B,), rotation angles in radians.
    - is_normalized: bool, whether the axes are already normalized.

    Returns:
    - rot_matrix: np.ndarray, shape (B, 3, 3), rotation matrices.
    """
    if not is_normalized:
        norm_axis = np.linalg.norm(axis, axis=1, keepdims=True)
        axis = axis / norm_axis

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    cos = np.cos(angle)
    sin = np.sin(angle)
    one_minus_cos = 1 - cos

    xs = x * sin
    ys = y * sin
    zs = z * sin
    xC = x * one_minus_cos
    yC = y * one_minus_cos
    zC = z * one_minus_cos
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot_matrix = np.stack(
        [
            x * xC + cos,
            xyC - zs,
            zxC + ys,
            xyC + zs,
            y * yC + cos,
            yzC - xs,
            zxC - ys,
            yzC + xs,
            z * zC + cos,
        ],
        axis=1,
    ).reshape(-1, 3, 3)  # Batch x 3 x 3
    return rot_matrix


def inplane_pose_2D_rotation(pose, rotation_angle):
    """
    Update the 6D pose with 2D rotation applied to the patch image, using ground truth pose translation.

    Parameters:
    - pose: np.ndarray, shape (4, 4), initial 6D pose in homogeneous coordinates.
    - rotation_angle: float, rotation angle in radians (2D image rotation).

    Returns:
    - updated_pose: np.ndarray, shape (4, 4), updated 6D pose in homogeneous coordinates.
    """
    # Extract rotation (R) and translation (t) from the 4x4 pose matrix
    R = pose[:3, :3]  # 3x3 rotation matrix
    t = pose[:3, 3]   # Translation vector (3,)

    # # Calculate alpha_x and alpha_y using ground truth translation
    alpha_x = -np.arctan2(t[1], t[2])  # YZ plane
    alpha_y = np.arctan2(t[0], np.linalg.norm(t[1:3]))  # ZX plane

    # Generate compensation rotation matrices
    Rx = axangle2mat(np.array([[1.0, 0.0, 0.0]]), np.array([alpha_x]), is_normalized=True)[0]
    Ry = axangle2mat(np.array([[0.0, 1.0, 0.0]]), np.array([alpha_y]), is_normalized=True)[0]

    # Generate 2D rotation matrix (around Z-axis)
    Rz = axangle2mat(np.array([[0.0, 0.0, 1.0]]), np.array([rotation_angle]), is_normalized=True)[0]

    # Final rotation update
    # R_canonical = Rz @ Rx.T @ Ry.T @ R 
    # R_new = Ry @ Rx @ R
    # R_new = Rz @ Rx.T @ Ry.T @ R
    # R_new = Rz @ Ry @ Rx @ R
    # R_new = Ry @ Rx @ Rz @ R
    # R_new = Ry @ Rx @ Rz @ R

    # Rz_canonical = Rz @ Ry.T @ Rx.T
    # Rz_canonical = Rx @ Ry @ Rz
    # R_new = Rz_canonical @ R
    # R_new = Ry @ Rx @ Rz @ Rx.T @ Ry.T @ R
    # R_new = Rz @ Rx.T @ Ry.T @ R
    # R_new = R
    R_new = Ry @ Rx @ Rz @ np.linalg.inv(Rx) @ np.linalg.inv(Ry) @ R
    # Translation remains unchanged
    t_new = t

    # Assemble the updated 4x4 pose matrix
    updated_pose = pose.copy()
    updated_pose[:3, :3] = R_new
    updated_pose[:3, 3] = t_new

    return updated_pose

# from scipy.spatial.transform import Rotation as R
# def inplane_pose_2D_rotation(pose, rotation_angle):
#     """
#     Update the 6D pose with a 2D in-plane rotation applied to the patch image.

#     Parameters:
#     - pose: np.ndarray, shape (4, 4), initial 6D pose in homogeneous coordinates.
#     - rotation_angle: float, rotation angle in radians (2D image rotation).

#     Returns:
#     - updated_pose: np.ndarray, shape (4, 4), updated 6D pose in homogeneous coordinates.
#     """
#     # Extract rotation (R) and translation (t) from the 4x4 pose matrix
#     R_mat = pose[:3, :3]  # 3x3 rotation matrix
#     t = pose[:3, 3]       # Translation vector (3,)

#     # Generate the 2D rotation matrix (around Z-axis)
#     Rz = R.from_euler('z', rotation_angle).as_matrix()

#     # Update the rotation matrix
#     R_new = Rz @ R_mat  # Apply the in-plane rotation to the original rotation

#     # Translation remains unchanged
#     t_new = t

#     # Assemble the updated 4x4 pose matrix
#     updated_pose = pose.copy()
#     updated_pose[:3, :3] = R_new
#     updated_pose[:3, 3] = t_new

#     return updated_pose


def flip_image(color, depth, mask, flip_code, intrinsic, fill_color=(0, 0, 0)):
    """
    Flip RGB, depth, and mask images.

    Parameters:
    - rgb_image: np.ndarray, shape (H, W, 3), RGB image.
    - depth_image: np.ndarray, shape (H, W), depth image.
    - mask_image: np.ndarray, shape (H, W), mask image.
    - flip_code: int, flip direction.
        0: Flip vertically.
        1: Flip horizontally.
        2: Flip both vertically and horizontally.

    Returns:
    - flipped_rgb: np.ndarray, flipped RGB image.
    - flipped_depth: np.ndarray, flipped depth image.
    - flipped_mask: np.ndarray, flipped mask image.
    """
    # Flip images using OpenCV's flip function
    h, w = color.shape[:2]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # 定义翻转类型对应的镜像矩阵
    if flip_code == 1:
        mirror_matrix = np.array([
            [-1,  0, 2 * cx],
            [ 0,  1,      0],
            [ 0,  0,      1]
        ], dtype=np.float32)
    elif flip_code == 0:
        mirror_matrix = np.array([
            [ 1,  0,      0],
            [ 0, -1, 2 * cy],
            [ 0,  0,      1]
        ], dtype=np.float32)
    elif flip_code == 2:
        mirror_matrix = np.array([
            [-1,  0, 2 * cx],
            [ 0, -1, 2 * cy],
            [ 0,  0,      1]
        ], dtype=np.float32)

    affine_mirror = mirror_matrix[:2, :]
    
    # 执行仿射变换
    flipped_rgb = cv2.warpAffine(
        color, affine_mirror, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill_color
    )
    flipped_depth = cv2.warpAffine(
        depth, affine_mirror, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )
    flipped_mask = cv2.warpAffine(
        mask, affine_mirror, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )

    return flipped_rgb, flipped_depth, flipped_mask

def rotate_image(color, depth, mask, rotation_angle, center=None, fill_color=(0, 0, 0)):
    """
    Rotate the patch image around its center.

    Parameters:
    - image: np.ndarray, the input image (H x W x C).
    - rotation_angle: float, rotation angle in degrees.
    - center: tuple, center of rotation (x, y). If None, use image center.
    - fill_color: tuple, fill color for areas outside the original image.

    Returns:
    - rotated_image: np.ndarray, the rotated image.
    """
    h, w = color.shape[:2]
    rotation_angle = np.rad2deg(rotation_angle)
    
    # Default center is the center of the image
    if center is None:
        center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale=1.0)

    # Perform the rotation
    rotated_color = cv2.warpAffine(
        color, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill_color
    )

    rotated_depth = cv2.warpAffine(
        depth, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )

    rotated_mask = cv2.warpAffine(
        mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=fill_color
    )
    return rotated_color, rotated_depth, rotated_mask


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


def visualize_bbox_with_center(image, rmin, rmax, cmin, cmax, center, index):
    """
    Visualize the bounding box and projection center on the given image using OpenCV.

    Parameters:
    - image: np.ndarray, shape (H, W, C) or (H, W), input image to visualize.
    - rmin, rmax, cmin, cmax: int, bounding box coordinates.
    - center: tuple, (u_center, v_center), the projection center in image coordinates.
    - title: str, title for saving the visualization (optional, used for debugging).

    Returns:
    - vis_image: np.ndarray, the image with bounding box and center drawn.
    """
    # Make a copy of the input image to avoid modifying the original
    vis_image = np.copy(image)

    # Ensure the image is in 3-channel format for visualization
    if len(vis_image.shape) == 2:  # Grayscale image
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    # Convert image to uint8 format if needed
    if vis_image.dtype != np.uint8:
        vis_image = (vis_image / vis_image.max() * 255).astype(np.uint8)

    # Draw the bounding box in red
    cv2.rectangle(vis_image, (cmin, rmin), (cmax, rmax), (0, 0, 255), 2)

    # Draw the projection center in green
    u_center, v_center = center
    cv2.circle(vis_image, (int(u_center), int(v_center)), 5, (0, 255, 0), -1)

    # Display the image with OpenCV
    cv2.imwrite('{}_bbox_save.png'.format(index), vis_image)
    # cv2.imshow(title, vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return vis_image


class GraspNetDataset(Dataset):
    def __init__(self, root, big_file_root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=1024,
                 remove_outlier=False, remove_invisible=True, multi_modal_pose_augment=False, point_augment=False, denoise=False, load_label=True, real_data=True, syn_data=False, visib_threshold=0.0, voxel_size=0.005, match_point_num=350):
        self.root = root
        if big_file_root is None:
            self.big_file_root = root
        else:
            self.big_file_root = big_file_root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        # self.pose_augment = pose_augment
        # self.inplane_pose_augment = inplane_pose_augment
        self.multi_modal_pose_augment = multi_modal_pose_augment
        self.point_augment = point_augment
        self.denoise = denoise
        self.denoise_pre_sample_num = int(self.num_points * 1.5)
        self.load_label = load_label    
        self.collision_labels = {}
        self.voxel_size = voxel_size
        self.minimum_num_pt = 50
        self.real_data = real_data
        self.syn_data = syn_data
        self.visib_threshold = visib_threshold
        self.match_point_num = match_point_num
        if split == 'train':
            self.sceneIds = list(range(100))
            # self.sceneIds = list(range(79, 80))
            # self.sceneIds = list(range(14, 15))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = ['scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds]

        self.resize_shape = (224, 224)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.colorpath = []
        self.depthpath = []
        self.labelpath = []
        self.metapath = []
        self.scenename = []
        self.frameid = []
        self.visibpath = []
        self.real_flags = []
        # self.graspnesspath = []
        # self.normalpath = []
        for x in tqdm(self.sceneIds, desc = 'Loading data path and collision labels...'):
            for img_num in range(256):
                if self.real_data:
                    self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb', str(img_num).zfill(4)+'.png'))
                    self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', str(img_num).zfill(4)+'.png'))
                    # self.depthpath.append(os.path.join(root, 'restored_depth',  x, camera, str(img_num).zfill(4)+'.png'))
                    self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', str(img_num).zfill(4)+'.png'))
                    self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                    self.visibpath.append(os.path.join(root, 'visib_info', x, camera, str(img_num).zfill(4)+'.mat'))
                    self.scenename.append(x.strip())
                    self.frameid.append(img_num)
                    self.real_flags.append(True)
                         
                if self.syn_data:
                    self.colorpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_rgb.png'))
                    self.depthpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_depth.png'))
                    self.labelpath.append(os.path.join(root, 'virtual_scenes', x, camera, str(img_num).zfill(4)+'_label.png'))
                    self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta', str(img_num).zfill(4)+'.mat'))
                    self.visibpath.append(os.path.join(root, 'visib_info', x, camera, str(img_num).zfill(4)+'.mat'))                    
                    self.scenename.append(x.strip())
                    self.frameid.append(img_num)
                    self.real_flags.append(False)
                    
            if self.load_label:
                collision_labels = np.load(os.path.join(self.big_file_root, 'collision_label', x.strip(), 'collision_labels.npz'))
                # collision_labels = h5py.File(os.path.join(root, 'collision_label_hdf5', x.strip(), 'collision_labels.hdf5'), "r")
                self.collision_labels[x.strip()] = {}
                for i in range(len(collision_labels)):
                    self.collision_labels[x.strip()][i] = collision_labels['arr_{}'.format(i)]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.depthpath)

    def instance_pose_augment(self, point_cloud, object_pose):
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.array([[-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 1]])
            point_cloud = transform_point_cloud(point_cloud, flip_mat, '3x3')
            object_pose = np.dot(flip_mat, object_pose).astype(np.float32)

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])
                
        point_cloud = transform_point_cloud(point_cloud, rot_mat, '3x3')
        object_pose = np.dot(rot_mat, object_pose).astype(np.float32)
        return point_cloud, object_pose

    def scene_pose_augment(self, images, object_poses, intrinsic, obj_idxs):
        (color, depth, mask) = images
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.eye(4)
            flip_mat[:3, :3] = np.array([[-1, 0, 0],
                                         [ 0, 1, 0],
                                         [ 0, 0, 1]])
            # point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            color, depth, mask = flip_image(color, depth, mask, 1, intrinsic)
            for i in range(len(object_poses)):
                object_pose = np.eye(4)
                object_pose[:3, :] = object_poses[i]
                object_poses[i] = (flip_mat @ object_pose)[:3, :].astype(np.float32)

        # Rotation along up-axis/Z-axis
        # rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        rot_angle = (np.random.random()*np.pi/2) - np.pi/4  # -45 ~ +45 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.eye(4)
        rot_mat[:3, :3] = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])

        object_pixel_num = [len(np.where(mask == obj_id)[0]) for obj_id in obj_idxs]
        color, depth, mask = rotate_image(color, depth, mask, -rot_angle, (intrinsic[0][2], intrinsic[1][2]))
        object_pixel_num_rot = [len(np.where(mask == obj_id)[0]) for obj_id in obj_idxs]
        # object_occ_rate = [object_pixel_num_rot[i] / object_pixel_num[i] for i in range(len(object_pixel_num))]
        object_occ_rate = np.array(object_pixel_num_rot) / (np.array(object_pixel_num) + 1e-8)
        # point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        for i in range(len(object_poses)):
            object_pose = np.eye(4)
            object_pose[:3, :] = object_poses[i]
            object_poses[i] = (rot_mat @ object_pose)[:3, :].astype(np.float32)

        return (color, depth, mask), object_poses, object_occ_rate
    
    # def inplane_pose_transform(self, color, depth, mask, object_pose):
    #     # rot_angle = (np.random.random()*np.deg2rad(180)) - np.deg2rad(90) # -90 ~ +90 degree
    #     rot_angle = np.deg2rad(90)
    #     augment_pose = inplane_pose_2D_rotation(object_pose, rot_angle)
    #     rot_color, rot_depth, rot_mask = rotate_image(color, depth, mask, -rot_angle, fill_color=(0, 0, 0))
    #     return rot_color, rot_depth, rot_mask, augment_pose
    
    def obj_idx_select(self, seg, obj_idxs, visib_info):
        max_visib_fract = 0  # 初始化最大可见性比例
        max_visib_idx = 0  # 初始化最大可见性对应的索引

        obj_idxs_list = list(range(len(obj_idxs)))
        np.random.shuffle(obj_idxs_list)
        for idx in obj_idxs_list:
            inst_mask = seg == obj_idxs[idx]
            inst_mask_len = inst_mask.sum()
            # inst_visib_fract = float(visib_info[str(obj_idxs[idx])]['visib_fract'])
            inst_visib_fract = float(inst_mask_len / visib_info[str(obj_idxs[idx])]['px_count_all'])

            # 更新最大可见性索引
            if inst_visib_fract > max_visib_fract:
                max_visib_fract = inst_visib_fract
                max_visib_idx = idx

            # 如果满足条件，立即返回
            if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
                return idx
        return max_visib_idx
            
    def __getitem__(self, index):
        if self.load_label:
            return self.get_data_label(index)
        else:
            return self.get_data(index)
    
    def get_resized_idxs(self, idxs, orig_shape):
        orig_width, orig_length = orig_shape
        scale_x = self.resize_shape[1] / orig_length
        scale_y = self.resize_shape[0] / orig_width
        coords = np.unravel_index(idxs, (orig_width, orig_length))
        new_coords_y = np.clip((coords[0] * scale_y).astype(int), 0, self.resize_shape[0]-1)
        new_coords_x = np.clip((coords[1] * scale_x).astype(int), 0, self.resize_shape[1]-1)
        new_idxs = np.ravel_multi_index((new_coords_y, new_coords_x), self.resize_shape)
        return new_idxs

    def get_data(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        visib_info = scio.loadmat(self.visibpath[index])
        scene = self.scenename[index]
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth']
        except Exception as e:
            print(repr(e))
            print(scene)

        # get valid points
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask # (720, 1280)

        seg_masked = seg * mask
        
        while 1:
            choose_idx = np.random.choice(np.arange(len(obj_idxs)))
            inst_mask = seg_masked == obj_idxs[choose_idx]
            inst_mask_len = inst_mask.sum()
            inst_visib_fract = float(visib_info[str(obj_idxs[choose_idx])]['visib_fract'])
            if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
                break

        inst_mask = inst_mask.astype(np.uint8)
        object_pose = poses[:, :, choose_idx]
        rmin, rmax, cmin, cmax, center = get_bbox(inst_mask, object_pose[:3, 3], intrinsic)
        bbox = (rmin, rmax, cmin, cmax)

        patch_color = color[rmin:rmax, cmin:cmax, :]
        patch_depth = depth[rmin:rmax, cmin:cmax]
        patch_mask = inst_mask[rmin:rmax, cmin:cmax]
        
        choose = patch_mask.flatten().nonzero()[0]
        sampled_idxs = sample_points(len(choose), self.num_points)
        choose = choose[sampled_idxs]
        
        inst_cloud = get_patch_point_cloud(patch_depth, intrinsic, bbox, choose, factor_depth)
        inst_color = patch_color.reshape(-1, 3)[choose]
                
        ret_dict = {}
        ret_dict['point_clouds'] = inst_cloud.astype(np.float32)
        ret_dict['cloud_colors'] = inst_color.astype(np.float32)
        
        ret_dict['coors'] = inst_cloud.astype(np.float32) / self.voxel_size
        # ret_dict['feats'] = inst_color.astype(np.float32)
        ret_dict['feats'] = np.ones_like(inst_cloud).astype(np.float32)
        return ret_dict

    # def get_data_label(self, index):
    #     color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
    #     depth = np.array(Image.open(self.depthpath[index]))
    #     seg = np.array(Image.open(self.labelpath[index]))
    #     meta = scio.loadmat(self.metapath[index])
    #     visib_info = scio.loadmat(self.visibpath[index])
    #     scene = self.scenename[index]
    #     real_flag = self.real_flags[index]
    #     # graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
    #     # normal = np.load(self.normalpath[index])['normals']
        
    #     try:
    #         obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    #         poses = meta['poses']
    #         intrinsic = meta['intrinsic_matrix']
    #         factor_depth = meta['factor_depth'][0]
    #     except Exception as e:
    #         print(repr(e))
    #         print(scene)

    #     # get valid points
    #     depth_mask = (depth > 0)
    #     seg_mask = (seg > 0)
    #     if self.remove_outlier:
    #         camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
    #         align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
    #         trans = np.dot(align_mat, camera_poses[self.frameid[index]])
    #         workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
    #         mask = (depth_mask & workspace_mask)
    #     else:
    #         mask = depth_mask

    #     seg_masked = seg * mask
        
    #     while 1:
    #         choose_idx = np.random.choice(np.arange(len(obj_idxs)))
    #         inst_mask = seg_masked == obj_idxs[choose_idx]
    #         inst_mask_len = inst_mask.sum()
    #         inst_visib_fract = float(visib_info[str(obj_idxs[choose_idx])]['visib_fract'])
    #         if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
    #             break

    #     inst_mask = inst_mask.astype(np.uint8)
    #     object_pose = poses[:, :, choose_idx]
    #     rmin, rmax, cmin, cmax, center = get_bbox_center(inst_mask, object_pose[:3, 3], intrinsic)
    #     bbox = (rmin, rmax, cmin, cmax)

    #     patch_color = color[rmin:rmax, cmin:cmax, :]
    #     patch_depth = depth[rmin:rmax, cmin:cmax]
    #     patch_mask = inst_mask[rmin:rmax, cmin:cmax]

    #     # cv2.imwrite("{}_color.png".format(index), patch_color*255.0)
    #     choose = patch_mask.flatten().nonzero()[0]
    #     sampled_idxs = sample_points(len(choose), self.num_points)
    #     choose = choose[sampled_idxs]
        
    #     inst_cloud = get_patch_point_cloud(patch_depth, intrinsic, bbox, choose, factor_depth)
    #     inst_color = patch_color.reshape(-1, 3)[choose]
        
    #     # inst_pc_vis = o3d.geometry.PointCloud()
    #     # inst_pc_vis.points = o3d.utility.Vector3dVector(inst_cloud.astype(np.float32))
    #     # # inst_pc_vis.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32))
    #     # inst_pc_vis.paint_uniform_color([1.0, 0.0, 0.0])
    #     # combin_vis = scene_vis + inst_pc_vis
    #     # o3d.io.write_point_cloud('{0}_combine.ply'.format(index), combin_vis)
    
    #     if self.point_augment:
    #         inst_cloud, dropout_idx = random_point_dropout(inst_cloud, min_num=self.minimum_num_pt,num_points_to_drop=2, radius_percent=0.1)
    #         inst_color = inst_color[dropout_idx]
    #         if not real_flag:
    #             inst_cloud = add_noise_point_cloud(inst_cloud.astype(np.float32), level=0.003, valid_min_z=0.1)
                
    #     orig_width, orig_length, _ = patch_color.shape
    #     resized_idxs = self.get_resized_idxs(choose, (orig_width, orig_length))
    #     img = self.img_transforms(patch_color)
        
    #     # inst_idxs_img = np.zeros_like(img)
    #     # inst_idxs_img = inst_idxs_img.reshape(-1, 3)
    #     # inst_idxs_img[resized_idxs] = inst_color
    #     # inst_idxs_img = inst_idxs_img.reshape((224, 224, 3))
    #     # cv2.imwrite("{}_inst_input.png".format(index), inst_idxs_img*255.)
        
    #     # inst_pc_vis = o3d.geometry.PointCloud()
    #     # inst_pc_vis.points = o3d.utility.Vector3dVector(inst_cloud.astype(np.float32))
    #     # inst_pc_vis.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32))
    #     # o3d.io.write_point_cloud('{0}_input.ply'.format(index), inst_pc_vis)
        
    #     points, offsets, scores = self.grasp_labels[obj_idxs[choose_idx]]
    #     collision = self.collision_labels[scene][choose_idx] #(Np, V, A, D)
    #     # grasp_idxs = np.random.choice(len(points), min(max(int(len(points)/4), 300),len(points)), replace=False)
        
    #     if self.pose_augment:
    #         inst_cloud, object_pose = self.points_pose_augment(inst_cloud, object_pose)
        
    #     grasp_idxs = np.sort(np.random.choice(len(points), self.match_point_num, replace=False))
    #     # grasp_idxs = np.random.choice(len(points), min(max(int(len(points) / 4), self.match_point_num), len(points)), replace=False)
    #     grasp_points = points[grasp_idxs]
    #     grasp_offsets = offsets[grasp_idxs]
    #     collision = collision[grasp_idxs].copy()
    #     scores = scores[grasp_idxs].copy()
    #     scores[collision] = 0
    #     grasp_scores = scores
        
    #     ret_dict = {}
    #     ret_dict['point_clouds'] = inst_cloud.astype(np.float32)
    #     ret_dict['cloud_colors'] = inst_color.astype(np.float32)
        
    #     # ret_dict['cloud_normals'] = inst_normal.astype(np.float32)
    #     ret_dict['coors'] = inst_cloud.astype(np.float32) / self.voxel_size
    #     # ret_dict['feats'] = inst_color.astype(np.float32)
    #     ret_dict['feats'] = np.ones_like(inst_cloud).astype(np.float32)
        
    #     ret_dict['img'] = img
    #     ret_dict['img_idxs'] = resized_idxs.astype(np.int64)
    #     # ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
    #     # ret_dict['objectness_label'] = objectness_label.astype(np.int64)
    #     # ret_dict['object_poses_list'] = object_poses_list
    #     # ret_dict['grasp_points_list'] = grasp_points_list
    #     # ret_dict['grasp_offsets_list'] = grasp_offsets_list
    #     # ret_dict['grasp_labels_list'] = grasp_scores_list
    #     ret_dict['object_pose'] = object_pose.astype(np.float32)
    #     ret_dict['grasp_points'] = grasp_points.astype(np.float32)
    #     ret_dict['grasp_offsets'] = grasp_offsets.astype(np.float32)
    #     ret_dict['grasp_labels'] = grasp_scores.astype(np.float32)
    #     return ret_dict
    
    
    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg = np.array(Image.open(self.labelpath[index]))
        meta = scio.loadmat(self.metapath[index])
        visib_info = scio.loadmat(self.visibpath[index])
        scene = self.scenename[index]
        real_flag = self.real_flags[index]
        # graspness = np.load(self.graspnesspath[index])  # for each point in workspace masked point cloud
        # normal = np.load(self.normalpath[index])['normals']
        
        try:
            obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
            poses = meta['poses']
            intrinsic = meta['intrinsic_matrix']
            factor_depth = meta['factor_depth'][0]
        except Exception as e:
            print(repr(e))
            print(scene)

        depth_mask = (depth > 0)
        # get valid points
        if self.remove_outlier:
            camera_info = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
            cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[self.frameid[index]])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth_mask & workspace_mask)
        else:
            mask = depth_mask

        poses = poses.transpose(2, 0, 1)
        seg = seg * mask
       
        if self.multi_modal_pose_augment:
            (color, depth, seg), poses, rot_occ_rate = self.scene_pose_augment((color, depth, seg), poses, intrinsic, obj_idxs)
        else:
            rot_occ_rate = np.ones(len(obj_idxs))
                
        while 1:
            choose_idx = np.random.choice(np.arange(len(obj_idxs)))
            inst_mask = seg == obj_idxs[choose_idx]
            inst_mask_len = inst_mask.sum()
            inst_visib_fract = float(rot_occ_rate[choose_idx] * visib_info[str(obj_idxs[choose_idx])]['visib_fract'])
            # inst_visib_fract = float(inst_mask_len / visib_info[str(obj_idxs[choose_idx])]['px_count_all'])
            if inst_mask_len > self.minimum_num_pt and inst_visib_fract > self.visib_threshold:
                break

        # choose_idx = self.obj_idx_select(seg, obj_idxs, visib_info)
        # inst_mask = seg == obj_idxs[choose_idx]
        # inst_mask = inst_mask.astype(np.uint8)
        object_pose = poses[choose_idx, :, :]
        rmin, rmax, cmin, cmax = get_bbox(inst_mask)
        
        bbox = (rmin, rmax, cmin, cmax)
        patch_color = color[rmin:rmax, cmin:cmax, :]
        patch_depth = depth[rmin:rmax, cmin:cmax]
        patch_mask = inst_mask[rmin:rmax, cmin:cmax]
        
        # cv2.imwrite("{}_before.png".format(index), patch_color[:, :, ::-1]*255.)
        # if self.pose_augment:
        #     patch_cloud, object_pose = self.points_pose_augment(patch_cloud, object_pose)

        # cv2.imwrite("{}_before.png".format(index), patch_color*255.0)
        
        # if self.inplane_pose_augment:
        #     patch_color, patch_depth, patch_mask, object_pose = self.inplane_pose_transform(patch_color, patch_depth, patch_mask, object_pose)

        choose = patch_mask.flatten().nonzero()[0]
        # if len(choose) <= 0:
        #     rmin, rmax, cmin, cmax, center = get_bbox(inst_mask, object_pose[:3, 3], intrinsic)
        #     print(self.colorpath[index], choose_idx)
        #     print(rmin, rmax, cmin, cmax)
        #     visualize_bbox_with_center(color, rmin, rmax, cmin, cmax, center, index)
        #     cv2.imwrite("{}_color.png".format(index), color*255.0)
        #     cv2.imwrite("{}_mask.png".format(index), inst_mask * 255.0)
            
        sampled_idxs = sample_points(len(choose), self.num_points)
        choose = choose[sampled_idxs]

        patch_cloud = get_patch_point_cloud(patch_depth, intrinsic, bbox, factor_depth)
        inst_cloud = patch_cloud[choose]
        inst_color = patch_color.reshape(-1, 3)[choose]
        
        # inst_pc_vis = o3d.geometry.PointCloud()
        # inst_pc_vis.points = o3d.utility.Vector3dVector(inst_cloud.astype(np.float32))
        # # inst_pc_vis.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32))
        # inst_pc_vis.paint_uniform_color([1.0, 0.0, 0.0])
        # combin_vis = scene_vis + inst_pc_vis
        # o3d.io.write_point_cloud('{0}_combine.ply'.format(index), combin_vis)
    
        if self.point_augment:
            inst_cloud, dropout_idx = random_point_dropout(inst_cloud, min_num=self.minimum_num_pt,num_points_to_drop=2, radius_percent=0.1)
            inst_color = inst_color[dropout_idx]
            if not real_flag:
                inst_cloud = add_noise_point_cloud(inst_cloud.astype(np.float32), level=0.003, valid_min_z=0.1)
                
        orig_width, orig_length, _ = patch_color.shape
        resized_idxs = self.get_resized_idxs(choose, (orig_width, orig_length))
        img = self.img_transforms(patch_color)
        
        # cv2.imwrite("{}_after.png".format(index), patch_color[:, :, ::-1]*255.)
        # inst_idxs_img = np.zeros_like(img)
        # inst_idxs_img = inst_idxs_img.reshape(-1, 3)
        # inst_idxs_img[resized_idxs] = inst_color[:, ::-1]
        # inst_idxs_img = inst_idxs_img.reshape((224, 224, 3))
        # cv2.imwrite("{}_inst_input.png".format(index), inst_idxs_img*255.)
        
        # inst_pc_vis = o3d.geometry.PointCloud()
        # inst_pc_vis.points = o3d.utility.Vector3dVector(inst_cloud.astype(np.float32))
        # inst_pc_vis.colors = o3d.utility.Vector3dVector(inst_color.astype(np.float32))
        # o3d.io.write_point_cloud('{0}_input.ply'.format(index), inst_pc_vis)
        
        points, offsets, scores = self.grasp_labels[obj_idxs[choose_idx]]
        collision = self.collision_labels[scene][choose_idx] #(Np, V, A, D)
        # grasp_idxs = np.random.choice(len(points), min(max(int(len(points)/4), 300),len(points)), replace=False)
        
        if len(points) < self.match_point_num:
            grasp_idxs = np.arange(len(points))
            grasp_idxs = np.concatenate([grasp_idxs, np.random.choice(grasp_idxs, self.match_point_num - len(points), replace=False)])
        else:
            grasp_idxs = np.sort(np.random.choice(len(points), self.match_point_num, replace=False))
        grasp_points = points[grasp_idxs]
        grasp_offsets = offsets[grasp_idxs]
        collision = collision[grasp_idxs].copy()
        scores = scores[grasp_idxs].copy()
        scores[collision] = 0
        grasp_scores = scores
        
        ret_dict = {}
        ret_dict['point_clouds'] = inst_cloud.astype(np.float32)
        ret_dict['cloud_colors'] = inst_color.astype(np.float32)
        
        # ret_dict['cloud_normals'] = inst_normal.astype(np.float32)
        ret_dict['coors'] = inst_cloud.astype(np.float32) / self.voxel_size
        # ret_dict['feats'] = inst_color.astype(np.float32)
        ret_dict['feats'] = np.ones_like(inst_cloud).astype(np.float32)
        
        ret_dict['img'] = img
        ret_dict['img_idxs'] = resized_idxs.astype(np.int64)
        # ret_dict['graspness_label'] = graspness_sampled.astype(np.float32)
        # ret_dict['objectness_label'] = objectness_label.astype(np.int64)
        # ret_dict['object_poses_list'] = object_poses_list
        # ret_dict['grasp_points_list'] = grasp_points_list
        # ret_dict['grasp_offsets_list'] = grasp_offsets_list
        # ret_dict['grasp_labels_list'] = grasp_scores_list
        ret_dict['object_pose'] = object_pose.astype(np.float32)
        ret_dict['grasp_points'] = grasp_points.astype(np.float32)
        ret_dict['grasp_offsets'] = grasp_offsets.astype(np.float32)
        ret_dict['grasp_labels'] = grasp_scores.astype(np.float32)
        return ret_dict


class GraspNetMultiDataset(Dataset):
    def __init__(self, root, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, voxel_size=0.005, remove_invisible=True, augment=False, load_label=True):
        assert num_points <= 50000
        self.root = root
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.augment = augment
        self.load_label = load_label
        self.collision_labels = {}
        self.voxel_size = voxel_size

        if split == 'train':
            self.sceneIds = list(range(100))
        elif split == 'test':
            self.sceneIds = list(range(100, 190))
        elif split == 'test_seen':
            self.sceneIds = list(range(100, 130))
        elif split == 'test_similar':
            self.sceneIds = list(range(130, 160))
        elif split == 'test_novel':
            self.sceneIds = list(range(160, 190))
        self.sceneIds = [f"scene_{str(x).zfill(4)}" for x in self.sceneIds]

        # multi-modal
        self.resize_shape = (448, 448)  # (H, W)
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize_shape),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.colorpath, self.depthpath, self.labelpath, self.metapath = [], [], [], []
        self.scenename, self.frameid, self.graspnesspath = [], [], []
        for x in tqdm(self.sceneIds, desc='Loading data path and collision labels...'):
            for img_num in range(256):
                self.colorpath.append(os.path.join(root, 'scenes', x, camera, 'rgb',  f'{img_num:04d}.png'))
                self.depthpath.append(os.path.join(root, 'scenes', x, camera, 'depth', f'{img_num:04d}.png'))
                self.labelpath.append(os.path.join(root, 'scenes', x, camera, 'label', f'{img_num:04d}.png'))
                self.metapath.append(os.path.join(root, 'scenes', x, camera, 'meta',  f'{img_num:04d}.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)
                if self.load_label:
                    self.graspnesspath.append(os.path.join(root, 'graspness', x, camera, f'{img_num:04d}.npy'))

            if self.load_label:
                collision_labels = np.load(os.path.join(root, 'collision_label', x.strip(), 'collision_labels.npz'))
                self.collision_labels[x.strip()] = {i: collision_labels[f'arr_{i}'] for i in range(len(collision_labels))}

    def __len__(self):
        return len(self.depthpath)

    def __getitem__(self, index):
        return self.get_data_label(index) if self.load_label else self.get_data(index)

    def scene_pose_augment(self, images, object_poses, intrinsic, obj_idxs):
        (color, depth, mask) = images
        aug_T = np.eye(4, dtype=np.float32)
        # Flipping along the YZ plane
        if np.random.random() > 0.5:
            flip_mat = np.eye(4)
            flip_mat[:3, :3] = np.array([[-1, 0, 0],
                                         [ 0, 1, 0],
                                         [ 0, 0, 1]])
            aug_T = flip_mat @ aug_T
            # point_clouds = transform_point_cloud(point_clouds, flip_mat, '3x3')
            color, depth, mask = flip_image(color, depth, mask, 1, intrinsic)
            for i in range(len(object_poses)):
                object_pose = np.eye(4)
                object_pose[:3, :] = object_poses[i]
                object_poses[i] = (flip_mat @ object_pose)[:3, :].astype(np.float32)

        # Rotation along up-axis/Z-axis
        # rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
        rot_angle = (np.random.random()*np.pi/2) - np.pi/4  # -45 ~ +45 degree
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.eye(4, dtype=np.float32)
        rot_mat[:3, :3] = np.array([[c, -s, 0],
                                    [s, c, 0],
                                    [0, 0, 1]])

        color, depth, mask = rotate_image(color, depth, mask, -rot_angle, (intrinsic[0][2], intrinsic[1][2]))
        # object_occ_rate = [object_pixel_num_rot[i] / object_pixel_num[i] for i in range(len(object_pixel_num))]
        # point_clouds = transform_point_cloud(point_clouds, rot_mat, '3x3')
        aug_T = rot_mat @ aug_T
        for i in range(len(object_poses)):
            object_pose = np.eye(4)
            object_pose[:3, :] = object_poses[i]
            object_poses[i] = (rot_mat @ object_pose)[:3, :].astype(np.float32)

        return (color, depth, mask), object_poses, aug_T

    def get_resized_idxs_from_flat(self, flat_idxs, orig_hw):
        """flat_idxs: flatten indices in original (H*W). -> flatten indices in resized (448*448)."""
        H, W = orig_hw
        scale_x = self.resize_shape[1] / W
        scale_y = self.resize_shape[0] / H
        ys, xs = np.unravel_index(flat_idxs, (H, W))
        new_y = np.clip((ys * scale_y).astype(np.int64), 0, self.resize_shape[0] - 1)
        new_x = np.clip((xs * scale_x).astype(np.int64), 0, self.resize_shape[1] - 1)
        return (new_y * self.resize_shape[1] + new_x).astype(np.int64)

    def _build_mask(self, depth, seg, cloud, scene, frameid, aug_T_cam=None):
        depth_mask = (depth > 0)
        if self.remove_outlier:
            camera_poses = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'camera_poses.npy'))
            align_mat = np.load(os.path.join(self.root, 'scenes', scene, self.camera, 'cam0_wrt_table.npy'))
            trans = np.dot(align_mat, camera_poses[frameid])  # cam -> table

            if aug_T_cam is not None:
                # cloud/poses 都在“增强后的相机坐标系”，workspace 变换也要一致
                trans = trans @ np.linalg.inv(aug_T_cam)

            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            return (depth_mask & workspace_mask)
        return depth_mask

    def get_data(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0  # (H,W,3)
        depth = np.array(Image.open(self.depthpath[index]))                            # (H,W)
        seg   = np.array(Image.open(self.labelpath[index]))                            # (H,W)
        meta  = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask = self._build_mask(depth, seg, cloud, scene, self.frameid[index])

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked = seg[mask]

        if cloud_masked.shape[0] == 0:
            raise RuntimeError(f"[{scene}/{self.frameid[index]}] mask has 0 points.")

        idxs = sample_points(len(cloud_masked), self.num_points)  # indices in masked-point order
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled   = seg_masked[idxs]

        # multi-modal: full image + point->pixel idx mapping
        H, W = depth.shape
        valid_flat = np.flatnonzero(mask)               # (mask_sum,)
        pix_flat = valid_flat[idxs]                     # (num_points,)
        resized_idxs = self.get_resized_idxs_from_flat(pix_flat, (H, W))
        img = self.img_transforms(color)                # full image resized

        return {
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),
            'coors': (cloud_sampled.astype(np.float32) / self.voxel_size),
            'feats': np.ones_like(cloud_sampled).astype(np.float32),

            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),

            # optional debug
            'seg': seg_sampled.astype(np.int32),
        }

    def get_data_label(self, index):
        color = np.array(Image.open(self.colorpath[index]), dtype=np.float32) / 255.0
        depth = np.array(Image.open(self.depthpath[index]))
        seg   = np.array(Image.open(self.labelpath[index]))
        meta  = scio.loadmat(self.metapath[index])
        scene = self.scenename[index]

        graspness = np.load(self.graspnesspath[index])
        graspness = graspness.squeeze()  # expect (mask_sum,)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        poses = meta['poses']
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        aug_T_cam = None
        if self.augment:
            poses_list = [poses[:, :, i].astype(np.float32) for i in range(poses.shape[2])]
            (color, depth, seg), poses_list, aug_T_cam = self.scene_pose_augment(
                (color, depth, seg), poses_list, intrinsic, obj_idxs
            )
        
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        mask = self._build_mask(depth, seg, cloud, scene, self.frameid[index], aug_T_cam=aug_T_cam)

        cloud_masked = cloud[mask]
        color_masked = color[mask]
        seg_masked   = seg[mask]

        if cloud_masked.shape[0] == 0:
            raise RuntimeError(f"[{scene}/{self.frameid[index]}] mask has 0 points.")

        idxs = sample_points(len(cloud_masked), self.num_points)  # indices in masked-point order
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        seg_sampled   = seg_masked[idxs]

        objectness_label = seg_sampled.copy()
        objectness_label[objectness_label > 1] = 1

        # multi-modal: full image + idx mapping
        H, W = depth.shape
        valid_flat = np.flatnonzero(mask)
        pix_flat = valid_flat[idxs]
        resized_idxs = self.get_resized_idxs_from_flat(pix_flat, (H, W))
        img = self.img_transforms(color)

        # scene-level labels (same “*_list” style)
        object_poses_list, grasp_points_list, grasp_offsets_list, grasp_scores_list = [], [], [], []
        for i, obj_idx in enumerate(obj_idxs):
            if obj_idx not in self.valid_obj_idxs:
                continue

            object_poses_list.append(poses[:, :, i])

            points, offsets, scores = self.grasp_labels[obj_idx]          # (Np,3), (Np,V,A,D), (Np,V,A,D)
            collision = self.collision_labels[scene][i]                    # (Np,V,A,D)

            # subsample grasp points per object (keep your original policy)
            sel = np.random.choice(len(points), min(max(int(len(points)/4), 300), len(points)), replace=False)

            points = points[sel]
            offsets = offsets[sel]
            scores = scores[sel].copy()
            col = collision[sel].copy()
            scores[col] = 0

            grasp_points_list.append(points)
            grasp_offsets_list.append(offsets)
            grasp_scores_list.append(scores)

        return {
            'point_clouds': cloud_sampled.astype(np.float32),
            'cloud_colors': color_sampled.astype(np.float32),

            'coors': (cloud_sampled.astype(np.float32) / self.voxel_size),
            'feats': np.ones_like(cloud_sampled).astype(np.float32),

            'img': img,
            'img_idxs': resized_idxs.astype(np.int64),

            'objectness_label': objectness_label.astype(np.int64),

            'object_poses_list': object_poses_list,
            'grasp_points_list': grasp_points_list,
            'grasp_offsets_list': grasp_offsets_list,
            'grasp_labels_list': grasp_scores_list,

            # optional debug
            'seg': seg_sampled.astype(np.int32),
        }

    
def load_grasp_labels(root):
    obj_names = list(range(88))
    # obj_names = [0, 2, 5, 14, 15, 20, 21, 22, 41, 43, 44, 46, 48, 52, 60, 62, 66, 70]
    # obj_names = [0, 2, 5, 20, 26, 37, 38, 51, 66]
    # obj_names = [ 8, 20, 26, 30, 41, 46, 56, 57, 60, 63, 66]
    # obj_names = [ 0, 9, 17, 51, 58, 61, 69, 70,]
    valid_obj_idxs = []
    grasp_labels = {}
    for obj_idx in tqdm(obj_names, desc='Loading grasping labels...'):
        # if i == 18: continue
        valid_obj_idxs.append(obj_idx+1) #here align with label png
        # tolerance = np.load(os.path.join(root, 'tolerance', '{}_tolerance.npy'.format(str(obj_idx).zfill(3))))
        # label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(i).zfill(3))))
        # grasp_labels[i + 1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
        #                         label['scores'].astype(np.float32), tolerance)
        # label = np.load(os.path.join(root, 'grasp_label', '{}_labels.npz'.format(str(obj_idx).zfill(3))))
        # grasp_labels[obj_idx+1] = (label['points'].astype(np.float32), label['offsets'].astype(np.float32),
        #                           label['scores'].astype(np.float32), tolerance)
        # label = h5py.File(os.path.join(root, 'grasp_label_simplified_hdf5', '{}_labels.hdf5'.format(str(obj_idx).zfill(3))), "r")
        label = np.load(os.path.join(root, 'grasp_label_simplified', '{}_labels.npz'.format(str(obj_idx).zfill(3))))
        grasp_labels[obj_idx+1] = (label['points'].astype(np.float32), label['width'].astype(np.float32),
                                  label['scores'].astype(np.float32))
    return valid_obj_idxs, grasp_labels


def collate_fn(batch):
    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)
    elif type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key:collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))


import MinkowskiEngine as ME
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data], dtype=torch.float32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if isinstance(batch[0], torch.Tensor):
            return torch.stack(batch, 0)
        elif type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res


def pt_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data], dtype=torch.float32)
    # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
    #     coordinates_batch, features_batch, return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        # "quantize2original": quantize2original
    }

    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)

    return res


if __name__ == "__main__":

    import random
    def setup_seed(seed):
         torch.manual_seed(seed)
         torch.cuda.manual_seed_all(seed)
         np.random.seed(seed)
         random.seed(seed)
         torch.backends.cudnn.deterministic = True
    setup_seed(0)
    
    root = '/media/gpuadmin/rcao/dataset/graspnet'
    valid_obj_idxs, grasp_labels = load_grasp_labels(root)
    train_dataset = GraspNetDataset(root, valid_obj_idxs, grasp_labels, num_points=1024, camera='realsense', split='train', multi_modal_pose_augment=True, point_augment=False, real_data=True, syn_data=False, visib_threshold=0.5, denoise=False, voxel_size=0.002)
    # print(len(train_dataset))

    scene_list = list(range(len(train_dataset)))
    # np.random.shuffle(scene_list)
    for scene_id in scene_list:
        end_points = train_dataset[scene_id]

        cloud = end_points['point_clouds']
        color = end_points['cloud_colors']
        pose = end_points['object_pose']
        grasp_point = end_points['grasp_points']
        grasp_point = transform_point_cloud(grasp_point, pose, '3x4')
        
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cloud)
        pc.colors = o3d.utility.Vector3dVector(color)

        pc_obj = o3d.geometry.PointCloud()
        pc_obj.points = o3d.utility.Vector3dVector(grasp_point)
        pc_obj.paint_uniform_color([1, 0, 0])
        pc_save = pc_obj + pc
        o3d.io.write_point_cloud('{}_combine.ply'.format(scene_id), pc_save)
        
        # o3d.visualization.draw_geometries([pc, pc_obj])
