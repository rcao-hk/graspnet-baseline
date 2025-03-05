import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import cv2
import time
import re
import glob
import argparse
import numpy as np
import torch
from PIL import Image, ImageEnhance
import scipy.io as scio
import open3d as o3d

from graspnetAPI import GraspGroup

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
setup_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--dataset_root', default='/media/user/data1/rcao/graspnet', help='Where dataset is')
parser.add_argument('--inst_pt_num', type=int, default=1024, help='Dump dir to save outputs')
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


def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):

    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def colorjitter(img, brightness, contrast, saturation, hue):
    img = Image.fromarray((img*255).astype(np.uint8))
    brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_factor = np.random.uniform(-hue, hue)

    img = adjust_brightness(img, brightness_factor)
    img = adjust_contrast(img, contrast_factor)
    img = adjust_saturation(img, saturation_factor)
    img = adjust_hue(img, hue_factor)
    img = np.array(img) / 255.0
    return img


def cutout(img, size_min=0.3, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    img = np.array(img)  # 转换为 NumPy 数组
    img_h, img_w, img_c = img.shape  # 处理 RGB 图像，形状 (H, W, 3)

    while True:
        size = np.random.uniform(size_min, size_max) * img_h * img_w
        ratio = np.random.uniform(ratio_1, ratio_2)
        erase_w = int(np.sqrt(size / ratio))
        erase_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_w)
        y = np.random.randint(0, img_h)

        if x + erase_w <= img_w and y + erase_h <= img_h:
            break

    # 生成全 0 遮挡区域
    value = np.zeros((erase_h, erase_w, img_c), dtype=img.dtype)

    # 应用遮挡
    img[y:y + erase_h, x:x + erase_w] = value
    return img

def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def defocus_blur(img, kernel_size=9):
    # c1 = random.randint(0,10)
    c2 = random.uniform(0,0.5)
    c = (kernel_size,c2)

    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(img[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
    img = np.clip(channels, 0, 1)
    return img


scene_idx = 0
# elapsed_time_list = []
for anno_idx in range(1):
    rgb_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
    depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))

    mask_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        
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
    seg_masked = seg[mask]
    seg_masked_org = seg * mask

    seg_idxs = np.unique(seg)
    for obj_idx in seg_idxs:
        if obj_idx == 0:
            continue

        inst_mask = seg_masked == obj_idx
        inst_mask_len = inst_mask.sum()
        if inst_mask_len < minimum_num_pt:
            continue
        inst_mask_org = seg_masked_org == obj_idx

        inst_cloud = cloud_masked[inst_mask]
        inst_color = color_masked[inst_mask]
        
        rmin, rmax, cmin, cmax = get_bbox(inst_mask_org.astype(np.uint8))
        
        img = color[rmin:rmax, cmin:cmax, :]
        inst_mask_org = inst_mask_org[rmin:rmax, cmin:cmax]
        inst_mask_choose = inst_mask_org.flatten().nonzero()[0]
        orig_width, orig_length, _ = img.shape
        
        # img = img_transforms(img)
        cutout_img = cutout(img)
        colorjitter_img = colorjitter(img, 0.3, 0.3, 0.3, 0.3)
        defocus_blur_img = defocus_blur(img)
        cv2.imwrite('raw_{}.png'.format(obj_idx), img[:, :, ::-1]*255.0)
        cv2.imwrite('cutout_{}.png'.format(obj_idx), cutout_img[:, :, ::-1]*255.0)
        cv2.imwrite('colorjitter_{}.png'.format(obj_idx), colorjitter_img[:, :, ::-1]*255.0)
        cv2.imwrite('defocus_blur_{}.png'.format(obj_idx), defocus_blur_img[:, :, ::-1]*255.0)
        
        # instance = o3d.geometry.PointCloud()
        # instance.points = o3d.utility.Vector3dVector(inst_cloud)
        # instance.colors = o3d.utility.Vector3dVector(inst_color)
        # o3d.visualization.draw_geometries([instance])
    
    # scene.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
    # scene.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
    # normal_masked = np.asarray(scene.normals)
