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

import numpy as np
# compatible with numpy >= 1.24.4
np.int = np.int32
np.float = np.float64
np.bool = np.bool_

import cv2
import time
import re
import glob
import argparse
import torch
from PIL import Image, ImageEnhance
import scipy.io as scio
import open3d as o3d
import MinkowskiEngine as ME

import json
from datetime import datetime
import platform

from graspnetAPI import GraspGroup

from utils.collision_detector import ModelFreeCollisionDetector, ModelFreeCollisionDetectorTorch
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_depth_map, apply_smoothing, random_point_dropout, find_large_missing_regions, apply_dropout_to_regions, depthaware_perlin_dropout_masks
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
parser.add_argument('--dataset_root', default='/media/gpuadmin/rcao/dataset/graspnet', help='Where dataset is')
parser.add_argument('--ckpt_root', default='/media/gpuadmin/rcao/result/ignet', help='Where checkpoint is')
parser.add_argument('--network_name', type=str, default='v0.8.0', help='Network version')
parser.add_argument('--dump_dir', type=str, default='ignet_v0.8.0', help='Dump dir to save outputs')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--m_point', type=int, default=1024, help='Number of sampled points for grasp prediction [default: 1024]')
parser.add_argument('--ckpt_epoch', type=int, default=48, help='Checkpoint epoch name of trained model')
parser.add_argument('--inst_denoise', action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--restored_depth', action='store_true', help='Flag to use restored depth [default: False]')
parser.add_argument('--depth_root',type=str, default='/media/gpuadmin/rcao/result/depth/v0.4', help='Restored depth path')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--fuse_type',type=str, default='early')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--data_type', type=str, default='real', choices=['real', 'syn', 'noise', 'rgb_noise'], help='Type of input data: real|syn|noise')
parser.add_argument('--smooth_size', type=int, default=1,
                    help='Box smoothing kernel size on depth (<=1 means off)')
parser.add_argument('--gaussian_noise_level', type=float, default=0.0,
                    help='Gaussian noise std in meters (0 means off)')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Depth-guided dropout: fraction of missing regions to DROP (0 means off)')
parser.add_argument('--dropout_min_size', type=int, default=200,
                    help='Min connected component size for missing regions (on FG mask)')
parser.add_argument('--rgb_noise', type=str, default='none',
                    help='RGB corruption type: none|cutout|blur|brightness|contrast')
parser.add_argument('--rgb_severity', type=int, default=0,
                    help='RGB corruption severity in [0,5], 0 means no corruption')

parser.add_argument('--pc_sparse_level', type=int, default=0,
                    help='Point sparsity level (only for data_type==noise): 0=off, 1..4 -> [512,1024,2048,5120]')
parser.add_argument('--sample_interval', type=int, default=1,
                    help='Sample 1 frame every K frames in each scene (e.g., 10 means 0,10,20,...)')
cfgs = parser.parse_args()

print(cfgs)
minimum_num_pt = 50
img_width = 720
img_length = 1280

resize_shape = (448, 448)
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def save_run_config_json(cfgs, dump_dir, split, camera, extra=None):
    """
    Save run args/config to: {dump_dir}/{split}_{camera}.json
    """
    os.makedirs(dump_dir, exist_ok=True)
    out_path = os.path.join(dump_dir, f"{split}_{camera}.json")

    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "cfgs": vars(cfgs),  # argparse.Namespace -> dict
        "env": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    }
    if extra is not None:
        payload["extra"] = extra

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[CFG] Saved run config to: {out_path}")


def get_resized_idxs(idxs, orig_shape, resize_shape):
    orig_width, orig_length = orig_shape
    scale_x = resize_shape[1] / orig_length
    scale_y = resize_shape[0] / orig_width
    coords = np.unravel_index(idxs, (orig_width, orig_length))
    new_coords_y = np.clip((coords[0] * scale_y).astype(int), 0, resize_shape[0]-1)
    new_coords_x = np.clip((coords[1] * scale_x).astype(int), 0, resize_shape[1]-1)
    new_idxs = np.ravel_multi_index((new_coords_y, new_coords_x), resize_shape)
    return new_idxs
    

def get_resized_idxs_from_flat(flat_idxs, orig_shape, resize_shape):
    """flat_idxs: flatten indices in original (H*W). -> flatten indices in resized (448*448)."""
    H, W = orig_shape
    scale_x = resize_shape[1] / W
    scale_y = resize_shape[0] / H
    ys, xs = np.unravel_index(flat_idxs, (H, W))
    new_y = np.clip((ys * scale_y).astype(np.int64), 0, resize_shape[0] - 1)
    new_x = np.clip((xs * scale_x).astype(np.int64), 0, resize_shape[1] - 1)
    return (new_y * resize_shape[1] + new_x).astype(np.int64)


def defocus_blur(image, kernel_size=9):
    """
    Apply defocus blur (a type of Gaussian blur) to the image.
    
    Parameters:
    - image: Input image (numpy array).
    - kernel_size: Size of the kernel used for Gaussian blur (must be odd).
    
    Returns:
    - Defocus blurred image.
    """
    # Apply Gaussian blur to simulate defocus
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def cutout(img, patch_size=64, mask_ratio=0.1, fill_value=0.0):
    """
    Patch-wise cutout on the whole scene image:
    split image into patches (patch_size x patch_size) and randomly mask out multiple patches.

    img: float ndarray in [0,1], shape (H,W,3)
    patch_size: int
    mask_ratio: ratio of patches to mask (0~1)
    fill_value: value to fill, default 0.0 (black)
    """
    img = np.asarray(img, dtype=np.float32).copy()
    H, W, C = img.shape

    ph = pw = int(patch_size)
    gh = int(np.ceil(H / ph))
    gw = int(np.ceil(W / pw))
    num_patches = gh * gw
    num_mask = int(np.round(mask_ratio * num_patches))
    if num_mask <= 0:
        return img

    # choose patches to mask
    patch_ids = np.random.choice(num_patches, num_mask, replace=False)

    for pid in patch_ids:
        i = pid // gw
        j = pid % gw
        y0, y1 = i * ph, min((i + 1) * ph, H)
        x0, x1 = j * pw, min((j + 1) * pw, W)
        img[y0:y1, x0:x1, :] = fill_value

    return img


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
    img = Image.fromarray((img * 255).astype(np.uint8))
    brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
    contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
    saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
    hue_factor = np.random.uniform(-hue, hue)

    img = adjust_brightness(img, brightness_factor)
    img = adjust_contrast(img, contrast_factor)
    img = adjust_saturation(img, saturation_factor)
    img = adjust_hue(img, hue_factor)
    img = np.asarray(img, dtype=np.float32) / 255.0
    return img

def _to_pil_uint8(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_u8)

def _from_pil_float01(img_pil):
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    return arr

def apply_rgb_corruption(img_float01, corr_type='none', severity=0):
    """
    img_float01: np.ndarray float32 in [0,1], (H,W,3) RGB
    corr_type: none|cutout|blur|brightness|contrast
    severity: int in [0,5]
    """
    if corr_type is None:
        return img_float01

    corr_type = str(corr_type).strip().lower()
    if corr_type in ['none', 'null', 'clean', 'no', 'na', 'n/a', 'nil', 'false', '0', '']:
        return img_float01

    severity = int(severity)
    if severity <= 0:
        return img_float01
    severity = min(severity, 5)

    img = np.asarray(img_float01, dtype=np.float32)

    # ---- severity design ----
    # blur_k = [5, 9, 11, 15, 17][severity - 1]
    blur_k = [7, 13, 21, 33, 45][severity - 1]

    cutout_ratio = [0.10, 0.20, 0.30, 0.40, 0.50][severity - 1]
    patch_size = 64  # fixed patch size for interpretability

    if corr_type == 'blur':
        out = defocus_blur(img, kernel_size=blur_k)
        return np.clip(out, 0.0, 1.0)

    if corr_type == 'cutout':
        out = cutout(img, patch_size=patch_size, mask_ratio=cutout_ratio, fill_value=0.0)
        return np.clip(out, 0.0, 1.0)

    # PIL enhance based
    pil = _to_pil_uint8(img)
    
    # factor = np.random.uniform(max(0.0, 1.0 - delta), 1.0 + delta)
    # more conservative deltas (fixed magnitude per severity)
    # delta_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
    # delta = delta_seq[severity - 1]

    # # random direction only: +1 or -1
    # sign = 1.0 if np.random.rand() < 0.5 else -1.0
    # factor = max(0.0, 1.0 + sign * delta)

    ratio_seq = [1.25, 1.50, 1.75, 2.00, 2.50]
    ratio = ratio_seq[severity - 1]
    if np.random.rand() < 0.5:
        factor = ratio
    else:
        factor = 1.0 / ratio
        
    if corr_type == 'brightness':
        pil = adjust_brightness(pil, factor)
    elif corr_type == 'contrast':
        factor = max(factor, 0.05)
        pil = adjust_contrast(pil, factor)
    elif corr_type == 'saturation':
        pil = adjust_saturation(pil, factor)
    else:
        raise ValueError(f"Unknown corr_type: {corr_type}")
 
    return _from_pil_float01(pil)


def visualize_rgb_corruptions(
    img_path,
    out_path='rgb_corruption_grid.png',
    seed=0,
    dpi=150
):
    """
    Layout:
      Row 0: Raw RGB | blur | cutout | brightness | saturation | contrast (single example each)
      Row 1..5: for each corruption in [blur, cutout, brightness, saturation, contrast],
                show severity s1..s5 (5 levels)

    Each subplot has a bottom label (bold, fontsize 24).
    """
    import matplotlib
    matplotlib.use('Agg')  # safe for headless
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    from PIL import Image

    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0  # RGB float [0,1]

    top_corrs = ['raw', 'blur', 'cutout', 'brightness', 'contrast']
    sev_corrs = ['blur', 'cutout', 'brightness', 'contrast']
    severities = [1, 2, 3, 4, 5]

    nrows = 1 + len(sev_corrs)          # 1 top row + 5 corruption rows
    ncols = len(top_corrs)              # 6 columns

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.2*nrows))

    # Ensure 2D
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    def _show(ax, img_show, label):
        ax.imshow(np.clip(img_show, 0.0, 1.0))
        ax.axis('off')
        # bottom label
        ax.text(
            0.5, -0.08, label,
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=24, fontweight='bold'
        )

    # -----------------
    # Row 0: single examples
    # -----------------
    for c, ct in enumerate(top_corrs):
        ax = axes[0, c]
        np.random.seed(seed + 1000*0 + 10*c)
        random.seed(seed + 1000*0 + 10*c)

        if ct == 'raw':
            out = img
            label = 'Raw RGB'
        else:
            # use a representative severity for the "single example" row:
            # pick s3 for a mid-level visible effect
            out = apply_rgb_corruption(img, ct, 3)
            label = ct  # blur/cutout/brightness/saturation/contrast

        _show(ax, out, label)

    # -----------------
    # Rows 1..: severity grids (s1..s5), left 5 columns used; last column left empty
    # -----------------
    for r, ct in enumerate(sev_corrs, start=1):
        for c, sv in enumerate(severities):
            ax = axes[r, c]
            np.random.seed(seed + 1000*r + 10*c + sv)
            random.seed(seed + 1000*r + 10*c + sv)

            out = apply_rgb_corruption(img, ct, sv)
            label = f's{sv}'
            _show(ax, out, label)

        # last column (col 5) left empty for clean layout
        axes[r, ncols - 1].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'[VIS] saved to {out_path}')


def _disable_corruptions(cfgs):
    cfgs.smooth_size = 1
    cfgs.gaussian_noise_level = 0.0
    cfgs.dropout_rate = 0.0
    cfgs.dropout_min_size = 0
    cfgs.rgb_noise = 'none'
    cfgs.rgb_severity = 0
    cfgs.pc_sparse_level = 0


def save_5_severity_dropout_pointclouds_colorcoded(
    out_dir: str,
    scene_idx: int,
    anno_idx: int,
    depth_used: np.ndarray,          # (H,W) 用来生成点云 + 分母
    depth_raw: np.ndarray,           # (H,W) raw depth (0 indicates missing)
    seg: np.ndarray,                 # (H,W) instance label
    color_float01: np.ndarray,       # (H,W,3) float [0,1]
    workspace_mask: np.ndarray,      # (H,W) bool
    camera_info,
    create_point_cloud_from_depth_image,
    severity_rates=(0.1, 0.2, 0.4, 0.6, 0.8),
    seed=0,
    base_res=16, octaves=4, persistence=0.5,
    strict_match=True,
    blue_rgb=(0.0, 0.4, 1.0),        # depth-guided dropout
    red_rgb=(1.0, 0.0, 0.0),         # perlin dropout
    write_ascii=False,
    instance_only=False              # True: 只看 seg>0 区域
):
    """
    Save 5 PLYs (severity 1..5):
      kept: original RGB
      depth-guided dropped: blue
      perlin dropped: red
    """
    os.makedirs(out_dir, exist_ok=True)

    cloud_org = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)  # (H,W,3)

    base_mask = (depth_used > 0) & workspace_mask
    if instance_only:
        base_mask &= (seg > 0)

    xyz_base = cloud_org[base_mask]         # (N,3)
    rgb_base = color_float01[base_mask]     # (N,3)

    saved_paths = []
    for s, rate in enumerate(severity_rates, start=1):
        drop_depth, drop_perlin = depthaware_perlin_dropout_masks(
            depth_raw=depth_raw,
            depth_clear=depth_used,
            seg=seg,
            dropout_rate=float(rate),
            seed=int(seed + 17 * s),
            base_res=base_res, octaves=octaves, persistence=persistence,
            strict_match=strict_match,
            use_bbox_local_noise=True
        )

        dd = drop_depth[base_mask]
        dp = drop_perlin[base_mask]

        colors = np.clip(rgb_base.copy(), 0.0, 1.0).astype(np.float64)
        # priority: if overlap ever happens (shouldn't), mark depth-guided first
        colors[dp] = np.array(red_rgb, dtype=np.float64)
        colors[dd] = np.array(blue_rgb, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_base.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        fn = f"scene_{scene_idx:04d}_anno_{anno_idx:04d}_sev{s}_depthBLUE_perlinRED.ply"
        path = os.path.join(out_dir, fn)
        o3d.io.write_point_cloud(path, pcd, write_ascii=write_ascii)
        saved_paths.append(path)


    return saved_paths

data_type = cfgs.data_type # syn
restored_depth = cfgs.restored_depth
restored_depth_root = cfgs.depth_root
inst_denoise = cfgs.inst_denoise

split = cfgs.split
camera = cfgs.camera
dataset_root = cfgs.dataset_root
voxel_size = cfgs.voxel_size
network_name = cfgs.network_name
ckpt_root = cfgs.ckpt_root
dump_dir = os.path.join(cfgs.dump_dir)
ckpt_epoch = cfgs.ckpt_epoch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
    
PC_SPARSE_LEVELS = [5120, 2048, 1024, 512]
if data_type in ['real', 'syn']:
    _disable_corruptions(cfgs)
elif data_type == 'noise' and int(cfgs.pc_sparse_level) > 0:
    lv = int(cfgs.pc_sparse_level)
    assert 1 <= lv <= 4, f'pc_sparse_level must be in [0,4], got {lv}'
    cfgs.num_point = PC_SPARSE_LEVELS[lv - 1]

if network_name.startswith('mmgnet'):
    from models.IGNet_v0_9 import IGNet, pred_decode
    # from models.IGNet_v0_10 import IGNet, pred_decode
    net = IGNet(m_point=cfgs.m_point, num_view=300, seed_feat_dim=cfgs.seed_feat_dim, img_feat_dim=cfgs.img_feat_dim, is_training=False, multi_scale_grouping=cfgs.multi_scale_grouping, fuse_type=cfgs.fuse_type)
elif network_name.startswith('gsnet'):
    from models.GSNet import GraspNet_multimodal, pred_decode
    net = GraspNet_multimodal(seed_feat_dim=cfgs.seed_feat_dim, img_feat_dim=64, is_training=False)
    
pattern = re.compile(rf'(epoch_{ckpt_epoch}_.+\.tar|checkpoint_{ckpt_epoch}\.tar|epoch{ckpt_epoch}\.tar)$')
ckpt_files = glob.glob(os.path.join(ckpt_root, network_name, cfgs.camera, '*.tar'))
ckpt_name = None
for ckpt_path in ckpt_files:
    if pattern.search(os.path.basename(ckpt_path)):
        ckpt_name = ckpt_path
        break

try :
    assert ckpt_name is not None
    print('Load checkpoint from {}'.format(ckpt_name))
except :
    raise FileNotFoundError

net.to(device)
net.eval()
checkpoint = torch.load(ckpt_name, map_location=device)

try:
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
except:
    net.load_state_dict(checkpoint, strict=True)
eps = 1e-8

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

save_run_config_json(
    cfgs,
    dump_dir=dump_dir,
    split=cfgs.split,
    camera=cfgs.camera,
    extra={
        "ckpt_path": ckpt_name,
        "resolved_dump_dir": os.path.abspath(dump_dir),
        "seed": 0,
    }
)

def sample_indices_exact(n_src, n_tgt):
    """Return indices of length n_tgt from [0, n_src). Replace if needed."""
    if n_src <= 0:
        return np.array([], dtype=np.int64)
    replace = (n_src < n_tgt)
    return np.random.choice(n_src, n_tgt, replace=replace).astype(np.int64)
    
    
def inference(scene_idx):
    interval = int(max(1, cfgs.sample_interval))

    for anno_idx in range(256):
        if (anno_idx % interval) != 0:
            continue
        if data_type in ['real', 'rgb_noise']:
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            if restored_depth:
                depth_path = os.path.join(
                    restored_depth_root, '{}/scene_{:04d}/{:04d}.png'.format(camera, scene_idx, anno_idx))
            else:
                depth_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
                
            mask_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/label/{:04d}.png'.format(scene_idx, camera, anno_idx))
        elif data_type == 'syn':
            rgb_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_rgb.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
        
        elif data_type == 'noise':
            rgb_path = os.path.join(dataset_root,
                                    'scenes/scene_{:04d}/{}/rgb/{:04d}.png'.format(scene_idx, camera, anno_idx))
            depth_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_depth.png'.format(scene_idx, camera, anno_idx))
            depth_raw_path = os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/depth/{:04d}.png'.format(scene_idx, camera, anno_idx))
            mask_path = os.path.join(dataset_root, 'virtual_scenes/scene_{:04d}/{}/{:04d}_label.png'.format(scene_idx, camera, anno_idx))
        meta_path = os.path.join(dataset_root,
                                'scenes/scene_{:04d}/{}/meta/{:04d}.mat'.format(scene_idx, camera, anno_idx))

        color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
        color = apply_rgb_corruption(color, cfgs.rgb_noise, cfgs.rgb_severity)
        # visualize_rgb_corruptions(rgb_path, out_path=os.path.join('vis', '{}_{}_rgb_corruption.png'.format(scene_idx, anno_idx)))

        depth = np.array(Image.open(depth_path))
        seg = np.array(Image.open(mask_path))
        meta = scio.loadmat(meta_path)

        obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
        intrinsics = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera_info = CameraInfo(img_length, img_width, intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth)

        cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

        depth_mask = (depth > 0)
        camera_poses = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/camera_poses.npy'.format(scene_idx, camera)))
        align_mat = np.load(
            os.path.join(dataset_root, 'scenes/scene_{:04d}/{}/cam0_wrt_table.npy'.format(scene_idx, camera)))
        trans = np.dot(align_mat, camera_poses[anno_idx])
        workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
        mask = (depth_mask & workspace_mask)

        # ---------------- Apply point corruptions in depth domain ----------------
        depth_used = depth.copy()   # uint16 / or float later
        dropout_mask = None
        noisy_cloud = None
        # (A) smoothing (box blur)
        if cfgs.smooth_size is not None and int(cfgs.smooth_size) > 1:
            depth_used = apply_smoothing(depth_used, size=int(cfgs.smooth_size))
            noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)

        # (B) gaussian noise (in meters, then back to uint16)
        if cfgs.gaussian_noise_level is not None and float(cfgs.gaussian_noise_level) > 0:
            depth_noisy = add_gaussian_noise_depth_map(
                depth_used.astype(np.float32),
                scale=factor_depth,
                level=float(cfgs.gaussian_noise_level),
                valid_min_depth=0.1
            )
            depth_used = np.clip(depth_noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)
            noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)
            
        # (C) depth-guided dropout (find missing regions on RAW depth by default)
        if cfgs.dropout_rate is not None and float(cfgs.dropout_rate) > 0:
            # foreground_mask = (seg > 0)
            real_depth = np.array(Image.open(depth_raw_path))

            # large_missing_regions, labeled, filtered_labels = find_large_missing_regions(
            #     real_depth, foreground_mask, min_size=int(cfgs.dropout_min_size)
            # )
            # dropout_regions = apply_dropout_to_regions(
            #     large_missing_regions, labeled, filtered_labels, float(cfgs.dropout_rate)
            # )
            # dropout_mask = (dropout_regions > 0)

            # r_list = [0.1, 0.2, 0.4, 0.6]
            # lvl = int(cfgs.dropout_rate)
            # lvl = max(1, min(lvl, 5))
            # dropout_rate = r_list[lvl - 1]

            drop_depth_mask, drop_perlin_mask = depthaware_perlin_dropout_masks(
                depth_raw=real_depth,
                depth_clear=depth_used,          # 分母：clear depth 的有效像素
                seg=seg,
                dropout_rate=cfgs.dropout_rate,
                seed=0,
                strict_match=True,               # 保证 severity 可控（instance-level 精确比例）
                base_res=16, octaves=4, persistence=0.5,
                use_bbox_local_noise=True
            )

            # 总 dropout
            dropout_mask = drop_depth_mask | drop_perlin_mask

        # if data_type == 'noise' and float(cfgs.dropout_rate) > 0:
        #     out_dir = os.path.join('vis', "dropout_vis")
        #     paths = save_5_severity_dropout_pointclouds_colorcoded(
        #         out_dir=out_dir,
        #         scene_idx=scene_idx,
        #         anno_idx=anno_idx,
        #         depth_used=depth_used,
        #         depth_raw=real_depth,
        #         seg=seg,
        #         color_float01=color,
        #         workspace_mask=workspace_mask,
        #         camera_info=camera_info,
        #         create_point_cloud_from_depth_image=create_point_cloud_from_depth_image,
        #         seed=0,
        #         strict_match=True,
        #         instance_only=False   # 可选：只看物体点
        #     )
        #     print("[VIS] saved:", paths)
        
        if dropout_mask is not None:
            mask = mask & (~dropout_mask)

        if noisy_cloud is not None:
            cloud_masked = noisy_cloud[mask]
        else:
            cloud_masked = cloud[mask]
        color_masked = color[mask]
        
        # normal_masked = normal
        # 采样到 cfgs.num_point 个点（如果 cloud_masked 不足，会自动重复采样）
        idxs = sample_points(len(cloud_masked), cfgs.num_point)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        
        H, W = depth.shape
        valid_flat = np.flatnonzero(mask)               # (mask_sum,)
        pix_flat = valid_flat[idxs]                     # (num_points,)
        resized_idxs = get_resized_idxs_from_flat(pix_flat, (H, W), resize_shape)  # (num_points,)
        img = img_transforms(color)                # full image resized
        img = img.to(device)
        
        cloud_tensor = torch.tensor(cloud_sampled, dtype=torch.float32, device=device)
        color_tensor = torch.tensor(color_sampled, dtype=torch.float32, device=device)
        coors_tensor = torch.tensor(cloud_sampled / cfgs.voxel_size, dtype=torch.int32, device=device)
        feats_tensor = torch.ones_like(cloud_tensor).float().to(device)
        
        resized_idxs_tensor = torch.tensor(resized_idxs, dtype=torch.int64, device=device)
        # coordinates_batch, features_batch = ME.utils.sparse_collate([coors_tensor], [feats_tensor],
        #                                                             dtype=torch.float32)
        # coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        #     coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device)

        batch_data_label = {"point_clouds": cloud_tensor.unsqueeze(0),
                            "cloud_colors": color_tensor.unsqueeze(0),
                            'img': img.unsqueeze(0),
                            'img_idxs': resized_idxs_tensor.unsqueeze(0),
                            
                            "coors": coors_tensor.unsqueeze(0),
                            "feats": feats_tensor.unsqueeze(0),
                            # "quantize2original": quantize2original,
                            }
        
        with torch.no_grad():
            end_points = net(batch_data_label)
            grasp_preds = pred_decode(end_points)
            preds = grasp_preds[0]
            gg = GraspGroup(preds)
            # torch.cuda.empty_cache()

        # torch.cuda.empty_cache()
        # collision detection
        # 记录时间并执行前向传播
        # start.record()
        
        if cfgs.collision_thresh > 0:            
            mfcdetector = ModelFreeCollisionDetectorTorch(cloud.reshape(-1, 3), voxel_size=cfgs.collision_voxel_size)
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)
        # print('Inference Time:', elapsed_time)
        # elapsed_time_list.append(elapsed_time)
        # downsampled_scene = scene.voxel_down_sample(voxel_size=0.005)
        # gg = gg.sort_by_score()
        # gg_vis = gg.random_sample(100)
        # gg_vis = gg[:500]
        # gg_vis_geo = gg.to_open3d_geometry_list()
        # o3d.visualization.draw_geometries([scene] + gg_vis_geo)

        # save grasps
        save_dir = os.path.join(dump_dir, 'scene_%04d'%scene_idx, cfgs.camera)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '%04d'%anno_idx+'.npy')
        gg.save_npy(save_path)
        print('Saving {}, {}'.format(scene_idx, anno_idx))
    
    # print(f"Mean Inference Time：{np.mean(elapsed_time_list[1:]):.3f} ms")


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