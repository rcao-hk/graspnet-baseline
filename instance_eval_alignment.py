#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute instance-level depth MAE and cross-modal alignment metrics (CKA / R^2)
for MMGNet / GSNet-style models on GraspNet frames.

One CSV row = one object instance in one frame.
Main merge keys with compute_object_topk_success_fixedB.py:
    - scene_id, ann_id, object_local_id / seg_instance_id
      unified object key matching the label image and the recomputed
      compute_object_topk_success.py output

This script is adapted from the user's scene-level CKA / R^2 analysis script,
but changed to:
  1. compute multiple instance-level reliability metrics:
     - inst_depth_mae_excluding_missing_m: pixel-wise depth MAE on GT-valid & sensor-valid overlap only
     - inst_depth_mae_including_missing_m: pixel-wise depth MAE on all GT-valid instance pixels, so sensor zeros are penalized
     - inst_point_l1_mae_m: sampled-point xyz L1 MAE using sampled input points vs sampled GT points at the same pixels
  2. compute instance-level alignment by masking sampled points per instance
  3. save one row per (scene_id, ann_id, object_local_id), ready to merge with
     object-level top-k success CSV.
"""

from __future__ import annotations

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
hard_limit = rlimit[1]
soft_limit = min(500000, hard_limit)
print("soft limit:", soft_limit, "hard limit:", hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

import sys
import re
import glob
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

import numpy as np
np.int = np.int32   # compatibility
np.float = np.float64
np.bool = np.bool_

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import scipy.io as scio
import MinkowskiEngine as ME
from PIL import Image, ImageEnhance
from torchvision import transforms
import pandas as pd

from utils.data_utils import (
    CameraInfo,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
    add_gaussian_noise_depth_map,
    apply_smoothing,
    find_large_missing_regions,
    apply_dropout_to_regions,
)

import random


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# Deterministic helpers
# -----------------------------------------------------------------------------
def make_base_seed(seed0: int, scene_idx: int, anno_idx: int) -> int:
    return int(seed0) * 1_000_000 + int(scene_idx) * 1_000 + int(anno_idx)


def sample_points_seeded(points_len: int, sample_num: int, seed: int):
    if points_len <= 0 or sample_num <= 0:
        return np.zeros((0,), dtype=np.int64)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    if points_len >= sample_num:
        idxs = rng.choice(points_len, sample_num, replace=False)
    else:
        idxs1 = np.arange(points_len)
        idxs2 = rng.choice(points_len, sample_num - points_len, replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    return idxs.astype(np.int64)


# -----------------------------------------------------------------------------
# RGB corruption helpers
# -----------------------------------------------------------------------------
def defocus_blur(image, kernel_size=9):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def cutout(img, patch_size=64, mask_ratio=0.1, fill_value=0.0):
    img = np.asarray(img, dtype=np.float32).copy()
    H, W, C = img.shape
    ph = pw = int(patch_size)
    gh = int(np.ceil(H / ph))
    gw = int(np.ceil(W / pw))
    num_patches = gh * gw
    num_mask = int(np.round(mask_ratio * num_patches))
    if num_mask <= 0:
        return img
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
    return enhancer.enhance(brightness_factor)


def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(contrast_factor)


def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(saturation_factor)


def _to_pil_uint8(img_float01):
    img_u8 = np.clip(img_float01 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img_u8)


def _from_pil_float01(img_pil):
    arr = np.asarray(img_pil, dtype=np.float32) / 255.0
    return arr


def apply_rgb_corruption(img_float01, corr_type='none', severity=0, seed: int = 0):
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
    blur_k = [5, 9, 11, 15, 17][severity - 1]
    cutout_ratio = [0.10, 0.20, 0.30, 0.40, 0.50][severity - 1]

    if corr_type == 'blur':
        return np.clip(defocus_blur(img, kernel_size=blur_k), 0.0, 1.0)
    if corr_type == 'cutout':
        return np.clip(cutout(img, patch_size=64, mask_ratio=cutout_ratio, fill_value=0.0), 0.0, 1.0)

    pil = _to_pil_uint8(img)
    delta_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
    delta = delta_seq[severity - 1]
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    sign = 1.0 if (rng.randint(0, 2) == 0) else -1.0
    factor = max(0.0, 1.0 + sign * delta)

    if corr_type == 'brightness':
        pil = adjust_brightness(pil, factor)
    elif corr_type == 'contrast':
        pil = adjust_contrast(pil, max(factor, 0.05))
    elif corr_type == 'saturation':
        pil = adjust_saturation(pil, factor)
    else:
        raise ValueError(f'Unknown corr_type: {corr_type}')
    return _from_pil_float01(pil)


# -----------------------------------------------------------------------------
# Index / depth helpers
# -----------------------------------------------------------------------------
def get_resized_idxs_from_flat(flat_idxs, orig_shape, resize_shape):
    H, W = orig_shape
    scale_x = resize_shape[1] / W
    scale_y = resize_shape[0] / H
    ys, xs = np.unravel_index(flat_idxs, (H, W))
    new_y = np.clip((ys * scale_y).astype(np.int64), 0, resize_shape[0] - 1)
    new_x = np.clip((xs * scale_x).astype(np.int64), 0, resize_shape[1] - 1)
    return (new_y * resize_shape[1] + new_x).astype(np.int64)


def depth_to_meters(depth_raw: np.ndarray, factor_depth: float, unit: str = 'raw') -> np.ndarray:
    unit = str(unit).lower()
    depth_raw = depth_raw.astype(np.float32)
    if unit == 'raw':
        return depth_raw / float(factor_depth)
    if unit == 'mm':
        return depth_raw / 1000.0
    if unit == 'm':
        return depth_raw
    raise ValueError(f'Unsupported gt_depth_unit: {unit}')


def create_point_cloud_from_depth_meters(depth_m: np.ndarray, camera_info: CameraInfo) -> np.ndarray:
    """Create an organized point cloud (H,W,3) from metric depth in meters."""
    assert depth_m.ndim == 2
    H, W = depth_m.shape
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    z = depth_m.astype(np.float32)
    x = (xs - float(camera_info.cx)) / float(camera_info.fx) * z
    y = (ys - float(camera_info.cy)) / float(camera_info.fy) * z
    return np.stack([x, y, z], axis=-1)


# -----------------------------------------------------------------------------
# Feature hooks / sparse helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def sparse_unique_from_coors_int(coors_int: torch.Tensor, device):
    assert coors_int.ndim == 2 and coors_int.shape[1] == 3
    N = coors_int.shape[0]
    feats = torch.ones((N, 1), dtype=torch.float32)
    coords_b, feats_b = ME.utils.sparse_collate([coors_int.cpu()], [feats])
    coords_b = coords_b.to(device)
    feats_b = feats_b.to(device)
    coords_u, _, uq_map, _ = ME.utils.sparse_quantize(
        coords_b, feats_b, return_index=True, return_inverse=True, device=device
    )
    return coords_u, uq_map.to(device)


@torch.no_grad()
def gather_img_feats_from_pyramid(pyr: dict, flat_idxs: torch.Tensor, base_hw=(448, 448)):
    H1, W1 = base_hw
    ys = (flat_idxs // W1).to(torch.int64)
    xs = (flat_idxs % W1).to(torch.int64)
    out = {}
    for key, feat in pyr.items():
        Hk, Wk = feat.shape[2], feat.shape[3]
        sy = H1 // Hk
        sx = W1 // Wk
        yk = torch.clamp(ys // sy, 0, Hk - 1)
        xk = torch.clamp(xs // sx, 0, Wk - 1)
        f = feat[0, :, yk, xk].transpose(0, 1).contiguous()
        out[key] = f
    return out


def register_pc_hooks(minkunet):
    feats = {}
    handles = []

    def _mk_hook(name):
        def hook(module, inp, out):
            feats[name] = out
        return hook

    for name in ["block1", "block2", "block3", "block4", "final"]:
        if hasattr(minkunet, name):
            handles.append(getattr(minkunet, name).register_forward_hook(_mk_hook(name)))
    return feats, handles


@torch.no_grad()
def query_sparse_feats_by_join(st: ME.SparseTensor, coords_u: torch.Tensor):
    C = st.C
    Nu = coords_u.shape[0]
    s = int(st.tensor_stride[0])
    q = coords_u.to(device=C.device, dtype=C.dtype).clone()
    if s != 1:
        sample = C[:min(2000, C.shape[0]), 1:]
        is_original_space = bool((sample % s).abs().max().item() == 0)
        if is_original_space:
            q[:, 1:] = (q[:, 1:] // s) * s
        else:
            q[:, 1:] = q[:, 1:] // s

    C64 = C.to(torch.int64)
    q64 = q.to(torch.int64)
    min_xyz = torch.minimum(C64[:, 1:].amin(0), q64[:, 1:].amin(0))
    max_xyz = torch.maximum(C64[:, 1:].amax(0), q64[:, 1:].amax(0))
    range_xyz = (max_xyz - min_xyz + 1)

    Cx = C64[:, 1] - min_xyz[0]
    Cy = C64[:, 2] - min_xyz[1]
    Cz = C64[:, 3] - min_xyz[2]
    Qx = q64[:, 1] - min_xyz[0]
    Qy = q64[:, 2] - min_xyz[1]
    Qz = q64[:, 3] - min_xyz[2]

    rx, ry, rz = range_xyz[0].item(), range_xyz[1].item(), range_xyz[2].item()
    keyC = (((C64[:, 0] * rx + Cx) * ry + Cy) * rz + Cz)
    keyQ = (((q64[:, 0] * rx + Qx) * ry + Qy) * rz + Qz)

    keyC_sorted, idxC_sorted = torch.sort(keyC)
    pos = torch.searchsorted(keyC_sorted, keyQ)
    in_range = pos < keyC_sorted.numel()
    pos_safe = pos.clone()
    pos_safe[~in_range] = 0
    hit = in_range & (keyC_sorted[pos_safe] == keyQ)

    Cdim = st.F.shape[1]
    feats = torch.zeros((Nu, Cdim), device=st.F.device, dtype=st.F.dtype)
    if hit.any():
        hit_idx = torch.nonzero(hit, as_tuple=False).squeeze(1)
        st_rows = idxC_sorted[pos[hit_idx]]
        st_rows = st_rows.to(device=st.F.device, dtype=torch.long)
        feats[hit_idx.to(device=st.F.device)] = st.F[st_rows]
    return feats, hit


@torch.no_grad()
def snap_coords_to_layer(st: ME.SparseTensor, coords_u: torch.Tensor):
    C = st.C
    s = int(st.tensor_stride[0])
    q = coords_u.to(device=C.device, dtype=C.dtype).clone()
    if s != 1:
        sample = C[:min(2000, C.shape[0]), 1:]
        is_original_space = bool((sample % s).abs().max().item() == 0)
        if is_original_space:
            q[:, 1:] = (q[:, 1:] // s) * s
        else:
            q[:, 1:] = q[:, 1:] // s
    return q.contiguous()


@torch.no_grad()
def group_mean(feat: torch.Tensor, inv: torch.Tensor, n_groups: int):
    if inv.device != feat.device:
        inv = inv.to(feat.device)
    inv = inv.long()
    N, C = feat.shape
    out = torch.zeros((n_groups, C), device=feat.device, dtype=feat.dtype)
    cnt = torch.zeros((n_groups, 1), device=feat.device, dtype=feat.dtype)
    out.index_add_(0, inv, feat)
    ones = torch.ones((N, 1), device=feat.device, dtype=feat.dtype)
    cnt.index_add_(0, inv, ones)
    return out / cnt.clamp_min(1.0)


# -----------------------------------------------------------------------------
# Alignment metrics
# -----------------------------------------------------------------------------
@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps=1e-8):
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    XtY = X.t() @ Y
    hsic = (XtY * XtY).sum()
    XtX = X.t() @ X
    YtY = Y.t() @ Y
    normx = torch.sqrt((XtX * XtX).sum() + eps)
    normy = torch.sqrt((YtY * YtY).sum() + eps)
    return (hsic / (normx * normy + eps)).clamp(0.0, 1.0)


@torch.no_grad()
def linear_predictability_r2(
    X, Y,
    train_ratio=0.8,
    l2=1.0,
    seed=0,
    std_thr=1e-3,
    ss_tot_thr=1e-6,
    clip_min=-1.0,
    clip_max=1.0,
):
    def _ridge_r2(A, B, seed_local):
        N = A.shape[0]
        if N < 128:
            return float('nan')
        g = torch.Generator(device=A.device)
        g.manual_seed(int(seed_local))
        perm = torch.randperm(N, generator=g, device=A.device)
        ntr = max(64, int(N * train_ratio))
        tr = perm[:ntr]
        te = perm[ntr:]
        if te.numel() < 32:
            return float('nan')

        A_tr = A[tr].float()
        B_tr = B[tr].float()
        A_te = A[te].float()
        B_te = B[te].float()

        muA = A_tr.mean(0, keepdim=True)
        sdA = A_tr.std(0, unbiased=False, keepdim=True)
        muB = B_tr.mean(0, keepdim=True)
        sdB = B_tr.std(0, unbiased=False, keepdim=True)
        maskA = (sdA.squeeze(0) > std_thr)
        maskB = (sdB.squeeze(0) > std_thr)
        if maskA.sum() < 8 or maskB.sum() < 8:
            return float('nan')

        A_tr = (A_tr[:, maskA] - muA[:, maskA]) / (sdA[:, maskA] + 1e-12)
        A_te = (A_te[:, maskA] - muA[:, maskA]) / (sdA[:, maskA] + 1e-12)
        B_tr = (B_tr[:, maskB] - muB[:, maskB]) / (sdB[:, maskB] + 1e-12)
        B_te = (B_te[:, maskB] - muB[:, maskB]) / (sdB[:, maskB] + 1e-12)

        A_tr64 = A_tr.double()
        B_tr64 = B_tr.double()
        XtX = A_tr64.t() @ A_tr64
        I = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        W = torch.linalg.solve(XtX + float(l2) * I, A_tr64.t() @ B_tr64)

        B_hat = (A_te.double() @ W).float()
        B_te = B_te.float()
        ss_res = ((B_te - B_hat) ** 2).sum(0)
        ss_tot = ((B_te - B_te.mean(0, keepdim=True)) ** 2).sum(0)
        dim_mask = ss_tot > ss_tot_thr
        if dim_mask.sum() < 8:
            return float('nan')
        r2_dim = 1.0 - ss_res[dim_mask] / (ss_tot[dim_mask] + 1e-12)
        return float(r2_dim.mean().clamp(min=clip_min, max=clip_max).item())

    r2_x2y = _ridge_r2(X, Y, seed_local=seed)
    r2_y2x = _ridge_r2(Y, X, seed_local=seed + 17)
    return r2_x2y, r2_y2x


# -----------------------------------------------------------------------------
# Path / split helpers
# -----------------------------------------------------------------------------
SPLIT_TO_SCENES = {
    'test': list(range(100, 190)),
    'test_seen': list(range(100, 130)),
    'test_similar': list(range(130, 160)),
    'test_novel': list(range(160, 190)),
}


def resolve_scene_list(split: str) -> List[int]:
    if split not in SPLIT_TO_SCENES:
        raise ValueError(f'Invalid split: {split}')
    return SPLIT_TO_SCENES[split]


def resolve_input_paths(dataset_root: str, camera: str, scene_idx: int, anno_idx: int, data_type: str):
    if data_type == 'real':
        rgb_path = os.path.join(dataset_root, f'scenes/scene_{scene_idx:04d}/{camera}/rgb/{anno_idx:04d}.png')
        depth_path = os.path.join(dataset_root, f'scenes/scene_{scene_idx:04d}/{camera}/depth/{anno_idx:04d}.png')
        mask_path = os.path.join(dataset_root, f'scenes/scene_{scene_idx:04d}/{camera}/label/{anno_idx:04d}.png')
    elif data_type == 'syn':
        rgb_path = os.path.join(dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_rgb.png')
        depth_path = os.path.join(dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_depth.png')
        mask_path = os.path.join(dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_label.png')
    elif data_type == 'noise':
        rgb_path = os.path.join(dataset_root, f'scenes/scene_{scene_idx:04d}/{camera}/rgb/{anno_idx:04d}.png')
        depth_path = os.path.join(dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_depth.png')
        mask_path = os.path.join(dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_label.png')
    else:
        raise ValueError(f'Unsupported data_type: {data_type}')
    meta_path = os.path.join(dataset_root, f'scenes/scene_{scene_idx:04d}/{camera}/meta/{anno_idx:04d}.mat')
    return rgb_path, depth_path, mask_path, meta_path


def resolve_gt_depth_path(args, scene_idx: int, anno_idx: int) -> Optional[str]:
    if args.gt_depth_root is None:
        if args.data_type == 'syn':
            return os.path.join(args.dataset_root, f'virtual_scenes/scene_{scene_idx:04d}/{args.camera}/{anno_idx:04d}_depth.png')
        return None

    if args.gt_depth_template:
        candidate = args.gt_depth_template.format(
            gt_depth_root=args.gt_depth_root,
            camera=args.camera,
            scene_idx=scene_idx,
            anno_idx=anno_idx,
        )
        return candidate

    candidates = [
        os.path.join(args.gt_depth_root, f'{args.camera}/scene_{scene_idx:04d}/{anno_idx:04d}.png'),
        os.path.join(args.gt_depth_root, f'scene_{scene_idx:04d}/{args.camera}/depth/{anno_idx:04d}.png'),
        os.path.join(args.gt_depth_root, f'scene_{scene_idx:04d}', args.camera, 'depth', f'{anno_idx:04d}.png'),
        os.path.join(args.gt_depth_root, f'{scene_idx:04d}', f'{anno_idx:04d}.png'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# -----------------------------------------------------------------------------
# Output schema
# -----------------------------------------------------------------------------
PC_LAYER_NAMES = ["block1", "block2", "block3", "block4", "final"]
IMG_KEYS = ["p1", "p2", "p4", "p8", "p16"]


def build_fieldnames() -> List[str]:
    fields = [
        'scene_id', 'ann_id', 'object_local_id', 'seg_instance_id', 'object_label',
        'network_name', 'fuse_type', 'split', 'camera', 'data_type',
        'inst_num_pixels_total', 'inst_num_pixels_eval', 'inst_num_valid_overlap',
        'inst_missing_ratio',
        'inst_depth_mae_m',
        'inst_depth_mae_excluding_missing_m',
        'inst_depth_mae_including_missing_m',
        'inst_point_l1_mae_m',
        'inst_num_sampled_points', 'inst_num_unique_voxels_input',
    ]
    for pl in PC_LAYER_NAMES:
        fields.append(f'inst_num_unique_{pl}')
        for ik in IMG_KEYS:
            fields.append(f'cka_{pl}_{ik}')
            fields.append(f'r2_x2y_{pl}_{ik}')
            fields.append(f'r2_y2x_{pl}_{ik}')
    return fields


def make_empty_metric_dict() -> Dict[str, float]:
    out = {}
    for pl in PC_LAYER_NAMES:
        out[f'inst_num_unique_{pl}'] = 0
        for ik in IMG_KEYS:
            out[f'cka_{pl}_{ik}'] = float('nan')
            out[f'r2_x2y_{pl}_{ik}'] = float('nan')
            out[f'r2_y2x_{pl}_{ik}'] = float('nan')
    return out


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_model(args, device):
    if args.network_name.startswith('mmgnet'):
        from models.IGNet_v0_9 import IGNet  # type: ignore
        net = IGNet(
            m_point=args.m_point,
            num_view=300,
            seed_feat_dim=args.seed_feat_dim,
            img_feat_dim=args.img_feat_dim,
            is_training=False,
            multi_scale_grouping=args.multi_scale_grouping,
            fuse_type=args.fuse_type,
        )
    elif args.network_name.startswith('gsnet'):
        from models.GSNet import GraspNet_multimodal  # type: ignore
        net = GraspNet_multimodal(seed_feat_dim=args.seed_feat_dim, img_feat_dim=64, is_training=False)
    else:
        raise ValueError(f'Unsupported network_name: {args.network_name}')

    pattern = re.compile(rf'(epoch_{args.ckpt_epoch}_.+\.tar|checkpoint_{args.ckpt_epoch}\.tar|epoch{args.ckpt_epoch}\.tar)$')
    ckpt_files = glob.glob(os.path.join(args.ckpt_root, args.network_name, args.camera, '*.tar'))
    ckpt_name = None
    for ckpt_path in ckpt_files:
        if pattern.search(os.path.basename(ckpt_path)):
            ckpt_name = ckpt_path
            break
    if ckpt_name is None:
        raise FileNotFoundError(f'Cannot find checkpoint epoch={args.ckpt_epoch} under {os.path.join(args.ckpt_root, args.network_name, args.camera)}')

    print('Load checkpoint from', ckpt_name)
    net.to(device)
    net.eval()
    checkpoint = torch.load(ckpt_name, map_location=device)
    try:
        net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except Exception:
        net.load_state_dict(checkpoint, strict=True)

    if not hasattr(net, 'img_backbone') or not hasattr(net, 'point_backbone'):
        raise RuntimeError('The loaded model does not expose img_backbone and point_backbone, which are required for alignment computation.')
    return net, ckpt_name


# -----------------------------------------------------------------------------
# Core per-frame computation
# -----------------------------------------------------------------------------
@torch.no_grad()
def compute_instance_alignment_for_frame(
    args,
    net,
    device,
    img_transforms,
    scene_idx: int,
    anno_idx: int,
) -> List[Dict]:
    base_seed = make_base_seed(args.seed, scene_idx, anno_idx)
    img_width, img_length = 720, 1280
    resize_shape = (448, 448)

    rgb_path, depth_path, mask_path, meta_path = resolve_input_paths(
        args.dataset_root, args.camera, scene_idx, anno_idx, args.data_type
    )
    if args.restored_depth:
        depth_path = os.path.join(args.depth_root, f'{args.camera}/scene_{scene_idx:04d}/{anno_idx:04d}.png')

    gt_depth_path = resolve_gt_depth_path(args, scene_idx, anno_idx)

    color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    if args.rgb_noise != 'none' and int(args.rgb_severity) > 0:
        color = apply_rgb_corruption(color, args.rgb_noise, args.rgb_severity, seed=base_seed + 19)

    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(mask_path))
    meta = scio.loadmat(meta_path)
    obj_idxs = meta['cls_indexes'].flatten().astype(np.int32)
    obj_idx_set = set(int(x) for x in obj_idxs.tolist())
    intrinsics = meta['intrinsic_matrix']
    factor_depth = float(meta['factor_depth'])

    camera_info = CameraInfo(
        img_length, img_width,
        intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2], factor_depth
    )

    # Predicted / used depth for geometry
    depth_used = depth.copy()
    noisy_cloud = None
    if int(args.smooth_size) > 1:
        depth_used = apply_smoothing(depth_used, size=int(args.smooth_size))
        noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)
    if float(args.gaussian_noise_level) > 0:
        depth_noisy = add_gaussian_noise_depth_map(
            depth_used.astype(np.float32), scale=factor_depth,
            level=float(args.gaussian_noise_level), valid_min_depth=0.1
        )
        depth_used = np.clip(depth_noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)

    # Geometry / workspace masks
    cloud_raw = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
    depth_mask = (depth_used > 0)
    camera_poses = np.load(os.path.join(args.dataset_root, f'scenes/scene_{scene_idx:04d}/{args.camera}/camera_poses.npy'))
    align_mat = np.load(os.path.join(args.dataset_root, f'scenes/scene_{scene_idx:04d}/{args.camera}/cam0_wrt_table.npy'))
    trans = np.dot(align_mat, camera_poses[anno_idx])
    workspace_mask = get_workspace_mask(cloud_raw, seg, trans=trans, organized=True, outlier=0.02)
    mask = depth_mask & workspace_mask

    dropout_mask = None
    if float(args.dropout_rate) > 0:
        if args.data_type == 'noise':
            depth_raw_path = os.path.join(args.dataset_root, f'scenes/scene_{scene_idx:04d}/{args.camera}/depth/{anno_idx:04d}.png')
            real_depth = np.array(Image.open(depth_raw_path))
        else:
            real_depth = depth.copy()
        foreground_mask = (seg > 0)
        large_missing_regions, labeled, filtered_labels = find_large_missing_regions(
            real_depth, foreground_mask, min_size=int(args.dropout_min_size)
        )
        dropout_regions = apply_dropout_to_regions(
            large_missing_regions, labeled, filtered_labels, float(args.dropout_rate)
        )
        dropout_mask = (dropout_regions > 0)
        mask = mask & (~dropout_mask)

    cloud_for_sampling = noisy_cloud if noisy_cloud is not None else create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)
    depth_eval_used = depth_used.copy()
    if dropout_mask is not None:
        depth_eval_used = depth_eval_used.copy()
        depth_eval_used[dropout_mask] = 0

    # GT depth / GT cloud for reliability metrics
    gt_depth_m = None
    gt_cloud_m = None
    pred_depth_m = depth_to_meters(depth_eval_used, factor_depth=factor_depth, unit='raw')
    if gt_depth_path is not None and os.path.exists(gt_depth_path):
        gt_depth_raw = np.array(Image.open(gt_depth_path))
        gt_depth_m = depth_to_meters(gt_depth_raw, factor_depth=factor_depth, unit=args.gt_depth_unit)
        gt_cloud_m = create_point_cloud_from_depth_meters(gt_depth_m, camera_info)

    # Build per-instance base rows first (depth metrics independent of feature extraction)
    local_ids = [int(x) for x in np.unique(seg) if int(x) > 0]
    rows: List[Dict] = []
    row_map: Dict[int, Dict] = {}
    for local_id in local_ids:
        seg_instance_id = int(local_id)
        # In GraspNet label images, the positive pixel values are already the object ids.
        # We therefore expose the same unified key through both object_local_id and
        # seg_instance_id so it matches the recomputed success CSV directly.
        object_label = int(seg_instance_id) if seg_instance_id in obj_idx_set else int(seg_instance_id)
        inst_mask_total = (seg == local_id)
        inst_mask_eval = inst_mask_total & workspace_mask if args.mae_use_workspace_mask else inst_mask_total

        base_row = {
            'scene_id': int(scene_idx),
            'ann_id': int(anno_idx),
            'object_local_id': int(seg_instance_id),
            'seg_instance_id': int(seg_instance_id),
            'object_label': int(object_label),
            'network_name': args.network_name,
            'fuse_type': args.fuse_type,
            'split': args.split,
            'camera': args.camera,
            'data_type': args.data_type,
            'inst_num_pixels_total': int(inst_mask_total.sum()),
            'inst_num_pixels_eval': int(inst_mask_eval.sum()),
            'inst_num_valid_overlap': 0,
            'inst_missing_ratio': float('nan'),
            # legacy alias: by default point to the overlap-only depth MAE below
            'inst_depth_mae_m': float('nan'),
            'inst_depth_mae_excluding_missing_m': float('nan'),
            'inst_depth_mae_including_missing_m': float('nan'),
            'inst_point_l1_mae_m': float('nan'),
            'inst_num_sampled_points': 0,
            'inst_num_unique_voxels_input': 0,
        }
        base_row.update(make_empty_metric_dict())

        if gt_depth_m is not None:
            gt_valid = gt_depth_m > 0
            pred_valid = pred_depth_m > 0
            valid_gt_only = inst_mask_eval & gt_valid
            valid_overlap = valid_gt_only & pred_valid
            n_gt = int(valid_gt_only.sum())
            n_overlap = int(valid_overlap.sum())
            base_row['inst_num_valid_overlap'] = n_overlap
            if n_gt > 0:
                base_row['inst_missing_ratio'] = float(1.0 - (n_overlap / max(n_gt, 1)))
            if n_overlap >= max(1, int(args.min_inst_valid_pixels)):
                depth_mae_excl = float(np.mean(np.abs(pred_depth_m[valid_overlap] - gt_depth_m[valid_overlap])))
                base_row['inst_depth_mae_excluding_missing_m'] = depth_mae_excl
                base_row['inst_depth_mae_m'] = depth_mae_excl
            if n_gt >= max(1, int(args.min_inst_valid_pixels)):
                # Include sensor zeros / missing values by comparing all GT-valid instance pixels.
                depth_mae_incl = float(np.mean(np.abs(pred_depth_m[valid_gt_only] - gt_depth_m[valid_gt_only])))
                base_row['inst_depth_mae_including_missing_m'] = depth_mae_incl

        rows.append(base_row)
        row_map[local_id] = base_row

    # If no valid cloud points remain, return depth-only rows.
    cloud_masked = cloud_for_sampling[mask]
    if cloud_masked.shape[0] == 0:
        return rows

    color_masked = color[mask]
    seg_masked = seg[mask]
    idxs = sample_points_seeded(len(cloud_masked), args.num_point, seed=base_seed + 11)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]
    seg_sampled = seg_masked[idxs]

    valid_flat = np.flatnonzero(mask)
    pix_flat = valid_flat[idxs]
    resized_idxs = get_resized_idxs_from_flat(pix_flat, depth.shape, resize_shape)

    img = img_transforms(color).to(device)
    cloud_tensor = torch.tensor(cloud_sampled, dtype=torch.float32, device=device)
    color_tensor = torch.tensor(color_sampled, dtype=torch.float32, device=device)
    coors_tensor = torch.tensor(cloud_sampled / args.voxel_size, dtype=torch.int32, device=device)
    feats_tensor = torch.ones_like(cloud_tensor).float().to(device)
    resized_idxs_tensor = torch.tensor(resized_idxs, dtype=torch.int64, device=device)

    # sampled GT points at the exact same sampled pixels; this avoids penalizing missing points
    # and measures only point distortion on the actually sampled input points.
    gt_cloud_sampled = None
    gt_valid_sampled = None
    if gt_cloud_m is not None:
        gt_cloud_sampled = gt_cloud_m.reshape(-1, 3)[pix_flat]
        gt_valid_sampled = gt_cloud_sampled[:, 2] > 0

    batch_data = {
        'point_clouds': cloud_tensor.unsqueeze(0),
        'cloud_colors': color_tensor.unsqueeze(0),
        'img': img.unsqueeze(0),
        'img_idxs': resized_idxs_tensor.unsqueeze(0),
        'coors': coors_tensor.unsqueeze(0),
        'feats': feats_tensor.unsqueeze(0),
    }

    coords_u, uq_map = sparse_unique_from_coors_int(coors_tensor, device=device)
    img_idxs_u = resized_idxs_tensor[uq_map]
    sampled_local_ids_u = torch.tensor(seg_sampled, device=device, dtype=torch.int64)[uq_map]

    pyr = net.img_backbone(batch_data['img'], return_pyramid=True)
    img_feats = gather_img_feats_from_pyramid(pyr, img_idxs_u, base_hw=resize_shape)

    pc_hook_feats, pc_handles = register_pc_hooks(net.point_backbone)
    _ = net(batch_data)
    for h in pc_handles:
        h.remove()

    # per-instance sampled-point MAE and point counts on input unique voxels
    seg_sampled_np = np.asarray(seg_sampled)
    for local_id in local_ids:
        inst_mask_sampled = (seg_sampled_np == int(local_id))
        inst_mask_u = (sampled_local_ids_u == int(local_id))
        n_sampled = int(inst_mask_sampled.sum())
        row_map[local_id]['inst_num_sampled_points'] = n_sampled
        row_map[local_id]['inst_num_unique_voxels_input'] = int(inst_mask_u.sum().item())

        if gt_cloud_sampled is not None and gt_valid_sampled is not None and n_sampled > 0:
            valid_pairs = inst_mask_sampled & gt_valid_sampled
            n_valid_pairs = int(valid_pairs.sum())
            row_map[local_id]['inst_num_valid_overlap'] = n_valid_pairs
            row_map[local_id]['inst_missing_ratio'] = float(1.0 - (n_valid_pairs / max(n_sampled, 1)))
            if n_valid_pairs >= max(1, int(args.min_inst_valid_pixels)):
                diff = np.abs(cloud_sampled[valid_pairs].astype(np.float32) - gt_cloud_sampled[valid_pairs].astype(np.float32))
                row_map[local_id]['inst_point_l1_mae_m'] = float(diff.mean())

    # Layer-wise alignment per instance
    pc_stage_to_seed = {"block1": 1, "block2": 2, "block3": 3, "block4": 4, "final": 5}
    img_key_to_seed = {"p1": 1, "p2": 2, "p4": 3, "p8": 4, "p16": 5}

    for pl_name, st_layer in pc_hook_feats.items():
        q = snap_coords_to_layer(st_layer, coords_u)
        q_unique, inv = torch.unique(q, dim=0, return_inverse=True)
        X_unique, hit_u = query_sparse_feats_by_join(st_layer, q_unique)
        assert int(hit_u.sum()) == q_unique.shape[0]
        Y_unique_dict = {k: group_mean(v, inv, q_unique.shape[0]) for k, v in img_feats.items()}
        local_ids_unique = group_mean(sampled_local_ids_u.float().unsqueeze(1), inv, q_unique.shape[0]).squeeze(1).round().long()

        for local_id in local_ids:
            inst_mask_unique = (local_ids_unique == int(local_id))
            n_unique_inst = int(inst_mask_unique.sum().item())
            row_map[local_id][f'inst_num_unique_{pl_name}'] = n_unique_inst
            if n_unique_inst < max(8, int(args.min_inst_unique_voxels)):
                continue

            X_inst_all = X_unique[inst_mask_unique]
            valid_x = (X_inst_all.abs().sum(1) > 0)
            pc_id = pc_stage_to_seed.get(pl_name, 0)

            for ikey, Y_unique in Y_unique_dict.items():
                Y_inst_all = Y_unique[inst_mask_unique].to(X_inst_all.device)
                valid = valid_x & (Y_inst_all.abs().sum(1) > 0)
                n_valid = int(valid.sum().item())
                if n_valid < max(16, int(args.min_inst_valid_pairs)):
                    continue

                Xv = X_inst_all[valid].float()
                Yv = Y_inst_all[valid].float()

                # deterministic subsample within instance for speed / fairness
                maxN = int(args.max_points_per_instance_metric)
                if maxN > 0 and Xv.shape[0] > maxN:
                    g_sel = torch.Generator(device=Xv.device)
                    img_id = img_key_to_seed.get(ikey, 0)
                    g_sel.manual_seed(int(base_seed + 1000 * pc_id + 10 * img_id + int(local_id)))
                    sel = torch.randperm(Xv.shape[0], generator=g_sel, device=Xv.device)[:maxN]
                    Xv = Xv[sel]
                    Yv = Yv[sel]

                if Xv.shape[0] >= 64:
                    row_map[local_id][f'cka_{pl_name}_{ikey}'] = float(linear_cka(Xv, Yv).item())
                if Xv.shape[0] >= 256:
                    img_id = img_key_to_seed.get(ikey, 0)
                    pair_seed = int(base_seed + 1000 * pc_id + 10 * img_id + int(local_id))
                    r2x2y, r2y2x = linear_predictability_r2(
                        Xv, Yv,
                        train_ratio=0.8,
                        l2=1e-1,
                        seed=pair_seed,
                        std_thr=1e-3,
                        ss_tot_thr=1e-6,
                        clip_min=-1.0,
                        clip_max=1.0,
                    )
                    row_map[local_id][f'r2_x2y_{pl_name}_{ikey}'] = float(r2x2y)
                    row_map[local_id][f'r2_y2x_{pl_name}_{ikey}'] = float(r2y2x)

    return rows


# -----------------------------------------------------------------------------
# Parser / main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test_seen')
    parser.add_argument('--camera', default='realsense')
    parser.add_argument('--seed_feat_dim', default=256, type=int)
    parser.add_argument('--img_feat_dim', default=64, type=int)
    parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet')
    parser.add_argument('--gt_depth_root', type=str, default=None,
                        help='Root path of GT depth. If unset and data_type=syn, virtual depth is used as GT; otherwise MAE is left NaN.')
    parser.add_argument('--gt_depth_template', type=str, default=None,
                        help='Optional format template for GT depth path, e.g. {gt_depth_root}/{camera}/scene_{scene_idx:04d}/{anno_idx:04d}.png')
    parser.add_argument('--gt_depth_unit', type=str, default='raw', choices=['raw', 'mm', 'm'])
    parser.add_argument('--ckpt_root', default='log')
    parser.add_argument('--network_name', type=str, default='mmgnet_scene')
    parser.add_argument('--ckpt_epoch', type=int, default=24)
    parser.add_argument('--num_point', type=int, default=20000)
    parser.add_argument('--m_point', type=int, default=2048)
    parser.add_argument('--multi_scale_grouping', action='store_true')
    parser.add_argument('--fuse_type', type=str, default='early')
    parser.add_argument('--voxel_size', type=float, default=0.002)
    parser.add_argument('--restored_depth', action='store_true')
    parser.add_argument('--depth_root', type=str, default='/media/gpuadmin/rcao/result/depth/v0.4')
    parser.add_argument('--data_type', type=str, default='real', choices=['real', 'syn', 'noise'])
    parser.add_argument('--smooth_size', type=int, default=1)
    parser.add_argument('--gaussian_noise_level', type=float, default=0.0)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--dropout_min_size', type=int, default=200)
    parser.add_argument('--rgb_noise', type=str, default='none', help='none|cutout|blur|brightness|saturation|contrast')
    parser.add_argument('--rgb_severity', type=int, default=0)
    parser.add_argument('--sample_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mae_use_workspace_mask', action='store_true',
                        help='If set, evaluate MAE only inside instance mask intersected with workspace mask.')
    parser.add_argument('--min_inst_valid_pixels', type=int, default=1,
                        help='Minimum valid GT/pred overlap pixels needed to report MAE.')
    parser.add_argument('--min_inst_unique_voxels', type=int, default=16,
                        help='Minimum unique voxels for reporting alignment on an instance/layer.')
    parser.add_argument('--min_inst_valid_pairs', type=int, default=32,
                        help='Minimum valid point pairs for reporting a metric on an instance/layer.')
    parser.add_argument('--max_points_per_instance_metric', type=int, default=2048,
                        help='Max points used per instance/layer metric computation for speed.')
    parser.add_argument('--output_csv', type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    resize_shape = (448, 448)
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_shape),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    net, ckpt_name = load_model(args, device)
    scene_list = resolve_scene_list(args.split)
    interval = int(max(1, args.sample_interval))

    rows_all: List[Dict] = []
    for scene_idx in scene_list:
        for anno_idx in range(256):
            if (anno_idx % interval) != 0:
                continue
            rows = compute_instance_alignment_for_frame(args, net, device, img_transforms, scene_idx, anno_idx)
            rows_all.extend(rows)
            print(f'[frame] scene={scene_idx} anno={anno_idx} rows+={len(rows)} total={len(rows_all)}')

    fieldnames = build_fieldnames()
    df = pd.DataFrame(rows_all)
    if len(df) == 0:
        print('[WARN] No rows generated; writing empty CSV with schema only.')
        df = pd.DataFrame(columns=fieldnames)
    else:
        # Ensure column order; keep any extra cols at the end.
        extra_cols = [c for c in df.columns if c not in fieldnames]
        df = df[fieldnames + extra_cols]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = {
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'network_name': args.network_name,
        'fuse_type': args.fuse_type,
        'split': args.split,
        'camera': args.camera,
        'data_type': args.data_type,
        'sample_interval': int(interval),
        'num_rows': int(len(df)),
        'num_scene_ann': int(df[['scene_id', 'ann_id']].drop_duplicates().shape[0]) if len(df) > 0 else 0,
        'num_instances': int(df[['scene_id', 'ann_id', 'object_local_id']].drop_duplicates().shape[0]) if len(df) > 0 else 0,
        'output_csv': str(output_csv.resolve()),
        'checkpoint': ckpt_name,
        'gt_depth_root': args.gt_depth_root,
        'gt_depth_template': args.gt_depth_template,
        'gt_depth_unit': args.gt_depth_unit,
        'mae_use_workspace_mask': bool(args.mae_use_workspace_mask),
        'min_inst_valid_pixels': int(args.min_inst_valid_pixels),
        'mae_definition': {
            'inst_depth_mae_excluding_missing_m': 'pixel-wise depth MAE on GT-valid & sensor-valid overlap only (sensor zeros excluded)',
            'inst_depth_mae_including_missing_m': 'pixel-wise depth MAE on all GT-valid instance pixels (sensor zeros included / penalized)',
            'inst_point_l1_mae_m': 'sampled-point xyz coordinate-wise L1 MAE between sampled input points and GT points at the same sampled pixels',
            'inst_depth_mae_m': 'legacy alias of inst_depth_mae_excluding_missing_m',
        },
        'min_inst_unique_voxels': int(args.min_inst_unique_voxels),
        'min_inst_valid_pairs': int(args.min_inst_valid_pairs),
        'max_points_per_instance_metric': int(args.max_points_per_instance_metric),
    }
    summary_path = output_csv.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'[DONE] Saved CSV to: {output_csv}')
    print(f'[DONE] Saved summary to: {summary_path}')
    print(f'[DONE] Rows: {len(df)}')


if __name__ == '__main__':
    main()
