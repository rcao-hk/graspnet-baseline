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
from datetime import datetime
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

from graspnetAPI import GraspGroup
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask, sample_points, points_denoise, add_gaussian_noise_depth_map, apply_smoothing, random_point_dropout, find_large_missing_regions, apply_dropout_to_regions
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


# -------------------------
# Utils: find submodules by class-name
# -------------------------
def find_first_module_by_classname(net, class_name: str):
    for m in net.modules():
        if m.__class__.__name__ == class_name:
            return m
    return None

def find_first_module_by_classname_prefix(net, prefix: str):
    for m in net.modules():
        if m.__class__.__name__.startswith(prefix):
            return m
    return None

# -------------------------
# Step0-A: voxel-unique coords for Minkowski
# -------------------------

@torch.no_grad()
def sparse_unique_from_coors_int(coors_int: torch.Tensor, device):
    """
    coors_int: (N,3) int32 on GPU/CPU, already = cloud/voxel_size
    return:
      coords_u: (Nu,4) int32 (batch,x,y,z)
      uq_map:   (Nu,) int64 indices into original N points
    """
    assert coors_int.ndim == 2 and coors_int.shape[1] == 3
    N = coors_int.shape[0]

    # dummy feats only for quantize; channel doesn't matter
    feats = torch.ones((N, 1), dtype=torch.float32)

    coords_b, feats_b = ME.utils.sparse_collate([coors_int.cpu()], [feats])  # CPU
    coords_b = coords_b.to(device)
    feats_b = feats_b.to(device)

    coords_u, _, uq_map, _ = ME.utils.sparse_quantize(
        coords_b, feats_b, return_index=True, return_inverse=True, device=device
    )
    # uq_map maps unique rows -> original rows (single batch)
    return coords_u, uq_map.to(device)

@torch.no_grad()
def gather_img_feats_from_pyramid(pyr: dict, flat_idxs: torch.Tensor, base_hw=(448, 448)):
    """
    pyr: {"p1":(1,C,H,W), "p2":..., "p16":...}
    flat_idxs: (Nu,) flatten indices in p1 resolution (448*448)
    return: dict {"p1":(Nu,C), ...}
    """
    H1, W1 = base_hw
    ys = (flat_idxs // W1).long()
    xs = (flat_idxs %  W1).long()

    out = {}
    for k, feat in pyr.items():
        # feat: (1,C,Hk,Wk)
        Hk, Wk = feat.shape[2], feat.shape[3]
        sy = H1 // Hk
        sx = W1 // Wk
        yk = torch.clamp(ys // sy, 0, Hk - 1)
        xk = torch.clamp(xs // sx, 0, Wk - 1)

        # (Nu,C)
        f = feat[0, :, yk, xk].transpose(0, 1).contiguous()
        out[k] = f
    return out


@torch.no_grad()
def build_sparse_input_unique(cloud_xyz: torch.Tensor, voxel_size: float, device):
    """
    cloud_xyz: (N,3) float32, in meters
    returns:
      sinput: ME.SparseTensor (unique voxels)
      coords_u: (Nu,4) int32 coordinates with batch index
      uq_map: (Nu,) indices mapping unique voxels -> original points
    """
    assert cloud_xyz.ndim == 2 and cloud_xyz.shape[1] == 3
    N = cloud_xyz.shape[0]

    # quantized voxel coords in "grid units"
    coords = torch.floor(cloud_xyz / voxel_size).to(torch.int32)  # (N,3)

    # build ME batched coords (Nu,4) with batch index at dim0
    # ME.utils.sparse_collate expects list per batch
    feats = torch.ones((N, 1), dtype=torch.float32, device=device)
    coords_b, feats_b = ME.utils.sparse_collate([coords.cpu()], [feats.cpu()])  # on CPU
    coords_b = coords_b.to(device)
    feats_b = feats_b.to(device)

    # unique voxelization
    coords_u, feats_u, uq_map, _ = ME.utils.sparse_quantize(
        coords_b, feats_b, return_index=True, return_inverse=True, device=device
    )
    # uq_map indexes coords_b rows -> original points (single batch), good for indexing your arrays
    sinput = ME.SparseTensor(features=feats_u, coordinates=coords_u, device=device)
    return sinput, coords_u, uq_map

# -------------------------
# Step0-B: sample image pyramid features at points
# -------------------------
@torch.no_grad()
def gather_img_feats_from_pyramid(pyr: dict, flat_idxs: torch.Tensor, base_hw=(448, 448)):
    """
    pyr: {"p1":(B,C,H,W), "p2":..., "p16":...} with B=1
    flat_idxs: (Nu,) flatten indices in base resolution (p1 resolution, 448x448)
    returns dict: {"p1":(Nu,C), "p2":(Nu,C), ...}
    """
    assert flat_idxs.dtype in (torch.int64, torch.int32)
    H1, W1 = base_hw
    ys = (flat_idxs // W1).to(torch.int64)
    xs = (flat_idxs % W1).to(torch.int64)

    out = {}
    for key, feat in pyr.items():
        # feat: (1,C,Hk,Wk)
        C = feat.shape[1]
        Hk, Wk = feat.shape[2], feat.shape[3]
        # scale factor relative to p1
        sy = H1 // Hk
        sx = W1 // Wk
        yk = torch.clamp(ys // sy, 0, Hk - 1)
        xk = torch.clamp(xs // sx, 0, Wk - 1)

        # gather: (Nu,C)
        f = feat[0, :, yk, xk].transpose(0, 1).contiguous()
        out[key] = f
    return out

# -------------------------
# Step0-C: hook MinkUNet internal layers and read per-point features
# -------------------------
def register_minkunet_hooks(minkunet):
    """
    capture SparseTensor outputs of specific blocks
    """
    feats = {}
    handles = []

    def _mk_hook(name):
        def hook(module, inp, out):
            feats[name] = out
        return hook

    # pick a few representative layers (stride changes)
    # block1: ~p2, block2: ~p4, block3: ~p8, block4: ~p16, final: ~p1 output
    for name in ["block1", "block2", "block3", "block4", "final"]:
        if hasattr(minkunet, name):
            handles.append(getattr(minkunet, name).register_forward_hook(_mk_hook(name)))

    return feats, handles


# -------------------------
# Step1: linear CKA
# -------------------------
def register_pc_hooks(minkunet):
    """
    hook outputs at key layers; adjust if you want more layers
    """
    feats = {}
    handles = []

    def _mk_hook(name):
        def hook(module, inp, out):
            feats[name] = out  # out is ME.SparseTensor
        return hook

    for name in ["block1", "block2", "block3", "block4", "final"]:
        if hasattr(minkunet, name):
            handles.append(getattr(minkunet, name).register_forward_hook(_mk_hook(name)))

    return feats, handles

@torch.no_grad()
def query_sparse_feats_at_coords(st: ME.SparseTensor, coords_u: torch.Tensor):
    s = int(st.tensor_stride[0])

    # 基础：dtype/device 对齐
    q = coords_u.to(device=st.C.device, dtype=st.C.dtype).contiguous().clone()

    if s != 1:
        C = st.C  # (Nl,4), usually on CPU
        # 判断该层坐标是否基本都是 stride 的倍数（= 原始坐标空间）
        # 取少量点判断即可，避免太慢
        sample = C[:min(2000, C.shape[0]), 1:]
        is_original_space = bool((sample % s).abs().max().item() == 0)

        if is_original_space:
            # snap 到该层 lattice（关键）
            q[:, 1:] = (q[:, 1:] // s) * s
        else:
            # 坐标已在下采样空间
            q[:, 1:] = q[:, 1:] // s

    return st.features_at_coordinates(q)

@torch.no_grad()
def query_sparse_feats_by_join(st: ME.SparseTensor, coords_u: torch.Tensor):
    """
    st: ME.SparseTensor
    coords_u: (Nu,4) query coords at stride=1 lattice (batch,x,y,z), int32/int64
    return:
      feats: (Nu,C) on st.F.device
      hit:   (Nu,) bool on CPU (哪些 query 坐标在 st.C 里能找到)
    """
    # coords live where st.C lives (often CPU)
    C = st.C  # (Nl,4), usually int32 CPU
    Nu = coords_u.shape[0]

    # --- build query coords at this tensor_stride ---
    s = int(st.tensor_stride[0])
    q = coords_u.to(device=C.device, dtype=C.dtype).clone()

    if s != 1:
        # 自动判断该层坐标空间：是否是 stride 倍数（原始坐标空间）
        sample = C[:min(2000, C.shape[0]), 1:]
        is_original_space = bool((sample % s).abs().max().item() == 0)
        if is_original_space:
            q[:, 1:] = (q[:, 1:] // s) * s
        else:
            q[:, 1:] = q[:, 1:] // s

    # --- make collision-free integer keys ---
    C64 = C.to(torch.int64)
    q64 = q.to(torch.int64)

    # 取 xyz 的 min/max，做 offset，避免负数导致碰撞
    min_xyz = torch.minimum(C64[:, 1:].amin(0), q64[:, 1:].amin(0))
    max_xyz = torch.maximum(C64[:, 1:].amax(0), q64[:, 1:].amax(0))
    range_xyz = (max_xyz - min_xyz + 1)  # (3,)

    Cx = C64[:, 1] - min_xyz[0]
    Cy = C64[:, 2] - min_xyz[1]
    Cz = C64[:, 3] - min_xyz[2]

    Qx = q64[:, 1] - min_xyz[0]
    Qy = q64[:, 2] - min_xyz[1]
    Qz = q64[:, 3] - min_xyz[2]

    # batch 一般都是 0，但保留也行（确保 batch 也不碰撞）
    rb, rx, ry, rz = (C64[:,0].max().item() + 1, range_xyz[0].item(), range_xyz[1].item(), range_xyz[2].item())

    keyC = (((C64[:, 0] * rx + Cx) * ry + Cy) * rz + Cz)
    keyQ = (((q64[:, 0] * rx + Qx) * ry + Qy) * rz + Qz)
    
    # --- sort & search ---
    keyC_sorted, idxC_sorted = torch.sort(keyC)  # CPU
    pos = torch.searchsorted(keyC_sorted, keyQ)  # CPU

    in_range = pos < keyC_sorted.numel()
    pos_safe = pos.clone()
    pos_safe[~in_range] = 0

    hit = in_range & (keyC_sorted[pos_safe] == keyQ)  # CPU bool

    # --- gather features ---
    Cdim = st.F.shape[1]
    feats = torch.zeros((Nu, Cdim), device=st.F.device, dtype=st.F.dtype)

    if hit.any():
        hit_idx = torch.nonzero(hit, as_tuple=False).squeeze(1)           # CPU
        st_rows = idxC_sorted[pos[hit_idx]]                               # CPU
        st_rows = st_rows.to(device=st.F.device, dtype=torch.long)        # GPU long
        feats[hit_idx.to(device=st.F.device)] = st.F[st_rows]

    return feats, hit


@torch.no_grad()
def _standardize_train_mask(Xtr, Xte, std_thr=1e-3, eps=1e-6):
    mu = Xtr.mean(0, keepdim=True)
    sd = Xtr.std(0, keepdim=True)
    keep = (sd > std_thr).squeeze(0)
    if keep.sum().item() < 8:
        return None, None
    sd = sd.clamp_min(eps)
    Xtr_s = (Xtr - mu) / sd
    Xte_s = (Xte - mu) / sd
    return Xtr_s[:, keep], Xte_s[:, keep]

@torch.no_grad()
def ridge_fit_predict(Xtr, Ytr, Xte, l2=1e-1):
    n, d = Xtr.shape
    XtX = Xtr.T @ Xtr
    tr = torch.trace(XtX).clamp_min(1e-6)
    lam = l2 * tr / d
    A = XtX + lam * torch.eye(d, device=Xtr.device, dtype=Xtr.dtype)
    B = Xtr.T @ Ytr
    W = torch.linalg.solve(A, B)
    return Xte @ W

@torch.no_grad()
def r2_score_multi_masked(Y, Yhat, ss_tot_thr=1e-6):
    Yc = Y - Y.mean(0, keepdim=True)
    ss_tot = (Yc * Yc).sum(0)
    ss_res = ((Y - Yhat) ** 2).sum(0)
    keep = ss_tot > ss_tot_thr
    if keep.sum().item() < 8:
        return torch.tensor(float("nan"), device=Y.device)
    r2 = 1.0 - ss_res[keep] / ss_tot[keep]
    return r2.mean()


@torch.no_grad()
def _ridge_fit_predict_r2(X, Y, train_ratio=0.8, l2=1e-1, seed=0,
                          std_thr=1e-3, ss_tot_thr=1e-6, clip_min=-1.0):
    """
    X: (N, dx), Y: (N, dy), float tensor on same device
    返回： (r2_mean, valid_dim_count, total_dim)
    - per-dim R2 on TEST, then average over valid dims
    - masks dims with tiny target variance (std/SST)
    - ridge with adaptive strength when ill-conditioned
    """
    assert X.ndim == 2 and Y.ndim == 2
    N = X.shape[0]
    if N < 256:
        return float("nan"), 0, int(Y.shape[1])

    # ---- split ----
    g = torch.Generator(device=X.device)
    g.manual_seed(int(seed))
    perm = torch.randperm(N, generator=g, device=X.device)
    ntr = int(round(N * train_ratio))
    ntr = max(128, min(ntr, N - 64))
    tr_idx = perm[:ntr]
    te_idx = perm[ntr:]

    Xtr = X[tr_idx].double()
    Ytr = Y[tr_idx].double()
    Xte = X[te_idx].double()
    Yte = Y[te_idx].double()

    # ---- center X/Y using TRAIN mean (for stable ridge, implicit bias) ----
    mx = Xtr.mean(0, keepdim=True)
    my = Ytr.mean(0, keepdim=True)
    Xtrc = Xtr - mx
    Ytrc = Ytr - my
    Xtec = Xte - mx
    Yte_centered_for_sst = Yte - Yte.mean(0, keepdim=True)  # R2 baseline用 test mean

    # ---- mask dims with tiny train std (almost-constant targets) ----
    y_std_tr = Ytr.std(0)  # (dy,)
    dim_mask_std = y_std_tr > std_thr

    # ---- ridge: adaptive lambda ----
    # base l2 乘一个 scale，避免不同层/不同N导致过弱
    dx = Xtrc.shape[1]
    # trace(X^T X)/dx 作为尺度（均方能量）
    x_energy = (Xtrc.pow(2).sum() / max(1, Xtrc.numel())).clamp_min(1e-12)
    # 当 dx 接近/超过 ntr 时，增大正则（防止权重爆炸）
    ratio = float(dx) / float(Xtrc.shape[0])
    boost = 1.0 if ratio < 0.5 else (2.0 if ratio < 1.0 else 5.0)
    lam = (l2 * boost) * x_energy

    # ---- solve (X^T X + lam I) W = X^T Y ----
    XtX = Xtrc.T @ Xtrc
    XtY = Xtrc.T @ Ytrc
    I = torch.eye(dx, device=X.device, dtype=torch.float64)
    W = torch.linalg.solve(XtX + lam * I, XtY)  # (dx, dy)

    # ---- predict on TEST in original Y scale ----
    Yhat = (Xtec @ W) + my  # (Nte, dy)

    # ---- per-dim R2 on TEST ----
    sse = (Yte - Yhat).pow(2).sum(0)                 # (dy,)
    sst = Yte_centered_for_sst.pow(2).sum(0)         # (dy,)
    dim_mask_sst = sst > ss_tot_thr
    dim_mask = dim_mask_std & dim_mask_sst

    if int(dim_mask.sum()) == 0:
        return float("nan"), 0, int(Y.shape[1])

    r2_dim = 1.0 - sse / (sst + 1e-12)

    # 可选：为了避免极端 outlier 影响热图，可 clip 到 [-1, 1]（不影响“好坏趋势”）
    if clip_min is not None:
        r2_dim = torch.clamp(r2_dim, min=float(clip_min), max=1.0)

    r2 = r2_dim[dim_mask].mean().item()
    return float(r2), int(dim_mask.sum().item()), int(Y.shape[1])


# @torch.no_grad()
# def linear_predictability_r2(X, Y, train_ratio=0.8, l2=1e-1, seed=0,
#                              std_thr=1e-3, ss_tot_thr=1e-6, clip_min=-1.0):
#     """
#     返回 (r2_x2y, r2_y2x)
#     """
#     r2_x2y, _, _ = _ridge_fit_predict_r2(
#         X, Y, train_ratio=train_ratio, l2=l2, seed=seed,
#         std_thr=std_thr, ss_tot_thr=ss_tot_thr, clip_min=clip_min
#     )
#     r2_y2x, _, _ = _ridge_fit_predict_r2(
#         Y, X, train_ratio=train_ratio, l2=l2, seed=seed + 17,
#         std_thr=std_thr, ss_tot_thr=ss_tot_thr, clip_min=clip_min
#     )
#     return r2_x2y, r2_y2x


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
    """
    Return: (r2_x2y, r2_y2x)
    - multi-output ridge regression with per-dim R2 + variance masking
    - robust to near-constant channels
    """
    def _ridge_r2(A, B):
        # A: (N, Da) -> B: (N, Db)
        N = A.shape[0]
        if N < 128:
            return float("nan")

        g = torch.Generator(device=A.device)
        g.manual_seed(int(seed))
        perm = torch.randperm(N, generator=g, device=A.device)

        ntr = max(64, int(N * train_ratio))
        tr = perm[:ntr]
        te = perm[ntr:]
        if te.numel() < 32:
            return float("nan")

        A_tr = A[tr].float()
        B_tr = B[tr].float()
        A_te = A[te].float()
        B_te = B[te].float()

        # ---- z-score by train stats ----
        muA = A_tr.mean(0, keepdim=True)
        sdA = A_tr.std(0, unbiased=False, keepdim=True)
        muB = B_tr.mean(0, keepdim=True)
        sdB = B_tr.std(0, unbiased=False, keepdim=True)

        maskA = (sdA.squeeze(0) > std_thr)
        maskB = (sdB.squeeze(0) > std_thr)
        if maskA.sum() < 8 or maskB.sum() < 8:
            return float("nan")

        A_tr = (A_tr[:, maskA] - muA[:, maskA]) / (sdA[:, maskA] + 1e-12)
        A_te = (A_te[:, maskA] - muA[:, maskA]) / (sdA[:, maskA] + 1e-12)
        B_tr = (B_tr[:, maskB] - muB[:, maskB]) / (sdB[:, maskB] + 1e-12)
        B_te = (B_te[:, maskB] - muB[:, maskB]) / (sdB[:, maskB] + 1e-12)

        # ---- ridge solve (float64 for stability) ----
        A_tr64 = A_tr.double()
        B_tr64 = B_tr.double()
        XtX = A_tr64.t() @ A_tr64
        I = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        W = torch.linalg.solve(XtX + float(l2) * I, A_tr64.t() @ B_tr64)  # (Da, Db)

        # ---- predict ----
        B_hat = (A_te.double() @ W).float()
        B_te = B_te.float()

        # ---- per-dim R2 with SST mask (use test mean) ----
        ss_res = ((B_te - B_hat) ** 2).sum(0)                   # (Db,)
        ss_tot = ((B_te - B_te.mean(0, keepdim=True)) ** 2).sum(0)  # (Db,)

        dim_mask = ss_tot > ss_tot_thr
        if dim_mask.sum() < 8:
            return float("nan")

        r2_dim = 1.0 - ss_res[dim_mask] / (ss_tot[dim_mask] + 1e-12)
        r2 = r2_dim.mean().clamp(min=clip_min, max=clip_max).item()
        return float(r2)

    # x2y: pc -> img
    r2_x2y = _ridge_r2(X, Y)
    # y2x: img -> pc
    r2_y2x = _ridge_r2(Y, X)
    return r2_x2y, r2_y2x

@torch.no_grad()
def linear_cka(X: torch.Tensor, Y: torch.Tensor, eps=1e-8):
    """
    X:(N,d), Y:(N,c)
    """
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
    """
    feat: (N, C)  每个点的特征（这里 N = coords_u.shape[0]）
    inv : (N,)    每个点对应的 group id（0..n_groups-1），来自 torch.unique(..., return_inverse=True)
    n_groups: int group 数（这里 = q_unique.shape[0]）
    return: (n_groups, C) 按 group 平均后的特征
    """
    if inv.device != feat.device:
        inv = inv.to(feat.device)
    inv = inv.long()

    N, C = feat.shape
    out = torch.zeros((n_groups, C), device=feat.device, dtype=feat.dtype)
    cnt = torch.zeros((n_groups, 1), device=feat.device, dtype=feat.dtype)

    # sum
    out.index_add_(0, inv, feat)

    # count
    ones = torch.ones((N, 1), device=feat.device, dtype=feat.dtype)
    cnt.index_add_(0, inv, ones)

    return out / cnt.clamp_min(1.0)


parser = argparse.ArgumentParser()
parser.add_argument('--split', default='test_seen', help='Dataset split [default: test_seen]')
parser.add_argument('--camera', default='realsense', help='Camera to use [kinect | realsense]')
parser.add_argument('--seed_feat_dim', default=256, type=int, help='Point wise feature dim')
parser.add_argument('--img_feat_dim', default=64, type=int, help='Image feature dim')
parser.add_argument('--dataset_root', default='/data/robotarm/dataset/graspnet', help='Where dataset is')
parser.add_argument('--ckpt_root', default='log', help='Where checkpoint is')
parser.add_argument('--network_name', type=str, default='mmgnet_scene', help='Network version')
parser.add_argument('--dump_dir', type=str, default='experiment/mmgnet_scene_vis', help='Dump dir to save outputs')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--m_point', type=int, default=2048, help='Number of sampled points for grasp prediction [default: 1024]')
parser.add_argument('--ckpt_epoch', type=int, default=24, help='Checkpoint epoch name of trained model')
parser.add_argument('--inst_denoise', action='store_true', help='Denoise instance points during training and testing [default: False]')
parser.add_argument('--restored_depth', action='store_true', help='Flag to use restored depth [default: False]')
parser.add_argument('--depth_root',type=str, default='/media/gpuadmin/rcao/result/depth/v0.4', help='Restored depth path')
parser.add_argument('--multi_scale_grouping', action='store_true', help='Multi-scale grouping [default: False]')
parser.add_argument('--fuse_type',type=str, default='early')
parser.add_argument('--voxel_size', type=float, default=0.002, help='Voxel Size to quantize point cloud [default: 0.005]')
parser.add_argument('--collision_voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--data_type', type=str, default='real', choices=['real', 'syn', 'noise'], help='Type of input data: real|syn|noise')
parser.add_argument('--smooth_size', type=int, default=1,
                    help='Box smoothing kernel size on depth (<=1 means off)')
parser.add_argument('--gaussian_noise_level', type=float, default=0.0,
                    help='Gaussian noise std in meters (0 means off)')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Depth-guided dropout: fraction of missing regions to DROP (0 means off)')
parser.add_argument('--dropout_min_size', type=int, default=200,
                    help='Min connected component size for missing regions (on FG mask)')
parser.add_argument('--rgb_noise', type=str, default='none',
                    help='RGB corruption type: none|cutout|blur|brightness|saturation|contrast')
parser.add_argument('--rgb_severity', type=int, default=0,
                    help='RGB corruption severity in [0,5], 0 means no corruption')
parser.add_argument('--sample_interval', type=int, default=10,
                    help='Sample 1 frame every K frames in each scene (e.g., 10 means 0,10,20,...)')
cfgs = parser.parse_args()

print(cfgs)
FEAT_VIS_DIR = os.path.join(ROOT_DIR, 'vis', 'feat_redun_vis')
os.makedirs(FEAT_VIS_DIR, exist_ok=True)

def _auto_json_name(cfgs):
    # 避免 early/late 覆盖同名文件（对比时你必须要两个json）
    return f"{cfgs.network_name}_{cfgs.fuse_type}_{cfgs.split}_{cfgs.sample_interval}.json"

json_name = _auto_json_name(cfgs)
json_path = os.path.join(FEAT_VIS_DIR, json_name)


def _save_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # 原子替换，避免写一半崩
    
img_width = 720
img_length = 1280

resize_shape = (448, 448)
img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(resize_shape),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cka_results = {
    "meta": {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "network_name": cfgs.network_name,
        "fuse_type": cfgs.fuse_type,
        "split": cfgs.split,
        "camera": cfgs.camera,
        "data_type": cfgs.data_type,
        "sample_interval": int(max(1, cfgs.sample_interval)),
        "num_point": int(cfgs.num_point),
        "m_point": int(cfgs.m_point),
        "voxel_size": float(cfgs.voxel_size),
        "img_feat_dim": int(cfgs.img_feat_dim),
        "seed_feat_dim": int(cfgs.seed_feat_dim),
        "resize_shape": list(resize_shape),
        "rgb_noise": cfgs.rgb_noise,
        "rgb_severity": int(cfgs.rgb_severity),
        "depth_corrupt": {
            "smooth_size": int(cfgs.smooth_size),
            "gaussian_noise_level": float(cfgs.gaussian_noise_level),
            "dropout_rate": float(cfgs.dropout_rate),
            "dropout_min_size": int(cfgs.dropout_min_size),
        },
    },
    "records": []
}


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
    corr_type: none|cutout|blur|brightness|saturation|contrast
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
    blur_k = [5, 9, 11, 15, 17][severity - 1]

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
    delta_seq = [0.1, 0.2, 0.3, 0.4, 0.5]
    delta = delta_seq[severity - 1]

    # random direction only: +1 or -1
    sign = 1.0 if np.random.rand() < 0.5 else -1.0
    factor = max(0.0, 1.0 + sign * delta)

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
    corr_types=('blur', 'cutout', 'brightness', 'saturation', 'contrast'),
    severities=(0, 1, 2, 3, 4, 5),
    seed=0,
    dpi=150
):
    """
    Visualize the same scene image under different corruptions and severities.
    Saves a grid figure to out_path.

    Rows: corruption types
    Cols: severity levels (including 0=clean)
    """
    import matplotlib
    matplotlib.use('Agg')  # safe for headless
    import matplotlib.pyplot as plt

    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0  # RGB float [0,1]

    nrows = len(corr_types)
    ncols = len(severities)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0*ncols, 3.0*nrows))

    # axes shape handling
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for r, ct in enumerate(corr_types):
        for c, sv in enumerate(severities):
            ax = axes[r, c]
            # make randomness reproducible per-cell
            np.random.seed(seed + 1000*r + 10*c + sv)
            random.seed(seed + 1000*r + 10*c + sv)

            if sv == 0:
                out = img
                title = f'{ct} | s0(clean)'
            else:
                out = apply_rgb_corruption(img, ct, sv)
                title = f'{ct} | s{sv}'

            ax.imshow(np.clip(out, 0.0, 1.0))
            ax.set_title(title, fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f'[VIS] saved to {out_path}')


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

if network_name.startswith('mmgnet'):
    from models.IGNet_v0_9 import IGNet, pred_decode
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
# print(net)

try:
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
except:
    net.load_state_dict(checkpoint, strict=True)
eps = 1e-8

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def _disable_corruptions(cfgs):
    cfgs.smooth_size = 1
    cfgs.gaussian_noise_level = 0.0
    cfgs.dropout_rate = 0.0
    cfgs.dropout_min_size = 0
    cfgs.rgb_noise = 'none'
    cfgs.rgb_severity = 0
    
    
def inference(scene_idx):
    interval = int(max(1, cfgs.sample_interval))

    for anno_idx in range(256):
        if (anno_idx % interval) != 0:
            continue
        if data_type == 'real':
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
        if data_type == 'noise':
            color = apply_rgb_corruption(color, cfgs.rgb_noise, cfgs.rgb_severity)
        else:
            _disable_corruptions(cfgs)
        # visualize_rgb_corruptions(rgb_path, out_path=os.path.join('vis', '{}_rgb_corruption.png'.format(scene_idx)))

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
            foreground_mask = (seg > 0)

            real_depth = np.array(Image.open(depth_raw_path))

            large_missing_regions, labeled, filtered_labels = find_large_missing_regions(
                real_depth, foreground_mask, min_size=int(cfgs.dropout_min_size)
            )
            dropout_regions = apply_dropout_to_regions(
                large_missing_regions, labeled, filtered_labels, float(cfgs.dropout_rate)
            )
            dropout_mask = (dropout_regions > 0)
            
        # cv2.imwrite('test_seg_{}_{}.png'.format(scene_idx, anno_idx), (net_seg.astype(np.float32)/net_seg.max()*255.0).astype(np.uint8))
        if dropout_mask is not None:
            mask = mask & (~dropout_mask)

        if noisy_cloud is not None:
            cloud_masked = noisy_cloud[mask]
        else:
            cloud_masked = cloud[mask]
        color_masked = color[mask]
        # normal_masked = normal

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
        
        # ---------- Step0: unique voxels + aligned img idxs ----------
        coords_u, uq_map = sparse_unique_from_coors_int(coors_tensor, device=device)  # coors_tensor:(N,3)
        img_idxs_u = resized_idxs_tensor[uq_map]  # (Nu,)

        # ---------- Step0: image pyramid feats at same voxels ----------
        with torch.no_grad():
            pyr = net.img_backbone(batch_data_label["img"], return_pyramid=True)  # {"p1","p2","p4","p8","p16"}
            img_feats = gather_img_feats_from_pyramid(pyr, img_idxs_u, base_hw=resize_shape)

        # ---------- Step1: hook pc layers, run ONE full net forward ----------
        pc_hook_feats, pc_handles = register_pc_hooks(net.point_backbone)

        with torch.no_grad():
            end_points = net(batch_data_label)   # normal forward (early fusion inside)

        for h in pc_handles:
            h.remove()

        # ---------- Step1: query pc feats at coords_u, then compute CKA ----------
        pc_layers = {}
        for lname, st in pc_hook_feats.items():
            # pc_layers[lname] = query_sparse_feats_at_coords(st, coords_u)  # (Nu,C)
            # pc_layers[lname] = query_sparse_feats_at_coords(st, coords_u)

            Xall, hit = query_sparse_feats_by_join(st, coords_u)
            pc_layers[lname] = Xall
            print(f"[JOIN] {lname}: hit {int(hit.sum())}/{hit.numel()} stride={st.tensor_stride} C={tuple(st.F.shape)}")

        # subsample for speed
        # Nu = coords_u.shape[0]
        # maxN = 4096
        # sel = torch.randperm(Nu, device=device)[:maxN] if Nu > maxN else torch.arange(Nu, device=device)

        cka_map = {}
        r2_x2y_map = {}   # pc -> img
        r2_y2x_map = {}   # img -> pc
        
        for pl_name, st_layer in pc_hook_feats.items():
            # 1) snap 到该层坐标 + unique
            q = snap_coords_to_layer(st_layer, coords_u)                       # (Nu,4) on CPU (same as st_layer.C device)
            q_unique, inv = torch.unique(q, dim=0, return_inverse=True)        # q_unique:(Nu2,4), inv:(Nu,)

            # 2) pc feats on unique coords
            X_unique, hit_u = query_sparse_feats_by_join(st_layer, q_unique)   # (Nu2,Cpc) on GPU
            # （可选）看下 unique 命中数
            # print(f"[JOIN-U] {pl_name}: hit {int(hit_u.sum())}/{hit_u.numel()} stride={st_layer.tensor_stride} Nu2={q_unique.shape[0]} C={tuple(st_layer.F.shape)}")
            assert int(hit_u.sum()) == q_unique.shape[0]
            
            # 3) img feats: 把每点的 img feat 聚合到 unique voxel
            Y_unique_dict = {k: group_mean(v, inv, q_unique.shape[0]) for k, v in img_feats.items()}  # (Nu2,Cimg)

            for k, Yu in Y_unique_dict.items():
                sd = Yu.float().std(0)
                print(pl_name, k, "Y std min/med =", float(sd.min()), float(sd.median()), "num<1e-3 =", int((sd < 1e-3).sum()))
    
            # 4) CKA on unique voxels
            Nu2 = q_unique.shape[0]
            # row = {}

            # per-layer rows
            row_cka = {}
            row_r2x2y = {}
            row_r2y2x = {}

            if Nu2 < 128:
                for ikey in img_feats.keys():
                    row_cka[ikey] = float("nan")
                    row_r2x2y[ikey] = float("nan")
                    row_r2y2x[ikey] = float("nan")
                cka_map[pl_name] = row_cka
                r2_x2y_map[pl_name] = row_r2x2y
                r2_y2x_map[pl_name] = row_r2y2x
                continue

            sel2 = torch.randperm(Nu2, device=X_unique.device)[:min(2048, Nu2)]

            X = X_unique[sel2].float()
            valid_x = (X.abs().sum(1) > 0)

            r2_row_x2y = {}
            r2_row_y2x = {}

            for ikey, Y_u in Y_unique_dict.items():
                Y = Y_u.to(X.device)[sel2].float()
                valid = valid_x & (Y.abs().sum(1) > 0)

                # --- CKA ---
                row_cka[ikey] = float(linear_cka(X[valid], Y[valid]).item()) if valid.sum() >= 64 else float("nan")

                # --- R2 (redundancy / linear predictability) ---
                if valid.sum() >= 256:
                    # r2x2y, r2y2x = linear_predictability_r2(
                    #     X[valid], Y[valid],
                    #     train_ratio=0.8, l2=1e-1, seed=int(scene_idx * 1000 + anno_idx),
                    #     std_thr=1e-3, ss_tot_thr=1e-6
                    # )

                    r2x2y, r2y2x = linear_predictability_r2(
                        X[valid], Y[valid],
                        train_ratio=0.8, l2=1e-1,
                        seed=int(scene_idx * 1000 + anno_idx),
                        std_thr=1e-3, ss_tot_thr=1e-6,
                        clip_min=-1.0
                    )
                else:
                    r2x2y, r2y2x = float("nan"), float("nan")

                r2_row_x2y[ikey] = r2x2y
                r2_row_y2x[ikey] = r2y2x

            cka_map[pl_name] = row_cka
            r2_x2y_map[pl_name] = r2_row_x2y
            r2_y2x_map[pl_name] = r2_row_y2x

        print(f"[CKA] scene {scene_idx} anno {anno_idx}: {cka_map}")
        print(f"[R2 x2y] scene {scene_idx} anno {anno_idx}: {r2_x2y_map}")
        print(f"[R2 y2x] scene {scene_idx} anno {anno_idx}: {r2_y2x_map}")

        # ---- 记录到 JSON results ----
        cka_results["records"].append({
            "scene_idx": int(scene_idx),
            "anno_idx": int(anno_idx),
            "split": cfgs.split,
            "cka": cka_map,
            "r2_x2y": r2_x2y_map,
            "r2_y2x": r2_y2x_map
        })

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


for scene_idx in scene_list:
    inference(scene_idx)
    _save_json(json_path, cka_results)
    print(f"[CKA-JSON] saved to {json_path} (records={len(cka_results['records'])})")