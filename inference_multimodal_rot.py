#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
import argparse
from typing import Dict, List, Optional, Sequence, Tuple
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))

from dataset.ignet_multi_dataset import GraspNetMultiDataset, collate_fn, load_grasp_labels
from models.IGNet_v0_9 import IGNet

# =========================================================
# Basic utils
# =========================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_device(batch: Dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt
    sd = strip_module_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[CKPT] loaded from {ckpt_path}")
    print(f"[CKPT] missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 0:
        print("[CKPT][missing]", missing[:20])
    if len(unexpected) > 0:
        print("[CKPT][unexpected]", unexpected[:20])


# =========================================================
# Collate
# =========================================================

def collate_fn_graspnet_multi(batch: List[Dict]):
    """
    保持 scene-level list labels，不去 default_collate 它们。
    其余数值型数组/张量按 batch 维堆叠。
    """
    out = {}
    keys = batch[0].keys()

    keep_as_list = {
        "object_poses_list",
        "grasp_points_list",
        "grasp_offsets_list",
        "grasp_labels_list",
        "scene",
        "frameid",
        "index",
    }

    for k in keys:
        vals = [b[k] for b in batch]
        if k in keep_as_list:
            out[k] = vals
            continue

        v0 = vals[0]

        if torch.is_tensor(v0):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, np.ndarray):
            # 数值型 ndarray -> tensor
            if v0.dtype.kind in ("f", "i", "u", "b"):
                out[k] = torch.from_numpy(np.stack(vals, axis=0))
            else:
                out[k] = vals
        elif isinstance(v0, (int, np.integer)):
            out[k] = torch.tensor(vals, dtype=torch.long)
        elif isinstance(v0, (float, np.floating)):
            out[k] = torch.tensor(vals, dtype=torch.float32)
        else:
            out[k] = vals

    return out


# =========================================================
# Feature gather
# =========================================================

@torch.no_grad()
def gather_local_img_feat_from_model(
    model: torch.nn.Module,
    img: torch.Tensor,          # (B,3,448,448)
    img_idxs: torch.Tensor,     # (B,N), flattened on 448x448
    feat_level: str = "p2",
):
    """
    不改 forward，直接复用 model.img_backbone 和 model._gather_2d_to_points
    再跑一次 2D backbone，取指定 pyramid level 的局部视觉特征。
    """
    assert hasattr(model, "img_backbone"), "model has no img_backbone"
    assert hasattr(model, "_gather_2d_to_points"), "model has no _gather_2d_to_points"
    assert feat_level in ["p1", "p2", "p4", "p8", "p16"]

    pyr = model.img_backbone(img, return_pyramid=True)
    feat2d = pyr[feat_level]  # (B,C,Hf,Wf)
    H0, W0 = img.shape[-2], img.shape[-1]
    feat_per_point = model._gather_2d_to_points(feat2d, img_idxs, base_hw=(H0, W0))  # (B,N,C)
    return feat_per_point


@torch.no_grad()
def gather_by_inds(feat: torch.Tensor, inds: torch.Tensor):
    """
    feat: (B,N,C)
    inds: (B,M)
    -> (B,M,C)
    """
    B, N, C = feat.shape
    M = inds.shape[1]
    return torch.gather(feat, 1, inds.unsqueeze(-1).expand(-1, -1, C)).contiguous()


# =========================================================
# Dense target -> probability
# =========================================================

def _safe_log(x: torch.Tensor, eps: float = 1e-12):
    return torch.log(x.clamp_min(eps))


def entropy_abs(p: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Absolute entropy: H(p) = -sum p log p
    """
    return -(p * _safe_log(p, eps=eps)).sum(dim=dim)


def entropy_normalized(p: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Normalized entropy in [0,1]:
        H(p) / log(K)
    where K is support size on `dim`.
    """
    k = p.shape[dim]
    h = entropy_abs(p, dim=dim, eps=eps)
    denom = math.log(max(k, 2))
    return h / denom


def effective_support_size(p: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    """
    Effective support size = exp(H(p)).
    Can be interpreted as the 'effective number of active bins'.
    """
    h = entropy_abs(p, dim=dim, eps=eps)
    return torch.exp(h)


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12):
    m = 0.5 * (p + q)
    kl_pm = (p * (_safe_log(p, eps) - _safe_log(m, eps))).sum(dim=-1)
    kl_qm = (q * (_safe_log(q, eps) - _safe_log(m, eps))).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def reshape_full_score(score: torch.Tensor) -> torch.Tensor:
    """
    支持:
      (B,M,R,D)
      (B,M,V,A,D) -> (B,M,R,D)
    """
    if score.ndim == 4:
        return score
    elif score.ndim == 5:
        B, M, V, A, D = score.shape
        return score.reshape(B, M, V * A, D)
    else:
        raise ValueError(
            f"Unsupported dense score shape: {tuple(score.shape)}; "
            f"expect (B,M,R,D) or (B,M,V,A,D)"
        )


def reshape_mask_like(mask: Optional[torch.Tensor], score_rd: torch.Tensor):
    if mask is None:
        return None
    if mask.shape == score_rd.shape:
        return mask
    if mask.ndim == 5 and score_rd.ndim == 4:
        B, M, V, A, D = mask.shape
        return mask.reshape(B, M, V * A, D)
    raise ValueError(
        f"Mask shape {tuple(mask.shape)} incompatible with score {tuple(score_rd.shape)}"
    )


def convert_dense_score(
    score: torch.Tensor,
    mode: str = "positive",
):
    """
    TODO:
    这个地方很重要，必须和你的 GT 语义一致。

    mode='positive':
        表示 score 越大越好，直接使用。
        适合你已经在 helper 里把 GT 转成正向 grasp quality 的情况。

    mode='graspnet_friction':
        表示原始值是 friction coefficient，数值越小越好。
        这里用一个常见的单调变换: quality = max(0, 1.1 - score)
        但这要求 invalid / collision 必须由 mask 去掉，否则 raw 0 会变成最大值。
    """
    if mode == "positive":
        q = score
    elif mode == "graspnet_friction":
        # valid scores are discrete in {0.1, 0.2, ..., 1.0}
        # smaller score = better grasp
        q = 1.1 - score
        q = torch.clamp(q, min=0.0)
    else:
        raise ValueError(f"Unknown score_mode={mode}")
    return q


def build_full_and_rot_probs(
    dense_score: torch.Tensor,
    dense_mask: Optional[torch.Tensor] = None,
    score_mode: str = "positive",
    eps: float = 1e-12,
):
    """
    dense_score:
      (B,M,V,A,D) or (B,M,R,D)
    dense_mask:
      same shape, optional

    Returns:
      full_prob: (B,M,R*D)
      rot_prob : (B,M,R)
      valid    : (B,M)
    """
    score_rd = reshape_full_score(dense_score).float()
    mask_rd = reshape_mask_like(dense_mask, score_rd)

    score_rd = convert_dense_score(score_rd, mode=score_mode)

    if mask_rd is not None:
        score_rd = score_rd * mask_rd.float()

    score_rd = torch.clamp(score_rd, min=0.0)

    B, M, R, D = score_rd.shape

    full = score_rd.reshape(B, M, R * D)
    full_sum = full.sum(dim=-1, keepdim=True)
    full_prob = full / full_sum.clamp_min(eps)

    rot = score_rd.sum(dim=-1)  # (B,M,R)
    rot_sum = rot.sum(dim=-1, keepdim=True)
    rot_prob = rot / rot_sum.clamp_min(eps)

    valid = (full_sum.squeeze(-1) > eps) & (rot_sum.squeeze(-1) > eps)
    return full_prob, rot_prob, valid


# =========================================================
# Offline analysis
# =========================================================

def knn_cosine_chunked(feat: np.ndarray, k: int = 16, chunk_size: int = 256) -> np.ndarray:
    x = torch.from_numpy(feat).float()
    x = F.normalize(x, dim=1)
    N = x.shape[0]
    outs = []

    with torch.no_grad():
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            sim = x[s:e] @ x.T  # (chunk,N)
            rows = torch.arange(s, e)
            sim[torch.arange(e - s), rows] = -1e9
            idx = torch.topk(sim, k=k, dim=1, largest=True, sorted=True).indices
            outs.append(idx.cpu().numpy())

    return np.concatenate(outs, axis=0)


def append_chunk_to_lists(
    feat_list: list,
    full_prob_list: list,
    rot_prob_list: list,
    scene_id_list: list,
    frame_id_list: list,
    feat: np.ndarray,
    full_prob: np.ndarray,
    rot_prob: np.ndarray,
    scene_ids,
    frame_ids,
):
    feat_list.append(feat.astype(np.float16, copy=False))
    full_prob_list.append(full_prob.astype(np.float16, copy=False))
    rot_prob_list.append(rot_prob.astype(np.float16, copy=False))
    scene_id_list.extend([str(x) for x in scene_ids])
    frame_id_list.extend([int(x) for x in frame_ids])


def save_full_pool(
    out_path: str,
    feat_list: list,
    full_prob_list: list,
    rot_prob_list: list,
    scene_id_list: list,
    frame_id_list: list,
):
    assert len(feat_list) > 0, "No samples collected."
    mkdir(os.path.dirname(out_path))

    feat = np.concatenate(feat_list, axis=0)
    full_prob = np.concatenate(full_prob_list, axis=0)
    rot_prob = np.concatenate(rot_prob_list, axis=0)
    scene_id = np.asarray(scene_id_list, dtype=object)
    frame_id = np.asarray(frame_id_list, dtype=np.int32)

    np.savez_compressed(
        out_path,
        feat=feat,
        full_prob=full_prob,
        rot_prob=rot_prob,
        scene_id=scene_id,
        frame_id=frame_id,
        meta=json.dumps({
            "n_samples": int(feat.shape[0]),
            "feat_dim": int(feat.shape[1]),
            "full_dim": int(full_prob.shape[1]),
            "rot_dim": int(rot_prob.shape[1]),
        }),
    )
    print(f"[Pool] saved {feat.shape[0]} samples -> {out_path}")
    
    
def analyze_visual_neighbor_dispersion(
    pool_npz: str,
    out_json: str,
    k: int = 16,
    chunk_size: int = 256,
):
    data = np.load(pool_npz, allow_pickle=True)
    feat = data["feat"].astype(np.float32)
    full_prob = data["full_prob"].astype(np.float32)
    rot_prob = data["rot_prob"].astype(np.float32)

    N = feat.shape[0]
    print(f"[Analyze] loaded {N} samples")
    nn_idx = knn_cosine_chunked(feat, k=k, chunk_size=chunk_size)
    print(f"[Analyze] kNN done, k={k}")

    full_prob_t = torch.from_numpy(full_prob)
    rot_prob_t = torch.from_numpy(rot_prob)
    nn_idx_t = torch.from_numpy(nn_idx).long()

    # per-anchor stats
    full_ent_norm_list, rot_ent_norm_list = [], []
    full_ent_abs_list, rot_ent_abs_list = [], []
    full_eff_list, rot_eff_list = [], []
    full_js_list, rot_js_list = [], []
    full_agree_list, rot_agree_list = [], []

    # neighbor-mean distribution stats
    full_nei_mean_ent_norm_list, rot_nei_mean_ent_norm_list = [], []
    full_nei_mean_ent_abs_list, rot_nei_mean_ent_abs_list = [], []
    full_nei_mean_eff_list, rot_nei_mean_eff_list = [], []

    full_top1 = torch.argmax(full_prob_t, dim=1)
    rot_top1 = torch.argmax(rot_prob_t, dim=1)

    with torch.no_grad():
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            idx_chunk = nn_idx_t[s:e]  # (Bc,k)
            Bc = e - s

            full_anchor = full_prob_t[s:e]      # (Bc,Kf)
            rot_anchor = rot_prob_t[s:e]        # (Bc,Kr)
            full_nei = full_prob_t[idx_chunk]   # (Bc,k,Kf)
            rot_nei = rot_prob_t[idx_chunk]     # (Bc,k,Kr)

            # --------------------------------------------------
            # A) Anchor distribution concentration
            # --------------------------------------------------
            full_ent_abs = entropy_abs(full_anchor)
            rot_ent_abs = entropy_abs(rot_anchor)

            full_ent_norm = entropy_normalized(full_anchor)
            rot_ent_norm = entropy_normalized(rot_anchor)

            full_eff = torch.exp(full_ent_abs)
            rot_eff = torch.exp(rot_ent_abs)

            # --------------------------------------------------
            # B) Neighbor-mean distribution concentration
            #    mean over {anchor + k neighbors}
            # --------------------------------------------------
            full_set = torch.cat([full_anchor.unsqueeze(1), full_nei], dim=1)  # (Bc,k+1,Kf)
            rot_set = torch.cat([rot_anchor.unsqueeze(1), rot_nei], dim=1)     # (Bc,k+1,Kr)

            full_mean = full_set.mean(dim=1)
            rot_mean = rot_set.mean(dim=1)

            full_mean_ent_abs = entropy_abs(full_mean)
            rot_mean_ent_abs = entropy_abs(rot_mean)

            full_mean_ent_norm = entropy_normalized(full_mean)
            rot_mean_ent_norm = entropy_normalized(rot_mean)

            full_mean_eff = torch.exp(full_mean_ent_abs)
            rot_mean_eff = torch.exp(rot_mean_ent_abs)

            # --------------------------------------------------
            # C) Anchor-neighbor JS divergence
            # --------------------------------------------------
            full_anchor_exp = full_anchor.unsqueeze(1).expand_as(full_nei)
            rot_anchor_exp = rot_anchor.unsqueeze(1).expand_as(rot_nei)

            full_js = js_divergence(
                full_anchor_exp.reshape(-1, full_anchor.shape[-1]),
                full_nei.reshape(-1, full_anchor.shape[-1]),
            ).reshape(Bc, k).mean(dim=1)

            rot_js = js_divergence(
                rot_anchor_exp.reshape(-1, rot_anchor.shape[-1]),
                rot_nei.reshape(-1, rot_anchor.shape[-1]),
            ).reshape(Bc, k).mean(dim=1)

            # --------------------------------------------------
            # D) Top-1 agreement
            # --------------------------------------------------
            full_agree = (full_top1[idx_chunk] == full_top1[s:e].unsqueeze(1)).float().mean(dim=1)
            rot_agree = (rot_top1[idx_chunk] == rot_top1[s:e].unsqueeze(1)).float().mean(dim=1)

            # collect
            full_ent_abs_list.append(full_ent_abs.cpu().numpy())
            rot_ent_abs_list.append(rot_ent_abs.cpu().numpy())
            full_ent_norm_list.append(full_ent_norm.cpu().numpy())
            rot_ent_norm_list.append(rot_ent_norm.cpu().numpy())
            full_eff_list.append(full_eff.cpu().numpy())
            rot_eff_list.append(rot_eff.cpu().numpy())

            full_nei_mean_ent_abs_list.append(full_mean_ent_abs.cpu().numpy())
            rot_nei_mean_ent_abs_list.append(rot_mean_ent_abs.cpu().numpy())
            full_nei_mean_ent_norm_list.append(full_mean_ent_norm.cpu().numpy())
            rot_nei_mean_ent_norm_list.append(rot_mean_ent_norm.cpu().numpy())
            full_nei_mean_eff_list.append(full_mean_eff.cpu().numpy())
            rot_nei_mean_eff_list.append(rot_mean_eff.cpu().numpy())

            full_js_list.append(full_js.cpu().numpy())
            rot_js_list.append(rot_js.cpu().numpy())
            full_agree_list.append(full_agree.cpu().numpy())
            rot_agree_list.append(rot_agree.cpu().numpy())

    # concat
    full_ent_abs = np.concatenate(full_ent_abs_list, axis=0)
    rot_ent_abs = np.concatenate(rot_ent_abs_list, axis=0)
    full_ent_norm = np.concatenate(full_ent_norm_list, axis=0)
    rot_ent_norm = np.concatenate(rot_ent_norm_list, axis=0)
    full_eff = np.concatenate(full_eff_list, axis=0)
    rot_eff = np.concatenate(rot_eff_list, axis=0)

    full_nei_mean_ent_abs = np.concatenate(full_nei_mean_ent_abs_list, axis=0)
    rot_nei_mean_ent_abs = np.concatenate(rot_nei_mean_ent_abs_list, axis=0)
    full_nei_mean_ent_norm = np.concatenate(full_nei_mean_ent_norm_list, axis=0)
    rot_nei_mean_ent_norm = np.concatenate(rot_nei_mean_ent_norm_list, axis=0)
    full_nei_mean_eff = np.concatenate(full_nei_mean_eff_list, axis=0)
    rot_nei_mean_eff = np.concatenate(rot_nei_mean_eff_list, axis=0)

    full_js = np.concatenate(full_js_list, axis=0)
    rot_js = np.concatenate(rot_js_list, axis=0)
    full_agree = np.concatenate(full_agree_list, axis=0)
    rot_agree = np.concatenate(rot_agree_list, axis=0)

    report = {
        "n_samples": int(N),
        "k": int(k),

        # --------------------------------------------------
        # Anchor distribution
        # --------------------------------------------------
        "full_anchor_entropy_abs_mean": float(full_ent_abs.mean()),
        "full_anchor_entropy_abs_std": float(full_ent_abs.std()),
        "rot_anchor_entropy_abs_mean": float(rot_ent_abs.mean()),
        "rot_anchor_entropy_abs_std": float(rot_ent_abs.std()),

        "full_anchor_entropy_norm_mean": float(full_ent_norm.mean()),
        "full_anchor_entropy_norm_std": float(full_ent_norm.std()),
        "rot_anchor_entropy_norm_mean": float(rot_ent_norm.mean()),
        "rot_anchor_entropy_norm_std": float(rot_ent_norm.std()),

        "full_anchor_effective_support_mean": float(full_eff.mean()),
        "full_anchor_effective_support_std": float(full_eff.std()),
        "rot_anchor_effective_support_mean": float(rot_eff.mean()),
        "rot_anchor_effective_support_std": float(rot_eff.std()),

        # --------------------------------------------------
        # Neighbor-mean distribution
        # --------------------------------------------------
        "full_neighbor_mean_entropy_abs_mean": float(full_nei_mean_ent_abs.mean()),
        "full_neighbor_mean_entropy_abs_std": float(full_nei_mean_ent_abs.std()),
        "rot_neighbor_mean_entropy_abs_mean": float(rot_nei_mean_ent_abs.mean()),
        "rot_neighbor_mean_entropy_abs_std": float(rot_nei_mean_ent_abs.std()),

        "full_neighbor_mean_entropy_norm_mean": float(full_nei_mean_ent_norm.mean()),
        "full_neighbor_mean_entropy_norm_std": float(full_nei_mean_ent_norm.std()),
        "rot_neighbor_mean_entropy_norm_mean": float(rot_nei_mean_ent_norm.mean()),
        "rot_neighbor_mean_entropy_norm_std": float(rot_nei_mean_ent_norm.std()),

        "full_neighbor_mean_effective_support_mean": float(full_nei_mean_eff.mean()),
        "full_neighbor_mean_effective_support_std": float(full_nei_mean_eff.std()),
        "rot_neighbor_mean_effective_support_mean": float(rot_nei_mean_eff.mean()),
        "rot_neighbor_mean_effective_support_std": float(rot_nei_mean_eff.std()),

        # --------------------------------------------------
        # Neighbor consistency
        # --------------------------------------------------
        "full_anchor_neighbor_js_mean": float(full_js.mean()),
        "full_anchor_neighbor_js_std": float(full_js.std()),
        "rot_anchor_neighbor_js_mean": float(rot_js.mean()),
        "rot_anchor_neighbor_js_std": float(rot_js.std()),

        "full_anchor_neighbor_top1_agreement_mean": float(full_agree.mean()),
        "full_anchor_neighbor_top1_agreement_std": float(full_agree.std()),
        "rot_anchor_neighbor_top1_agreement_mean": float(rot_agree.mean()),
        "rot_anchor_neighbor_top1_agreement_std": float(rot_agree.std()),

        # --------------------------------------------------
        # Relative summaries
        # --------------------------------------------------
        "anchor_entropy_abs_reduction_pct": float(
            (full_ent_abs.mean() - rot_ent_abs.mean()) / max(full_ent_abs.mean(), 1e-12) * 100.0
        ),
        "anchor_effective_support_reduction_pct": float(
            (full_eff.mean() - rot_eff.mean()) / max(full_eff.mean(), 1e-12) * 100.0
        ),
        "neighbor_mean_entropy_abs_reduction_pct": float(
            (full_nei_mean_ent_abs.mean() - rot_nei_mean_ent_abs.mean()) / max(full_nei_mean_ent_abs.mean(), 1e-12) * 100.0
        ),
        "neighbor_mean_effective_support_reduction_pct": float(
            (full_nei_mean_eff.mean() - rot_nei_mean_eff.mean()) / max(full_nei_mean_eff.mean(), 1e-12) * 100.0
        ),
        "js_relative_reduction_pct": float(
            (full_js.mean() - rot_js.mean()) / max(full_js.mean(), 1e-12) * 100.0
        ),
        "top1_agreement_relative_gain_pct": float(
            (rot_agree.mean() - full_agree.mean()) / max(full_agree.mean(), 1e-12) * 100.0
        ),
    }

    mkdir(os.path.dirname(out_json))
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("[Analyze] report:")
    print(json.dumps(report, indent=2))
    print(f"[Analyze] saved -> {out_json}")


# =========================================================
# Main
# =========================================================

def build_dataset(args):
    valid_obj_idxs, grasp_labels = load_grasp_labels(args.dataset_root)

    ds = GraspNetMultiDataset(
        root=args.dataset_root,
        valid_obj_idxs=valid_obj_idxs,
        grasp_labels=grasp_labels,
        camera=args.camera,
        split=args.split,
        num_points=args.num_points,
        remove_outlier=args.remove_outlier,
        voxel_size=args.voxel_size,
        remove_invisible=args.remove_invisible,
        augment=False,
        load_label=True,  # exp1 必须开 GT
    )
    return ds


def build_model(args):
    model = IGNet(
        m_point=args.m_point,
        num_view=args.num_view,
        num_angle=args.num_angle,
        num_depth=args.num_depth,
        seed_feat_dim=args.seed_feat_dim,
        img_feat_dim=args.img_feat_dim,
        is_training=True,              # 这里要 True，确保 forward 走 GT 处理分支
        multi_scale_grouping=args.multi_scale_grouping,
        fuse_type=args.fuse_type,
    )
    return model


def main():
    parser = argparse.ArgumentParser("Experiment-1: visual-neighbor label dispersion")

    # dataset / model
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--camera", type=str, default="realsense", choices=["kinect", "realsense"])
    parser.add_argument("--split", type=str, default="test", choices=["test", "test_seen", "test_similar", "test_novel"])
    parser.add_argument("--checkpoint", type=str, required=True)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=1)   # 建议先 1
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num_points", type=int, default=20000)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--remove_outlier", action="store_true")
    parser.add_argument("--remove_invisible", action="store_true")

    # model args
    parser.add_argument("--fuse_type", type=str, default="intermediate")
    parser.add_argument("--m_point", type=int, default=1024)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--num_angle", type=int, default=12)
    parser.add_argument("--num_depth", type=int, default=4)
    parser.add_argument("--seed_feat_dim", type=int, default=256)
    parser.add_argument("--img_feat_dim", type=int, default=64)
    parser.add_argument("--multi_scale_grouping", action="store_true")

    # exp1
    parser.add_argument("--exp1_feat_level", type=str, default="p2", choices=["p1", "p2", "p4", "p8", "p16"])
    parser.add_argument("--pool_path", type=str, required=True)
    parser.add_argument("--report_path", type=str, required=True)
    parser.add_argument("--capacity", type=int, default=8000)
    parser.add_argument("--per_frame_max", type=int, default=8)
    parser.add_argument("--knn_k", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=256)

    # dense GT keys
    parser.add_argument("--full_score_key", type=str, default="exp1_full_score")
    parser.add_argument("--full_mask_key", type=str, default="exp1_full_mask")
    parser.add_argument("--score_mode", type=str, default="positive", choices=["positive", "graspnet_friction"])

    # misc
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strict_load", action="store_true")

    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Take one sample every N valid samples within each frame. "
            "For example, 10 means keep idx[::10]."
    )

    args = parser.parse_args()

    assert args.fuse_type == "intermediate", "This script is currently written for fuse_type=intermediate."

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    ds_full = build_dataset(args)
    sampled_indices = list(range(0, len(ds_full), args.sample_interval))
    ds = Subset(ds_full, sampled_indices)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn_graspnet_multi,
    )
    print(f"[Data] split={args.split}, len={len(ds)}")

    model = build_model(args)
    load_checkpoint(model, args.checkpoint, strict=args.strict_load)
    model = model.to(device)
    model.eval()

    feat_list = []
    full_prob_list = []
    rot_prob_list = []
    scene_id_list = []
    frame_id_list = []

    t0 = time.time()
    n_ok = 0
    n_skip = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch = to_device(batch, device)

            try:
                # --------------- forward ---------------
                end_points = model(batch)

                # --------------- local visual evidence ---------------
                img = batch["img"]               # (B,3,448,448)
                img_idxs = batch["img_idxs"]     # (B,N)
                local_feat_all = gather_local_img_feat_from_model(
                    model=model,
                    img=img,
                    img_idxs=img_idxs,
                    feat_level=args.exp1_feat_level,
                )  # (B,N,C)

                # selected seeds
                assert "graspable_inds" in end_points, "forward must expose end_points['graspable_inds']"
                inds = end_points["graspable_inds"].long()   # (B,M)
                local_feat_sel = gather_by_inds(local_feat_all, inds)  # (B,M,C)

                if args.full_score_key not in end_points:
                    raise KeyError(
                        f"Missing end_points['{args.full_score_key}'].\n"
                        f"TODO: expose aligned dense GT full target before top-rot slicing."
                    )

                dense_score = end_points[args.full_score_key]
                dense_mask = end_points.get(args.full_mask_key, None)

                # print("dense_score:", dense_score.shape, dense_score.min().item(), dense_score.max().item())
                # print("dense_mask :", dense_mask.shape, dense_mask.float().mean().item())

                # valid_scores = dense_score[dense_mask > 0]

                # print("dense_mask ratio:", dense_mask.float().mean().item())
                # print("valid_scores num:", valid_scores.numel())

                # if valid_scores.numel() > 0:
                #     print("valid min/max:", valid_scores.min().item(), valid_scores.max().item())
                #     print("valid mean/std:", valid_scores.mean().item(), valid_scores.std().item())

                #     uq = torch.unique(valid_scores)
                #     print("num unique:", uq.numel())
                #     print("first unique values:", uq[:20].cpu().tolist())

                #     q = torch.quantile(valid_scores, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=valid_scores.device))
                #     print("quantiles:", q.cpu().tolist())
    
                full_prob, rot_prob, valid = build_full_and_rot_probs(
                    dense_score=dense_score,
                    dense_mask=dense_mask,
                    score_mode=args.score_mode,
                )

                # ---------------- sanity check: only print once or a few times ----------------
                if batch_idx < 3:
                    print("full_prob shape:", full_prob.shape)
                    print("rot_prob  shape:", rot_prob.shape)
                    print("valid ratio:", valid.float().mean().item())

                    vp = valid[0]
                    if vp.any():
                        fp = full_prob[0][vp]   # (Nv, 14400)
                        rp = rot_prob[0][vp]    # (Nv, 3600)

                        full_ent_abs = entropy_abs(fp)
                        rot_ent_abs = entropy_abs(rp)

                        full_ent_norm = entropy_normalized(fp)
                        rot_ent_norm = entropy_normalized(rp)

                        full_eff = effective_support_size(fp)
                        rot_eff = effective_support_size(rp)

                        print("full_prob max mean:", fp.max(dim=-1).values.mean().item())
                        print("rot_prob  max mean:", rp.max(dim=-1).values.mean().item())

                        print("full entropy abs mean:", full_ent_abs.mean().item())
                        print("rot  entropy abs mean:", rot_ent_abs.mean().item())

                        print("full entropy norm mean:", full_ent_norm.mean().item())
                        print("rot  entropy norm mean:", rot_ent_norm.mean().item())

                        print("full effective support mean:", full_eff.mean().item())
                        print("rot  effective support mean:", rot_eff.mean().item())
                                    
                B, M, C = local_feat_sel.shape

                # --------------- per-frame subsample ---------------
                feat_np_list = []
                full_np_list = []
                rot_np_list = []
                scene_list = []
                frame_list = []

                scenes = batch["scene"]     # list[str], len=B
                frameids = batch["frameid"] # list[int], len=B  or tensor/list
                if torch.is_tensor(frameids):
                    frameids = frameids.cpu().tolist()

                for b in range(B):
                    idx = torch.where(valid[b])[0]
                    if idx.numel() == 0:
                        continue

                    if args.per_frame_max > 0 and idx.numel() > args.per_frame_max:
                        perm = torch.randperm(idx.numel(), device=idx.device)[:args.per_frame_max]
                        idx = idx[perm]

                    feat_np_list.append(local_feat_sel[b, idx].detach().cpu().numpy().astype(np.float32))
                    full_np_list.append(full_prob[b, idx].detach().cpu().numpy().astype(np.float32))
                    rot_np_list.append(rot_prob[b, idx].detach().cpu().numpy().astype(np.float32))

                    scene_list.extend([str(scenes[b])] * idx.numel())
                    frame_list.extend([int(frameids[b])] * idx.numel())

                if len(feat_np_list) == 0:
                    n_skip += 1
                    continue

                feat_np = np.concatenate(feat_np_list, axis=0)
                full_np = np.concatenate(full_np_list, axis=0)
                rot_np = np.concatenate(rot_np_list, axis=0)

                append_chunk_to_lists(
                    feat_list=feat_list,
                    full_prob_list=full_prob_list,
                    rot_prob_list=rot_prob_list,
                    scene_id_list=scene_id_list,
                    frame_id_list=frame_id_list,
                    feat=feat_np,
                    full_prob=full_np,
                    rot_prob=rot_np,
                    scene_ids=scene_list,
                    frame_ids=frame_list,
                )

                n_ok += 1
                if n_ok % 10 == 0:
                    n_collected = sum(x.shape[0] for x in feat_list) if len(feat_list) > 0 else 0
                    print(
                        f"[processed={n_ok}] raw_batch_idx={batch_idx+1}/{len(loader)}, "
                        f"skip={n_skip}, collected={n_collected}",
                        flush=True,
                    )
                    
            except Exception as e:
                n_skip += 1
                print(f"[WARN] batch {batch_idx} failed: {repr(e)}")
                continue

    save_full_pool(
        out_path=args.pool_path,
        feat_list=feat_list,
        full_prob_list=full_prob_list,
        rot_prob_list=rot_prob_list,
        scene_id_list=scene_id_list,
        frame_id_list=frame_id_list,
    )

    analyze_visual_neighbor_dispersion(
        pool_npz=args.pool_path,
        out_json=args.report_path,
        k=args.knn_k,
        chunk_size=args.chunk_size,
    )
    
    dt = time.time() - t0
    print(f"[Done] time={dt/60.0:.1f} min, ok={n_ok}, skip={n_skip}")


if __name__ == "__main__":
    main()