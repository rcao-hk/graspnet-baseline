#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene-level MMGNet visualization for the framework figure.

Default target:
    GraspNet-1Billion / RealSense / scene_0100 / frame 0000

Three sequential Open3D windows are shown:
    1) Scene Points
    2) Grasp-Shaded Grouping
    3) Resulted Grasps

The camera view chosen interactively in Window 1 is reused for Windows 2/3.
Screenshots and the chosen Open3D camera parameters are saved automatically.

Important:
- Preprocessing follows the current scene-level inference: workspace crop ->
  sample scene points -> RGB crop/resize to 448x448 -> pixel-point indices.
- The grouping visualization applies grasp NMS first, then selects up to
  ``--group_num`` NMS-surviving grasps. Dense predicted-foreground scene points falling inside
  the corresponding grasp-shaded cuboids are colored yellow by default. The model's
  exact K sampled seed points are used only to verify/infer the cuboid frame.
- The final visualization applies collision filtering, official grasp NMS,
  score sorting, and then shows Top-M grasps.
- Gripper geometries are rendered through the official graspnetAPI Grasp
  Open3D visualization routine, preserving score-based colors.
"""

import os
import sys
import re
import glob
import random
import argparse
import importlib
from pathlib import Path

import numpy as np

# graspnetAPI compatibility with recent NumPy.
if not hasattr(np, "int"):
    np.int = np.int32
if not hasattr(np, "float"):
    np.float = np.float64
if not hasattr(np, "bool"):
    np.bool = np.bool_

import torch
from PIL import Image
import scipy.io as scio
import open3d as o3d
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))

from graspnetAPI import GraspGroup
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.data_utils import (
    CameraInfo,
    create_point_cloud_from_depth_image,
    get_workspace_mask,
    sample_points,
)

RESIZE_HW = (448, 448)
IMG_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(RESIZE_HW),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def setup_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def crop_box_from_mask(mask):
    H, W = mask.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, W, H
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return int(x0), int(y0), int(x1), int(y1)


def get_resized_idxs_from_flat_crop(pix_flat, orig_hw, crop_box, out_hw=RESIZE_HW):
    """Map original HxW flat pixel indices into the cropped/resized image."""
    H, W = orig_hw
    outH, outW = out_hw
    x0, y0, x1, y1 = crop_box
    cw, ch = (x1 - x0), (y1 - y0)

    ys, xs = np.unravel_index(pix_flat, (H, W))
    xs = xs.astype(np.float32) - float(x0)
    ys = ys.astype(np.float32) - float(y0)
    xs = np.clip(xs, 0, cw - 1e-6)
    ys = np.clip(ys, 0, ch - 1e-6)

    xf = np.floor(xs * (outW / float(cw))).astype(np.int64)
    yf = np.floor(ys * (outH / float(ch))).astype(np.int64)
    xf = np.clip(xf, 0, outW - 1)
    yf = np.clip(yf, 0, outH - 1)
    return (yf * outW + xf).astype(np.int64)


def load_scene_frame(cfg):
    scene_idx, anno_idx, camera = cfg.scene, cfg.anno, cfg.camera
    root = cfg.dataset_root

    rgb_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/rgb/{anno_idx:04d}.png")
    depth_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/depth/{anno_idx:04d}.png")
    label_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/label/{anno_idx:04d}.png")
    meta_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/meta/{anno_idx:04d}.mat")
    poses_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/camera_poses.npy")
    align_path = os.path.join(root, f"scenes/scene_{scene_idx:04d}/{camera}/cam0_wrt_table.npy")

    for p in [rgb_path, depth_path, label_path, meta_path, poses_path, align_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    color = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    seg = np.array(Image.open(label_path))
    meta = scio.loadmat(meta_path)

    intrinsics = meta["intrinsic_matrix"]
    factor_depth = float(np.asarray(meta["factor_depth"]).squeeze())
    H, W = depth.shape
    camera_info = CameraInfo(
        W, H,
        intrinsics[0][0], intrinsics[1][1],
        intrinsics[0][2], intrinsics[1][2],
        factor_depth,
    )
    cloud_org = create_point_cloud_from_depth_image(depth, camera_info, organized=True)

    camera_poses = np.load(poses_path)
    align_mat = np.load(align_path)
    trans = np.dot(align_mat, camera_poses[anno_idx])
    workspace_mask = get_workspace_mask(
        cloud_org, seg, trans=trans, organized=True, outlier=0.02
    )
    mask = (depth > 0) & workspace_mask

    cloud_masked = cloud_org[mask]
    color_masked = color[mask]
    if cloud_masked.shape[0] == 0:
        raise RuntimeError("No valid scene points after workspace masking.")

    # Network input: sampled full-scene points.
    idxs = np.asarray(sample_points(len(cloud_masked), cfg.num_point), dtype=np.int64)
    cloud_sampled = cloud_masked[idxs].astype(np.float32)
    color_sampled = color_masked[idxs].astype(np.float32)

    # Pixel-point correspondence for exactly the sampled points.
    valid_flat = np.flatnonzero(mask)
    pix_flat = valid_flat[idxs]
    crop_box = crop_box_from_mask(mask)
    x0, y0, x1, y1 = crop_box
    color_crop = color[y0:y1, x0:x1].copy()
    img = IMG_TRANSFORMS(color_crop)
    resized_idxs = get_resized_idxs_from_flat_crop(
        pix_flat, (H, W), crop_box, RESIZE_HW
    )

    return {
        "cloud_org": cloud_org,
        "cloud_masked": cloud_masked.astype(np.float32),
        "color_masked": color_masked.astype(np.float32),
        "cloud_sampled": cloud_sampled,
        "color_sampled": color_sampled,
        "img": img,
        "img_idxs": resized_idxs,
    }


def resolve_ckpt(cfg):
    if cfg.ckpt:
        ckpt = os.path.abspath(os.path.expanduser(cfg.ckpt))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        return ckpt

    if not cfg.network_name:
        raise ValueError("Provide --ckpt, or --network_name for automatic checkpoint lookup.")

    pattern = re.compile(
        rf"(epoch_{cfg.ckpt_epoch}_.+\.tar|checkpoint_{cfg.ckpt_epoch}\.tar|epoch{cfg.ckpt_epoch}\.tar)$"
    )
    ckpt_dir = os.path.join(cfg.ckpt_root, cfg.network_name, cfg.camera)
    for p in sorted(glob.glob(os.path.join(ckpt_dir, "*.tar"))):
        if pattern.search(os.path.basename(p)):
            return p
    raise FileNotFoundError(
        f"No epoch-{cfg.ckpt_epoch} checkpoint found under {ckpt_dir}"
    )


def build_model(cfg, device):
    model_mod = importlib.import_module(cfg.model_module)
    IGNet = getattr(model_mod, "IGNet")
    pred_decode = getattr(model_mod, "pred_decode")

    kwargs = dict(
        m_point=cfg.m_point,
        num_view=300,
        seed_feat_dim=cfg.seed_feat_dim,
        img_feat_dim=cfg.img_feat_dim,
        is_training=False,
        multi_scale_grouping=cfg.multi_scale_grouping,
        fuse_type=cfg.fuse_type,
        grouping_type=cfg.grouping_type,
    )
    if cfg.grouping_nsample is not None:
        kwargs["grouping_nsample"] = cfg.grouping_nsample

    net = IGNet(**kwargs).to(device).eval()
    ckpt_path = resolve_ckpt(cfg)
    print(f"[CKPT] {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    net.load_state_dict(state_dict, strict=True)
    return net, pred_decode, model_mod, ckpt_path


def make_end_points(scene_data, cfg, device):
    cloud = scene_data["cloud_sampled"]
    colors = scene_data["color_sampled"]
    cloud_tensor = torch.tensor(cloud, dtype=torch.float32, device=device)
    return {
        "point_clouds": cloud_tensor.unsqueeze(0),
        "cloud_colors": torch.tensor(colors, dtype=torch.float32, device=device).unsqueeze(0),
        "img": scene_data["img"].to(device).unsqueeze(0),
        "img_idxs": torch.tensor(
            scene_data["img_idxs"], dtype=torch.int64, device=device
        ).unsqueeze(0),
        "coors": torch.tensor(
            cloud / cfg.voxel_size, dtype=torch.int32, device=device
        ).unsqueeze(0),
        "feats": torch.ones_like(cloud_tensor, dtype=torch.float32).unsqueeze(0),
    }


@torch.no_grad()
def recover_exact_grouped_seed_indices(net, end_points, model_mod, query_seed_inds):
    """
    Recover the exact K sampled source seed IDs used by RectangularQueryAndGroup.

    We pass source IDs as a dummy one-channel point feature, invoke the same
    grouper as the model, then read those IDs back from the grouped output.
    """
    if getattr(net, "multi_scale_grouping", False):
        crop_module = net.crop_op_list[-1]  # largest scale = 1.0
        crop_scale = float(net.crop_scales[-1])
    else:
        crop_module = net.crop
        crop_scale = 1.0

    if getattr(crop_module, "grouping_type", None) != "rectangular":
        raise RuntimeError("Grasp-shaded visualization requires rectangular grouping.")

    seed_xyz = end_points["xyz_graspable"]  # [1,M,3]
    B, M, _ = seed_xyz.shape
    if B != 1:
        raise RuntimeError("Visualization expects batch size 1.")

    qidx = torch.as_tensor(query_seed_inds, dtype=torch.long, device=seed_xyz.device)
    query_xyz = seed_xyz[:, qidx, :]

    if "grasp_top_rot" in end_points:
        query_rot = end_points["grasp_top_rot"][:, qidx, :, :]
    else:
        rot_bank = getattr(model_mod, "grasp_rot").to(seed_xyz.device)
        query_rot = rot_bank[end_points["grasp_top_rot_inds"][:, qidx]]

    # Keep crop_size exactly consistent with IGNet.forward().
    base_depth = float(getattr(model_mod, "base_depth", 0.04))
    grasp_max_width = float(getattr(model_mod, "GRASP_MAX_WIDTH", 0.10))
    Q = qidx.numel()
    crop_length = (0.04 + base_depth) * torch.ones(
        (1, Q, 1), dtype=seed_xyz.dtype, device=seed_xyz.device
    )
    crop_width = (crop_scale * grasp_max_width) * torch.ones_like(crop_length)
    crop_height = 0.02 * torch.ones_like(crop_length)
    crop_size = torch.cat([crop_length, crop_width, crop_height], dim=-1).contiguous()

    id_features = torch.arange(
        M, dtype=seed_xyz.dtype, device=seed_xyz.device
    ).view(1, 1, M)

    grouped = crop_module.grouper(
        seed_xyz, query_xyz, query_rot, crop_size, id_features
    )
    # Standard QueryAndGroup with use_xyz=True -> [B,3+C,Q,K].
    if grouped.ndim != 4 or grouped.shape[1] < 4:
        raise RuntimeError(
            f"Unexpected RectangularQueryAndGroup output: {tuple(grouped.shape)}"
        )

    grouped_ids = grouped[0, -1].detach().round().long().cpu().numpy()
    grouped_ids = np.clip(grouped_ids, 0, M - 1)
    return np.unique(grouped_ids.reshape(-1)), grouped_ids


def _match_subset_rows_to_original(subset_rows, original_rows):
    """Map unchanged NMS output rows back to original row indices."""
    subset_rows = np.asarray(subset_rows, dtype=np.float64)
    original_rows = np.asarray(original_rows, dtype=np.float64)
    used = np.zeros(len(original_rows), dtype=bool)
    out = []
    for row in subset_rows:
        exact = np.where(
            (~used) & np.all(np.isclose(original_rows, row[None], rtol=0.0, atol=1e-10), axis=1)
        )[0]
        if exact.size == 0:
            # Defensive fallback for tiny serialization/extension differences.
            dist = np.max(np.abs(original_rows - row[None]), axis=1)
            dist[used] = np.inf
            j = int(np.argmin(dist))
            if not np.isfinite(dist[j]) or dist[j] > 1e-6:
                raise RuntimeError("Could not map an NMS grasp back to its source seed.")
        else:
            j = int(exact[0])
        used[j] = True
        out.append(j)
    return np.asarray(out, dtype=np.int64)


def nms_seed_grasps(preds, translation_thresh, rotation_thresh_rad):
    """
    Apply the official graspnetAPI GraspGroup.nms() to one decoded grasp per seed
    and return both the NMS grasp array and the corresponding original seed IDs.
    """
    preds = np.asarray(preds, dtype=np.float64)
    order = np.argsort(preds[:, 0])[::-1]
    preds_sorted = preds[order]
    gg = GraspGroup(preds_sorted.copy())
    gg_nms = gg.nms(
        translation_thresh=float(translation_thresh),
        rotation_thresh=float(rotation_thresh_rad),
    )
    gg_nms.sort_by_score()
    local_idx = _match_subset_rows_to_original(
        gg_nms.grasp_group_array, preds_sorted
    )
    seed_inds = order[local_idx]
    return gg_nms.grasp_group_array.copy(), seed_inds


def _get_query_rotations(end_points, model_mod, query_seed_inds):
    qidx = torch.as_tensor(
        query_seed_inds,
        dtype=torch.long,
        device=end_points["xyz_graspable"].device,
    )
    if "grasp_top_rot" in end_points:
        rots = end_points["grasp_top_rot"][:, qidx, :, :]
    else:
        rot_bank = getattr(model_mod, "grasp_rot").to(qidx.device)
        rots = rot_bank[end_points["grasp_top_rot_inds"][:, qidx]]
    return rots[0].detach().float().cpu().numpy()


def _rect_membership(local_xyz, crop_length, crop_width, crop_height, x_rule, yz_rule):
    """Boolean membership in one candidate interpretation of the rectangular crop."""
    x = local_xyz[:, 0]
    y = local_xyz[:, 1]
    z = local_xyz[:, 2]

    if x_rule == "minus04_to_base":
        # Current IGNet passes crop_length = 0.04 + base_depth. This interpretation
        # corresponds to the common grasp crop spanning 4 cm behind the seed to
        # base_depth in front of it.
        x_min = -0.04
        x_max = crop_length - 0.04
    elif x_rule == "symmetric":
        x_min, x_max = -0.5 * crop_length, 0.5 * crop_length
    elif x_rule == "zero_to_length":
        x_min, x_max = 0.0, crop_length
    elif x_rule == "minus_length_to_zero":
        x_min, x_max = -crop_length, 0.0
    else:
        raise ValueError(x_rule)

    if yz_rule == "width_height":
        y_half, z_half = 0.5 * crop_width, 0.5 * crop_height
    elif yz_rule == "height_width":
        y_half, z_half = 0.5 * crop_height, 0.5 * crop_width
    else:
        raise ValueError(yz_rule)

    eps = 1e-6
    return (
        (x >= x_min - eps) & (x <= x_max + eps) &
        (np.abs(y) <= y_half + eps) &
        (np.abs(z) <= z_half + eps)
    )


def infer_rectangular_crop_rule(
    seed_xyz,
    query_seed_inds,
    query_rots,
    grouped_seed_inds_per_query,
    crop_length,
    crop_width,
    crop_height,
):
    """
    Infer the world->grasp transform convention and rectangular x-range by
    checking which candidate rule contains the exact K points returned by the
    model's RectangularQueryAndGroup.

    This keeps the visualization robust to the rotation convention used by the
    custom pointnet2 rectangular query implementation without approximating K.
    """
    candidates = []
    for rot_mode in ("rel_R", "rel_RT"):
        for x_rule in (
            "minus04_to_base",
            "symmetric",
            "zero_to_length",
            "minus_length_to_zero",
        ):
            for yz_rule in ("width_height", "height_width"):
                inside_count = 0
                total_count = 0
                for qi, seed_idx in enumerate(query_seed_inds):
                    center = seed_xyz[int(seed_idx)]
                    R = query_rots[qi]
                    ids = grouped_seed_inds_per_query[qi]
                    pts = seed_xyz[ids]
                    rel = pts - center[None, :]
                    local = rel @ (R if rot_mode == "rel_R" else R.T)
                    inside = _rect_membership(
                        local,
                        crop_length,
                        crop_width,
                        crop_height,
                        x_rule,
                        yz_rule,
                    )
                    inside_count += int(inside.sum())
                    total_count += int(inside.size)
                ratio = inside_count / max(total_count, 1)
                candidates.append((ratio, rot_mode, x_rule, yz_rule))

    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    print(
        "[GROUP-CUBOID] inferred rule: "
        f"containment={best[0]*100:.2f}% | rot={best[1]} | x={best[2]} | yz={best[3]}"
    )
    if best[0] < 0.80:
        print(
            "[WARN] Cuboid-rule validation is below 80%. The custom rectangular "
            "query may use a different boundary convention; inspect the yellow "
            "region before using it in the paper."
        )
    return {
        "containment": best[0],
        "rot_mode": best[1],
        "x_rule": best[2],
        "yz_rule": best[3],
    }


def all_scene_points_in_cuboids(
    scene_xyz,
    seed_xyz,
    query_seed_inds,
    query_rots,
    crop_length,
    crop_width,
    crop_height,
    crop_rule,
    candidate_mask=None,
):
    """
    Return dense scene-point indices inside the selected cuboids.

    ``candidate_mask`` can restrict visualization to predicted foreground
    scene points. This is more faithful to the current scene-level network:
    CloudCrop is called on ``xyz_sel`` / ``xyz_graspable`` after objectness-
    guided foreground selection, so background/table points are not actual
    grouping candidates even when the geometric cuboid overlaps them.
    """
    scene_xyz = np.asarray(scene_xyz, dtype=np.float32)
    union = np.zeros(len(scene_xyz), dtype=bool)
    per_query_counts = []

    if candidate_mask is None:
        candidate_mask = np.ones(len(scene_xyz), dtype=bool)
    else:
        candidate_mask = np.asarray(candidate_mask, dtype=bool).reshape(-1)
        if candidate_mask.shape[0] != len(scene_xyz):
            raise ValueError(
                f"candidate_mask has {candidate_mask.shape[0]} points but "
                f"scene_xyz has {len(scene_xyz)}."
            )

    for qi, seed_idx in enumerate(query_seed_inds):
        center = seed_xyz[int(seed_idx)]
        R = query_rots[qi]
        rel = scene_xyz - center[None, :]
        local = rel @ (R if crop_rule["rot_mode"] == "rel_R" else R.T)
        inside = _rect_membership(
            local,
            crop_length,
            crop_width,
            crop_height,
            crop_rule["x_rule"],
            crop_rule["yz_rule"],
        )
        inside &= candidate_mask
        union |= inside
        per_query_counts.append(int(inside.sum()))

    return (
        np.flatnonzero(union).astype(np.int64),
        np.asarray(per_query_counts, dtype=np.int32),
    )


def get_full_scene_foreground_mask(end_points, prob_thresh=None):
    """
    Foreground mask over the original N_f scene points used by the network.

    If ``prob_thresh`` is None, use the model's native argmax foreground
    decision (class 1). Otherwise, use softmax foreground probability >= thresh.
    """
    if "objectness_score" not in end_points:
        raise KeyError("end_points does not contain 'objectness_score'.")

    logits = end_points["objectness_score"]
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise RuntimeError(
            f"Unexpected objectness_score shape: {tuple(logits.shape)}; "
            "expected [1, 2, N_f]."
        )

    if prob_thresh is None:
        mask = torch.argmax(logits, dim=1)[0] == 1
        prob = torch.softmax(logits, dim=1)[0, 1]
    else:
        prob = torch.softmax(logits, dim=1)[0, 1]
        mask = prob >= float(prob_thresh)

    return (
        mask.detach().cpu().numpy().astype(bool),
        prob.detach().float().cpu().numpy(),
    )


def make_pcd(points, colors=None, uniform_color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if uniform_color is not None:
        pcd.paint_uniform_color(list(uniform_color))
    elif colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
        )
    return pcd


def make_scene_pcd(scene_data):
    return make_pcd(scene_data["cloud_sampled"], scene_data["color_sampled"])


def make_dim_scene_pcd(scene_data, dim_factor=0.55):
    colors = np.clip(scene_data["color_sampled"] * float(dim_factor), 0.0, 1.0)
    return make_pcd(scene_data["cloud_sampled"], colors)


def make_group_colored_scene_pcd(
    scene_data,
    group_inds,
    dim_factor=0.55,
    group_color=(1.0, 0.82, 0.0),
):
    """
    Build ONE point cloud for the grouping visualization.

    Do not render a dim scene cloud and a second yellow cloud at exactly the
    same XYZ coordinates. Legacy Open3D can depth-test coincident points
    against each other, so the background cloud may visually hide the yellow
    cloud. Overwrite the grouped points' colors in the same PointCloud instead.
    """
    xyz = np.asarray(scene_data["cloud_sampled"], dtype=np.float64)
    colors = np.clip(
        np.asarray(scene_data["color_sampled"], dtype=np.float64)
        * float(dim_factor),
        0.0,
        1.0,
    )

    group_inds = np.asarray(group_inds, dtype=np.int64).reshape(-1)
    if group_inds.size > 0:
        valid = (group_inds >= 0) & (group_inds < len(xyz))
        group_inds = group_inds[valid]
        colors[group_inds] = np.asarray(group_color, dtype=np.float64)

    return make_pcd(xyz, colors)


def official_grasp_geometries(pred_array):
    """
    Use the official graspnetAPI visualization without overriding ``color``.

    graspnetAPI forwards each grasp's predicted score to ``plot_gripper_pro_max``.
    With the default ``color=None``, the official renderer maps the score to:

        RGB = (score, 0, 1 - score)

    so high-score grasps are red and low-score grasps are blue.
    """
    gg = GraspGroup(np.asarray(pred_array, dtype=np.float64))
    return gg.to_open3d_geometry_list()


def build_grouping_visual_grasps_from_best_rotation(
    end_points,
    model_mod,
    query_seed_inds,
    decoded_preds=None,
):
    """
    Build grasp arrays for the *grouping-stage* visualization.

    Logic requested by the user:
    - randomly select seed points for the grouping figure;
    - for each selected seed point, use that point's BEST rotation
      (``grasp_top_rot`` / ``grasp_top_rot_inds``).

    For display size/color, we keep the decoded score/width/height/depth of the
    same seed if ``decoded_preds`` is provided. Only the rotation is replaced by
    the best-rotation output. This keeps the grouping-stage overlay readable
    while ensuring the orientation matches the rotational graspness branch.
    """
    query_seed_inds = np.asarray(query_seed_inds, dtype=np.int64).reshape(-1)
    centers = (
        end_points["xyz_graspable"][0, query_seed_inds]
        .detach().float().cpu().numpy()
    )
    rots = _get_query_rotations(end_points, model_mod, query_seed_inds)

    Q = len(query_seed_inds)
    if decoded_preds is not None:
        scores = np.asarray(decoded_preds[query_seed_inds, 0], dtype=np.float64)
        widths = np.asarray(decoded_preds[query_seed_inds, 1], dtype=np.float64)
        heights = np.asarray(decoded_preds[query_seed_inds, 2], dtype=np.float64)
        depths = np.asarray(decoded_preds[query_seed_inds, 3], dtype=np.float64)
    else:
        # Fallback sizes if decoded grasps are not provided.
        scores = np.full(Q, 0.8, dtype=np.float64)
        widths = np.full(Q, min(float(getattr(model_mod, "GRASP_MAX_WIDTH", 0.10)), 0.06), dtype=np.float64)
        heights = np.full(Q, 0.02, dtype=np.float64)
        depths = np.full(Q, float(getattr(model_mod, "base_depth", 0.04)), dtype=np.float64)

    object_ids = -np.ones(Q, dtype=np.float64)
    arr = np.concatenate(
        [
            scores[:, None],
            widths[:, None],
            heights[:, None],
            depths[:, None],
            rots.reshape(Q, 9).astype(np.float64),
            centers.astype(np.float64),
            object_ids[:, None],
        ],
        axis=1,
    )
    return arr


def _set_render_options(vis, point_size=2.5):
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True


def _add_geometries(vis, geometries):
    first = True
    for geo in geometries:
        vis.add_geometry(geo, reset_bounding_box=first)
        first = False


def show_capture_view(title, geometries, cfg, screenshot_path, view_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=cfg.window_width,
        height=cfg.window_height,
        visible=True,
    )
    _add_geometries(vis, geometries)
    _set_render_options(vis, cfg.point_size)

    print("\n" + "=" * 80)
    print(f"[WINDOW 1] {title}")
    print("Adjust the camera to the view you want, then press Q / close the window.")
    print("The exact camera will be reused by the next two windows.")
    print("=" * 80)
    vis.run()

    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.capture_screen_image(str(screenshot_path), do_render=True)
    o3d.io.write_pinhole_camera_parameters(str(view_path), camera_params)
    vis.destroy_window()
    print(f"[SAVE] {screenshot_path}")
    print(f"[SAVE] {view_path}")
    return camera_params


def show_with_view(title, geometries, camera_params, cfg, screenshot_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=title,
        width=cfg.window_width,
        height=cfg.window_height,
        visible=True,
    )
    _add_geometries(vis, geometries)
    _set_render_options(vis, cfg.point_size)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    try:
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(camera_params)

    print("\n" + "=" * 80)
    print(f"[WINDOW] {title}")
    print("Camera view copied from Window 1. Press Q / close when finished.")
    print("=" * 80)
    vis.run()
    vis.capture_screen_image(str(screenshot_path), do_render=True)
    vis.destroy_window()
    print(f"[SAVE] {screenshot_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize scene-level MMGNet framework stages.")

    # Requested sample defaults.
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    p.add_argument("--scene", type=int, default=100)
    p.add_argument("--anno", type=int, default=0)

    # Model/checkpoint.
    p.add_argument("--model_module", default="models.IGNet_v0_9")
    p.add_argument("--ckpt", default=None, help="Direct checkpoint path (recommended).")
    p.add_argument("--ckpt_root", default="/media/gpuadmin/rcao/result/ignet")
    p.add_argument("--network_name", default=None, help="Only for automatic ckpt lookup.")
    p.add_argument("--ckpt_epoch", type=int, default=48)
    p.add_argument("--seed_feat_dim", type=int, default=256)
    p.add_argument("--img_feat_dim", type=int, default=256)
    p.add_argument("--num_point", type=int, default=40000)
    p.add_argument("--m_point", type=int, default=1024)
    p.add_argument("--voxel_size", type=float, default=0.002)
    p.add_argument("--fuse_type", default="intermediate")
    p.add_argument("--grouping_type", default="rectangular", choices=["rectangular", "cylinder"])
    p.add_argument("--grouping_nsample", type=int, default=None)
    p.add_argument("--multi_scale_grouping", action="store_true")

    # Visualization.
    p.add_argument("--group_num", type=int, default=20)
    p.add_argument("--result_topk", type=int, default=10)
    p.add_argument(
        "--nms_translation_thresh", type=float, default=0.04,
        help="graspnetAPI NMS translation threshold in meters (default: 0.03).",
    )
    p.add_argument(
        "--nms_rotation_thresh_deg", type=float, default=40.0,
        help="graspnetAPI NMS rotation threshold in degrees (default: 30).",
    )
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--window_width", type=int, default=1600)
    p.add_argument("--window_height", type=int, default=900)
    p.add_argument("--point_size", type=float, default=4)
    p.add_argument("--group_scene_dim", type=float, default=0.55)
    p.add_argument(
        "--group_all_scene_points",
        action="store_true",
        help=(
            "Color every scene point geometrically inside the grouping cuboids. "
            "Default is safer/more faithful: only color full-scene points that "
            "the objectness head predicts as foreground."
        ),
    )
    p.add_argument(
        "--group_objectness_thresh",
        type=float,
        default=None,
        help=(
            "Optional foreground probability threshold for dense grouping "
            "visualization. Default None uses the same argmax foreground rule "
            "as graspable-point selection."
        ),
    )
    p.add_argument("--save_dir", default="vis/mmgnet_framework_scene0100_0000")

    # Resulted-grasp post-processing, matching current inference.
    p.add_argument("--collision_voxel_size", type=float, default=0.01)
    p.add_argument(
        "--collision_thresh", type=float, default=0.01,
        help="<=0 disables model-free collision filtering for Resulted Grasps."
    )
    return p.parse_args()


def main():
    cfg = parse_args()
    setup_seed(cfg.random_seed)

    if cfg.grouping_type != "rectangular":
        raise ValueError(
            "Grasp-shaded visualization is defined for rectangular grouping; "
            "run with --grouping_type rectangular."
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] {cfg.camera} scene_{cfg.scene:04d} frame_{cfg.anno:04d}")
    scene_data = load_scene_frame(cfg)
    print(
        f"[DATA] workspace={len(scene_data['cloud_masked'])} points; "
        f"network input={len(scene_data['cloud_sampled'])} points"
    )

    net, pred_decode, model_mod, ckpt_path = build_model(cfg, device)
    batch_data = make_end_points(scene_data, cfg, device)

    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
        preds = np.asarray(grasp_preds[0], dtype=np.float64)

    if preds.ndim != 2 or preds.shape[1] != 17:
        raise RuntimeError(f"Unexpected grasp array shape {preds.shape}; expected [M,17].")
    M = preds.shape[0]
    if M != int(end_points["xyz_graspable"].shape[1]):
        raise RuntimeError("Decoded grasp rows are not aligned with xyz_graspable seeds.")
    print(f"[INFER] {M} seed points -> one best decoded grasp per seed")

    # ------------------------------------------------------------------
    # 1) Scene Points
    # ------------------------------------------------------------------
    scene_pcd = make_scene_pcd(scene_data)
    o3d.io.write_point_cloud(str(save_dir / "scene_points.ply"), scene_pcd)
    camera_params = show_capture_view(
        "1/3  Scene Points",
        [scene_pcd],
        cfg,
        save_dir / "01_scene_points.png",
        save_dir / "open3d_view.json",
    )

    # ------------------------------------------------------------------
    # 2) Grasp-Shaded Grouping
    # Logic:
    #   all M=1024 seed points
    #   -> use each point's BEST rotation
    #   -> model-free collision filtering
    #   -> official grasp NMS
    #   -> randomly sample group_num survivors for visualization
    #
    # Yellow = dense foreground scene points inside the corresponding cuboids.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(cfg.random_seed)
    all_seed_inds = np.arange(M, dtype=np.int64)

    # Build one visualization grasp for every seed using that seed's BEST
    # rotational-graspness rotation. Width/depth/score are borrowed from the
    # decoded grasp of the same seed only for gripper size/color visualization.
    grouping_all_grasps = build_grouping_visual_grasps_from_best_rotation(
        end_points,
        model_mod,
        all_seed_inds,
        decoded_preds=preds,
    )
    grouping_candidate_seed_inds = all_seed_inds.copy()
    grouping_candidate_grasps = grouping_all_grasps

    # 1) Collision check BEFORE grouping NMS, as requested.
    if cfg.collision_thresh > 0:
        group_mfcdetector = ModelFreeCollisionDetectorTorch(
            scene_data["cloud_org"].reshape(-1, 3),
            voxel_size=cfg.collision_voxel_size,
        )
        group_gg = GraspGroup(grouping_candidate_grasps.copy())
        group_collision_mask = group_mfcdetector.detect(
            group_gg,
            approach_dist=0.05,
            collision_thresh=cfg.collision_thresh,
        )
        if torch.is_tensor(group_collision_mask):
            group_collision_mask = group_collision_mask.detach().cpu().numpy()
        group_collision_mask = np.asarray(group_collision_mask, dtype=bool)

        grouping_candidate_grasps = grouping_candidate_grasps[~group_collision_mask]
        grouping_candidate_seed_inds = grouping_candidate_seed_inds[~group_collision_mask]

        print(
            f"[GROUP-COLLISION] {M} -> {len(grouping_candidate_seed_inds)} "
            f"collision-free seed grasps "
            f"(voxel={cfg.collision_voxel_size:.3f} m, "
            f"thresh={cfg.collision_thresh:.3f})"
        )
    else:
        print(
            f"[GROUP-COLLISION] disabled; keeping all {M} seed grasps"
        )

    if len(grouping_candidate_seed_inds) == 0:
        raise RuntimeError(
            "No grouping grasps remain after collision filtering. "
            "Reduce --collision_thresh or inspect the predicted grasps."
        )

    # 2) NMS on the collision-free best-rotation grasps.
    group_nms_rot_rad = np.deg2rad(float(cfg.nms_rotation_thresh_deg))
    group_nms_rows, group_nms_local_inds = nms_seed_grasps(
        grouping_candidate_grasps,
        cfg.nms_translation_thresh,
        group_nms_rot_rad,
    )
    group_nms_seed_inds = grouping_candidate_seed_inds[group_nms_local_inds]

    print(
        f"[GROUP-NMS] {len(grouping_candidate_seed_inds)} -> "
        f"{len(group_nms_seed_inds)} grasps "
        f"(t={cfg.nms_translation_thresh:.3f} m, "
        f"r={cfg.nms_rotation_thresh_deg:.1f} deg)"
    )

    if len(group_nms_seed_inds) == 0:
        raise RuntimeError("No grouping grasps remain after NMS.")

    # 3) Randomly choose the visualized seed points AFTER collision + NMS.
    Q = min(int(cfg.group_num), len(group_nms_seed_inds))
    picked_local = rng.choice(
        len(group_nms_seed_inds),
        size=Q,
        replace=False,
    ).astype(np.int64)
    query_seed_inds = np.sort(group_nms_seed_inds[picked_local])

    print(
        f"[GROUP-SAMPLE] randomly selected {Q}/{len(group_nms_seed_inds)} "
        f"collision-free NMS survivors; "
        f"rotation of each point = its best rotation."
    )

    # Recover the model's exact K samples only to infer/verify the rectangular
    # crop convention. These K points are NOT what we color yellow anymore.
    _, grouped_per_query = recover_exact_grouped_seed_indices(
        net, end_points, model_mod, query_seed_inds
    )
    seed_xyz_np = end_points["xyz_graspable"][0].detach().float().cpu().numpy()
    query_rots_np = _get_query_rotations(
        end_points, model_mod, query_seed_inds
    )

    base_depth = float(getattr(model_mod, "base_depth", 0.04))
    grasp_max_width = float(getattr(model_mod, "GRASP_MAX_WIDTH", 0.10))
    crop_scale = 1.0
    crop_length = 0.04 + base_depth
    crop_width = crop_scale * grasp_max_width
    crop_height = 0.02

    crop_rule = infer_rectangular_crop_rule(
        seed_xyz_np,
        query_seed_inds,
        query_rots_np,
        grouped_per_query,
        crop_length,
        crop_width,
        crop_height,
    )

    # Dense visualization should follow the model's actual grouping source.
    # The network groups from xyz_sel / xyz_graspable, which was selected from
    # objectness-predicted foreground points. Therefore, by default we color
    # only predicted-foreground full-scene points inside each cuboid.
    fg_mask, fg_prob = get_full_scene_foreground_mask(
        end_points,
        prob_thresh=cfg.group_objectness_thresh,
    )
    if cfg.group_all_scene_points:
        dense_candidate_mask = None
        dense_candidate_name = "all scene points"
    else:
        dense_candidate_mask = fg_mask
        dense_candidate_name = "predicted foreground scene points"

    all_group_scene_inds, cuboid_point_counts = all_scene_points_in_cuboids(
        scene_data["cloud_sampled"],
        seed_xyz_np,
        query_seed_inds,
        query_rots_np,
        crop_length,
        crop_width,
        crop_height,
        crop_rule,
        candidate_mask=dense_candidate_mask,
    )
    np.save(save_dir / "scene_objectness_fg_prob.npy", fg_prob)
    np.save(save_dir / "scene_objectness_fg_mask.npy", fg_mask)
    print(
        f"[GROUP-DENSE] candidates={dense_candidate_name}; "
        f"foreground={int(fg_mask.sum())}/{len(fg_mask)} full-scene points"
    )

    if len(all_group_scene_inds) == 0:
        raise RuntimeError(
            "No full-scene points were classified inside the grasp-shaded "
            "cuboids. The cuboid membership calculation returned an empty set. "
            "Check the printed [GROUP-CUBOID] containment/rule."
        )

    # Color grouped points IN PLACE in the same scene point cloud.
    # A second coincident PointCloud can be hidden by Open3D depth testing.
    group_scene = make_group_colored_scene_pcd(
        scene_data,
        all_group_scene_inds,
        dim_factor=cfg.group_scene_dim,
        group_color=(1.0, 0.82, 0.0),
    )
    grouping_vis_grasps = build_grouping_visual_grasps_from_best_rotation(
        end_points,
        model_mod,
        query_seed_inds,
        decoded_preds=preds,
    )
    grouping_grasp_geos = official_grasp_geometries(
        grouping_vis_grasps
    )

    o3d.io.write_point_cloud(
        str(save_dir / "grasp_shaded_scene_colored.ply"),
        group_scene,
    )
    np.save(save_dir / "group_query_seed_inds_random.npy", query_seed_inds)
    np.save(save_dir / "grouping_vis_grasps_best_rotation.npy", grouping_vis_grasps)
    np.save(save_dir / "group_nms_seed_inds.npy", group_nms_seed_inds)
    np.save(save_dir / "group_collision_free_seed_inds.npy", grouping_candidate_seed_inds)
    np.save(save_dir / "grouped_K_seed_inds_per_query_debug.npy", grouped_per_query)
    np.save(save_dir / "grouped_all_scene_inds.npy", all_group_scene_inds)
    np.save(save_dir / "grouped_all_scene_count_per_query.npy", cuboid_point_counts)
    pred_widths = preds[query_seed_inds, 1]
    pred_depths = preds[query_seed_inds, 3]
    print(
        "[GROUP-GEOMETRY] grouping uses the model crop before the depth/width "
        "head: "
        f"crop_length={crop_length:.3f} m, crop_width={crop_width:.3f} m, "
        f"crop_height={crop_height:.3f} m. "
        f"Displayed final grasps have predicted width "
        f"{pred_widths.min():.3f}-{pred_widths.max():.3f} m and depth "
        f"{pred_depths.min():.3f}-{pred_depths.max():.3f} m, so the grouping "
        "region can legitimately be larger than the final gripper."
    )

    print(
        f"[GROUP] collision+NMS survivors={len(group_nms_seed_inds)}; visualized={Q}; "
        f"cuboid-selected={len(all_group_scene_inds)}/{len(scene_data['cloud_sampled'])} "
        "scene points -> colored yellow in-place"
    )
    if len(cuboid_point_counts) > 0:
        print(
            f"[GROUP] scene points/cuboid: min={cuboid_point_counts.min()}, "
            f"mean={cuboid_point_counts.mean():.1f}, max={cuboid_point_counts.max()}"
        )

    show_with_view(
        "2/3  Grasp-Shaded Grouping",
        [group_scene] + grouping_grasp_geos,
        camera_params,
        cfg,
        save_dir / "02_grasp_shaded_grouping.png",
    )

    # ------------------------------------------------------------------
    # 3) Resulted Grasps:
    # collision filter -> official grasp NMS -> score sort -> Top-M
    # ------------------------------------------------------------------
    gg_final = GraspGroup(preds.copy())
    if cfg.collision_thresh > 0:
        mfcdetector = ModelFreeCollisionDetectorTorch(
            scene_data["cloud_org"].reshape(-1, 3),
            voxel_size=cfg.collision_voxel_size,
        )
        collision_mask = mfcdetector.detect(
            gg_final, approach_dist=0.05, collision_thresh=cfg.collision_thresh
        )
        if torch.is_tensor(collision_mask):
            collision_mask = collision_mask.detach().cpu().numpy()
        collision_mask = np.asarray(collision_mask, dtype=bool)
        gg_final = gg_final[~collision_mask]
        print(f"[COLLISION] kept {len(gg_final)}/{len(preds)} grasps")

    before_nms = len(gg_final)
    gg_final.sort_by_score()

    # Grouping-stage NMS was removed when the grouping visualization changed
    # to random seed sampling. Compute the Resulted-Grasp NMS threshold here.
    result_nms_rot_rad = np.deg2rad(float(cfg.nms_rotation_thresh_deg))
    gg_final = gg_final.nms(
        translation_thresh=float(cfg.nms_translation_thresh),
        rotation_thresh=float(result_nms_rot_rad),
    )
    gg_final.sort_by_score()
    print(
        f"[RESULT-NMS] {before_nms} -> {len(gg_final)} grasps "
        f"(t={cfg.nms_translation_thresh:.3f} m, "
        f"r={cfg.nms_rotation_thresh_deg:.1f} deg)"
    )

    topk = min(int(cfg.result_topk), len(gg_final))
    gg_top = gg_final[:topk]
    result_grasp_geos = official_grasp_geometries(
        gg_top.grasp_group_array
    )
    gg_top.save_npy(str(save_dir / f"resulted_top{topk}_grasps_after_nms.npy"))

    show_with_view(
        f"3/3  Resulted Grasps — NMS Top {topk}",
        [make_scene_pcd(scene_data)] + result_grasp_geos,
        camera_params,
        cfg,
        save_dir / "03_resulted_grasps.png",
    )

    print("\n[DONE]")
    print(f"checkpoint: {ckpt_path}")
    print(f"outputs:    {save_dir.resolve()}")
    print("01_scene_points.png")
    print("02_grasp_shaded_grouping.png  # random seed points + each point's best rotation")
    print("03_resulted_grasps.png         # collision-filtered + NMS + Top-M")
    print("open3d_view.json")


if __name__ == "__main__":
    main()
