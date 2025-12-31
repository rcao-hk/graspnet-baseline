#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))

import json
import time
import copy
import numpy as np
import cv2
import open3d as o3d
import torch
import MinkowskiEngine as ME
import pyrealsense2 as rs

from aubo.auboi5_controller import (
    AuboController,
    shift_pose,
    posevec2mat,
)

from demo.gripper_control import Gripper

from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import plot_gripper_pro_max
from models.GSNet import GraspNet, pred_decode

np.set_printoptions(precision=6, suppress=True)

# -------------------------
# cfg
# -------------------------
DEFAULT_CFG = dict(
    robot_ip="192.168.1.115",
    robot_eef_offset=[0.0, 0.0, 0.12],

    sensing_joint_pose=[
        float(np.deg2rad(-19.19)),
        float(np.deg2rad(-37.32)),
        float(np.deg2rad(-102.487)),
        0.0,
        float(np.deg2rad(-83.69)),
        float(np.deg2rad(-17.93)),
    ],

    place_joint_pose=[
        float(np.deg2rad(-38.30)),
        float(np.deg2rad(25.64)),
        float(np.deg2rad(-83.99)),
        float(np.deg2rad(-18.57)),
        float(np.deg2rad(-90.48)),
        float(np.deg2rad(-38.97)),
    ],

    pick_speed=0.2,
    place_speed=0.2,

    # eMc (4x4) eef->cam
    handeye_tf=np.eye(4).tolist(),

    rs_w=1280,
    rs_h=720,
    rs_fps=30,

    checkpoint_path="log/gsnet_base/checkpoint.tar",
    seed_feat_dim=512,
    num_point=15000,
    voxel_size=0.005,
    collision_thresh=0.05,
    collision_voxel_size=0.01,

    # camera_crop_x_left=316,
    # camera_crop_x_right=772,
    # camera_crop_y_top=202,
    # camera_crop_y_bottom=637,

    camera_crop_x_left=346, camera_crop_x_right=796, camera_crop_y_top=325, camera_crop_y_bottom=717,

    # --------- filtering / execution control ---------
    rotation_filtering=True,
    filter_angle_deg=40.0,

    # --------- width gate ---------
    min_grasp_width_m=0.06,   # 5cm: filter out grasps with width < this
    
    min_grasp_score=0.2,
    max_attempts=20,
    max_exec=20,

    VIS=False,
    DO_PLACE=True,

    planning_cfg=dict(
        eef_offset=[0.0, 0.0, 0.06],
        pre_pick_offset=[0.0, 0.0, -0.05],
        free_move_height=0.25,
    ),

    # =======================
    # Depth restoration config
    # =======================
    use_restored_depth=True,
    dr_project_root="/home/hkclr-user/projects/object_depth_percetion",
    dr_method="dreds_clearpose_hiss_50k_dav2_complete_obs_iter_unc_cali_convgru_l1_only_scale_norm_robust_init_wo_soft_fuse_l1+grad_sigma_conf_518x518",
    dr_encoder="vitl",
    dr_ckpt_epoch=None,
    dr_latest_ckpt=False,

    dr_iter_num=5,
    dr_refine_downsample=2,
    dr_min_depth=0.001,
    dr_max_depth=5.0,
    dr_input_width=518,
    dr_input_height=518,

    dr_scale_norm=True,
    dr_sn_mode="logbias",
    dr_robust_init=True,

    dr_depth_factor=1000.0,
    dr_save_debug=False,
    dr_debug_dir="debug_depth_restoration",

    enable_conf_reweight=True,     # <<< 主开关：用 DR 的 unc_map 重权重 grasp score（NMS 前）
    conf_reweight_debug=False,     # 打印一些调试信息
    conf_reweight_use_mean_fallback=True,  # 投影出界/无效Z用均值回填
    
    # =======================
    # Data saving
    # =======================
    save_data=True,                 # <<< 主开关
    save_root="demo_saved_data",     # run dir root
    save_only_on_success=True,      # False: 每次 attempt 都存（即使没 grasp 或 score gate 失败）
    save_unc_pmin = 1.0,
    save_unc_pmax = 99.0,
    
    # --------- NEW: save grasp visualization pointcloud ---------
    save_grasp_vis_pcd=True,           # <<< 是否额外输出 pcd_with_grasps.ply
    save_grasp_vis_voxel=0.005,
    save_grasp_vis_topk=1,
    save_grasp_vis_points_per_grasp=2000,
    save_grasp_vis_use_filtered=True, # True: 用 angle/collision/nms 后的 grasps_filtered; False: 用 grasps_raw
)


def load_cfg(path="demo/config.json"):
    cfg = copy.deepcopy(DEFAULT_CFG)
    if os.path.exists(path):
        with open(path, "r") as f:
            user = json.load(f)
        if "planning_cfg" in user and isinstance(user["planning_cfg"], dict):
            cfg["planning_cfg"].update(user["planning_cfg"])
            user = {k: v for k, v in user.items() if k != "planning_cfg"}
        cfg.update(user)

    cfg["handeye_tf"] = np.array(cfg["handeye_tf"], dtype=np.float64)
    return cfg


# -------------------------
# small io helpers
# -------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _jsonable_cfg(cfg: dict):
    # avoid dumping huge/np types
    out = {}
    for k, v in cfg.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (list, tuple, dict, str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _normalize_to_uint8(x: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-6):
    """
    Robust normalize by percentiles -> uint8 [0,255].
    """
    x = np.asarray(x)
    x = np.squeeze(x)

    # handle invalid
    m = np.isfinite(x)
    if not np.any(m):
        return np.zeros_like(x, dtype=np.uint8)

    vals = x[m].astype(np.float32)
    lo = float(np.percentile(vals, pmin))
    hi = float(np.percentile(vals, pmax))
    if hi <= lo:
        hi = lo + eps

    y = (x.astype(np.float32) - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y_u8 = (y * 255.0 + 0.5).astype(np.uint8)
    return y_u8


class DataSaver:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enable = bool(cfg.get("save_data", False))
        self.only_on_success = bool(cfg.get("save_only_on_success", False))
        self.run_dir = None
        if not self.enable:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        root = str(cfg.get("save_root", "demo_saved_data"))
        self.run_dir = os.path.join(root, ts)
        ensure_dir(self.run_dir)

        with open(os.path.join(self.run_dir, "cfg.json"), "w") as f:
            json.dump(_jsonable_cfg(cfg), f, indent=2, ensure_ascii=False)

        print(f"[SAVE] enabled. run_dir = {self.run_dir}")

    def _write_u16_png(self, path, depth_u16):
        if depth_u16 is None:
            return
        depth_u16 = np.asarray(depth_u16)
        if depth_u16.dtype != np.uint16:
            depth_u16 = depth_u16.astype(np.uint16)
        cv2.imwrite(path, depth_u16)

    def _write_npy(self, path, arr):
        if arr is None:
            return
        np.save(path, np.asarray(arr))

    # =========================
    # NEW: uncertainty saving
    # =========================
    def _normalize_to_uint8(self, x: np.ndarray, pmin=1.0, pmax=99.0, eps=1e-6):
        """
        Robust normalize (percentile) -> uint8 [0,255]
        NOTE: uncertainty larger => less reliable, we keep monotonic mapping:
              larger uncertainty => larger intensity => more "yellow" in viridis.
        """
        x = np.asarray(x)
        x = np.squeeze(x).astype(np.float32)

        m = np.isfinite(x)
        if not np.any(m):
            return np.zeros_like(x, dtype=np.uint8)

        vals = x[m]
        lo = float(np.percentile(vals, pmin))
        hi = float(np.percentile(vals, pmax))
        if hi <= lo:
            hi = lo + eps

        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        y_u8 = (y * 255.0 + 0.5).astype(np.uint8)
        return y_u8

    def _write_uncertainty(self, base_path_no_ext: str, unc_map: np.ndarray):
        """
        Save:
          - {base}.npy (float32 raw)
          - {base}_viridis.png (normalized viridis)
        """
        if unc_map is None:
            return

        unc = np.asarray(unc_map).astype(np.float32)
        np.save(base_path_no_ext + ".npy", unc)

        u8 = self._normalize_to_uint8(
            unc,
            pmin=float(self.cfg.get("save_unc_pmin", 1.0)),
            pmax=float(self.cfg.get("save_unc_pmax", 99.0)),
        )
        vis = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(base_path_no_ext + "_viridis.png", vis)

    def _build_o3d_scene_from_crop(self, crop_cloud, crop_color_bgr):
        """
        crop_cloud: (H,W,3) float (camera frame)
        crop_color_bgr: (H,W,3) uint8
        """
        if crop_cloud is None:
            return None

        pts = np.asarray(crop_cloud).reshape(-1, 3).astype(np.float64)
        if pts.size == 0:
            return None

        m = np.isfinite(pts).all(axis=1) & (pts[:, 2] > 1e-6)
        pts = pts[m]
        if pts.shape[0] < 10:
            return None

        scene = o3d.geometry.PointCloud()
        scene.points = o3d.utility.Vector3dVector(pts)

        if crop_color_bgr is not None:
            col = np.asarray(crop_color_bgr).reshape(-1, 3).astype(np.float64) / 255.0
            col = col[m]
            col = col[:, ::-1].copy()  # BGR -> RGB
            scene.colors = o3d.utility.Vector3dVector(col)

        return scene

    def _save_grasp_vis_pcd(self, save_path, scene, grasps_arr):
        """
        Save a single pointcloud containing:
          - downsampled scene points
          - sampled points from grasp geometries (top-k after sort+nms)
        """
        if scene is None or grasps_arr is None:
            return False
        grasps_arr = np.asarray(grasps_arr)
        if grasps_arr.ndim != 2 or grasps_arr.shape[1] != 17 or grasps_arr.shape[0] == 0:
            return False

        try:
            voxel = float(self.cfg.get("save_grasp_vis_voxel", 0.005))
            topk = int(self.cfg.get("save_grasp_vis_topk", 50))
            npts = int(self.cfg.get("save_grasp_vis_points_per_grasp", 2000))

            downsampled_scene = scene.voxel_down_sample(voxel_size=voxel)

            gg = GraspGroup(grasps_arr)
            gg = gg.sort_by_score()
            gg = gg.nms()
            gg_vis = gg[:min(topk, len(gg))]

            gg_vis_geo = gg_vis.to_open3d_geometry_list()

            pcd_vis = o3d.geometry.PointCloud(downsampled_scene)
            for g in gg_vis_geo:
                try:
                    pcd_vis += g.sample_points_uniformly(number_of_points=npts)
                except Exception:
                    if isinstance(g, o3d.geometry.PointCloud):
                        pcd_vis += g

            ok = o3d.io.write_point_cloud(save_path, pcd_vis)
            return bool(ok)
        except Exception as e:
            print(f"[SAVE][WARN] save_grasp_vis_pcd failed: {repr(e)}")
            return False

    def save_attempt(self, attempt_idx: int, payload: dict):
        if not self.enable:
            return

        status = payload.get("status", "unknown")
        if self.only_on_success and status != "ok":
            return

        ddir = os.path.join(self.run_dir)
        ensure_dir(ddir)

        # -------- raw / used image & depth --------
        if payload.get("raw_color_bgr", None) is not None:
            cv2.imwrite(os.path.join(ddir, f"{attempt_idx:03d}_raw_color.png"), payload["raw_color_bgr"])
            self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_raw_color.npy"), payload["raw_color_bgr"])

        self._write_u16_png(os.path.join(ddir, f"{attempt_idx:03d}_raw_depth_u16.png"), payload.get("raw_depth_u16", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_raw_depth_u16.npy"), payload.get("raw_depth_u16", None))
        self._write_u16_png(os.path.join(ddir, f"{attempt_idx:03d}_used_depth_u16.png"), payload.get("used_depth_u16", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_used_depth_u16.npy"), payload.get("used_depth_u16", None))

        if payload.get("restored_depth_u16", None) is not None:
            self._write_u16_png(os.path.join(ddir, f"{attempt_idx:03d}_restored_depth_u16.png"), payload["restored_depth_u16"])
            self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_restored_depth_u16.npy"), payload["restored_depth_u16"])

        # -------- crop image & crop depth --------
        if payload.get("crop_color_bgr", None) is not None:
            cv2.imwrite(os.path.join(ddir, f"{attempt_idx:03d}_crop_color.png"), payload["crop_color_bgr"])
            self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_crop_color.npy"), payload["crop_color_bgr"])

        self._write_u16_png(os.path.join(ddir, f"{attempt_idx:03d}_crop_depth_u16.png"), payload.get("crop_depth_u16", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_crop_depth_u16.npy"), payload.get("crop_depth_u16", None))

        # =========================
        # NEW: uncertainty map save
        # =========================
        # full-res uncertainty
        if payload.get("uncertainty_map", None) is not None:
            base = os.path.join(ddir, f"{attempt_idx:03d}_uncertainty")
            self._write_uncertainty(base, payload["uncertainty_map"])

        # optional crop-res uncertainty (if you provide it)
        if payload.get("uncertainty_map_crop", None) is not None:
            base = os.path.join(ddir, f"{attempt_idx:03d}_uncertainty_crop")
            self._write_uncertainty(base, payload["uncertainty_map_crop"])

        # -------- grasps --------
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_grasps_raw.npy"), payload.get("grasps_raw", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_grasps_filtered.npy"), payload.get("grasps_filtered", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_grasp_best_row.npy"), payload.get("grasp_best_row", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_cTg_raw.npy"), payload.get("cTg_raw", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_cTg_aligned.npy"), payload.get("cTg_aligned", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_bTe.npy"), payload.get("bTe", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_eMc.npy"), payload.get("eMc", None))
        self._write_npy(os.path.join(ddir, f"{attempt_idx:03d}_K.npy"), payload.get("K", None))

        # -------- pointcloud + grasps visualization --------
        if bool(self.cfg.get("save_grasp_vis_pcd", False)):
            crop_cloud = payload.get("crop_cloud", None)
            crop_color = payload.get("crop_color_bgr", None)

            scene = self._build_o3d_scene_from_crop(crop_cloud, crop_color)
            use_filtered = bool(self.cfg.get("save_grasp_vis_use_filtered", True))
            grasps_arr = payload.get("grasps_filtered", None) if use_filtered else payload.get("grasps_raw", None)

            vis_path = os.path.join(ddir, f"{attempt_idx:03d}_pcd_with_grasps.ply")
            ok = self._save_grasp_vis_pcd(vis_path, scene, grasps_arr)
            if ok:
                try:
                    o3d.io.write_point_cloud(
                        os.path.join(ddir, f"{attempt_idx:03d}_pcd_scene_downsampled.ply"),
                        scene.voxel_down_sample(voxel_size=float(self.cfg.get("save_grasp_vis_voxel", 0.005)))
                    )
                except Exception:
                    pass

        # -------- meta --------
        meta = payload.get("meta", {})
        meta_out = {
            "status": status,
            "attempt_idx": int(attempt_idx),
            "ts": time.time(),
            **meta,
        }
        with open(os.path.join(ddir, f"{attempt_idx:03d}_meta.json"), "w") as f:
            json.dump(meta_out, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] attempt {attempt_idx:03d} -> {ddir} (status={status})")
        
# -------------------------
# Realsense
# -------------------------
def rs_init(w, h, fps):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipe.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    align = rs.align(rs.stream.color)

    color_stream = profile.get_stream(rs.stream.color)
    intr = color_stream.as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0, intr.fy, intr.ppy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    for _ in range(10):
        pipe.wait_for_frames()

    return pipe, align, depth_scale, K


def rs_get_frame(pipe, align):
    frames = pipe.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None
    depth = np.asanyarray(depth_frame.get_data())   # uint16
    color = np.asanyarray(color_frame.get_data())   # BGR uint8
    return color, depth


# -------------------------
# grasp alignment: +90 around +Y then -90 around +Z (local)
# -------------------------
def _Ry(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float64)


def _Rz(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float64)


_R_OFF = _Ry(+90.0) @ _Rz(-90.0)


def align_grasp_pose(cTg: np.ndarray) -> np.ndarray:
    cTg_aligned = cTg.copy()
    cTg_aligned[:3, :3] = cTg[:3, :3] @ _R_OFF
    return cTg_aligned


# -------------------------
# angle filter (base -Z in camera as gravity-down)
# -------------------------
def filter_by_approach_angle(
    gg: GraspGroup,
    angle_deg: float,
    bTe: np.ndarray,
    eMc: np.ndarray,
    apply_grasp_alignment: bool = True,
):
    if len(gg) == 0:
        return gg

    bTe = np.asarray(bTe, dtype=np.float64)
    eMc = np.asarray(eMc, dtype=np.float64)

    bTc_raw = bTe @ eMc
    cTb = np.linalg.inv(bTc_raw)

    z_base_cam = cTb[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    z_base_cam = z_base_cam / (np.linalg.norm(z_base_cam) + 1e-12)
    base_minus_z_cam = -z_base_cam

    R_all = np.asarray(gg.rotation_matrices, dtype=np.float64)  # (N,3,3)

    if apply_grasp_alignment:
        v = _R_OFF[:, 2]
        z_grasp_cam = np.einsum("nij,j->ni", R_all, v)
    else:
        z_grasp_cam = R_all[:, :, 2]

    z_grasp_cam = z_grasp_cam / (np.linalg.norm(z_grasp_cam, axis=1, keepdims=True) + 1e-12)

    cosv = z_grasp_cam @ base_minus_z_cam
    cosv = np.clip(cosv, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosv))

    keep = np.where(ang <= float(angle_deg))[0].tolist()
    return gg[keep] if len(keep) > 0 else gg[[]]


def filter_by_min_width(gg: GraspGroup, min_width_m: float, debug: bool = False):
    """
    Keep grasps whose predicted width >= min_width_m.
    Assumes gg.widths is in meters (typical in GraspNetAPI / GSNet decode).
    """
    if len(gg) == 0:
        return gg

    w = np.asarray(gg.widths, dtype=np.float64)  # (N,)
    keep = np.where(w >= float(min_width_m))[0].tolist()

    if debug:
        print(f"[WIDTH] min_width={float(min_width_m):.3f} m | kept {len(keep)}/{len(gg)} | "
              f"w(min/mean/max)=({w.min():.3f}/{w.mean():.3f}/{w.max():.3f})")

    return gg[keep] if len(keep) > 0 else gg[[]]


def reweight_grasps_by_uncertainty(
    gg: GraspGroup,
    conf_map: np.ndarray,          # (H, W) float32, unc map (bigger = more uncertain)
    camera_info: CameraInfo,       # has fx, fy, cx, cy
    debug: bool = False,
):
    if gg is None or len(gg) == 0:
        return gg
    if conf_map is None:
        return gg

    conf_map = np.asarray(conf_map)
    conf_map = np.squeeze(conf_map)
    if conf_map.ndim != 2:
        if debug:
            print(f"[REWEIGHT][WARN] conf_map shape invalid: {conf_map.shape}, skip.")
        return gg

    gg_arr = gg.grasp_group_array.copy()  # (M, 17)
    scores = gg_arr[:, 0].astype(np.float32)
    centers = gg_arr[:, 13:16].astype(np.float32)  # (M,3) in camera frame

    fx = float(camera_info.fx)
    fy = float(camera_info.fy)
    cx = float(camera_info.cx)
    cy = float(camera_info.cy)

    H, W = conf_map.shape
    conf_mean = float(np.mean(conf_map))

    X = centers[:, 0]
    Y = centers[:, 1]
    Z = centers[:, 2]

    eps = 1e-6
    valid_z = Z > eps

    u = fx * X / np.maximum(Z, eps) + cx
    v = fy * Y / np.maximum(Z, eps) + cy
    u_int = np.rint(u).astype(np.int32)
    v_int = np.rint(v).astype(np.int32)

    in_bounds = (
        (u_int >= 0) & (u_int < W) &
        (v_int >= 0) & (v_int < H) &
        valid_z
    )

    grasp_unc = np.full((len(centers),), conf_mean, dtype=np.float32)
    if np.any(in_bounds):
        grasp_unc[in_bounds] = conf_map[v_int[in_bounds], u_int[in_bounds]].astype(np.float32)

    u_min = float(np.min(grasp_unc))
    u_max = float(np.max(grasp_unc))
    denom = (u_max - u_min) + 1e-6
    u_norm = (grasp_unc - u_min) / denom     # [0,1]
    w = 1.0 - u_norm                         # 越不确定 -> 权重越小

    new_scores = scores * w
    gg_arr[:, 0] = new_scores

    if debug:
        print("[REWEIGHT] conf stats:",
              f"unc[min,max,mean]=[{u_min:.4f},{u_max:.4f},{float(grasp_unc.mean()):.4f}]",
              f"w[min,max,mean]=[{float(w.min()):.4f},{float(w.max()):.4f},{float(w.mean()):.4f}]",
              f"score[old_mean,new_mean]=[{float(scores.mean()):.4f},{float(new_scores.mean()):.4f}]")

    return GraspGroup(gg_arr)

# -------------------------
# visualization (camera frame)
# -------------------------
def visualize_camera_scene_with_frames(cfg, crop_cloud, crop_color, cTg, grasp_meta, bTe, eMc):
    score, width, depth = grasp_meta

    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(crop_cloud.reshape(-1, 3))
    scene.colors = o3d.utility.Vector3dVector((crop_color.reshape(-1, 3) / 255.0).astype(np.float64))

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    bTe = np.asarray(bTe, dtype=np.float64)
    eMc = np.asarray(eMc, dtype=np.float64)
    bTc = bTe @ eMc
    cTb = np.linalg.inv(bTc)
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    base_frame.transform(cTb)

    cT_eef = np.linalg.inv(eMc)
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    eef_frame.transform(cT_eef)

    tool_offset = np.array(cfg.get("robot_eef_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    eefTtool = np.eye(4, dtype=np.float64)
    eefTtool[:3, 3] = tool_offset
    cT_tool = cT_eef @ eefTtool
    tool_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    tool_frame.transform(cT_tool)

    cTg = np.asarray(cTg, dtype=np.float64)
    cTg_aligned = align_grasp_pose(cTg)

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    grasp_frame.transform(cTg_aligned)

    gr = plot_gripper_pro_max(
        cTg[:3, 3],
        cTg[:3, :3],
        float(width), float(depth), float(score)
    )

    o3d.visualization.draw_geometries(
        [scene, cam_frame, base_frame, eef_frame, tool_frame, grasp_frame, gr],
        width=1536, height=864
    )
    return cTg_aligned


# -------------------------
# Planner
# -------------------------
class BinPickingPlanner:
    def __init__(self, tf_handeye, planning_cfg):
        self.eef_offset_ = np.array(planning_cfg["eef_offset"], dtype=np.float64)
        self.pre_pick_offset_ = np.array(planning_cfg["pre_pick_offset"], dtype=np.float64)
        self.tf_handeye_ = tf_handeye
        self.planning_cfg = planning_cfg

    def plan_pick(self, obj_pose_robot):
        pick_pose = shift_pose(obj_pose_robot, self.eef_offset_)
        pre_pick_pose = shift_pose(pick_pose, self.pre_pick_offset_)

        free_move_pose = copy.deepcopy(pre_pick_pose)
        free_move_pose[2, 3] = float(self.planning_cfg["free_move_height"])

        pick_pose_array = [free_move_pose, pre_pick_pose, pick_pose]
        after_pick_pose_array = [pick_pose, pre_pick_pose, free_move_pose]
        return pick_pose_array, after_pick_pose_array


# -------------------------
# IK symmetry: nearest wrist solution
# -------------------------
def _wrap_to_pi(x):
    x = np.asarray(x, dtype=np.float64)
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def joint_distance(q, q_ref):
    q = np.asarray(q, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    dq = _wrap_to_pi(q - q_ref)
    return float(np.linalg.norm(dq))


def pose_equivalent_flip_tool_z(T):
    Rz_pi = np.array([[-1.0,  0.0, 0.0],
                      [ 0.0, -1.0, 0.0],
                      [ 0.0,  0.0, 1.0]], dtype=np.float64)
    T_alt = T.copy()
    T_alt[:3, :3] = T[:3, :3] @ Rz_pi
    return T_alt


def choose_nearest_pose_by_ik(aubo, T, q_seed):
    q1 = aubo.compute_pose_tcp(T, q_seed)
    ok1 = (isinstance(q1, (list, tuple)) and len(q1) == 6)

    T2 = pose_equivalent_flip_tool_z(T)
    q2 = aubo.compute_pose_tcp(T2, q_seed)
    ok2 = (isinstance(q2, (list, tuple)) and len(q2) == 6)

    if (not ok1) and (not ok2):
        return None, None, False
    if ok1 and (not ok2):
        return T, q1, False
    if ok2 and (not ok1):
        return T2, q2, True

    d1 = joint_distance(q1, q_seed)
    d2 = joint_distance(q2, q_seed)
    return (T2, q2, True) if d2 < d1 else (T, q1, False)


def resolve_waypoints_nearest_gripper_pose(aubo, pose_list, q_start=None, verbose=False):
    if q_start is None:
        q_start = aubo.get_jq()
    q_curr = list(q_start)

    new_list = []
    for i, T in enumerate(pose_list):
        T_best, q_best, used_alt = choose_nearest_pose_by_ik(aubo, T, q_curr)
        if T_best is None:
            raise RuntimeError(f"[IK FAIL] waypoint {i} unreachable for both T and T@Rz(pi).\nT=\n{T}")
        new_list.append(T_best)
        if verbose:
            tag = "ALT" if used_alt else "ORI"
            print(f"[NEAREST] wp{i:02d}: {tag}, dist={joint_distance(q_best, q_curr):.4f}")
        q_curr = list(q_best)

    return new_list, q_curr


# ==========================================================
# Depth Restoration: init + run
# ==========================================================
def _maybe_add_sys_path(p: str):
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.insert(0, p)


def build_depth_restorer(cfg: dict, device: torch.device):
    if not bool(cfg.get("use_restored_depth", False)):
        return None

    dr_root = cfg.get("dr_project_root", "")
    if not dr_root or (not os.path.isdir(dr_root)):
        print(f"[DR][WARN] dr_project_root not found: {dr_root}. Disable restored depth.")
        return None

    _maybe_add_sys_path(dr_root)

    try:
        from model.dv2_res_conv import (
            DV2_Two_Branch_Unc_ConvGRU,
            DV2_Two_Branch_Unc_Iter_ConvGRU,
            DV2_Two_Branch_Unc_Filter_ConvGRU,
            DV2_Two_Branch_ConvGRU,
            DV2_Two_Branch_Unc_Norm_Iter_ConvGRU,
        )
    except Exception as e:
        print(f"[DR][WARN] import dv2_res_conv failed: {repr(e)}. Disable restored depth.")
        return None

    method = str(cfg["dr_method"])
    encoder = str(cfg["dr_encoder"])
    iter_num = int(cfg["dr_iter_num"])
    refine_downsample = int(cfg["dr_refine_downsample"])
    min_depth = float(cfg["dr_min_depth"])
    max_depth = float(cfg["dr_max_depth"])

    if cfg.get("dr_ckpt_epoch", None) is not None:
        ckpt_name = os.path.join(dr_root, "log", method, f"{encoder}_epoch_{int(cfg['dr_ckpt_epoch'])}.pth")
    elif bool(cfg.get("dr_latest_ckpt", False)):
        ckpt_name = os.path.join(dr_root, "log", method, f"{encoder}_latest.pth")
    else:
        ckpt_name = os.path.join(dr_root, "log", method, f"{encoder}_best.pth")

    if not os.path.isfile(ckpt_name):
        print(f"[DR][WARN] ckpt not found: {ckpt_name}. Disable restored depth.")
        return None

    print(f"[DR] Loading checkpoint: {ckpt_name}")
    checkpoint = torch.load(ckpt_name, map_location="cpu")

    try:
        if "none" in method:
            model = DV2_Two_Branch_ConvGRU(
                encoder=encoder, output_dim=1,
                iter_num=iter_num, refine_downsample=refine_downsample,
                min_depth=min_depth, max_depth=max_depth
            )
        elif "vanilla" in method:
            model = DV2_Two_Branch_Unc_ConvGRU(
                encoder=encoder, output_dim=1,
                iter_num=iter_num, refine_downsample=refine_downsample,
                min_depth=min_depth, max_depth=max_depth
            )
        elif "iter" in method:
            model = DV2_Two_Branch_Unc_Iter_ConvGRU(
                encoder=encoder, output_dim=1,
                iter_num=iter_num, refine_downsample=refine_downsample,
                min_depth=min_depth, max_depth=max_depth
            )
            if bool(cfg.get("dr_scale_norm", False)):
                model = DV2_Two_Branch_Unc_Norm_Iter_ConvGRU(
                    encoder=encoder, output_dim=2,
                    iter_num=iter_num, refine_downsample=refine_downsample,
                    min_depth=min_depth, max_depth=max_depth,
                    use_scale_norm=True,
                    sn_align_mode=str(cfg.get("dr_sn_mode", "logbias")),
                    noisy_robust_init=bool(cfg.get("dr_robust_init", False)),
                )
        elif "filter" in method:
            model = DV2_Two_Branch_Unc_Filter_ConvGRU(
                encoder=encoder, output_dim=2,
                iter_num=iter_num, refine_downsample=refine_downsample,
                min_depth=min_depth, max_depth=max_depth
            )
        else:
            print(f"[DR][WARN] Unrecognized dr_method: {method}. Disable restored depth.")
            return None
    except Exception as e:
        print(f"[DR][WARN] model build failed: {repr(e)}. Disable restored depth.")
        return None

    try:
        sd = checkpoint["model"] if isinstance(checkpoint, dict) and ("model" in checkpoint) else checkpoint
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[DR][WARN] load_state_dict failed: {repr(e)}. Disable restored depth.")
        return None

    model = model.to(device).eval()
    print("[DR] Depth restorer ready.")
    return model


@torch.no_grad()
def run_depth_restoration(
    cfg: dict,
    dr_model,
    color_bgr: np.ndarray,
    depth_raw_u16: np.ndarray,
    depth_scale: float,
    factor_depth: float,
    step_tag: str = "",
):
    """
    Returns:
      depth_used_u16: uint16 (same unit as RealSense raw)
      conf_full: (H,W) float32 uncertainty/conf map in the SAME pixel grid as input color/depth
                (if model doesn't provide conf -> None)
    """
    if dr_model is None:
        return depth_raw_u16, None

    try:
        H0, W0 = depth_raw_u16.shape[:2]

        depth_m = depth_raw_u16.astype(np.float32) * float(depth_scale)
        depth_mm = depth_m * float(cfg.get("dr_depth_factor", 1000.0))

        input_size = (int(cfg["dr_input_width"]), int(cfg["dr_input_height"]))

        out = dr_model.infer_image(color_bgr, depth_mm, input_size, True)

        if not (isinstance(out, (list, tuple)) and len(out) >= 4):
            raise RuntimeError("infer_image returned unexpected format.")

        pred_depth = out[3]
        conf = out[4] if len(out) >= 5 else None  # 你脚本里叫 conf，但你这里当 unc map 用

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.detach().cpu().numpy()
        pred_depth = np.squeeze(pred_depth).astype(np.float32)  # meters, at model output size

        # resize pred depth back
        pred_depth_m = cv2.resize(pred_depth, (W0, H0), interpolation=cv2.INTER_NEAREST)
        pred_depth_m = np.clip(pred_depth_m, float(cfg["dr_min_depth"]), float(cfg["dr_max_depth"]))

        depth_units = pred_depth_m * float(factor_depth)  # meters -> raw units
        depth_units = np.where(pred_depth_m > 0, depth_units, 0.0)
        depth_used_u16 = np.clip(depth_units, 0, 65535).astype(np.uint16)

        conf_full = None
        if conf is not None:
            if isinstance(conf, torch.Tensor):
                conf = conf.detach().cpu().numpy()
            conf = np.squeeze(conf).astype(np.float32)
            # resize to full-res (match color/depth pixel grid)
            conf_full = cv2.resize(conf, (W0, H0), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        if bool(cfg.get("dr_save_debug", False)):
            os.makedirs(cfg.get("dr_debug_dir", "debug_depth_restoration"), exist_ok=True)
            tag = step_tag if step_tag else "frame"
            cv2.imwrite(os.path.join(cfg["dr_debug_dir"], f"{tag}_color.png"), color_bgr)
            cv2.imwrite(os.path.join(cfg["dr_debug_dir"], f"{tag}_depth_raw_u16.png"), depth_raw_u16)
            cv2.imwrite(os.path.join(cfg["dr_debug_dir"], f"{tag}_depth_used_u16.png"), depth_used_u16)
            if conf_full is not None:
                np.save(os.path.join(cfg["dr_debug_dir"], f"{tag}_unc_map.npy"), conf_full)

        return depth_used_u16, conf_full

    except Exception as e:
        print(f"[DR][WARN] restoration failed, fallback to raw depth. err={repr(e)}")
        return depth_raw_u16, None


# -------------------------
# inference (returns more for saving)
# -------------------------
def _safe_gg_array(gg):
    if gg is None:
        return None
    if hasattr(gg, "grasp_group_array"):
        try:
            return np.asarray(gg.grasp_group_array)
        except Exception:
            return None
    return None


@torch.no_grad()
def infer_grasps(
    cfg,
    net,
    device,
    color_bgr,
    depth_used_u16,
    K,
    factor_depth,
    bTe,
    eMc,
    conf_map_full=None):
    """
    Returns dict:
      crop_color_bgr, crop_depth_u16, crop_cloud,
      grasps_raw (Nx17),
      grasps_filtered (Mx17 or None if empty),
      best_row (17,) or None,
      cTg_raw (4x4) or None,
      meta (score/width/depth) or None
    """
    H, W = depth_used_u16.shape[:2]
    cam_info = CameraInfo(
        W, H,
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
        float(factor_depth),
    )
    cloud = create_point_cloud_from_depth_image(depth_used_u16, cam_info, organized=True)

    y0, y1 = int(cfg["camera_crop_y_top"]), int(cfg["camera_crop_y_bottom"])
    x0, x1 = int(cfg["camera_crop_x_left"]), int(cfg["camera_crop_x_right"])

    crop_color = color_bgr[y0:y1, x0:x1, :].copy()
    crop_depth = depth_used_u16[y0:y1, x0:x1].copy()
    crop_cloud = cloud[y0:y1, x0:x1, :].copy()

    cloud_flat = crop_cloud.reshape(-1, 3)
    color_flat = crop_color.reshape(-1, 3)
    N = len(cloud_flat)
    num_point = int(cfg["num_point"])

    if N == 0:
        raise RuntimeError("Crop cloud is empty. Check crop params or depth validity.")

    if N >= num_point:
        idxs = np.random.choice(N, num_point, replace=False)
    else:
        idxs = np.concatenate([np.arange(N), np.random.choice(N, num_point - N, replace=True)], axis=0)

    pts = cloud_flat[idxs].astype(np.float32)
    cols = color_flat[idxs].astype(np.float32)

    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    cols_t = torch.tensor(cols, dtype=torch.float32, device=device)

    voxel_size = float(cfg["voxel_size"])
    coors_t = torch.tensor(pts / voxel_size, dtype=torch.int32, device=device)
    feats_t = torch.ones_like(pts_t).float()

    coordinates_batch, features_batch = ME.utils.sparse_collate([coors_t], [feats_t], dtype=torch.float32)
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch,
        features_batch,
        return_index=True,
        return_inverse=True,
        device=device,
    )

    batch = dict(
        point_clouds=pts_t.unsqueeze(0),
        cloud_colors=cols_t.unsqueeze(0),
        coors=coordinates_batch,
        feats=features_batch,
        quantize2original=quantize2original,
    )

    end_points = net(batch)
    grasp_preds = pred_decode(end_points)
    preds_raw = torch.stack(grasp_preds).reshape(-1, 17).detach().cpu().numpy()  # Nx17
    gg = GraspGroup(preds_raw)

    # collision (full cloud)
    if float(cfg["collision_thresh"]) > 0:
        mfcd = ModelFreeCollisionDetectorTorch(
            cloud.reshape(-1, 3),
            voxel_size=float(cfg["collision_voxel_size"])
        )
        cmask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=float(cfg["collision_thresh"]))
        cmask = cmask.detach().cpu().numpy()
        gg = gg[~cmask]

    # ======== NEW: conf/unc reweight BEFORE NMS ========
    if bool(cfg.get("enable_conf_reweight", False)) and (conf_map_full is not None) and (len(gg) > 0):
        gg = reweight_grasps_by_uncertainty(
            gg,
            conf_map=conf_map_full,
            camera_info=cam_info,
            debug=bool(cfg.get("conf_reweight_debug", False)),
        )
    
    gg = gg.sort_by_score().nms()

    min_w = float(cfg.get("min_grasp_width_m", 0.0))
    if min_w > 0:
        gg = filter_by_min_width(gg, min_w, debug=True)
        if len(gg) == 0:
            return None

    if bool(cfg["rotation_filtering"]):
        gg = filter_by_approach_angle(
            gg,
            float(cfg["filter_angle_deg"]),
            bTe=bTe,
            eMc=eMc,
            apply_grasp_alignment=True,
        )

    preds_filtered = _safe_gg_array(gg)
    if preds_filtered is None:
        # still allow saving raw predictions
        preds_filtered = None

    if len(gg) == 0:
        return dict(
            crop_color_bgr=crop_color,
            crop_depth_u16=crop_depth,
            crop_cloud=crop_cloud,
            grasps_raw=preds_raw,
            grasps_filtered=preds_filtered,
            best_row=None,
            cTg_raw=None,
            meta=None,
        )

    best = gg[0]
    best_row = None
    if hasattr(gg, "grasp_group_array"):
        try:
            best_row = np.asarray(gg.grasp_group_array)[0].copy()
        except Exception:
            best_row = None

    R = best.rotation_matrix.astype(np.float64)
    t = best.translation.astype(np.float64)
    width = float(best.width)
    depth = float(best.depth)
    score = float(best.score)

    cTg_raw = np.eye(4, dtype=np.float64)
    cTg_raw[:3, :3] = R
    cTg_raw[:3, 3] = t

    return dict(
        crop_color_bgr=crop_color,
        crop_depth_u16=crop_depth,
        crop_cloud=crop_cloud,
        grasps_raw=preds_raw,
        grasps_filtered=preds_filtered,
        best_row=best_row,
        cTg_raw=cTg_raw,
        meta=(score, width, depth),
    )


# -------------------------
# main loop
# -------------------------
def main():
    cfg = load_cfg("demo/config.json")
    saver = DataSaver(cfg)

    aubo = AuboController(robot_ip_=cfg["robot_ip"], eef_offset=cfg["robot_eef_offset"])
    gripper = Gripper(True)

    pipe, align, depth_scale, K = rs_init(cfg["rs_w"], cfg["rs_h"], cfg["rs_fps"])
    factor_depth = 1.0 / float(depth_scale)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # grasp net
    net = GraspNet(seed_feat_dim=int(cfg["seed_feat_dim"]), is_training=False).to(device).eval()
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    try:
        net.load_state_dict(ckpt)
    except Exception:
        net.load_state_dict(ckpt["model_state_dict"])

    # depth restorer (optional)
    dr_model = build_depth_restorer(cfg, device)
    eMc = np.array(cfg["handeye_tf"], dtype=np.float64)

    executed = 0
    attempts = 0

    try:
        gripper.config_gripper(open=True)

        while attempts < int(cfg["max_attempts"]) and executed < int(cfg["max_exec"]):
            attempts += 1
            print(f"\n========== Attempt {attempts}/{cfg['max_attempts']} | Exec {executed}/{cfg['max_exec']} ==========")

            # sensing pose
            aubo.moveJ(cfg["sensing_joint_pose"], speed=cfg["pick_speed"])

            # current base->eef
            _, bTe = aubo.get_current_state()
            bTe = np.array(bTe, dtype=np.float64)

            # capture
            color_bgr, depth_raw_u16 = rs_get_frame(pipe, align)
            if color_bgr is None:
                print("[WARN] Realsense capture failed, skip.")
                # save attempt (optional)
                saver.save_attempt(attempts, dict(
                    status="capture_fail",
                    raw_color_bgr=None,
                    raw_depth_u16=None,
                    used_depth_u16=None,
                    restored_depth_u16=None,
                    crop_color_bgr=None,
                    crop_depth_u16=None,
                    grasps_raw=None,
                    grasps_filtered=None,
                    grasp_best_row=None,
                    cTg_raw=None,
                    cTg_aligned=None,
                    bTe=bTe,
                    eMc=eMc,
                    K=K,
                    meta={"reason": "capture_fail"},
                ))
                continue

            # optional depth restoration
            depth_used_u16 = depth_raw_u16
            conf_map_full = None
            if bool(cfg.get("use_restored_depth", False)):
                depth_used_u16, conf_map_full = run_depth_restoration(
                    cfg, dr_model, color_bgr, depth_raw_u16,
                    depth_scale=depth_scale, factor_depth=factor_depth,
                    step_tag=f"att{attempts:02d}"
                )
                # if restoration enabled, treat used depth as restored (even if fallback happened)
                restored_depth_u16 = depth_used_u16
                
            # infer grasps (with debug outputs)
            try:
                out = infer_grasps(
                    cfg, net, device,
                    color_bgr, depth_used_u16,
                    K, factor_depth,
                    bTe=bTe, eMc=eMc, conf_map_full=conf_map_full
                )
            except Exception as ex:
                print(f"[WARN] inference failed: {repr(ex)}")
                saver.save_attempt(attempts, dict(
                    status="infer_fail",
                    raw_color_bgr=color_bgr,
                    raw_depth_u16=depth_raw_u16,
                    used_depth_u16=depth_used_u16,
                    restored_depth_u16=restored_depth_u16,
                    crop_color_bgr=None,
                    crop_depth_u16=None,
                    grasps_raw=None,
                    grasps_filtered=None,
                    grasp_best_row=None,
                    cTg_raw=None,
                    cTg_aligned=None,
                    bTe=bTe, eMc=eMc, K=K,
                    meta={"reason": "infer_fail", "err": repr(ex)},
                ))
                continue

            crop_color = out["crop_color_bgr"]
            crop_depth = out["crop_depth_u16"]
            crop_cloud = out["crop_cloud"]
            preds_raw = out["grasps_raw"]
            preds_filtered = out["grasps_filtered"]
            best_row = out["best_row"]
            cTg_raw = out["cTg_raw"]
            meta = out["meta"]  # (score,width,depth) or None

            # save (always attempt, unless save_only_on_success=True)
            base_payload = dict(
                raw_color_bgr=color_bgr,
                raw_depth_u16=depth_raw_u16,
                used_depth_u16=depth_used_u16,
                restored_depth_u16=restored_depth_u16 if bool(cfg.get("use_restored_depth", False)) else None,
                uncertainty_map=conf_map_full,
                crop_color_bgr=crop_color,
                crop_depth_u16=crop_depth,
                grasps_raw=preds_raw,
                grasps_filtered=preds_filtered,
                crop_cloud=crop_cloud,
                grasp_best_row=best_row,
                cTg_raw=cTg_raw,
                cTg_aligned=None,  # fill later if ok
                bTe=bTe,
                eMc=eMc,
                K=K,
                meta=dict(
                    use_restored_depth=bool(cfg.get("use_restored_depth", False)),
                    rotation_filtering=bool(cfg.get("rotation_filtering", False)),
                    filter_angle_deg=float(cfg.get("filter_angle_deg", 0.0)),
                    min_grasp_width_m=float(cfg.get("min_grasp_width_m", 0.0)),
                ),
            )

            if cTg_raw is None or meta is None:
                print("[SKIP] no grasp after filtering.")
                base_payload["status"] = "no_grasp"
                saver.save_attempt(attempts, base_payload)
                continue

            score, width, gdepth = meta
            print(f"[INFO] grasp score={score:.4f}, width={width:.4f}, depth={gdepth:.4f}")

            # score gate
            if float(score) < float(cfg["min_grasp_score"]):
                print(f"[SKIP] score {score:.4f} < min_grasp_score {cfg['min_grasp_score']:.4f}")
                base_payload["status"] = "score_gate"
                base_payload["meta"].update(dict(score=float(score), width=float(width), depth=float(gdepth)))
                saver.save_attempt(attempts, base_payload)
                continue

            # align for execution (+ optional visualize)
            if bool(cfg["VIS"]):
                cTg_aligned = visualize_camera_scene_with_frames(cfg, crop_cloud, crop_color, cTg_raw, meta, bTe, eMc)
            else:
                cTg_aligned = align_grasp_pose(cTg_raw)

            base_payload["cTg_aligned"] = cTg_aligned
            base_payload["meta"].update(dict(score=float(score), width=float(width), depth=float(gdepth)))
            base_payload["status"] = "ok"
            saver.save_attempt(attempts, base_payload)

            # compute base grasp pose
            bTg = bTe @ eMc @ cTg_aligned

            # plan
            planner = BinPickingPlanner(eMc, cfg["planning_cfg"])
            pick_wps, retreat_wps = planner.plan_pick(bTg)

            # nearest-wrist resolve
            q_now = aubo.get_jq()
            pick_wps, q_end_pick = resolve_waypoints_nearest_gripper_pose(aubo, pick_wps, q_start=q_now, verbose=False)
            retreat_wps, _ = resolve_waypoints_nearest_gripper_pose(aubo, retreat_wps, q_start=q_end_pick, verbose=False)

            # execute pick
            gripper.config_gripper(open=True)
            aubo.execute_trajectory(pick_wps, cfg["pick_speed"])
            gripper.config_gripper(open=False)
            aubo.execute_trajectory(retreat_wps, cfg["pick_speed"])

            # place
            if bool(cfg["DO_PLACE"]) and (cfg.get("place_joint_pose", None) is not None):
                aubo.moveJ(cfg["place_joint_pose"], speed=cfg["place_speed"])
                gripper.config_gripper(open=True)
                aubo.moveJ(cfg["sensing_joint_pose"], speed=cfg["place_speed"])
            else:
                aubo.moveJ(cfg["sensing_joint_pose"], speed=cfg["place_speed"])

            executed += 1
            print(f"[OK] executed pick&place #{executed}")

        print(f"\n[DONE] attempts={attempts}, executed={executed}")

    finally:
        try:
            pipe.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
