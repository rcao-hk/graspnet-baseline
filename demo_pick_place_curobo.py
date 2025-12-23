#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline:
  1) ZED camera capture (color/depth)
  2) (Optional) depth restoration
  3) grasp synthesis (GSNet + GraspGroup)
  4) UDP send cartesian pose (CuarmUdpThread)

Outputs (when cfgs.vis=True):
  - color/depth raw & cropped
  - point cloud raw & cropped (ply + npy)
  - grasp predictions (preds Nx17), filtered grasps, chosen grasp pose, grasp vis ply
"""

import os
import sys
import time
import json
import argparse
import datetime
from types import SimpleNamespace
import yaml

import numpy as np
import cv2

# ---------- ZED ----------
import pyzed.sl as sl

# ---------- Grasp synthesis deps (your repo) ----------
import torch
import MinkowskiEngine as ME
import open3d as o3d
from scipy.spatial.transform import Rotation as RotationR

# your repo relative imports (adjust if needed)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from graspnetAPI import GraspGroup
from graspnetAPI.utils.utils import plot_gripper_pro_max

# --- depth restoration project ---
DR_PROJECT_ROOT = "/mnt/ssd/robotarm/object_depth_percetion"
if DR_PROJECT_ROOT not in sys.path:
    sys.path.insert(0, DR_PROJECT_ROOT)

from model.dv2_res_conv import DV2_Two_Branch_Unc_Norm_Iter_ConvGRU

# ---------- UDP sender deps (your other repo layout) ----------
# (keep same pattern as your target_sender_cartesian.py)
def _append_udp_paths():
    repo_path = os.path.dirname(os.path.dirname(__file__))
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    script_path = repo_path + "/scripts"
    udp_repo_path = parent_path + "/user/ruby/curi_udp/python"
    # config_repo_share_path = parent_path + "/user/ruby/cuarm_configuration/share/python"
    curobo_root = parent_path + '/user/ruby'
    curobo_path = parent_path + '/user/ruby/cuarm_curobo/scripts'
    sys.path = [repo_path, parent_path, script_path, udp_repo_path, curobo_root,  curobo_path] + sys.path
    return parent_path

PARENT_PATH = _append_udp_paths()
from demo.cuarm_configuration.share.python.cuarm_udp import CuarmUdp, CuarmUdpThread
from demo.cuarm_configuration.share.python.cuarm_state import (
    TargetCommand, GripperMode, CuroboState, TargetCommandMode,
    TargetCommandGripperCommandType
)
from dataclasses import dataclass
T_base_cam_np = np.array([[-0.0104, -0.8878, 0.4601, 0.2132],
                  [-0.9999, 0.0096, -0.004, 0.0496],
                  [-0.0009, -0.4602, -0.8878, 0.5231],
                  [0, 0, 0, 1]])

def load_yaml(path: str) -> dict:
    """Plain YAML loader (no curobo dependency)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@dataclass
class CuroboConfig:
    config: dict
    world_cfg: dict
    robot_cfg: dict
    target_link_names: list
    target_size: int
    ee_link_name: str
    ee_link_idx_in_target_link_names: int
    joint_names: list
    joint_size: int
    arm_size: int
    arm_joint_size: list
    total_arm_joint_names: list
    total_arm_joint_size: int
    gripper_size: int
    gripper_types: list
    gripper_joint_size: list
    total_gripper_joint_names: list
    total_gripper_joint_size: int
    end_effector_offset: np.ndarray

    @staticmethod
    def load_from_yaml_path(config_path: str):
        config = load_yaml(config_path)

        world_cfg = config.get("world_cfg", {})
        robot_cfg = config.get("robot_cfg", {})

        # Ensure absolute path for URDF file
        urdf_path = robot_cfg.get("kinematics", {}).get("urdf_path", "")
        if urdf_path and not os.path.isabs(urdf_path):
            config_prefix = os.path.dirname(os.path.abspath(config_path))
            robot_cfg["kinematics"]["urdf_path"] = os.path.normpath(os.path.join(config_prefix, urdf_path))

        kin = robot_cfg.get("kinematics", {})
        target_link_names = kin.get("link_names", []) or []
        target_size = len(target_link_names) if target_link_names else 1

        ee_link_name = kin.get("ee_link", "")
        ee_link_idx_in_target_link_names = (
            target_link_names.index(ee_link_name) if (ee_link_name and ee_link_name in target_link_names) else 0
        )

        cspace = kin.get("cspace", {})
        joint_names = cspace.get("joint_names", []) or []
        joint_size = len(joint_names)

        arms = config.get("arm", []) or []
        arm_size = len(arms)
        arm_joint_size = [len(a.get("joint_names", []) or []) for a in arms]
        total_arm_joint_names = [j for a in arms for j in (a.get("joint_names", []) or [])]
        total_arm_joint_size = len(total_arm_joint_names)

        grippers = config.get("gripper", []) or []
        gripper_size = len(grippers)
        gripper_types = [g.get("type", "inspire_dexterous_hand") for g in grippers]
        gripper_joint_size = [len(g.get("joint_names", []) or []) for g in grippers]
        total_gripper_joint_names = [j for g in grippers for j in (g.get("joint_names", []) or [])]
        total_gripper_joint_size = len(total_gripper_joint_names)

        end_effector_offset = np.zeros((arm_size, 3), dtype=np.float32)
        for i, arm in enumerate(arms):
            if "end_effector_offset" in arm:
                end_effector_offset[i, :] = np.array(arm["end_effector_offset"], dtype=np.float32)

        return CuroboConfig(
            config=config,
            world_cfg=world_cfg,
            robot_cfg=robot_cfg,
            target_link_names=target_link_names,
            target_size=target_size,
            ee_link_name=ee_link_name,
            ee_link_idx_in_target_link_names=ee_link_idx_in_target_link_names,
            joint_names=joint_names,
            joint_size=joint_size,
            arm_size=arm_size,
            arm_joint_size=arm_joint_size,
            total_arm_joint_names=total_arm_joint_names,
            total_arm_joint_size=total_arm_joint_size,
            gripper_size=gripper_size,
            gripper_types=gripper_types,
            gripper_joint_size=gripper_joint_size,
            total_gripper_joint_names=total_gripper_joint_names,
            total_gripper_joint_size=total_gripper_joint_size,
            end_effector_offset=end_effector_offset,
        )
def now_string():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_npy(path, arr):
    ensure_dir(os.path.dirname(path))
    np.save(path, arr)

def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def depth_to_vis_uint8(depth_m, invalid_to_zero=True):
    """depth_m: float32 depth in meters"""
    d = depth_m.copy()
    if invalid_to_zero:
        d[~np.isfinite(d)] = 0.0
        d[d < 0] = 0.0
    valid = d[d > 0]
    if valid.size == 0:
        return np.zeros_like(d, dtype=np.uint8)
    p5, p95 = np.percentile(valid, 5), np.percentile(valid, 95)
    if p95 <= p5:
        p95 = p5 + 1e-6
    d = np.clip(d, p5, p95)
    vis = ((d - p5) / (p95 - p5) * 255.0).astype(np.uint8)
    return vis

def write_pcd_ply(path, pts, colors=None):
    """pts: (N,3) float, colors: (N,3) in [0,1] or [0,255]"""
    ensure_dir(os.path.dirname(path))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        pcd.colors = o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(path, pcd)

def load_T_base_cam(cfgs):
    if cfgs.T_base_cam_np and os.path.isfile(cfgs.T_base_cam_np):
        T = np.load(cfgs.T_base_cam_np)
        assert T.shape == (4, 4)
        return T.astype(np.float64)
    return np.eye(4, dtype=np.float64)

def apply_T(T, R, t):
    """T: 4x4, R: 3x3, t: (3,) -> (R', t')"""
    Tc = np.eye(4, dtype=np.float64)
    Tc[:3, :3] = R
    Tc[:3, 3] = t
    Tb = T @ Tc
    return Tb[:3, :3], Tb[:3, 3]

def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Return quaternion in (w,x,y,z).
    SciPy old versions: as_quat() -> (x,y,z,w) only.
    SciPy new versions: as_quat(scalar_first=True) -> (w,x,y,z).
    """
    r = RotationR.from_matrix(R)
    try:
        # New SciPy
        q_wxyz = r.as_quat(scalar_first=True)
    except TypeError:
        # Old SciPy: (x,y,z,w) -> (w,x,y,z)
        q_xyzw = r.as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=q_xyzw.dtype)

    # optional: normalize for safety
    q_wxyz = q_wxyz / (np.linalg.norm(q_wxyz) + 1e-12)
    return q_wxyz
# ------------------------- Camera -------------------------

class ZEDCamera:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.zed = sl.Camera()
        self.init = sl.InitParameters(
            depth_mode=sl.DEPTH_MODE.NEURAL,
            coordinate_units=sl.UNIT.METER,
            coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        )
        if cfgs.input_svo_file:
            self.init.set_from_svo_file(cfgs.input_svo_file)
        elif cfgs.ip_address:
            ip_str = cfgs.ip_address
            if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
                self.init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.')) == 4:
                self.init.set_from_stream(ip_str)

        res_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1200": sl.RESOLUTION.HD1200,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "SVGA": sl.RESOLUTION.SVGA,
            "VGA": sl.RESOLUTION.VGA,
        }
        if cfgs.resolution in res_map:
            self.init.camera_resolution = res_map[cfgs.resolution]

        status = self.zed.open(self.init)
        if status > sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED open failed: {repr(status)}")

        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()

        # intrinsics (rectified left)
        camera_info = self.zed.get_camera_information()
        cam_conf = camera_info.camera_configuration
        left = cam_conf.calibration_parameters.left_cam
        self.fx, self.fy, self.cx, self.cy = left.fx, left.fy, left.cx, left.cy

    def grab(self):
        return self.zed.grab() <= sl.ERROR_CODE.SUCCESS

    def get_frame(self):
        # left color
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        # aligned depth
        self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)

        img = self.image_mat.get_data()  # likely BGRA
        color = img[..., :3].copy()      # BGR uint8
        depth = self.depth_mat.get_data().astype(np.float32).copy()  # meters, may contain nan/inf

        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0
        return color, depth

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy

    def close(self):
        self.zed.close()

# ------------------------- Depth restoration (optional placeholder) -------------------------
class DepthRestorerDV2:
    def __init__(
        self,
        cfgs,
        device: str = "cuda",
    ):  
        self.cfgs = cfgs
        self.project_root = getattr(self.cfgs, "dr_project_root", None)
        self.method_dir = getattr(self.cfgs, "dr_method_dir", None)
        self.encoder = getattr(self.cfgs, "dr_encoder", 'vitl')
        self.iter_num = int(getattr(self.cfgs, "dr_iter_num", 5))
        self.refine_downsample = int(getattr(self.cfgs, "dr_refine_downsample", 2))
        self.min_depth = float(getattr(self.cfgs, "dr_min_depth", 0.01))
        self.max_depth = float(getattr(self.cfgs, "dr_max_depth", 5.0))
        self.input_size = (int(getattr(self.cfgs, "dr_input_width", 518)), int(getattr(self.cfgs, "dr_input_height", 518)))  # (W,H)
        self.sn_mode = getattr(self.cfgs, "dr_sn_mode", 'logbias')
        self.robust_init = getattr(self.cfgs, "dr_robust_init", True)
        self.depth_factor = getattr(self.cfgs, "dr_depth_factor", 1000)

        print(self.cfgs)
        self.device = (
            torch.device(device)
            if (device.startswith("cuda") and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.method_dir is None:
            raise ValueError("[DepthRestorerDV2] method_dir must be specified in cfgs.")
        
        # ---- resolve ckpt path ----
        if self.method_dir.startswith("log/") or self.method_dir.startswith("log\\"):
            method_path = os.path.join(self.project_root, self.method_dir)
        else:
            method_path = os.path.join(self.project_root, "log", self.method_dir)

        ckpt_path = os.path.join(method_path, f"{self.encoder}_best.pth")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"[DepthRestorerDV2] checkpoint not found: {ckpt_path}")

        print(f"[DepthRestorerDV2] Loading ckpt: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # ---- build model (fixed to DV2_Two_Branch_Unc_Norm_Iter_ConvGRU) ----
        self.model = DV2_Two_Branch_Unc_Norm_Iter_ConvGRU(
            encoder=self.encoder,
            output_dim=2,  # matches your infer script when scale_norm enabled
            iter_num=self.iter_num,
            refine_downsample=self.refine_downsample,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            use_scale_norm=True,
            sn_align_mode=self.sn_mode,
            noisy_robust_init=self.robust_init,
        )

        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def restore(self, color_bgr: np.ndarray, depth_m: np.ndarray):
        """
        Inputs:
          color_bgr: HxWx3 uint8 (BGR)
          depth_m:   HxW float32 (meters), invalid=0
        Returns:
          pred_depth_m: HxW float32 (meters), invalid=0
          conf_map:     HxW float32 or None
        """
        H0, W0 = depth_m.shape[:2]

        # DV2 infer_image expects obs_depth in "sensor units" (your script uses mm as float)
        obs_depth_mm = (depth_m.astype(np.float32) * self.depth_factor)

        # model.infer_image(image, obs_depth, input_size, True)
        init_depth, rel_align_depth, _, pred_depth, conf = self.model.infer_image(
            color_bgr, obs_depth_mm, self.input_size, True
        )

        if isinstance(pred_depth, torch.Tensor):
            pred_depth = pred_depth.detach().cpu().numpy()
        pred_depth = np.squeeze(pred_depth).astype(np.float32)  # meters

        conf_np = None
        if conf is not None:
            if isinstance(conf, torch.Tensor):
                conf_np = conf.detach().cpu().numpy()
            else:
                conf_np = conf
            conf_np = np.squeeze(conf_np).astype(np.float32)

        # resize back to original camera resolution for grasp synthesis (keep original intrinsics)
        pred_depth_m = cv2.resize(pred_depth, (W0, H0), interpolation=cv2.INTER_LINEAR)

        if conf_np is not None:
            conf_map = cv2.resize(conf_np, (W0, H0), interpolation=cv2.INTER_LINEAR)
        else:
            conf_map = None

        # sanitize
        pred_depth_m[~np.isfinite(pred_depth_m)] = 0.0
        pred_mask = pred_depth_m > 0
        pred_depth_m[pred_mask] = np.clip(pred_depth_m[pred_mask], self.min_depth, self.max_depth)

        return pred_depth_m, conf_map

# ------------------------- Grasp synthesis -------------------------

def batch_matrix_to_viewpoint(batch_matrix):
    towards = batch_matrix[:, :, 0]
    towards = towards / (np.linalg.norm(towards, axis=1, keepdims=True) + 1e-12)
    y_axis = batch_matrix[:, :, 1]
    cos_angle = y_axis[:, 1]
    sin_angle = -y_axis[:, 0]
    angle = np.arctan2(sin_angle, cos_angle)
    return towards, angle

def filter_rotation(approach, threshold=30):
    # keep those close to +z
    normal_dists = np.rad2deg(np.arccos(
        (np.dot(approach, np.array([0, 0, 1]))) / (np.linalg.norm(approach, axis=1) + 1e-12)
    ))
    return [i for i, d in enumerate(normal_dists) if d < threshold]

class GraspSynthesizer:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.device = torch.device(cfgs.device if torch.cuda.is_available() else "cpu")

        # import net
        from models.GSNet import GraspNet, pred_decode
        self.pred_decode = pred_decode
        self.net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)

        ckpt = torch.load(cfgs.checkpoint_path, map_location=self.device)
        try:
            self.net.load_state_dict(ckpt)
        except Exception:
            self.net.load_state_dict(ckpt["model_state_dict"])
        self.net.to(self.device).eval()

    def infer(self, color_bgr, depth_m, intrinsics):
        """
        color_bgr: HxWx3 uint8
        depth_m:   HxW float32 meters
        intrinsics: (fx,fy,cx,cy)
        returns:
          dict with chosen grasp pose (camera frame), and intermediate objects
        """
        fx, fy, cx, cy = intrinsics
        H, W = depth_m.shape[:2]

        # create organized cloud (meters)
        cam_info = CameraInfo(W, H, fx, fy, cx, cy, scale=1.0)
        cloud = create_point_cloud_from_depth_image(depth_m, cam_info, organized=True)  # HxWx3

        # crop
        H, W = depth_m.shape[:2]

        # crop params may not exist -> default 0
        x1 = int(getattr(self.cfgs, "crop_x_left", 0) or 0)
        x2 = int(getattr(self.cfgs, "crop_x_right", 0) or 0)
        y1 = int(getattr(self.cfgs, "crop_y_top", 0) or 0)
        y2 = int(getattr(self.cfgs, "crop_y_bottom", 0) or 0)

        # if all zeros -> no crop (use full image)
        if (x1, x2, y1, y2) == (0, 0, 0, 0):
            x1, x2, y1, y2 = 0, W, 0, H
        else:
            # allow "0" to mean "till end" for right/bottom (common in configs)
            if x2 == 0:
                x2 = W
            if y2 == 0:
                y2 = H

            # clamp to valid range
            x1 = max(0, min(W, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H, y1))
            y2 = max(0, min(H, y2))

            # ensure non-empty crop; if invalid, fallback to full
            if x2 <= x1 or y2 <= y1:
                x1, x2, y1, y2 = 0, W, 0, H

        crop_color = color_bgr[y1:y2, x1:x2, :]
        crop_depth  = depth_m[y1:y2, x1:x2]
        crop_cloud  = cloud[y1:y2, x1:x2, :]
        
        # valid points mask (avoid zeros)
        z = crop_cloud[..., 2]
        valid = (crop_depth > 0) & np.isfinite(crop_depth) & np.isfinite(z) & (z > 0)

        pts = crop_cloud.reshape(-1, 3)
        cols = crop_color.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        pts = pts[valid_flat]
        cols = cols[valid_flat]

        if pts.shape[0] < self.cfgs.minimum_num_pt:
            return {
                "ok": False,
                "reason": f"too few valid points: {pts.shape[0]}",
                "cloud_full": cloud,
                "crop": (crop_color, crop_depth, crop_cloud),
            }

        # sample points
        N = self.cfgs.num_point
        if pts.shape[0] >= N:
            idxs = np.random.choice(pts.shape[0], N, replace=False)
        else:
            idxs1 = np.arange(pts.shape[0])
            idxs2 = np.random.choice(pts.shape[0], N - pts.shape[0], replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        pts_s = pts[idxs]
        cols_s = cols[idxs]

        cloud_tensor = torch.tensor(pts_s, dtype=torch.float32, device=self.device)
        feats_tensor = torch.ones_like(cloud_tensor).float()

        coors_tensor = torch.tensor(pts_s / self.cfgs.voxel_size, dtype=torch.int32, device=self.device)
        coordinates_batch, features_batch = ME.utils.sparse_collate([coors_tensor], [feats_tensor], dtype=torch.float32)
        coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
            coordinates_batch, features_batch, return_index=True, return_inverse=True, device=self.device
        )

        batch = {
            "point_clouds": cloud_tensor.unsqueeze(0),
            "cloud_colors": torch.tensor(cols_s, dtype=torch.float32, device=self.device).unsqueeze(0),
            "coors": coordinates_batch,
            "feats": features_batch,
            "quantize2original": quantize2original,
        }

        with torch.no_grad():
            end_points = self.net(batch)
            grasp_preds = self.pred_decode(end_points)
            preds = torch.stack(grasp_preds).reshape(-1, 17).detach().cpu().numpy()

        gg = GraspGroup(preds)

        # collision detection
        if self.cfgs.collision_thresh > 0:
            det = ModelFreeCollisionDetectorTorch(
                pts.astype(np.float32),
                voxel_size=self.cfgs.collision_voxel_size
            )
            collision_mask = det.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
            collision_mask = collision_mask.detach().cpu().numpy()
            gg = gg[~collision_mask]

        if len(gg) == 0:
            return {
                "ok": False,
                "reason": "no grasps after collision filter",
                "preds": preds,
                "gg": None,
                "cloud_full": cloud,
                "crop": (crop_color, crop_depth, crop_cloud),
                "pts_valid": pts,
                "cols_valid": cols,
            }

        gg = gg.sort_by_score()
        gg = gg.nms()

        # rotation filtering
        scores = np.array(gg.scores)
        Rmats = np.array(gg.rotation_matrices)
        trans = np.array(gg.translations)
        widths = np.array(gg.widths)
        depths = np.array(gg.depths)

        if self.cfgs.rotation_filtering:
            dirs, _ = batch_matrix_to_viewpoint(Rmats)
            keep = filter_rotation(dirs, self.cfgs.filter_angle)
            if len(keep) == 0:
                return {
                    "ok": False,
                    "reason": "all grasps filtered by rotation",
                    "preds": preds,
                    "gg": gg,
                    "cloud_full": cloud,
                    "crop": (crop_color, crop_depth, crop_cloud),
                    "pts_valid": pts,
                    "cols_valid": cols,
                }
            scores = scores[keep]
            Rmats = Rmats[keep]
            trans = trans[keep]
            widths = widths[keep]
            depths = depths[keep]

        best_i = int(np.argmax(scores))
        chosen = {
            "score": float(scores[best_i]),
            "t_cam": trans[best_i].astype(float),
            "R_cam": Rmats[best_i].astype(float),
            "width": float(widths[best_i]),
            "depth": float(depths[best_i]),
        }

        return {
            "ok": True,
            "preds": preds,          # raw decoded preds
            "gg": gg,               # after nms (before rotation filter arrays slicing)
            "chosen": chosen,       # chosen grasp in camera frame
            "cloud_full": cloud,
            "crop": (crop_color, crop_depth, crop_cloud),
            "pts_valid": pts,
            "cols_valid": cols,
        }

# ------------------------- UDP Sender -------------------------

class UdpPoseSender:
    def __init__(self, cfgs):
        self.cfgs = cfgs
        config_path = cfgs.curobo_config_path
        self.config = CuroboConfig.load_from_yaml_path(config_path)
        
        self.target_command = TargetCommand()
        self.target_command.mode = TargetCommandMode.CARTESIAN
        self.target_command.gripper_command_type = TargetCommandGripperCommandType.JOINT_COMMAND
        self.target_command.gripper_size = self.config.gripper_size
        self.target_command.gripper_mode_command = [GripperMode.OPEN.value for _ in range(self.target_command.gripper_size)]
        self.target_command.gripper_joint_size = self.config.gripper_joint_size
        self.target_command.gripper_joint_position = [[0]*n for n in self.config.gripper_joint_size]

        self.udp = CuarmUdpThread(
            cfgs.udp_local_ip, cfgs.udp_local_port,
            cfgs.udp_remote_ip, cfgs.udp_remote_port,
            CuarmUdp.unpack_curobo_state, CuarmUdp.pack_target_command,
            recv_delay_ms=cfgs.udp_recv_delay_ms
        )

        self.arm_size = self.config.arm_size

    def send_pose_once_hold(self, pos_m, quat_wxyz, hold_s=1.0):
        """
        pos_m: (3,) meters
        quat_wxyz: (4,) (w,x,y,z)
        Strategy:
          - read current state; keep other arms at current pose; set target arm at pos/quat
          - send command repeatedly for hold_s seconds
        """
        t0 = time.time()
        arm_i = self.cfgs.control_arm_index

        while time.time() - t0 < hold_s:
            state: CuroboState = self.udp.receive()

            # default: keep everything at zero if no state
            target_pos = np.zeros((self.arm_size, 3), dtype=np.float32)
            target_quat = np.zeros((self.arm_size, 4), dtype=np.float32)

            if state is not None:
                # keep other arms unchanged
                target_pos[:] = state.arm_cartesian_position.astype(np.float32)
                target_quat[:] = state.arm_cartesian_orientation.astype(np.float32)

            # set controlled arm to new target
            target_pos[arm_i] = pos_m.astype(np.float32)
            target_quat[arm_i] = quat_wxyz.astype(np.float32)

            self.target_command.target_link_size = self.arm_size
            self.target_command.target_link_position = target_pos
            self.target_command.target_link_orientation = target_quat

            # gripper: keep open by default
            self.udp.send(self.target_command)
            time.sleep(self.cfgs.udp_send_period_s)

    def close(self):
        self.udp.terminate()

# ------------------------- Visualization dump -------------------------

def dump_vis(
    cfgs,
    frame_dir,
    color_bgr,
    depth_raw_m,
    cloud_raw_full,
    crop_pack,
    grasp_out,
    T_base_cam,
    restored_depth_m=None,
    restored_cloud_full=None,
):
    """
    Always save:
      - rgb
      - raw_depth (meters)
      - raw_points (from raw depth)
      - grasps outputs

    If restoration is used (restored_depth_m provided):
      - restored_depth (meters)
      - restored_points (from restored depth or provided restored_cloud_full)

    Notes:
      - crop_pack should correspond to the *grasp input* (typically from restored depth if used)
      - cloud_raw_full should correspond to the raw depth pointcloud (organized HxWx3)
    """
    ensure_dir(frame_dir)

    crop_color, crop_depth, crop_cloud = crop_pack

    # -------------------------
    # 1) RGB
    # -------------------------
    cv2.imwrite(os.path.join(frame_dir, "rgb.png"), color_bgr)
    cv2.imwrite(os.path.join(frame_dir, "rgb_crop.png"), crop_color)

    # -------------------------
    # 2) RAW depth (meters)
    # -------------------------
    save_npy(os.path.join(frame_dir, "raw_depth_m.npy"), depth_raw_m.astype(np.float32))
    cv2.imwrite(os.path.join(frame_dir, "raw_depth_vis.png"), depth_to_vis_uint8(depth_raw_m))

    # -------------------------
    # 3) RAW points (organized HxWx3)
    # -------------------------
    if cloud_raw_full is not None:
        save_npy(os.path.join(frame_dir, "raw_cloud_full.npy"), cloud_raw_full.astype(np.float32))
        write_pcd_ply(
            os.path.join(frame_dir, "raw_cloud_full.ply"),
            cloud_raw_full.reshape(-1, 3),
            color_bgr.reshape(-1, 3),
        )

    # -------------------------
    # 4) RESTORED depth + points (optional)
    # -------------------------
    use_restoration = restored_depth_m is not None

    if use_restoration:
        save_npy(os.path.join(frame_dir, "restored_depth_m.npy"), restored_depth_m.astype(np.float32))
        cv2.imwrite(os.path.join(frame_dir, "restored_depth_vis.png"), depth_to_vis_uint8(restored_depth_m))

        if restored_cloud_full is not None:
            cloud_rest_full = restored_cloud_full
        else:
            cloud_rest_full = None
            # try to re-project from restored depth using intrinsics if provided
            if hasattr(cfgs, "fx") and hasattr(cfgs, "fy") and hasattr(cfgs, "cx") and hasattr(cfgs, "cy"):
                H, W = restored_depth_m.shape[:2]
                cam_info = CameraInfo(W, H, cfgs.fx, cfgs.fy, cfgs.cx, cfgs.cy, factor_depth=1.0)
                cloud_rest_full = create_point_cloud_from_depth_image(restored_depth_m, cam_info, organized=True)

        if cloud_rest_full is not None:
            save_npy(os.path.join(frame_dir, "restored_cloud_full.npy"), cloud_rest_full.astype(np.float32))
            write_pcd_ply(
                os.path.join(frame_dir, "restored_cloud_full.ply"),
                cloud_rest_full.reshape(-1, 3),
                color_bgr.reshape(-1, 3),
            )

    # -------------------------
    # 5) Crop that was actually used for grasp (depth/points)
    # -------------------------
    # (This crop should be derived from the depth used for grasp synthesis)
    save_npy(os.path.join(frame_dir, "grasp_crop_depth_m.npy"), crop_depth.astype(np.float32))
    cv2.imwrite(os.path.join(frame_dir, "grasp_crop_depth_vis.png"), depth_to_vis_uint8(crop_depth))

    save_npy(os.path.join(frame_dir, "grasp_crop_cloud.npy"), crop_cloud.astype(np.float32))
    write_pcd_ply(
        os.path.join(frame_dir, "grasp_crop_cloud.ply"),
        crop_cloud.reshape(-1, 3),
        crop_color.reshape(-1, 3),
    )

    # -------------------------
    # 6) Grasps + meta
    # -------------------------
    meta = {
        "T_base_cam": T_base_cam.tolist(),
        "ok": bool(grasp_out.get("ok", False)),
        "use_restoration": bool(use_restoration),
    }

    if "reason" in grasp_out:
        meta["reason"] = grasp_out["reason"]

    if "preds" in grasp_out and grasp_out["preds"] is not None:
        save_npy(os.path.join(frame_dir, "grasps_preds_raw.npy"), grasp_out["preds"].astype(np.float32))

    if grasp_out.get("ok", False):
        chosen = grasp_out["chosen"]
        t_cam = np.array(chosen["t_cam"], dtype=np.float64)
        R_cam = np.array(chosen["R_cam"], dtype=np.float64)
        R_base, t_base = apply_T(T_base_cam, R_cam, t_cam)

        meta["chosen"] = {
            "score": float(chosen["score"]),
            "width": float(chosen["width"]),
            "depth": float(chosen["depth"]),
            "t_cam": t_cam.tolist(),
            "R_cam": R_cam.tolist(),
            "t_base": t_base.tolist(),
            "R_base": R_base.tolist(),
            "quat_base_wxyz": rot_to_quat_wxyz(R_base).tolist(),
        }

        # grasp vis ply
        try:
            scene_pts = grasp_out.get("pts_valid", None)
            scene_cols = grasp_out.get("cols_valid", None)

            # fallback: use crop points if pts_valid not present
            if scene_pts is None:
                scene_pts = crop_cloud.reshape(-1, 3)
            if scene_cols is None:
                scene_cols = crop_color.reshape(-1, 3)

            down = o3d.geometry.PointCloud()
            down.points = o3d.utility.Vector3dVector(scene_pts)
            down.colors = o3d.utility.Vector3dVector(scene_cols.astype(np.float32) / 255.0)

            gripper = plot_gripper_pro_max(
                t_cam, R_cam, chosen["width"], chosen["depth"], chosen["score"]
            )
            vis_pcd = down.voxel_down_sample(voxel_size=0.005)
            vis_pcd += gripper.sample_points_uniformly(number_of_points=2000)

            o3d.io.write_point_cloud(os.path.join(frame_dir, "grasp_vis.ply"), vis_pcd)
        except Exception as e:
            meta["grasp_vis_error"] = str(e)

    save_json(os.path.join(frame_dir, "meta.json"), meta)

# ------------------------- Main -------------------------

def build_cfgs():
    parser = argparse.ArgumentParser()

    # --- camera ---
    parser.add_argument('--input_svo_file', type=str, default='')
    parser.add_argument('--ip_address', type=str, default='')
    parser.add_argument('--resolution', type=str, default='HD720')

    # --- optional depth restoration ---
    parser.add_argument('--use_restored_depth', action='store_true')

    parser.add_argument('--dr_method_dir', type=str,
        default='log/dreds_clearpose_hiss_50k_dav2_complete_obs_iter_unc_cali_convgru_l1_only_scale_norm_robust_init_wo_soft_fuse_l1+grad_sigma_conf_518x518/')
    parser.add_argument('--dr_project_root', type=str, default='/mnt/ssd/robotarm/object_depth_percetion')  # path to DV2 project root
    parser.add_argument('--dr_encoder', type=str, default='vitl')

    parser.add_argument('--dr_iter_num', type=int, default=5)
    parser.add_argument('--dr_refine_downsample', type=int, default=2)
    parser.add_argument('--dr_min_depth', type=float, default=0.001)
    parser.add_argument('--dr_max_depth', type=float, default=5.0)
    parser.add_argument('--dr_input_width', type=int, default=518)
    parser.add_argument('--dr_input_height', type=int, default=518)
    parser.add_argument('--dr_sn_mode', type=str, default='logbias', choices=['logbias', 'z_affine'])
    parser.add_argument('--dr_robust_init', action='store_true', default=True)
    parser.add_argument('--dr_depth_factor', type=float, default=1000.0)  # factor to convert meters to sensor units (e.g., mm)
    
    # --- grasp net ---
    parser.add_argument('--checkpoint_path', type=str, default='log/gsnet_base/checkpoint.tar')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed_feat_dim', type=int, default=512)
    parser.add_argument('--num_point', type=int, default=15000)
    parser.add_argument('--voxel_size', type=float, default=0.005)
    parser.add_argument('--minimum_num_pt', type=int, default=100)
    parser.add_argument('--collision_thresh', type=float, default=0.01)
    parser.add_argument('--collision_voxel_size', type=float, default=0.005)
    parser.add_argument('--rotation_filtering', action='store_true', default=True)
    parser.add_argument('--filter_angle', type=float, default=25.0)

    # crop (defaults are your realsense numbers; adjust for ZED if needed)
    parser.add_argument('--crop_x_left', type=int, default=0)
    parser.add_argument('--crop_x_right', type=int, default=0)
    parser.add_argument('--crop_y_top', type=int, default=0)
    parser.add_argument('--crop_y_bottom', type=int, default=0)

    # --- extrinsics ---
    # parser.add_argument('--T_base_cam_np', type=str, default='')  # npy 4x4

        
    # --- udp ---
    parser.add_argument('--curobo_config_path', type=str,
                        default='/mnt/ssd/robotarm/graspnet-baseline/demo/cuarm_configuration/dual_v1/curobo/two_gripper.yaml')
    parser.add_argument('--udp_local_ip', type=str, default="127.0.0.1")
    parser.add_argument('--udp_local_port', type=int, default=10091)
    parser.add_argument('--udp_remote_ip', type=str, default="127.0.0.1")
    parser.add_argument('--udp_remote_port', type=int, default=10092)
    parser.add_argument('--udp_recv_delay_ms', type=int, default=10)
    parser.add_argument('--udp_send_period_s', type=float, default=0.02)
    parser.add_argument('--control_arm_index', type=int, default=1)  # which arm to command
    parser.add_argument('--hold_s', type=float, default=1.0)         # how long to stream the target

    # --- run ---
    parser.add_argument('--run_once', action='store_true', help='capture one frame, infer once, send once, exit')
    parser.add_argument('--max_frames', type=int, default=999999)
    parser.add_argument('--sleep_s', type=float, default=0.0)

    # --- vis dump ---
    parser.add_argument('--vis', action='store_true', help='save intermediate results')
    parser.add_argument('--vis_dir', type=str, default='vis')

    args = parser.parse_args()
    return SimpleNamespace(**vars(args))

def main():
    cfgs = build_cfgs()
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # T_base_cam = load_T_base_cam(cfgs)
    T_base_cam = T_base_cam_np
    
    cam = ZEDCamera(cfgs)
    grasp = GraspSynthesizer(cfgs)
    sender = UdpPoseSender(cfgs)
    restorer = None
    if cfgs.use_restored_depth:
        restorer = DepthRestorerDV2(cfgs, device=cfgs.device)
        
    run_id = now_string()
    base_vis_dir = os.path.join(cfgs.vis_dir, run_id) if cfgs.vis else None
    if cfgs.vis:
        ensure_dir(base_vis_dir)
        save_json(os.path.join(base_vis_dir, "cfgs.json"), vars(cfgs))
        save_npy(os.path.join(base_vis_dir, "T_base_cam.npy"), T_base_cam.astype(np.float64))

    print("[INFO] intrinsics fx,fy,cx,cy =", cam.get_intrinsics())
    print("[INFO] T_base_cam:\n", T_base_cam)

    frame_i = 0
    try:
        while frame_i < cfgs.max_frames:
            if not cam.grab():
                continue

            color_bgr, depth_raw_m = cam.get_frame()

            if restorer is not None:
                depth_used_m, depth_conf = restorer.restore(color_bgr, depth_raw_m)
            else:
                depth_used_m, depth_conf = depth_raw_m, None

            out = grasp.infer(color_bgr, depth_used_m, cam.get_intrinsics())

            if cfgs.vis:
                frame_dir = os.path.join(base_vis_dir, f"frame_{frame_i:06d}")
                dump_vis(cfgs, frame_dir, color_bgr, depth_raw_m, out['cloud_full'], out["crop"], out, T_base_cam)

            if out.get("ok", False):
                chosen = out["chosen"]
                t_cam = np.array(chosen["t_cam"], dtype=np.float64)
                R_cam = np.array(chosen["R_cam"], dtype=np.float64)

                # camera -> base
                R_base, t_base = apply_T(T_base_cam, R_cam, t_cam)
                quat_wxyz = rot_to_quat_wxyz(R_base)

                # send (pos in meters)
                sender.send_pose_once_hold(t_base.astype(np.float32), quat_wxyz.astype(np.float32), hold_s=cfgs.hold_s)

                print(f"[OK] frame={frame_i} score={chosen['score']:.4f} t_base(m)={t_base} quat(wxyz)={quat_wxyz}")
            else:
                print(f"[WARN] frame={frame_i} grasp failed: {out.get('reason', 'unknown')}")

            frame_i += 1
            if cfgs.run_once:
                break
            if cfgs.sleep_s > 0:
                time.sleep(cfgs.sleep_s)

    finally:
        sender.close()
        cam.close()

if __name__ == "__main__":
    main()
