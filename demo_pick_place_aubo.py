#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

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

    # tool offset mounted on eef: mainly z offset
    robot_eef_offset=[0.0, 0.0, 0.12],

    sensing_joint_pose=[float(np.deg2rad(-19.19)), float(np.deg2rad(-37.32)), float(np.deg2rad(-102.487)),
                        0.0, float(np.deg2rad(-83.69)), float(np.deg2rad(-17.93))],

    place_joint_pose=[float(np.deg2rad(-38.30)), float(np.deg2rad(25.64)), float(np.deg2rad(-83.99)),
                      float(np.deg2rad(-18.57)), float(np.deg2rad(-90.48)), float(np.deg2rad(-38.97))],

    pick_speed=0.15,
    place_speed=0.15,

    # IMPORTANT: keep consistent with your current working pipeline
    # eMc (4x4)
    handeye_tf=np.eye(4).tolist(),

    rs_w=1280,
    rs_h=720,
    rs_fps=30,

    checkpoint_path="log/gsnet_base/checkpoint.tar",
    seed_feat_dim=512,
    num_point=15000,
    voxel_size=0.005,
    collision_thresh=0.01,
    collision_voxel_size=0.005,

    camera_crop_x_left=316,
    camera_crop_x_right=772,
    camera_crop_y_top=202,
    camera_crop_y_bottom=637,

    # --------- filtering / execution control ---------
    rotation_filtering=True,
    filter_angle_deg=60.0,

    min_grasp_score=0.20,        # skip execution if best score < this
    max_attempts=10,             # max perception attempts
    max_exec=10,                 # max executed pick&place cycles (successful executions)

    VIS=True,
    DO_PLACE=True,

    planning_cfg=dict(
        eef_offset=[0.0, 0.0, 0.03],
        # local in pick frame
        pre_pick_offset=[0.0, 0.0, -0.05],
        free_move_height=0.2,
    ),
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
# Realsense
# -------------------------
def rs_init(w, h, fps):
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    profile = pipe.start(cfg)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
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
# cTg_aligned.R = cTg.R @ (Ry(+90) @ Rz(-90))
# -------------------------
def _Ry(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, 0,  s],
                     [ 0, 1,  0],
                     [-s, 0,  c]], dtype=np.float64)


def _Rz(deg):
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]], dtype=np.float64)


_R_OFF = _Ry(+90.0) @ _Rz(-90.0)


def align_grasp_pose(cTg: np.ndarray) -> np.ndarray:
    cTg_aligned = cTg.copy()
    cTg_aligned[:3, :3] = cTg[:3, :3] @ _R_OFF
    return cTg_aligned


# -------------------------
# angle filter
# - approach axis = R[:,0]
# - compute angle to either camera +Z or base +Z (recommended)
# -------------------------
def filter_by_approach_angle(
    gg: GraspGroup,
    angle_deg: float,
    bTe: np.ndarray,
    eMc: np.ndarray,
    use_gravity_down: bool = True,
    apply_grasp_alignment: bool = True,
    debug: bool = True,
    debug_topk: int = 10,
):
    """
    Keep grasps whose (aligned) grasp Z-axis is within angle_deg to gravity direction.

    Prints:
    - base +Z in camera, gravity direction in camera
    - alignment rotation summary (R_off, and its z-axis v = R_off[:,2])
    - angle stats and top-k (best) angles with their scores
    - which grasps are kept

    Note:
      Here we test (aligned grasp Z) vs gravity direction.
    """
    if len(gg) == 0:
        if debug:
            print("[ANGLE] empty gg")
        return gg

    bTe = np.asarray(bTe, dtype=np.float64)
    eMc = np.asarray(eMc, dtype=np.float64)

    # base->cam
    bTc = bTe @ eMc
    z_base_cam = bTc[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # gravity direction in camera
    g_cam = (-z_base_cam) if use_gravity_down else (z_base_cam)
    g_cam = g_cam / (np.linalg.norm(g_cam) + 1e-12)

    # grasp rotations in camera
    R_all = np.asarray(gg.rotation_matrices, dtype=np.float64)  # (N,3,3)

    v = None
    R_off = None
    if apply_grasp_alignment:
        th_y = np.deg2rad(90.0)
        th_z = np.deg2rad(-90.0)
        Ry = np.array([[ np.cos(th_y), 0.0, np.sin(th_y)],
                       [ 0.0,         1.0, 0.0        ],
                       [-np.sin(th_y), 0.0, np.cos(th_y)]], dtype=np.float64)
        Rz = np.array([[ np.cos(th_z), -np.sin(th_z), 0.0],
                       [ np.sin(th_z),  np.cos(th_z), 0.0],
                       [ 0.0,           0.0,          1.0]], dtype=np.float64)
        R_off = Ry @ Rz

        # z-axis of aligned grasp in camera: cRg @ (R_off[:,2])
        v = R_off[:, 2]
        z_grasp_cam = np.einsum("nij,j->ni", R_all, v)  # (N,3)
    else:
        z_grasp_cam = R_all[:, :, 2]

    # angle(z_grasp_cam, g_cam)
    zn = np.linalg.norm(z_grasp_cam, axis=1) + 1e-12
    cosv = (z_grasp_cam @ g_cam) / zn
    cosv = np.clip(cosv, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosv))  # (N,)

    keep = np.where(ang <= float(angle_deg))[0].tolist()

    if debug:
        N = len(gg)
        scores = np.asarray(gg.scores, dtype=np.float64) if hasattr(gg, "scores") else None

        print("\n[ANGLE] ===== angle filter debug =====")
        print(f"[ANGLE] N={N}, thresh={float(angle_deg):.2f} deg, "
              f"use_gravity_down={use_gravity_down}, apply_grasp_alignment={apply_grasp_alignment}")
        print(f"[ANGLE] z_base_cam = {z_base_cam}  (should match your debug)")
        print(f"[ANGLE] g_cam      = {g_cam}")

        if apply_grasp_alignment:
            print("[ANGLE] R_off = Ry(+90) @ Rz(-90):\n", R_off)
            print(f"[ANGLE] v = R_off[:,2] (aligned grasp Z in original grasp frame) = {v}")

        print(f"[ANGLE] ang stats: min={ang.min():.2f}, max={ang.max():.2f}, mean={ang.mean():.2f}")
        print(f"[ANGLE] kept {len(keep)}/{N}")

        # show top-k smallest angles (most aligned to gravity direction)
        order = np.argsort(ang)
        k = int(min(max(debug_topk, 1), N))
        print(f"[ANGLE] top-{k} smallest angles:")
        for rank in range(k):
            i = int(order[rank])
            if scores is not None:
                print(f"  idx={i:4d}  ang={ang[i]:7.2f}  score={scores[i]:.4f}")
            else:
                print(f"  idx={i:4d}  ang={ang[i]:7.2f}")

        # also show how close each kept grasp is to the threshold
        if len(keep) > 0:
            kept_ang = ang[keep]
            print(f"[ANGLE] kept angles: min={kept_ang.min():.2f}, max={kept_ang.max():.2f}")
        else:
            print("[ANGLE] kept angles: <none>")

        print("[ANGLE] ===== end debug =====\n")

    return gg[keep] if len(keep) > 0 else gg[[]]



# -------------------------
# visualization (camera frame)
# -------------------------
def visualize_camera_scene_with_frames(cfg, crop_cloud, crop_color, cTg, grasp_meta, bTe, eMc):
    """
    Visualize everything in *camera frame*:
      - cropped scene point cloud (camera)
      - camera frame (I)
      - base frame projected into camera: cTb = inv(bTc), where bTc = bTe @ eMc
      - raw eef frame in camera: cT_eef = inv(eMc)    (since eMc is eef->cam)
      - tool frame in camera: cT_tool = cT_eef @ eefTtool (cfg["robot_eef_offset"])
      - aligned grasp frame (cTg_aligned): rotate +90 about +Y then -90 about +Z in grasp local frame
      - gripper geometry plotted at aligned grasp pose
    """
    import numpy as np
    import open3d as o3d
    from graspnetAPI.utils.utils import plot_gripper_pro_max

    score, width, depth = grasp_meta

    # ---------------- scene (camera) ----------------
    scene = o3d.geometry.PointCloud()
    scene.points = o3d.utility.Vector3dVector(crop_cloud.reshape(-1, 3))
    scene.colors = o3d.utility.Vector3dVector((crop_color.reshape(-1, 3) / 255.0).astype(np.float64))

    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    # ---------------- base frame in camera ----------------
    # Assumption consistent with your chain: bTg = bTe @ eMc @ cTg
    # => bTe: base->eef, eMc: eef->cam, so bTc = base->cam = bTe @ eMc, and cTb = inv(bTc)
    bTe = np.asarray(bTe, dtype=np.float64)
    eMc = np.asarray(eMc, dtype=np.float64)
    bTc = bTe @ eMc
    cTb = np.linalg.inv(bTc)

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.12)
    base_frame.transform(cTb)

    # ---------------- eef/tool in camera ----------------
    # eMc: eef->cam  => cT_eef = inv(eMc)
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
    R_off = _Ry(+90.0) @ _Rz(-90.0)              # local: +Y then -Z
    cTg_aligned = cTg.copy()
    cTg_aligned[:3, :3] = cTg[:3, :3] @ R_off    # right-multiply => rotate about grasp local axes

    grasp_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    grasp_frame.transform(cTg_aligned)

    # gripper geometry at aligned grasp pose
    gr = plot_gripper_pro_max(
        cTg[:3, 3],
        cTg[:3, :3],
        float(width), float(depth), float(score)
    )

    # ---------------- optional debug print ----------------
    # base +Z expressed in camera (direction)
    z_base_in_cam = bTc[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    print("[VIS DEBUG] base +Z in camera =", z_base_in_cam)

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


# -------------------------
# inference: now supports base-frame angle filter
# -------------------------
@torch.no_grad()
def infer_best_grasp(cfg, net, device, color_bgr, depth_u16, K, factor_depth, bTe, eMc):
    H, W = depth_u16.shape[:2]
    cam_info = CameraInfo(
        W, H,
        float(K[0, 0]), float(K[1, 1]),
        float(K[0, 2]), float(K[1, 2]),
        float(factor_depth)
    )
    cloud = create_point_cloud_from_depth_image(depth_u16, cam_info, organized=True)

    y0, y1 = int(cfg["camera_crop_y_top"]), int(cfg["camera_crop_y_bottom"])
    x0, x1 = int(cfg["camera_crop_x_left"]), int(cfg["camera_crop_x_right"])
    crop_color = color_bgr[y0:y1, x0:x1, :]
    crop_cloud = cloud[y0:y1, x0:x1, :]

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
        coordinates_batch, features_batch, return_index=True, return_inverse=True, device=device
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
    preds = torch.stack(grasp_preds).reshape(-1, 17).detach().cpu().numpy()
    gg = GraspGroup(preds)

    # collision
    if float(cfg["collision_thresh"]) > 0:
        mfcd = ModelFreeCollisionDetectorTorch(
            cloud.reshape(-1, 3),
            voxel_size=float(cfg["collision_voxel_size"])
        )
        cmask = mfcd.detect(gg, approach_dist=0.05, collision_thresh=float(cfg["collision_thresh"]))
        cmask = cmask.detach().cpu().numpy()
        gg = gg[~cmask]

    gg = gg.sort_by_score().nms()

    # angle filter (camera or base)
    gg = gg.sort_by_score().nms()

    if bool(cfg["rotation_filtering"]):
        gg = filter_by_approach_angle(
            gg,
            float(cfg["filter_angle_deg"]),
            bTe=bTe,
            eMc=eMc,
            use_gravity_down=True,       # base +Z up, gravity is -Z
            apply_grasp_alignment=True,  # 和执行用的 grasp alignment 保持一致
            debug=True,
        )
        if len(gg) == 0:
            return None  # or raise / return empty per your flow

    best = gg[0]
    R = best.rotation_matrix.astype(np.float64)
    t = best.translation.astype(np.float64)
    width = float(best.width)
    depth = float(best.depth)
    score = float(best.score)

    cTg = np.eye(4, dtype=np.float64)
    cTg[:3, :3] = R
    cTg[:3, 3] = t

    return cTg, (score, width, depth), (crop_cloud, crop_color)


# -------------------------
# main loop
# -------------------------
def main():
    cfg = load_cfg("demo/config.json")

    aubo = AuboController(robot_ip_=cfg["robot_ip"], eef_offset=cfg["robot_eef_offset"])
    gripper = Gripper(True)

    pipe, align, depth_scale, K = rs_init(cfg["rs_w"], cfg["rs_h"], cfg["rs_fps"])
    factor_depth = 1.0 / depth_scale

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(seed_feat_dim=int(cfg["seed_feat_dim"]), is_training=False).to(device).eval()
    ckpt = torch.load(cfg["checkpoint_path"], map_location=device)
    try:
        net.load_state_dict(ckpt)
    except Exception:
        net.load_state_dict(ckpt["model_state_dict"])

    eMc = np.array(cfg["handeye_tf"], dtype=np.float64)

    executed = 0
    attempts = 0

    try:
        # always start with open gripper
        gripper.config_gripper(open=True)

        while attempts < int(cfg["max_attempts"]) and executed < int(cfg["max_exec"]):
            attempts += 1
            print(f"\n========== Attempt {attempts}/{cfg['max_attempts']} | Exec {executed}/{cfg['max_exec']} ==========")

            # move to sensing pose (clean start)
            aubo.moveJ(cfg["sensing_joint_pose"], speed=cfg["pick_speed"])

            # current base->eef (from robot)
            _, bTe = aubo.get_current_state()
            bTe = np.array(bTe, dtype=np.float64)

            # capture
            color_bgr, depth_u16 = rs_get_frame(pipe, align)
            if color_bgr is None:
                print("[WARN] Realsense capture failed, skip.")
                continue
            
            bTc = bTe @ eMc
            bRc = bTc[:3, :3]
            z_base_in_cam = bRc.T @ np.array([0.0, 0.0, 1.0])  # because bRc maps base->cam, so base vector to cam: bRc @ v_base
            # 更直接：z_base_in_cam = bRc @ [0,0,1]
            z_base_in_cam = bRc @ np.array([0.0, 0.0, 1.0])
            print("[DEBUG] base +Z expressed in camera =", z_base_in_cam)

            # infer
            try:
                cTg_raw, meta, crop_pack = infer_best_grasp(
                    cfg, net, device, color_bgr, depth_u16, K, factor_depth, bTe=bTe, eMc=eMc
                )
            except Exception as ex:
                print(f"[WARN] inference failed: {repr(ex)}")
                continue

            score, width, gdepth = meta
            print(f"[INFO] grasp score={score:.4f}, width={width:.4f}, depth={gdepth:.4f}")

            # score gate
            if float(score) < float(cfg["min_grasp_score"]):
                print(f"[SKIP] score {score:.4f} < min_grasp_score {cfg['min_grasp_score']:.4f}")
                continue

            crop_cloud, crop_color = crop_pack

            # visualize (optional) + align for execution
            if bool(cfg["VIS"]):
                cTg_aligned = visualize_camera_scene_with_frames(cfg, crop_cloud, crop_color, cTg_raw, meta, bTe, eMc)
            else:
                cTg_aligned = align_grasp_pose(cTg_raw)

            # compute base grasp pose (keep consistent with your current working execution)
            bTg = bTe @ eMc @ cTg_aligned

            # plan
            planner = BinPickingPlanner(eMc, cfg["planning_cfg"])
            pick_wps, retreat_wps = planner.plan_pick(bTg)

            # nearest-wrist resolve (per waypoint)
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
                # back
                aubo.moveJ(cfg["sensing_joint_pose"], speed=cfg["place_speed"])
            else:
                # still return
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
