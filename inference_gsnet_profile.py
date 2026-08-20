#!/usr/bin/env python3
"""GSNet inference with deployment-oriented latency profiling.

``grasp_inference_ms`` measures the path from prepared GPU tensors through
Minkowski sparse preprocessing, GSNet forward, ``pred_decode``, and
``GraspGroup`` construction. ``online_inference_ms`` additionally includes
model-free collision filtering. RGB-D loading, depth-to-point-cloud
construction, workspace masking, point sampling, tensor construction/transfer,
and result saving are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))

import cv2
import MinkowskiEngine as ME
import numpy as np
import scipy.io as scio
import torch
from PIL import Image

from graspnetAPI import GraspGroup
from models.GSNet import GraspNet, pred_decode
from utils.collision_detector import ModelFreeCollisionDetectorTorch
from utils.data_utils import (
    CameraInfo,
    add_gaussian_noise_depth_map,
    apply_smoothing,
    create_point_cloud_from_depth_image,
    depthaware_perlin_dropout_masks,
    get_workspace_mask,
    sample_points,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Avoid the PyTorch DataLoader "received 0 items of ancdata" failure.
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft_limit = min(500000, hard_limit)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
print("soft limit:", new_soft_limit, "hard limit:", hard_limit)


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _metric_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }


@dataclass
class LatencyProfiler:
    method: str
    enabled: bool
    warmup: int
    max_samples: int
    print_every: int
    output_dir: str
    seen_samples: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    def should_measure(self) -> bool:
        return (
            self.enabled
            and self.seen_samples >= self.warmup
            and (self.max_samples <= 0 or len(self.rows) < self.max_samples)
        )

    def add_row(self, row: Dict[str, Any], measured: bool) -> None:
        self.seen_samples += 1
        if self.enabled and self.seen_samples == self.warmup:
            print(f"[{self.method}-PROFILE] Warm-up finished: {self.warmup} samples")
        if not measured:
            return

        self.rows.append(row)
        n = len(self.rows)
        if self.print_every > 0 and (n == 1 or n % self.print_every == 0):
            vals = [float(r["online_inference_ms"]) for r in self.rows]
            print(
                f"[{self.method}-PROFILE] samples={n} "
                f"last={row['online_inference_ms']:.3f} ms "
                f"mean={np.mean(vals):.3f} ms"
            )

    def done(self) -> bool:
        return self.enabled and self.max_samples > 0 and len(self.rows) >= self.max_samples

    def summary(self, config: Dict[str, Any], scope: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {
            key: _metric_stats([float(row[key]) for row in self.rows])
            for key in ("grasp_inference_ms", "collision_ms", "online_inference_ms")
        }
        online_total = metrics.get("online_inference_ms", {}).get("total", 0.0)
        return {
            "method": self.method,
            "profiled_samples": len(self.rows),
            "warmup_samples": self.warmup,
            "processed_samples": self.seen_samples,
            "metrics_ms": metrics,
            "throughput_fps": (
                float(len(self.rows) * 1000.0 / online_total)
                if online_total > 0.0
                else None
            ),
            "measurement_scope": scope,
            "config": config,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu": (
                    torch.cuda.get_device_name(torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "sample_order": [
                {"scene_idx": int(row["scene_idx"]), "anno_idx": int(row["anno_idx"])}
                for row in self.rows
            ],
        }

    def print_summary(self, config: Dict[str, Any], scope: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        summary = self.summary(config=config, scope=scope)
        print("\n" + "=" * 88)
        print(f"[{self.method}-PROFILE] Final latency summary")
        if summary["profiled_samples"] == 0:
            print("No measured samples. Reduce the warm-up or process more frames.")
        else:
            metrics = summary["metrics_ms"]
            print(
                f"samples={summary['profiled_samples']} | "
                f"warmup={summary['warmup_samples']}"
            )
            print(f"grasp inference mean = {metrics['grasp_inference_ms']['mean']:.3f} ms")
            print(f"collision mean       = {metrics['collision_ms']['mean']:.3f} ms")
            print(f"online inference mean= {metrics['online_inference_ms']['mean']:.3f} ms")
        print("=" * 88)

    def export(self, config: Dict[str, Any], scope: Dict[str, Any]) -> Tuple[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        rows_path = os.path.join(self.output_dir, "gsnet_inference_profile_rows.csv")
        summary_path = os.path.join(self.output_dir, "gsnet_inference_profile_summary.json")
        fieldnames = [
            "scene_idx",
            "anno_idx",
            "num_input_points",
            "num_grasps_pre_collision",
            "num_grasps_final",
            "grasp_inference_ms",
            "collision_ms",
            "online_inference_ms",
        ]
        with open(rows_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary(config=config, scope=scope), f, indent=2, ensure_ascii=False)
        print(f"[{self.method}-PROFILE] Rows:    {rows_path}")
        print(f"[{self.method}-PROFILE] Summary: {summary_path}")
        return rows_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test_seen", choices=["test", "test_seen", "test_similar", "test_novel"])
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--seed_feat_dim", default=512, type=int)
    parser.add_argument("--dataset_root", default="/media/gpuadmin/rcao/dataset/graspnet")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--num_point", type=int, default=15000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--voxel_size", type=float, default=0.005)
    parser.add_argument("--collision_voxel_size", type=float, default=0.01)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--data_type", default="real", choices=["real", "syn", "noise"])
    parser.add_argument("--smooth_size", type=int, default=1)
    parser.add_argument("--gaussian_noise_level", type=float, default=0.0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--dropout_min_size", type=int, default=200)
    parser.add_argument("--pc_sparse_level", type=int, default=0)
    parser.add_argument("--sample_interval", type=int, default=16)
    parser.add_argument("--skip_save", action="store_true")

    parser.add_argument("--enable_inference_timer", action="store_true")
    parser.add_argument("--timer_warmup", type=int, default=20)
    parser.add_argument("--timer_max_samples", type=int, default=100)
    parser.add_argument("--timer_print_every", type=int, default=20)
    parser.add_argument("--timer_output_dir", type=str, default=None)
    parser.add_argument("--profile_only", action="store_true")
    args = parser.parse_args()
    args.sample_interval = max(1, int(args.sample_interval))
    args.timer_warmup = max(0, int(args.timer_warmup))
    return args


def get_scene_list(split: str) -> List[int]:
    if split == "test":
        return list(range(100, 190))
    if split == "test_seen":
        return list(range(100, 130))
    if split == "test_similar":
        return list(range(130, 160))
    if split == "test_novel":
        return list(range(160, 190))
    raise ValueError(f"Unsupported split: {split}")


def save_run_config_json(cfgs: argparse.Namespace, extra: Dict[str, Any]) -> None:
    os.makedirs(cfgs.dump_dir, exist_ok=True)
    path = os.path.join(cfgs.dump_dir, f"{cfgs.split}_{cfgs.camera}.json")
    payload = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv,
        "cfgs": vars(cfgs),
        "extra": extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"[CFG] Saved run config to: {path}")


def main() -> None:
    cfgs = parse_args()
    setup_seed(0)

    if cfgs.data_type in ("real", "syn"):
        cfgs.smooth_size = 1
        cfgs.gaussian_noise_level = 0.0
        cfgs.dropout_rate = 0.0
        cfgs.pc_sparse_level = 0
    elif cfgs.pc_sparse_level > 0:
        sparse_levels = [5120, 2048, 1024, 512]
        if not 1 <= cfgs.pc_sparse_level <= 4:
            raise ValueError("pc_sparse_level must be in [0, 4]")
        cfgs.num_point = sparse_levels[cfgs.pc_sparse_level - 1]

    print(cfgs)
    os.makedirs(cfgs.dump_dir, exist_ok=True)
    device = torch.device(
        f"cuda:{cfgs.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(cfgs.gpu_id)

    def sync_cuda() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    net = GraspNet(seed_feat_dim=cfgs.seed_feat_dim, is_training=False)
    net.to(device)
    net.eval()
    checkpoint = torch.load(cfgs.checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_cuda()

    save_run_config_json(
        cfgs,
        extra={
            "checkpoint_path": os.path.abspath(cfgs.checkpoint_path),
            "seed": 0,
        },
    )

    profiler = LatencyProfiler(
        method="GSNet",
        enabled=cfgs.enable_inference_timer,
        warmup=cfgs.timer_warmup,
        max_samples=cfgs.timer_max_samples,
        print_every=cfgs.timer_print_every,
        output_dir=(
            cfgs.timer_output_dir
            or os.path.join(cfgs.dump_dir, "inference_profile")
        ),
    )

    width, height = 1280, 720

    def process_scene(scene_idx: int) -> bool:
        for anno_idx in range(0, 256, cfgs.sample_interval):
            depth_raw_path: Optional[str] = None
            if cfgs.data_type == "real":
                rgb_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/rgb/{anno_idx:04d}.png")
                depth_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/depth/{anno_idx:04d}.png")
                mask_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/label/{anno_idx:04d}.png")
            elif cfgs.data_type == "syn":
                rgb_path = os.path.join(cfgs.dataset_root, f"virtual_scenes/scene_{scene_idx:04d}/{cfgs.camera}/{anno_idx:04d}_rgb.png")
                depth_path = os.path.join(cfgs.dataset_root, f"virtual_scenes/scene_{scene_idx:04d}/{cfgs.camera}/{anno_idx:04d}_depth.png")
                mask_path = os.path.join(cfgs.dataset_root, f"virtual_scenes/scene_{scene_idx:04d}/{cfgs.camera}/{anno_idx:04d}_label.png")
            else:
                rgb_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/rgb/{anno_idx:04d}.png")
                depth_path = os.path.join(cfgs.dataset_root, f"virtual_scenes/scene_{scene_idx:04d}/{cfgs.camera}/{anno_idx:04d}_depth.png")
                depth_raw_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/depth/{anno_idx:04d}.png")
                mask_path = os.path.join(cfgs.dataset_root, f"virtual_scenes/scene_{scene_idx:04d}/{cfgs.camera}/{anno_idx:04d}_label.png")

            meta_path = os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/meta/{anno_idx:04d}.mat")
            color = np.asarray(Image.open(rgb_path), dtype=np.float32) / 255.0
            depth = np.asarray(Image.open(depth_path))
            seg = np.asarray(Image.open(mask_path))
            meta = scio.loadmat(meta_path)
            intrinsics = meta["intrinsic_matrix"]
            factor_depth = meta["factor_depth"]
            camera_info = CameraInfo(
                width,
                height,
                intrinsics[0][0],
                intrinsics[1][1],
                intrinsics[0][2],
                intrinsics[1][2],
                factor_depth,
            )

            cloud = create_point_cloud_from_depth_image(depth, camera_info, organized=True)
            camera_poses = np.load(os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/camera_poses.npy"))
            align_mat = np.load(os.path.join(cfgs.dataset_root, f"scenes/scene_{scene_idx:04d}/{cfgs.camera}/cam0_wrt_table.npy"))
            trans = np.dot(align_mat, camera_poses[anno_idx])
            workspace_mask = get_workspace_mask(cloud, seg, trans=trans, organized=True, outlier=0.02)
            mask = (depth > 0) & workspace_mask

            depth_used = depth.copy()
            dropout_mask = None
            noisy_cloud = None
            if cfgs.smooth_size > 1:
                depth_used = apply_smoothing(depth_used, size=cfgs.smooth_size)
                noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)
            if cfgs.gaussian_noise_level > 0:
                depth_noisy = add_gaussian_noise_depth_map(
                    depth_used.astype(np.float32),
                    scale=factor_depth,
                    level=cfgs.gaussian_noise_level,
                    valid_min_depth=0.1,
                )
                depth_used = np.clip(depth_noisy, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                noisy_cloud = create_point_cloud_from_depth_image(depth_used, camera_info, organized=True)
            if cfgs.dropout_rate > 0:
                real_depth = (
                    np.asarray(Image.open(depth_raw_path))
                    if depth_raw_path is not None and os.path.exists(depth_raw_path)
                    else depth
                )
                drop_depth, drop_perlin = depthaware_perlin_dropout_masks(
                    depth_raw=real_depth,
                    depth_clear=depth_used,
                    seg=seg,
                    dropout_rate=cfgs.dropout_rate,
                    seed=0,
                    strict_match=True,
                    base_res=16,
                    octaves=4,
                    persistence=0.5,
                    use_bbox_local_noise=True,
                )
                dropout_mask = drop_depth | drop_perlin
            if dropout_mask is not None:
                mask &= ~dropout_mask

            if noisy_cloud is not None:
                cloud_masked = noisy_cloud[mask]
                scene_cloud_for_collision = noisy_cloud.reshape(-1, 3)
            else:
                cloud_masked = cloud[mask]
                scene_cloud_for_collision = cloud.reshape(-1, 3)
            color_masked = color[mask]
            idxs = sample_points(len(cloud_masked), cfgs.num_point)
            cloud_sampled = cloud_masked[idxs].astype(np.float32)
            color_sampled = color_masked[idxs].astype(np.float32)

            # Tensor construction and host-to-device transfer are kept outside
            # the timed region, matching the MMGNet online-inference boundary.
            cloud_tensor = torch.as_tensor(cloud_sampled, dtype=torch.float32, device=device)
            color_tensor = torch.as_tensor(color_sampled, dtype=torch.float32, device=device)
            coors_tensor = torch.as_tensor(cloud_sampled / cfgs.voxel_size, dtype=torch.int32, device=device)
            feats_tensor = torch.ones_like(cloud_tensor, dtype=torch.float32, device=device)

            measured = profiler.should_measure()
            online_start: Optional[float] = None
            if measured:
                sync_cuda()
                online_start = time.perf_counter()

            grasp_start = time.perf_counter() if measured else None
            coordinates_batch, features_batch = ME.utils.sparse_collate(
                [coors_tensor], [feats_tensor], dtype=torch.float32
            )
            coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
                coordinates_batch,
                features_batch,
                return_index=True,
                return_inverse=True,
                device=device,
            )
            batch_data = {
                "point_clouds": cloud_tensor.unsqueeze(0),
                "cloud_colors": color_tensor.unsqueeze(0),
                "coors": coordinates_batch,
                "feats": features_batch,
                "quantize2original": quantize2original,
            }
            with torch.inference_mode():
                end_points = net(batch_data)
                grasp_preds = pred_decode(end_points)
            preds = grasp_preds[0]
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            gg = GraspGroup(preds)

            if measured:
                sync_cuda()
                assert grasp_start is not None
                grasp_inference_ms = (time.perf_counter() - grasp_start) * 1000.0
            else:
                grasp_inference_ms = 0.0

            num_grasps_pre = len(gg)
            collision_start = time.perf_counter() if measured else None
            if cfgs.collision_thresh > 0 and len(gg) > 0:
                detector = ModelFreeCollisionDetectorTorch(
                    scene_cloud_for_collision,
                    voxel_size=cfgs.collision_voxel_size,
                )
                collision_mask = detector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                if torch.is_tensor(collision_mask):
                    collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~np.asarray(collision_mask, dtype=bool)]

            if measured:
                sync_cuda()
                assert collision_start is not None and online_start is not None
                collision_ms = (time.perf_counter() - collision_start) * 1000.0
                online_inference_ms = (time.perf_counter() - online_start) * 1000.0
            else:
                collision_ms = 0.0
                online_inference_ms = 0.0

            profiler.add_row(
                {
                    "scene_idx": int(scene_idx),
                    "anno_idx": int(anno_idx),
                    "num_input_points": int(cfgs.num_point),
                    "num_grasps_pre_collision": int(num_grasps_pre),
                    "num_grasps_final": int(len(gg)),
                    "grasp_inference_ms": float(grasp_inference_ms),
                    "collision_ms": float(collision_ms),
                    "online_inference_ms": float(online_inference_ms),
                },
                measured=measured,
            )

            if not cfgs.skip_save:
                save_dir = os.path.join(cfgs.dump_dir, f"scene_{scene_idx:04d}", cfgs.camera)
                os.makedirs(save_dir, exist_ok=True)
                gg.save_npy(os.path.join(save_dir, f"{anno_idx:04d}.npy"))

            if cfgs.profile_only and profiler.done():
                return True
        return False

    for scene_idx in get_scene_list(cfgs.split):
        if process_scene(scene_idx):
            break

    config = {
        "split": cfgs.split,
        "camera": cfgs.camera,
        "checkpoint_path": cfgs.checkpoint_path,
        "num_point": cfgs.num_point,
        "voxel_size": cfgs.voxel_size,
        "sample_interval": cfgs.sample_interval,
        "collision_thresh": cfgs.collision_thresh,
        "collision_voxel_size": cfgs.collision_voxel_size,
        "batch_size": 1,
    }
    scope = {
        "input_boundary": "prepared GPU point-cloud tensors",
        "grasp_inference": (
            "sparse collate/quantize + GSNet forward + pred_decode + GraspGroup"
        ),
        "collision": "ModelFreeCollisionDetectorTorch",
        "online_inference": "grasp_inference + collision",
        "excluded": [
            "RGB-D file I/O",
            "depth-to-point-cloud construction",
            "workspace masking",
            "point sampling",
            "tensor construction and host-to-device transfer",
            "result saving",
        ],
    }
    profiler.print_summary(config=config, scope=scope)
    if cfgs.enable_inference_timer:
        profiler.export(config=config, scope=scope)


if __name__ == "__main__":
    main()
