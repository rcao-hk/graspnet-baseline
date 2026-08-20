#!/usr/bin/env python3
"""Scale-Balanced Grasp inference with unified latency profiling.

For the official observation setting (``--obs``), ``grasp_inference_ms``
includes DSN instance segmentation, clustering, Scale-Balanced Grasp forward,
``pred_decode``, and ``GraspGroup`` construction. ``online_inference_ms``
additionally includes model-free collision filtering. Dataset loading,
host-to-device transfer, raw-cloud loading for collision, and result saving are
kept outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from graspnetAPI import GraspGroup
from torch.utils.data import DataLoader, Subset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "pointnet2"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "dataset"))

from dataset.scale_grasp_dataset import GraspNetDataset, collate_fn
from models.dsn import DSN, cluster
from models.scale_graspnet import GraspNet_MSCQ, pred_decode
from utils.collision_detector import ModelFreeCollisionDetectorTorch


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
                {
                    "dataset_index": int(row["dataset_index"]),
                    "scene_idx": int(row["scene_idx"]),
                    "anno_idx": int(row["anno_idx"]),
                }
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
                f"warmup={summary['warmup_samples']} | batch_size=1"
            )
            print(f"grasp inference mean = {metrics['grasp_inference_ms']['mean']:.3f} ms")
            print(f"collision mean       = {metrics['collision_ms']['mean']:.3f} ms")
            print(f"online inference mean= {metrics['online_inference_ms']['mean']:.3f} ms")
        print("=" * 88)

    def export(self, config: Dict[str, Any], scope: Dict[str, Any]) -> Tuple[str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        rows_path = os.path.join(self.output_dir, "scale_balanced_inference_profile_rows.csv")
        summary_path = os.path.join(self.output_dir, "scale_balanced_inference_profile_summary.json")
        fieldnames = [
            "dataset_index",
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
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--dataset_root", default="/data/robotarm/dataset/graspnet")
    parser.add_argument("--checkpoint_path", default="log/scale_grasp/log_full_model/checkpoint.tar")
    parser.add_argument("--seg_checkpoint_path", default="log/scale_grasp/log_insseg/checkpoint.tar")
    parser.add_argument("--dump_dir", default="experiment/scale_grasp.512")
    parser.add_argument("--camera", default="realsense", choices=["realsense", "kinect"])
    parser.add_argument("--num_point", type=int, default=20000)
    parser.add_argument("--num_view", type=int, default=300)
    parser.add_argument("--remove_outlier", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Latency profiling requires batch size 1")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--collision_thresh", type=float, default=0.01)
    parser.add_argument("--voxel_size", type=float, default=0.01,
                        help="Collision detector voxel size")
    parser.add_argument("--gaussian_noise_level", type=float, default=0.0)
    parser.add_argument("--smooth_size", type=int, default=0)
    parser.add_argument("--dropout_num", type=int, default=0)
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--downsample_voxel_size", type=float, default=0.0)
    parser.add_argument("--depth_type", default="virtual", choices=["real", "virtual"])
    parser.set_defaults(obs=True)
    parser.add_argument("--obs", dest="obs", action="store_true")
    parser.add_argument("--no_obs", dest="obs", action="store_false")
    parser.add_argument("--sample_interval", type=int, default=16)
    parser.add_argument("--num_inference", type=int, default=-1,
                        help="Number of selected samples; -1 means all")
    parser.add_argument("--skip_save", action="store_true")

    parser.add_argument("--enable_inference_timer", action="store_true")
    parser.add_argument("--timer_warmup", type=int, default=20,
                        help="Warm-up samples, not batches")
    parser.add_argument("--timer_max_samples", type=int, default=100)
    parser.add_argument("--timer_print_every", type=int, default=20)
    parser.add_argument("--timer_output_dir", type=str, default=None)
    parser.add_argument("--profile_only", action="store_true")
    args = parser.parse_args()
    args.sample_interval = max(1, int(args.sample_interval))
    args.timer_warmup = max(0, int(args.timer_warmup))
    if args.enable_inference_timer and args.batch_size != 1:
        raise ValueError(
            "Single-frame latency must be measured with --batch_size 1. "
            "Use larger batches only for throughput benchmarking."
        )
    return args


def move_batch_to_device(batch_data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for key, value in batch_data.items():
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device, non_blocking=False)
        elif torch.is_tensor(value):
            batch_data[key] = value.to(device, non_blocking=False)
    return batch_data


def parse_scene_idx(scene_name: str) -> int:
    # Expected form: scene_0100. Fall back to the trailing integer.
    try:
        return int(str(scene_name).split("_")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse scene index from {scene_name!r}") from exc


def main() -> None:
    cfgs = parse_args()
    setup_seed(0)
    print(cfgs)
    os.makedirs(cfgs.dump_dir, exist_ok=True)

    full_dataset = GraspNetDataset(
        cfgs.dataset_root,
        None,
        None,
        split=cfgs.split,
        camera=cfgs.camera,
        num_points=cfgs.num_point,
        gaussian_noise_level=cfgs.gaussian_noise_level,
        smooth_size=cfgs.smooth_size,
        dropout_num=cfgs.dropout_num,
        downsample_voxel_size=cfgs.downsample_voxel_size,
        dropout_rate=cfgs.dropout_rate,
        remove_outlier=cfgs.remove_outlier,
        augment=False,
        load_label=False,
        depth_type=cfgs.depth_type,
    )
    scene_list = full_dataset.scene_list()
    full_num_samples = len(full_dataset)

    # Match the scene/annotation order used by the other profilers. The
    # GraspNet dataset is ordered in contiguous blocks of 256 annotations.
    selected_indices = [
        idx for idx in range(full_num_samples)
        if (idx % 256) % cfgs.sample_interval == 0
    ]
    if cfgs.num_inference >= 0:
        selected_indices = selected_indices[: cfgs.num_inference]
    if cfgs.profile_only and cfgs.enable_inference_timer and cfgs.timer_max_samples > 0:
        needed = cfgs.timer_warmup + cfgs.timer_max_samples
        selected_indices = selected_indices[:needed]
    if not selected_indices:
        raise ValueError("No samples selected. Check split, interval, and num_inference.")

    test_dataset = Subset(full_dataset, selected_indices)

    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(0 + worker_id)

    dataloader = DataLoader(
        test_dataset,
        batch_size=cfgs.batch_size,
        shuffle=False,
        num_workers=cfgs.num_workers,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    print(
        f"Selected {len(selected_indices)} / {full_num_samples} samples "
        f"(interval={cfgs.sample_interval})"
    )

    device = torch.device(
        f"cuda:{cfgs.gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(cfgs.gpu_id)

    def sync_cuda() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    net = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=cfgs.num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.08,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
        obs=cfgs.obs,
    )
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", -1)
    print(f"-> loaded checkpoint {cfgs.checkpoint_path} (epoch: {epoch})")
    del checkpoint
    net.eval()

    seg_net: Optional[DSN] = None
    if cfgs.obs:
        seg_net = DSN(input_feature_dim=0)
        seg_net.to(device)
        checkpoint = torch.load(cfgs.seg_checkpoint_path, map_location="cpu", weights_only=False)
        seg_net.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        seg_net.eval()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync_cuda()

    profiler = LatencyProfiler(
        method="ScaleBalancedGrasp",
        enabled=cfgs.enable_inference_timer,
        warmup=cfgs.timer_warmup,
        max_samples=cfgs.timer_max_samples,
        print_every=cfgs.timer_print_every,
        output_dir=(
            cfgs.timer_output_dir
            or os.path.join(cfgs.dump_dir, "inference_profile")
        ),
    )

    subset_position = 0
    for batch_idx, batch_data_cpu in enumerate(dataloader):
        actual_batch_size = len(batch_data_cpu["point_clouds"])
        original_indices = selected_indices[
            subset_position : subset_position + actual_batch_size
        ]
        subset_position += actual_batch_size

        # Raw cloud loading is deliberately outside the timed region.
        raw_clouds: List[Optional[np.ndarray]] = []
        if cfgs.collision_thresh > 0:
            for original_idx in original_indices:
                cloud, _ = full_dataset.get_data(original_idx, return_raw_cloud=True)
                raw_clouds.append(cloud.reshape(-1, 3))
        else:
            raw_clouds = [None] * actual_batch_size

        # Host-to-device transfer is kept outside the timed region, matching
        # the MMGNet online-inference boundary.
        batch_data = move_batch_to_device(batch_data_cpu, device)

        measured = profiler.should_measure()
        online_start: Optional[float] = None
        if measured:
            sync_cuda()
            online_start = time.perf_counter()

        grasp_start = time.perf_counter() if measured else None
        with torch.inference_mode():
            if cfgs.obs:
                assert seg_net is not None
                end_points = seg_net(batch_data)
                batch_xyz_img = end_points["point_clouds"]
                batch_offsets = end_points["center_offsets"]
                batch_fg = torch.argmax(
                    F.softmax(end_points["foreground_logits"], dim=1),
                    dim=1,
                )
                clustered_imgs = []
                for i in range(actual_batch_size):
                    clustered_img, _ = cluster(
                        batch_xyz_img[i],
                        batch_offsets[i].permute(1, 0),
                        batch_fg[i],
                    )
                    clustered_imgs.append(clustered_img.unsqueeze(0))
                end_points["seed_cluster"] = torch.cat(clustered_imgs, dim=0)

            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Profiling mode enforces batch size one. The general loop below is
        # retained so the script still supports ordinary batched inference.
        grasp_groups: List[GraspGroup] = []
        for preds in grasp_preds:
            if torch.is_tensor(preds):
                preds = preds.detach().cpu().numpy()
            grasp_groups.append(GraspGroup(preds))

        if measured:
            sync_cuda()
            assert grasp_start is not None
            grasp_inference_ms = (time.perf_counter() - grasp_start) * 1000.0
        else:
            grasp_inference_ms = 0.0

        collision_start = time.perf_counter() if measured else None
        final_groups: List[GraspGroup] = []
        for gg, raw_cloud in zip(grasp_groups, raw_clouds):
            if cfgs.collision_thresh > 0 and len(gg) > 0:
                assert raw_cloud is not None
                detector = ModelFreeCollisionDetectorTorch(
                    raw_cloud,
                    voxel_size=cfgs.voxel_size,
                )
                collision_mask = detector.detect(
                    gg,
                    approach_dist=0.05,
                    collision_thresh=cfgs.collision_thresh,
                )
                if torch.is_tensor(collision_mask):
                    collision_mask = collision_mask.detach().cpu().numpy()
                gg = gg[~np.asarray(collision_mask, dtype=bool)]
            final_groups.append(gg)

        if measured:
            sync_cuda()
            assert collision_start is not None and online_start is not None
            collision_ms_total = (time.perf_counter() - collision_start) * 1000.0
            online_ms_total = (time.perf_counter() - online_start) * 1000.0
        else:
            collision_ms_total = 0.0
            online_ms_total = 0.0

        # With profiling enabled B=1, so these are exact per-sample values.
        per_sample_grasp_ms = grasp_inference_ms / actual_batch_size
        per_sample_collision_ms = collision_ms_total / actual_batch_size
        per_sample_online_ms = online_ms_total / actual_batch_size

        for local_idx, (original_idx, gg_pre, gg_final) in enumerate(
            zip(original_indices, grasp_groups, final_groups)
        ):
            scene_name = scene_list[original_idx]
            scene_idx = parse_scene_idx(scene_name)
            anno_idx = original_idx % 256
            profiler.add_row(
                {
                    "dataset_index": int(original_idx),
                    "scene_idx": int(scene_idx),
                    "anno_idx": int(anno_idx),
                    "num_input_points": int(cfgs.num_point),
                    "num_grasps_pre_collision": int(len(gg_pre)),
                    "num_grasps_final": int(len(gg_final)),
                    "grasp_inference_ms": float(per_sample_grasp_ms),
                    "collision_ms": float(per_sample_collision_ms),
                    "online_inference_ms": float(per_sample_online_ms),
                },
                measured=measured,
            )

            if not cfgs.skip_save:
                save_dir = os.path.join(cfgs.dump_dir, scene_name, cfgs.camera)
                os.makedirs(save_dir, exist_ok=True)
                gg_final.save_npy(os.path.join(save_dir, f"{anno_idx:04d}.npy"))

        if cfgs.profile_only and profiler.done():
            break

    config = {
        "split": cfgs.split,
        "camera": cfgs.camera,
        "checkpoint_path": cfgs.checkpoint_path,
        "seg_checkpoint_path": cfgs.seg_checkpoint_path if cfgs.obs else None,
        "obs": cfgs.obs,
        "num_point": cfgs.num_point,
        "sample_interval": cfgs.sample_interval,
        "collision_thresh": cfgs.collision_thresh,
        "collision_voxel_size": cfgs.voxel_size,
        "batch_size": cfgs.batch_size,
    }
    scope = {
        "input_boundary": "collated GPU dataset batch",
        "grasp_inference": (
            "DSN segmentation/clustering (when obs=True) + Scale-Balanced "
            "Grasp forward + pred_decode + GraspGroup"
        ),
        "collision": "ModelFreeCollisionDetectorTorch",
        "online_inference": "grasp_inference + collision",
        "excluded": [
            "DataLoader/dataset loading",
            "host-to-device transfer",
            "raw-cloud loading for collision",
            "result saving",
        ],
    }
    profiler.print_summary(config=config, scope=scope)
    if cfgs.enable_inference_timer:
        profiler.export(config=config, scope=scope)


if __name__ == "__main__":
    main()
