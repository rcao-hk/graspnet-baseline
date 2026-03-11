#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute object-level top-k grasp success statistics on GraspNet predictions.

This script follows the Scale-Balanced-Grasp evaluation flow up to the
object-wise evaluator output, then computes per-object top-k success metrics
instead of scene-level AP.

Main outputs (one CSV row per object instance in each annotation):
    - object_local_id / seg_instance_id:
        unified object key used for downstream merge, matching the instance id
        in the segmentation / label image (and in compute_instance_alignment_mae.py)
    - evaluator_object_local_id:
        original evaluator-side contiguous object index (0..num_objects-1)
    - object_label_zero_based:
        original evaluator-side zero-based object id; in GraspNet this is usually
        equal to seg_instance_id - 1
    - succ_num_topk:   number of successful grasps among the top-k grasps
    - succ_rate_topk:  succ_num_topk / k
    - succ_rate_avail_topk: succ_num_topk / min(k, num_pred_after_eval)

A grasp is considered successful when its evaluator score satisfies
    0 < score <= success_mu
which matches the force-closure thresholding style used by GraspNet eval.

Example
-------
python compute_object_topk_success.py \
    --graspnet_root /data/graspnet \
    --dump_folder logs/dump_full_model \
    --camera realsense \
    --split novel \
    --topk 10 \
    --success_mu 0.8 \
    --output_csv /tmp/object_top10_success_novel.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# GraspNet / Open3D imports
# -----------------------------
try:
    from graspnetAPI.graspnet_eval import GraspNetEval
    from graspnetAPI.grasp import GraspGroup
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.eval_utils import (
        get_scene_name,
        create_table_points,
        transform_points,
        voxel_sample_points,
        compute_closest_points,
        collision_detection,
        get_grasp_score,
        GraspQualityConfigFactory,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import graspnetAPI. Please install graspnetAPI in the current "
        "environment before running this script. Original error: %s" % (e,)
    )


# -----------------------------
# Helpers
# -----------------------------
SPLIT_TO_SCENES = {
    "seen": list(range(100, 130)),
    "similar": list(range(130, 160)),
    "novel": list(range(160, 190)),
    "all": list(range(100, 190)),
}


def parse_scene_ids(split: str, scene_ids: Optional[str]) -> List[int]:
    if scene_ids is None or scene_ids.strip() == "":
        if split not in SPLIT_TO_SCENES:
            raise ValueError(f"Unknown split: {split}")
        return SPLIT_TO_SCENES[split]

    out: List[int] = []
    parts = [x.strip() for x in scene_ids.split(",") if x.strip()]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                raise ValueError(f"Invalid range: {part}")
            out.extend(list(range(a, b + 1)))
        else:
            out.append(int(part))
    out = sorted(set(out))
    return out


def apply_scale_filter(gg_array: np.ndarray, scale: str) -> np.ndarray:
    if gg_array.size == 0:
        return gg_array
    if scale == "all":
        return gg_array
    widths = gg_array[:, 1]
    if scale == "small":
        mask = widths < 0.04
    elif scale == "medium":
        mask = (widths >= 0.04) & (widths < 0.07)
    elif scale == "large":
        mask = widths >= 0.07
    else:
        raise ValueError(f"Unknown scale: {scale}")
    return gg_array[mask]


def clip_width_inplace(gg_array: np.ndarray, max_width: float) -> np.ndarray:
    if gg_array.size == 0:
        return gg_array
    gg_array = gg_array.copy()
    gg_array[gg_array[:, 1] < 0, 1] = 0.0
    gg_array[gg_array[:, 1] > max_width, 1] = max_width
    return gg_array


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_summary(df: pd.DataFrame, args: argparse.Namespace) -> Dict:
    summary = {
        "num_rows": int(len(df)),
        "num_unique_scene_ann": int(df[["scene_id", "ann_id"]].drop_duplicates().shape[0])
        if len(df) > 0
        else 0,
        "num_unique_objects": int(df[["scene_id", "ann_id", "object_local_id"]].drop_duplicates().shape[0])
        if len(df) > 0
        else 0,
        "mean_succ_num_topk": float(df["succ_num_topk"].mean()) if len(df) > 0 else 0.0,
        "mean_succ_rate_topk": float(df["succ_rate_topk"].mean()) if len(df) > 0 else 0.0,
        "mean_succ_rate_avail_topk": float(df["succ_rate_avail_topk"].mean()) if len(df) > 0 else 0.0,
        "mean_num_pred_after_eval": float(df["num_pred_after_eval"].mean()) if len(df) > 0 else 0.0,
        "config": {
            "graspnet_root": args.graspnet_root,
            "dump_folder": args.dump_folder,
            "camera": args.camera,
            "split": args.split,
            "scene_ids": args.scene_ids,
            "scale": args.scale,
            "topk": args.topk,
            "success_mu": args.success_mu,
            "max_width": args.max_width,
            "ann_start": args.ann_start,
            "ann_end": args.ann_end,
            "sample_interval": args.sample_interval,
            "num_workers": args.num_workers,
            "mp_start_method": args.mp_start_method,
        },
    }

    if len(df) > 0:
        # mean object-level stats aggregated by scene-annotation
        per_ann = (
            df.groupby(["scene_id", "ann_id"], as_index=False)[
                ["succ_num_topk", "succ_rate_topk", "succ_rate_avail_topk"]
            ]
            .mean()
            .to_dict(orient="records")
        )
        summary["per_scene_ann_mean"] = per_ann[:50]  # keep summary compact
    return summary


# -----------------------------
# Evaluator
# -----------------------------
class ObjectTopKEval(GraspNetEval):
    """Object-level evaluator built on top of GraspNetEval."""

    @staticmethod
    def eval_grasp_scalebalanced(
        grasp_group: GraspGroup,
        models: Sequence[np.ndarray],
        dexnet_models: Sequence[object],
        poses: Sequence[np.ndarray],
        config: dict,
        table: Optional[np.ndarray] = None,
        voxel_size: float = 0.008,
        top_k_scene: int = 50,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Reproduce the custom object-wise preselection used in
        Scale-Balanced-Grasp's evaluate_scale.py before collision detection and
        force-closure scoring.
        """
        num_models = len(models)

        # Grasp NMS.
        grasp_group = grasp_group.nms(0.03, 30.0 / 180.0 * np.pi)

        if len(grasp_group) == 0:
            return ([np.empty((0, 17), dtype=np.float64) for _ in range(num_models)],
                    [np.empty((0,), dtype=np.float64) for _ in range(num_models)],
                    [np.empty((0,), dtype=bool) for _ in range(num_models)])

        # Assign grasps to the nearest object model point in the merged scene.
        model_trans_list: List[np.ndarray] = []
        seg_mask: List[np.ndarray] = []
        for i, model in enumerate(models):
            model_trans = transform_points(model, poses[i])
            seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
            model_trans_list.append(model_trans)
            seg_mask.append(seg)
        seg_mask_arr = np.concatenate(seg_mask, axis=0)
        merged_scene = np.concatenate(model_trans_list, axis=0)

        indices = compute_closest_points(grasp_group.translations, merged_scene)
        model_to_grasp = seg_mask_arr[indices]

        pre_grasp_list: List[np.ndarray] = []
        for i in range(num_models):
            grasp_i = grasp_group[model_to_grasp == i]
            if len(grasp_i) == 0:
                pre_grasp_list.append(np.empty((0, 17), dtype=np.float64))
                continue
            grasp_i.sort_by_score()
            pre_grasp_list.append(grasp_i[:5].grasp_group_array)

        nonempty = [x for x in pre_grasp_list if len(x) > 0]
        if len(nonempty) == 0:
            return ([np.empty((0, 17), dtype=np.float64) for _ in range(num_models)],
                    [np.empty((0,), dtype=np.float64) for _ in range(num_models)],
                    [np.empty((0,), dtype=bool) for _ in range(num_models)])

        all_grasp_list = np.vstack(nonempty)
        remain_mask = np.argsort(all_grasp_list[:, 0])[::-1]
        min_score = all_grasp_list[remain_mask[min(top_k_scene - 1, len(remain_mask) - 1)], 0]

        grasp_list: List[np.ndarray] = []
        for i in range(num_models):
            if len(pre_grasp_list[i]) == 0:
                grasp_list.append(np.empty((0, 17), dtype=np.float64))
                continue
            remain_mask_i = pre_grasp_list[i][:, 0] >= min_score
            grasp_list.append(pre_grasp_list[i][remain_mask_i])

        if table is not None:
            merged_scene_with_table = np.concatenate([merged_scene, table], axis=0)
        else:
            merged_scene_with_table = merged_scene

        collision_mask_list, _, dexgrasp_list = collision_detection(
            grasp_list,
            model_trans_list,
            dexnet_models,
            poses,
            merged_scene_with_table,
            outlier=0.05,
            return_dexgrasps=True,
        )

        force_closure_quality_config = {}
        fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
        for value_fc in fc_list:
            value_fc = round(float(value_fc), 2)
            config["metrics"]["force_closure"]["friction_coef"] = value_fc
            force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
                config["metrics"]["force_closure"]
            )

        score_list: List[np.ndarray] = []
        for i in range(num_models):
            collision_mask = collision_mask_list[i]
            dexgrasps = dexgrasp_list[i]
            dexnet_model = dexnet_models[i]
            scores: List[float] = []
            for grasp_id in range(len(dexgrasps)):
                if collision_mask[grasp_id]:
                    scores.append(-1.0)
                    continue
                if dexgrasps[grasp_id] is None:
                    scores.append(-1.0)
                    continue
                grasp = dexgrasps[grasp_id]
                score = get_grasp_score(
                    grasp, dexnet_model, fc_list, force_closure_quality_config
                )
                scores.append(score)
            score_list.append(np.array(scores, dtype=np.float64))

        return grasp_list, score_list, collision_mask_list

    def eval_scene_object_topk(
        self,
        scene_id: int,
        dump_folder: str,
        topk: int = 10,
        success_mu: float = 0.8,
        top_k_scene: int = 50,
        scale: str = "all",
        max_width: float = 0.1,
        ann_ids: Optional[Sequence[int]] = None,
        skip_missing: bool = False,
        verbose: bool = True,
    ) -> List[Dict]:
        """Evaluate one scene and return one row per object instance."""
        config = get_config()
        table = create_table_points(
            1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008
        )

        if ann_ids is None:
            ann_ids = list(range(256))

        # Same design as official evaluator / scale-balanced evaluator:
        # load scene models once using ann_id=0.
        model_list, dexmodel_list, _ = self.get_scene_models(scene_id, ann_id=0)
        model_sampled_list = [voxel_sample_points(model, 0.008) for model in model_list]
        num_models = len(model_sampled_list)

        rows: List[Dict] = []
        scene_name = get_scene_name(scene_id)

        for ann_id in ann_ids:
            npy_path = os.path.join(dump_folder, scene_name, self.camera, f"{ann_id:04d}.npy")
            if not os.path.exists(npy_path):
                msg = f"Missing prediction file: {npy_path}"
                if skip_missing:
                    print(f"[WARN] {msg}")
                    continue
                raise FileNotFoundError(msg)

            grasp_group = GraspGroup().from_npy(npy_path)
            cur_obj_ids, pose_list, camera_pose, align_mat = self.get_model_poses(scene_id, ann_id)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

            raw_count = len(grasp_group)
            gg_array = grasp_group.grasp_group_array
            gg_array = clip_width_inplace(gg_array, max_width=max_width)
            gg_array = apply_scale_filter(gg_array, scale=scale)
            filtered_count = int(len(gg_array))

            grasp_group.grasp_group_array = gg_array

            if filtered_count == 0:
                grasp_list = [np.empty((0, 17), dtype=np.float64) for _ in range(num_models)]
                score_list = [np.empty((0,), dtype=np.float64) for _ in range(num_models)]
                collision_mask_list = [np.empty((0,), dtype=bool) for _ in range(num_models)]
            else:
                grasp_list, score_list, collision_mask_list = self.eval_grasp_scalebalanced(
                    grasp_group=grasp_group,
                    models=model_sampled_list,
                    dexnet_models=dexmodel_list,
                    poses=pose_list,
                    config=config,
                    table=table_trans,
                    voxel_size=0.008,
                    top_k_scene=top_k_scene,
                )

            if len(cur_obj_ids) != num_models:
                raise RuntimeError(
                    f"Scene {scene_id} ann {ann_id}: object count mismatch. "
                    f"models_from_ann0={num_models}, current_ann={len(cur_obj_ids)}"
                )

            for evaluator_object_local_id in range(num_models):
                object_label_zero_based = int(cur_obj_ids[evaluator_object_local_id])
                # Unify the downstream merge key with the segmentation / label-image id.
                # In GraspNet, evaluator object ids are typically zero-based while the label
                # image uses positive ids, so we expose seg_instance_id = object_id + 1.
                seg_instance_id = int(object_label_zero_based + 1)
                object_local_id = int(seg_instance_id)

                grasps = grasp_list[evaluator_object_local_id]
                scores = score_list[evaluator_object_local_id]
                collision_mask = collision_mask_list[evaluator_object_local_id]

                if len(grasps) == 0:
                    row = {
                        "scene_id": int(scene_id),
                        "ann_id": int(ann_id),
                        "object_local_id": int(object_local_id),
                        "seg_instance_id": int(seg_instance_id),
                        "evaluator_object_local_id": int(evaluator_object_local_id),
                        "object_label_zero_based": int(object_label_zero_based),
                        "object_label": int(object_local_id),
                        "object_uid": f"scene_{scene_id:04d}_ann_{ann_id:04d}_obj_{object_local_id:03d}",
                        "scale": scale,
                        "topk": int(topk),
                        "success_mu": float(success_mu),
                        "raw_pred_count": int(raw_count),
                        "pred_count_after_scale": int(filtered_count),
                        "num_pred_after_eval": 0,
                        "num_pred_topk": 0,
                        "succ_num_topk": 0,
                        "succ_rate_topk": 0.0,
                        "succ_rate_avail_topk": 0.0,
                        "mean_conf_topk": np.nan,
                        "mean_fc_score_topk": np.nan,
                    }
                    rows.append(row)
                    continue

                conf = grasps[:, 0]
                order = np.argsort(-conf)
                grasps = grasps[order]
                scores = scores[order]
                collision_mask = collision_mask[order]

                topk_eff = min(topk, len(scores))
                success_mask = (scores > 0) & (scores <= success_mu)
                # Collision or invalid grasps are already set to negative scores by the evaluator.
                # We keep the explicit collision flag for logging only.
                succ_num = int(success_mask[:topk_eff].sum())
                succ_rate_topk = float(succ_num / float(topk)) if topk > 0 else 0.0
                succ_rate_avail_topk = float(succ_num / float(topk_eff)) if topk_eff > 0 else 0.0

                row = {
                    "scene_id": int(scene_id),
                    "ann_id": int(ann_id),
                    "object_local_id": int(object_local_id),
                    "seg_instance_id": int(seg_instance_id),
                    "evaluator_object_local_id": int(evaluator_object_local_id),
                    "object_label_zero_based": int(object_label_zero_based),
                    "object_label": int(object_local_id),
                    "object_uid": f"scene_{scene_id:04d}_ann_{ann_id:04d}_obj_{object_local_id:03d}",
                    "scale": scale,
                    "topk": int(topk),
                    "success_mu": float(success_mu),
                    "raw_pred_count": int(raw_count),
                    "pred_count_after_scale": int(filtered_count),
                    "num_pred_after_eval": int(len(scores)),
                    "num_pred_topk": int(topk_eff),
                    "succ_num_topk": int(succ_num),
                    "succ_rate_topk": float(succ_rate_topk),
                    "succ_rate_avail_topk": float(succ_rate_avail_topk),
                    "mean_conf_topk": float(np.mean(conf[order][:topk_eff])) if topk_eff > 0 else np.nan,
                    "mean_fc_score_topk": float(np.mean(scores[:topk_eff])) if topk_eff > 0 else np.nan,
                    "num_collision_topk": int(collision_mask[:topk_eff].sum()) if topk_eff > 0 else 0,
                    "num_success_topk_check": int(success_mask[:topk_eff].sum()) if topk_eff > 0 else 0,
                }
                rows.append(row)

            print(
                f"\r[scene {scene_id:04d}] ann {ann_id:04d} | raw={raw_count:4d} | "
                f"after_scale={filtered_count:4d} | rows={len(rows)}",
                end="",
                flush=True,
            )

        print()
        return rows


# -----------------------------
# Main
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute object-level top-k success statistics from GraspNet predictions."
    )
    parser.add_argument("--graspnet_root", type=str, required=True, help="Root directory of GraspNet.")
    parser.add_argument("--dump_folder", type=str, required=True, help="Folder with dumped prediction npy files.")
    parser.add_argument("--camera", type=str, default="realsense", choices=["realsense", "kinect"])
    parser.add_argument(
        "--split",
        type=str,
        default="novel",
        choices=["seen", "similar", "novel", "all"],
        help="Scene split to evaluate if --scene_ids is not given.",
    )
    parser.add_argument(
        "--scene_ids",
        type=str,
        default=None,
        help="Explicit scene ids, e.g. '160-189' or '100,101,105'. Overrides --split.",
    )
    parser.add_argument("--ann_start", type=int, default=0, help="Start annotation id (inclusive).")
    parser.add_argument("--ann_end", type=int, default=255, help="End annotation id (inclusive).")
    parser.add_argument("--topk", type=int, default=10, help="Object-level top-k to evaluate.")
    parser.add_argument(
        "--top_k_scene",
        type=int,
        default=50,
        help="Scene-level candidate cutoff used inside the evaluator (as in official AP eval).",
    )
    parser.add_argument(
        "--success_mu",
        type=float,
        default=0.8,
        help="A grasp is counted as success if 0 < evaluator_score <= success_mu.",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="all",
        choices=["all", "small", "medium", "large"],
        help="Optional width-based scale filter, following Scale-Balanced-Grasp.",
    )
    parser.add_argument("--max_width", type=float, default=0.1, help="Maximum width clip before evaluation.")
    parser.add_argument("--skip_missing", action="store_true", help="Skip missing prediction npy files.")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Uses scene-level parallelism when > 1.",
    )
    parser.add_argument(
        "--mp_start_method",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="multiprocessing start method when --num_workers > 1.",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Evaluate every K-th annotation only. Example: 10 -> 0,10,20,...",
    )
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the object-level CSV.")
    return parser


def _build_scene_ann_ids(args: argparse.Namespace) -> Dict[int, List[int]]:
    scene_ids = parse_scene_ids(args.split, args.scene_ids)
    ann_start = int(args.ann_start)
    ann_end = int(args.ann_end)
    if ann_end < ann_start:
        raise ValueError("ann_end must be >= ann_start")
    sample_interval = max(1, int(args.sample_interval))
    ann_ids = [i for i in range(ann_start, ann_end + 1) if ((i - ann_start) % sample_interval == 0)]
    return {scene_id: ann_ids for scene_id in scene_ids}


def _eval_single_scene(payload: Tuple[int, List[int], Dict]) -> List[Dict]:
    scene_id, ann_ids, kwargs = payload
    evaluator = ObjectTopKEval(root=kwargs["graspnet_root"], camera=kwargs["camera"], split="test")
    rows = evaluator.eval_scene_object_topk(
        scene_id=scene_id,
        dump_folder=kwargs["dump_folder"],
        topk=kwargs["topk"],
        success_mu=kwargs["success_mu"],
        top_k_scene=kwargs["top_k_scene"],
        scale=kwargs["scale"],
        max_width=kwargs["max_width"],
        ann_ids=ann_ids,
        skip_missing=kwargs["skip_missing"],
        verbose=False,
    )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    ensure_parent(Path(args.output_csv))

    scene_to_ann_ids = _build_scene_ann_ids(args)
    all_rows: List[Dict] = []

    if int(args.num_workers) <= 1:
        evaluator = ObjectTopKEval(root=args.graspnet_root, camera=args.camera, split="test")
        for scene_id, ann_ids in scene_to_ann_ids.items():
            rows = evaluator.eval_scene_object_topk(
                scene_id=scene_id,
                dump_folder=args.dump_folder,
                topk=args.topk,
                success_mu=args.success_mu,
                top_k_scene=args.top_k_scene,
                scale=args.scale,
                max_width=args.max_width,
                ann_ids=ann_ids,
                skip_missing=args.skip_missing,
                verbose=True,
            )
            all_rows.extend(rows)
    else:
        import multiprocessing as mp

        ctx = mp.get_context(args.mp_start_method)
        worker_kwargs = {
            "graspnet_root": args.graspnet_root,
            "camera": args.camera,
            "dump_folder": args.dump_folder,
            "topk": args.topk,
            "success_mu": args.success_mu,
            "top_k_scene": args.top_k_scene,
            "scale": args.scale,
            "max_width": args.max_width,
            "skip_missing": args.skip_missing,
        }
        payloads = [(scene_id, ann_ids, worker_kwargs) for scene_id, ann_ids in scene_to_ann_ids.items()]

        done = 0
        total = len(payloads)
        with ctx.Pool(processes=int(args.num_workers)) as pool:
            for rows in pool.imap_unordered(_eval_single_scene, payloads):
                all_rows.extend(rows)
                done += 1
                print(f"[mp] finished scenes {done}/{total} | accumulated rows={len(all_rows)}", flush=True)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        print("[WARN] No rows were produced. Writing empty CSV.")
    df.to_csv(args.output_csv, index=False)

    summary = build_summary(df, args)
    summary_path = str(Path(args.output_csv).with_suffix(".summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved CSV to: {args.output_csv}")
    print(f"[DONE] Saved summary to: {summary_path}")
    print(f"[DONE] Rows: {len(df)}")


if __name__ == "__main__":
    main()
