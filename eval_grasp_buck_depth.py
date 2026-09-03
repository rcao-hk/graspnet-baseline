#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from collections import defaultdict
import multiprocessing as mp
import imageio.v2 as imageio

from graspnetAPI.graspnet_eval import GraspNetEval
from graspnetAPI.grasp import GraspGroup
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.eval_utils import (
    create_table_points, transform_points,
    voxel_sample_points, collision_detection,
    GraspQualityConfigFactory, get_grasp_score,
    compute_closest_points
)

# -----------------------------
# IO helpers
# -----------------------------
def read_depth_png_m(path: str) -> np.ndarray:
    """Read uint16 depth png; return depth in meters (float32)."""
    d = imageio.imread(path).astype(np.float32)
    if np.nanmax(d) > 20.0:  # likely mm
        d = d / 1000.0
    return d

def read_label_png(path: str) -> np.ndarray:
    """Read label png (instance id per pixel)."""
    return imageio.imread(path).astype(np.int32)

# -----------------------------
# Fast instance MAE (vectorized)
# -----------------------------
def compute_inst_mae_bincount(label: np.ndarray, d_gt: np.ndarray, d_noisy: np.ndarray, min_pixels=50):
    """
    Return: inst_mae dict: {inst_id: mae_in_meters}
    """
    valid = (label > 0) & (d_gt > 0) & (d_noisy > 0)
    # valid = (label > 0) & (d_gt > 0)
    if valid.sum() == 0:
        return {}

    inst = label[valid].astype(np.int32)
    diff = np.abs(d_noisy[valid] - d_gt[valid]).astype(np.float64)

    max_id = int(inst.max())
    sum_diff = np.bincount(inst, weights=diff, minlength=max_id + 1)
    cnt = np.bincount(inst, minlength=max_id + 1)

    inst_mae = {}
    ids = np.nonzero(cnt > 0)[0]
    for i in ids:
        if i == 0:
            continue
        if cnt[i] < min_pixels:
            inst_mae[int(i)] = np.nan
        else:
            inst_mae[int(i)] = float(sum_diff[i] / cnt[i])
    return inst_mae

# -----------------------------
# Bucketing (endpoint convention configurable)
#   B0: (0, t1)
#   B1: [t1, t2]
#   B2: (t2, +inf)
# -----------------------------
def bucketize_depth_mae(mae_m: float, t1: float, t2: float) -> int:
    if mae_m is None or not np.isfinite(mae_m):
        return -1
    if mae_m <= 0.0:
        return -1
    if mae_m < t1 - 1e-12:
        return 0
    if mae_m <= t2 + 1e-12:
        return 1
    return 2

def bucket_names(t1: float, t2: float):
    return {
        0: f"(0, {t1})",
        1: f"[{t1}, {t2}]",
        2: f"({t2}, +inf)"
    }

# -----------------------------
# Eval grasp + return obj_id per grasp (avoid extra NN later)
# -----------------------------
def eval_grasp_scene_level(grasp_group, models, dexnet_models, poses, config, table=None, TOP_K=50):
    """
    Similar to GraspNet's eval_grasp:
      - NMS
      - assign grasps to object via NN in merged scene points
      - per-object top-10, then keep those >= global min_score (topK-th conf)
      - collision + force-closure score

    Return:
      grasp_arr (N,D), score_arr (N,), coll_arr (N,), obj_id_arr (N,)
    """
    num_models = len(models)
    grasp_group = grasp_group.nms(0.03, 30.0 / 180 * np.pi)

    # Build transformed scene points & seg ids
    model_trans_list = []
    seg_mask_list = []
    for i, model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        model_trans_list.append(model_trans)
        seg_mask_list.append(np.full((model_trans.shape[0],), i, dtype=np.int32))
    seg_mask = np.concatenate(seg_mask_list, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # Assign grasps to object
    idx_nn = compute_closest_points(grasp_group.translations, scene)
    model_to_grasp = seg_mask[idx_nn]

    # Per-object top-10
    pre_grasp_list = []
    for i in range(num_models):
        g_i = grasp_group[model_to_grasp == i]
        g_i.sort_by_score()
        pre_grasp_list.append(g_i[:10].grasp_group_array)

    nonempty = [g for g in pre_grasp_list if len(g) != 0]
    if len(nonempty) == 0:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
                np.zeros((0,), dtype=np.int32))

    all_grasp_list = np.vstack(nonempty)
    order = np.argsort(all_grasp_list[:, 0])[::-1]
    min_score = all_grasp_list[order[min(TOP_K - 1, len(order) - 1)], 0]

    # Keep per-object grasps above min_score (preserve list length = num_models)
    grasp_list = []
    obj_id_list = []
    # figure D from a nonempty grasp for empty placeholder
    D = all_grasp_list.shape[1]
    for i in range(num_models):
        g_i = pre_grasp_list[i]
        if len(g_i) == 0:
            grasp_list.append(np.zeros((0, D), dtype=np.float32))
            obj_id_list.append(np.zeros((0,), dtype=np.int32))
            continue
        keep = (g_i[:, 0] >= min_score)
        g_kept = g_i[keep]
        grasp_list.append(g_kept)
        obj_id_list.append(np.full((len(g_kept),), i, dtype=np.int32))

    if table is not None:
        scene_for_coll = np.concatenate([scene, table], axis=0)
    else:
        scene_for_coll = scene

    collision_mask_list, _, dexgrasp_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene_for_coll,
        outlier=0.05, return_dexgrasps=True
    )

    # Force-closure configs
    force_closure_quality_config = {}
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    for value_fc in fc_list:
        value_fc = round(float(value_fc), 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(
            config['metrics']['force_closure']
        )

    score_list = []
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        coll = collision_mask_list[i]
        dexgrasps = dexgrasp_list[i]
        scores = []
        for gid in range(len(dexgrasps)):
            if coll[gid] or dexgrasps[gid] is None:
                scores.append(-1.0)
                continue
            scores.append(get_grasp_score(dexgrasps[gid], dexnet_model, fc_list, force_closure_quality_config))
        score_list.append(np.array(scores, dtype=np.float32))

    # Concat
    g_nonempty = [g for g in grasp_list if len(g) != 0]
    if len(g_nonempty) == 0:
        return (np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=bool),
                np.zeros((0,), dtype=np.int32))

    grasp_arr = np.concatenate([g for g in grasp_list if len(g) != 0], axis=0)
    score_arr = np.concatenate([s for s in score_list if len(s) != 0], axis=0)
    coll_arr = np.concatenate([c for c in collision_mask_list if len(c) != 0], axis=0).astype(bool)
    obj_id_arr = np.concatenate([o for o in obj_id_list if len(o) != 0], axis=0)

    return grasp_arr, score_arr, coll_arr, obj_id_arr

# -----------------------------
# Conditional counts on global topK (for weighted aggregation)
# -----------------------------
def cond_counts_on_global_topk(grasp_arr, score_arr, coll_arr, obj_bucket,
                               topk=50, fric_list=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2)):
    """
    Returns:
      counts[b]["n"]             number of grasps from bucket b in global topK
      counts[b]["coll"]          collisions within those
      counts[b]["succ@fr"]       successes within those under friction threshold fr
      topk_len                   actual topK length used (=min(topk, N))
    """
    conf = grasp_arr[:, 0]
    order = np.argsort(-conf)[:min(topk, len(conf))]
    topk_len = int(len(order))

    counts = {b: {"n": 0, "coll": 0} for b in [0, 1, 2]}
    for b in [0, 1, 2]:
        for fr in fric_list:
            counts[b][f"succ@{fr}"] = 0

    for idx in order:
        b = int(obj_bucket[idx])
        if b not in [0, 1, 2]:
            continue
        counts[b]["n"] += 1
        counts[b]["coll"] += int(coll_arr[idx])
        for fr in fric_list:
            succ = (score_arr[idx] > 0) and (score_arr[idx] <= fr)
            counts[b][f"succ@{fr}"] += int(succ)

    return counts, topk_len

# -----------------------------
# Multiprocessing utils
# -----------------------------
def _init_worker():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _merge_scene_stats(dst, src):
    """
    dst/src format:
      {
        "frames": int,
        "topk_total": int,
        "buckets": {
          b: { "n": int, "coll": int, "succ@fr": int ... }
        }
      }
    """
    dst["frames"] += src["frames"]
    dst["topk_total"] += src["topk_total"]
    for b in [0, 1, 2]:
        for k, v in src["buckets"][b].items():
            dst["buckets"][b][k] += v

def _process_one_scene(scene_id: int, args_dict: dict):
    ge = GraspNetEval(root=args_dict["dataset_root"], camera=args_dict["camera"], split=args_dict["split"])
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

    fric_list = args_dict["fric_list"]
    step = args_dict["anno_step"]
    topk = args_dict["topk"]
    t1, t2 = args_dict["mae_t1"], args_dict["mae_t2"]

    model_list, dexmodel_list, _ = ge.get_scene_models(scene_id, ann_id=0)
    model_sampled_list = [voxel_sample_points(m, 0.008) for m in model_list]
    num_models = len(model_sampled_list)

    # assumption: label instance ids are 1..num_models in model order
    model_to_inst = {i: (i + 1) for i in range(num_models)}

    out = {
        "frames": 0,         # number of (scene,ann) frames actually evaluated (non-empty grasp_arr)
        "topk_total": 0,     # sum of topk_len across frames
        "buckets": {
            0: defaultdict(int),
            1: defaultdict(int),
            2: defaultdict(int),
        }
    }

    for ann_id in range(0, 256, step):
        npy_path = os.path.join(args_dict["dump_folder"], f"scene_{scene_id:04d}", args_dict["camera"], f"{ann_id:04d}.npy")
        if not os.path.exists(npy_path):
            continue

        grasp_group = GraspGroup().from_npy(npy_path)

        _, pose_list, camera_pose, align_mat = ge.get_model_poses(scene_id, ann_id)
        table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

        # clip widths
        gg_array = grasp_group.grasp_group_array
        gg_array[gg_array[:, 1] < 0, 1] = 0
        gg_array[gg_array[:, 1] > args_dict["max_width"], 1] = args_dict["max_width"]
        grasp_group.grasp_group_array = gg_array

        gt_path = args_dict["gt_depth_pattern"].format(root=args_dict["dataset_root"], scene_idx=scene_id, camera=args_dict["camera"], anno_idx=ann_id)
        noisy_path = args_dict["noisy_depth_pattern"].format(root=args_dict["dataset_root"], scene_idx=scene_id, camera=args_dict["camera"], anno_idx=ann_id)
        lab_path = args_dict["label_pattern"].format(root=args_dict["dataset_root"], scene_idx=scene_id, camera=args_dict["camera"], anno_idx=ann_id)
        if not (os.path.exists(gt_path) and os.path.exists(noisy_path) and os.path.exists(lab_path)):
            continue

        d_gt = read_depth_png_m(gt_path)
        d_noisy = read_depth_png_m(noisy_path)
        lab = read_label_png(lab_path)

        inst_mae = compute_inst_mae_bincount(lab, d_gt, d_noisy, min_pixels=50)

        model_bucket = {}
        for i in range(num_models):
            inst_id = model_to_inst[i]
            mae = inst_mae.get(inst_id, np.nan)
            model_bucket[i] = bucketize_depth_mae(mae, t1=t1, t2=t2)

        grasp_arr, score_arr, coll_arr, obj_id_arr = eval_grasp_scene_level(
            grasp_group, model_sampled_list, dexmodel_list, pose_list, config,
            table=table_trans, TOP_K=topk
        )
        if grasp_arr.shape[0] == 0:
            continue

        obj_bucket = np.array([model_bucket.get(int(o), -1) for o in obj_id_arr], dtype=np.int32)

        counts, topk_len = cond_counts_on_global_topk(
            grasp_arr, score_arr, coll_arr, obj_bucket, topk=topk, fric_list=fric_list
        )

        out["frames"] += 1
        out["topk_total"] += topk_len

        for b in [0, 1, 2]:
            out["buckets"][b]["n"] += counts[b]["n"]
            out["buckets"][b]["coll"] += counts[b]["coll"]
            for fr in fric_list:
                out["buckets"][b][f"succ@{fr}"] += counts[b][f"succ@{fr}"]

    # cast defaultdict->dict for pickling cleanliness
    out["buckets"] = {b: dict(out["buckets"][b]) for b in [0, 1, 2]}
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="/data/robotarm/dataset/graspnet")
    ap.add_argument("--dump_folder", type=str, default="experiment/mmgnet_scene_24")
    ap.add_argument("--camera", type=str, default="realsense")
    ap.add_argument("--split", type=str, default="test", choices=["test", "train"])
    ap.add_argument("--subset", type=str, default="seen", choices=["seen", "similar", "novel"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--anno_sample_ratio", type=float, default=0.1)
    ap.add_argument("--max_width", type=float, default=0.1)

    # thresholds for buckets
    ap.add_argument("--mae_t1", type=float, default=0.002)
    ap.add_argument("--mae_t2", type=float, default=0.005)

    # Patterns
    ap.add_argument("--gt_depth_pattern", type=str,
                    default="{root}/virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_depth.png")
    ap.add_argument("--noisy_depth_pattern", type=str,
                    default="{root}/scenes/scene_{scene_idx:04d}/{camera}/depth/{anno_idx:04d}.png")
    ap.add_argument("--label_pattern", type=str,
                    default="{root}/virtual_scenes/scene_{scene_idx:04d}/{camera}/{anno_idx:04d}_label.png")

    ap.add_argument("--proc", type=int, default=8)
    ap.add_argument("--print_fric", type=float, default=0.4)
    ap.add_argument("--out_json", type=str, default="", help="Optional path to save results as JSON.")

    args = ap.parse_args()

    # scene ids by subset (keep your current ranges)
    if args.subset == "seen":
        scene_ids = list(range(100, 130))
    elif args.subset == "similar":
        scene_ids = list(range(130, 160))
    else:
        scene_ids = list(range(160, 190))

    fric_list = (0.2, 0.4, 0.6, 0.8, 1.0, 1.2)
    step = max(1, int(1 / max(1e-9, args.anno_sample_ratio)))
    bnames = bucket_names(args.mae_t1, args.mae_t2)

    args_dict = {
        "dataset_root": args.dataset_root,
        "dump_folder": args.dump_folder,
        "camera": args.camera,
        "split": args.split,
        "topk": args.topk,
        "anno_step": step,
        "max_width": args.max_width,
        "fric_list": fric_list,
        "gt_depth_pattern": args.gt_depth_pattern,
        "noisy_depth_pattern": args.noisy_depth_pattern,
        "label_pattern": args.label_pattern,
        "mae_t1": args.mae_t1,
        "mae_t2": args.mae_t2,
    }

    ctx = mp.get_context("spawn")
    global_stats = {
        "frames": 0,
        "topk_total": 0,
        "buckets": {
            0: defaultdict(int),
            1: defaultdict(int),
            2: defaultdict(int),
        }
    }

    with ctx.Pool(processes=args.proc, initializer=_init_worker) as pool:
        results = pool.starmap(_process_one_scene, [(sid, args_dict) for sid in scene_ids], chunksize=1)

    # merge
    for r in results:
        # convert incoming dicts into our accumulator format
        src = {
            "frames": int(r["frames"]),
            "topk_total": int(r["topk_total"]),
            "buckets": {
                0: defaultdict(int, r["buckets"].get(0, {})),
                1: defaultdict(int, r["buckets"].get(1, {})),
                2: defaultdict(int, r["buckets"].get(2, {})),
            }
        }
        _merge_scene_stats(global_stats, src)

    frames = global_stats["frames"]
    topk_total = global_stats["topk_total"]

    # compute weighted metrics
    summary = {
        "meta": {
            "dataset_root": args.dataset_root,
            "dump_folder": args.dump_folder,
            "camera": args.camera,
            "split": args.split,
            "subset": args.subset,
            "scene_ids": scene_ids,
            "topk": args.topk,
            "anno_step": step,
            "anno_sample_ratio": args.anno_sample_ratio,
            "max_width": args.max_width,
            "mae_thresholds_m": {"t1": args.mae_t1, "t2": args.mae_t2},
            "bucket_definition": {
                "0": f"(0, {args.mae_t1})",
                "1": f"[{args.mae_t1}, {args.mae_t2}]",
                "2": f"({args.mae_t2}, +inf)"
            },
            "frames_evaluated": int(frames),
            "topk_total_grasps": int(topk_total),
            "proc": args.proc,
            "fric_list": list(fric_list),
        },
        "buckets": {}
    }

    print("\n=== Depth-noise Conditional-on-Global-TopK (Weighted by #grasps) ===")
    print(f"Subset: {args.subset} | topK={args.topk} | anno_step={step} | camera={args.camera} | proc={args.proc}")
    print(f"Frames evaluated: {frames} | Total topK grasps counted: {topk_total}")
    print(f"Buckets: B0 {bnames[0]}, B1 {bnames[1]}, B2 {bnames[2]}\n")

    for b in [0, 1, 2]:
        bdict = global_stats["buckets"][b]
        n = int(bdict.get("n", 0))
        coll = int(bdict.get("coll", 0))

        # E[n_b] over frames (include zeros)
        avg_n_in_topk = (n / frames) if frames > 0 else float("nan")
        # ratio_in_topk over all counted topK grasps
        ratio_in_topk = (n / topk_total) if topk_total > 0 else float("nan")
        # conditional rates within bucket
        coll_rate = (coll / n) if n > 0 else float("nan")

        succ_map = {}
        for fr in fric_list:
            succ = int(bdict.get(f"succ@{fr}", 0))
            succ_rate = (succ / n) if n > 0 else float("nan")
            succ_map[str(fr)] = {"succ_count": succ, "succ_rate": succ_rate}

        # print
        print(f"[Bucket {b}] MAE {bnames[b]}")
        print(f"  n_in_topk_total: {n}")
        print(f"  E[n_in_topk] (per-frame): {avg_n_in_topk:.3f}")
        print(f"  ratio_in_topk (weighted): {ratio_in_topk:.3f}")
        print(f"  collision_rate (weighted): {coll_rate:.3f}")

        fr = float(args.print_fric)
        fr_key = str(fr)
        if fr_key in succ_map:
            print(f"  success@fric{fr}: {succ_map[fr_key]['succ_rate']:.3f} (count={succ_map[fr_key]['succ_count']})")
        else:
            print(f"  success@fric{fr}: n/a")

        parts = []
        for fr in fric_list:
            parts.append(f"{fr}:{succ_map[str(fr)]['succ_rate']:.3f}")
        print("  all fric succ (weighted): " + " | ".join(parts))
        print()

        summary["buckets"][str(b)] = {
            "mae_range": bnames[b],
            "n_in_topk_total": n,
            "E_n_in_topk_per_frame": avg_n_in_topk,
            "ratio_in_topk_weighted": ratio_in_topk,
            "collision_count": coll,
            "collision_rate_weighted": coll_rate,
            "success": succ_map
        }

    # save json
    if args.out_json.strip():
        out_path = args.out_json
        os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved JSON to: {out_path}")

if __name__ == "__main__":
    main()