#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PC_ORDER = ["block1", "block2", "block3", "block4", "final"]
IMG_ORDER = ["p1", "p2", "p4", "p8", "p16"]
SPLITS = ["test_seen", "test_similar", "test_novel"]

METHOD_SPECS_DEFAULT = [
    ("mmgnet_scene",  "early",  "mmgnet_scene_early (early)"),
    ("mmgnet_scene_concat", "concat", "mmgnet_scene_concat (late (concat))"),
    ("mmgnet_scene_add",    "add",    "mmgnet_scene_add (late (add))"),
    ("mmgnet_scene_gate",   "gate",   "mmgnet_scene_gate (late (gate))"),
]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_float(x):
    try:
        v = float(x)
        return v
    except Exception:
        return np.nan

def extract_matrix_from_record(r, key):
    """key: 'cka' | 'r2_x2y' | 'r2_y2x' """
    d = r.get(key, {})
    mat = np.full((len(PC_ORDER), len(IMG_ORDER)), np.nan, dtype=np.float64)
    for i, pl in enumerate(PC_ORDER):
        row = d.get(pl, {})
        for j, ik in enumerate(IMG_ORDER):
            mat[i, j] = safe_float(row.get(ik, np.nan))
    return mat

def mean_stats_from_records(records, key, clip=None):
    """
    Return:
      mean_mat: (5,5)
      count_mat: finite counts
      frac_valid: count / N
      frac_neg: fraction of values < 0 among finite (only meaningful for R2)
    clip: None | (lo, hi)  applied PER-VALUE before accumulation
    """
    N = len(records)
    S = np.zeros((len(PC_ORDER), len(IMG_ORDER)), dtype=np.float64)
    C = np.zeros_like(S)
    Neg = np.zeros_like(S)  # count of v<0 among finite

    for r in records:
        mat = extract_matrix_from_record(r, key)  # (5,5)
        if clip is not None:
            lo, hi = clip
            # clip only finite entries
            fin = np.isfinite(mat)
            mat_clip = mat.copy()
            mat_clip[fin] = np.clip(mat_clip[fin], lo, hi)
            mat = mat_clip

        fin = np.isfinite(mat)
        S[fin] += mat[fin]
        C[fin] += 1
        # negative stats on raw values only makes sense when clip is None
        if clip is None and key.startswith("r2"):
            Neg[fin & (mat < 0)] += 1

    mean = np.full_like(S, np.nan)
    mask = C > 0
    mean[mask] = S[mask] / C[mask]

    frac_valid = np.zeros_like(S)
    if N > 0:
        frac_valid = C / float(N)

    frac_neg = None
    if key.startswith("r2") and clip is None:
        frac_neg = np.full_like(S, np.nan)
        m2 = C > 0
        frac_neg[m2] = Neg[m2] / C[m2]

    return mean, C, frac_valid, frac_neg, N

def merge_records(json_list):
    all_records = []
    for j in json_list:
        all_records.extend(j.get("records", []))
    return all_records

def plot_row_4methods(mats, titles, out_path, suptitle, vmin=None, vmax=None, show_numbers=True, cbar=True):
    """
    mats: list of 2D arrays (5x5) len=4
    """
    fig, axes = plt.subplots(1, 4, figsize=(4*4.8, 4.9), constrained_layout=True)

    im0 = None
    for idx, ax in enumerate(axes):
        mat = mats[idx]
        im0 = ax.imshow(mat, vmin=vmin, vmax=vmax)
        ax.set_title(titles[idx], fontsize=11)

        ax.set_xticks(range(len(IMG_ORDER)))
        ax.set_xticklabels(IMG_ORDER, fontsize=10)
        ax.set_yticks(range(len(PC_ORDER)))
        if idx == 0:
            ax.set_yticklabels(PC_ORDER, fontsize=10)
        else:
            ax.set_yticklabels([])

        if show_numbers:
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    if cbar:
        cbar_obj = fig.colorbar(im0, ax=axes, fraction=0.02, pad=0.02)
        cbar_obj.ax.tick_params(labelsize=10)

    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def build_json_path(root, method_name, fusion_type, split, interval):
    # preferred format: {method_name}_{fusion_type}_{split}_{interval}.json
    fname = f"{method_name}_{fusion_type}_{split}_{interval}.json"
    p = os.path.join(root, fname)
    if os.path.exists(p):
        return p
    # fallback (old format): {method_name}_{split}_{interval}.json
    alt = os.path.join(root, f"{method_name}_{split}_{interval}.json")
    if os.path.exists(alt):
        return alt
    return p  # default (will trigger FileNotFoundError upstream)

def collect_mats_for_split(data, method_specs, split, key, clip=None, want_counts=False):
    mats = []
    titles = []
    counts = []
    nrecs = []
    frac_valid_list = []
    frac_neg_list = []

    for method_name, fusion_type, title in method_specs:
        j = data[method_name][split]
        records = j.get("records", [])
        mean, C, frac_valid, frac_neg, N = mean_stats_from_records(records, key=key, clip=clip)
        mats.append(mean)
        titles.append(title)
        counts.append(C)
        frac_valid_list.append(frac_valid)
        frac_neg_list.append(frac_neg)
        nrecs.append(N)

    return mats, titles, counts, frac_valid_list, frac_neg_list, nrecs

def collect_mats_for_test_mean(data, method_specs, key, clip=None):
    mats = []
    titles = []
    counts = []
    frac_valid_list = []
    frac_neg_list = []
    nrecs = []

    for method_name, fusion_type, title in method_specs:
        jlist = [data[method_name][sp] for sp in SPLITS]
        records = merge_records(jlist)
        mean, C, frac_valid, frac_neg, N = mean_stats_from_records(records, key=key, clip=clip)
        mats.append(mean)
        titles.append(title)
        counts.append(C)
        frac_valid_list.append(frac_valid)
        frac_neg_list.append(frac_neg)
        nrecs.append(N)

    return mats, titles, counts, frac_valid_list, frac_neg_list, nrecs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_root", default="vis/feat_redun_vis", help="Folder containing json files")
    ap.add_argument("--out_dir", default="vis/feat_redun_vis", help="Output directory for figures")
    ap.add_argument("--interval", type=int, default=10, help="Interval suffix in json filename")
    ap.add_argument("--prefix", default="mmgnet", help="Figure filename prefix")
    ap.add_argument("--no_numbers", action="store_true", help="Disable writing values on heatmaps")

    # plotting switches
    ap.add_argument("--plot_cka", action="store_true", help="Plot CKA mean heatmaps")
    ap.add_argument("--plot_r2", action="store_true", help="Plot raw R2 mean heatmaps (x2y/y2x)")
    ap.add_argument("--plot_r2_clip", action="store_true", help="Plot clipped R2 mean heatmaps (x2y/y2x), clip per-value to [0,1]")
    ap.add_argument("--plot_neg_frac", action="store_true", help="Plot negative fraction heatmaps for raw R2 (x2y/y2x)")
    ap.add_argument("--plot_valid_frac", action="store_true", help="Plot valid fraction heatmaps (finite ratio) for each metric")

    # ranges
    ap.add_argument("--cka_vmin", type=float, default=0.0)
    ap.add_argument("--cka_vmax", type=float, default=1.0)
    ap.add_argument("--r2_vmin", type=float, default=-1.0)
    ap.add_argument("--r2_vmax", type=float, default=1.0)
    ap.add_argument("--r2clip_vmin", type=float, default=0.0)
    ap.add_argument("--r2clip_vmax", type=float, default=1.0)
    ap.add_argument("--frac_vmin", type=float, default=0.0)
    ap.add_argument("--frac_vmax", type=float, default=1.0)

    args = ap.parse_args()

    # default behavior: if user didn't specify any, plot all the main ones
    if not (args.plot_cka or args.plot_r2 or args.plot_r2_clip or args.plot_neg_frac or args.plot_valid_frac):
        args.plot_cka = True
        args.plot_r2 = True
        args.plot_r2_clip = True

    os.makedirs(args.out_dir, exist_ok=True)
    method_specs = METHOD_SPECS_DEFAULT

    # --------- Load all JSONs: method x split ----------
    data = {}
    for method_name, fusion_type, _title in method_specs:
        data[method_name] = {}
        for sp in SPLITS:
            p = build_json_path(args.json_root, method_name, fusion_type, sp, args.interval)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing json: {p}")
            data[method_name][sp] = load_json(p)
            print(f"[LOAD] {p}")

    def _plot_for_one_split(split_name, records_mode="split"):
        # helper to avoid duplication; returns dict with mats/stats per metric
        out = {}

        def _collect(key, clip=None):
            if records_mode == "split":
                mats, titles, counts, frac_valid_list, frac_neg_list, nrecs = collect_mats_for_split(
                    data, method_specs, split_name, key, clip=clip
                )
            else:
                mats, titles, counts, frac_valid_list, frac_neg_list, nrecs = collect_mats_for_test_mean(
                    data, method_specs, key, clip=clip
                )
            return mats, titles, counts, frac_valid_list, frac_neg_list, nrecs

        show_numbers = (not args.no_numbers)

        # ---- CKA ----
        if args.plot_cka:
            mats, titles, counts, frac_valid_list, _frac_neg_list, nrecs = _collect("cka", clip=None)
            out_path = os.path.join(args.out_dir, f"{args.prefix}_cka_{split_name}_row4.png")
            suptitle = f"Mean CKA heatmaps | {split_name}"
            plot_row_4methods(mats, titles, out_path, suptitle, vmin=args.cka_vmin, vmax=args.cka_vmax, show_numbers=show_numbers)
            out["cka"] = (mats, frac_valid_list)

            if args.plot_valid_frac:
                out_path = os.path.join(args.out_dir, f"{args.prefix}_cka_validfrac_{split_name}_row4.png")
                suptitle = f"Valid fraction (finite) | CKA | {split_name}"
                plot_row_4methods(frac_valid_list, titles, out_path, suptitle,
                                  vmin=args.frac_vmin, vmax=args.frac_vmax, show_numbers=show_numbers)

        # ---- R2 raw ----
        if args.plot_r2:
            for key in ["r2_x2y", "r2_y2x"]:
                mats, titles, counts, frac_valid_list, frac_neg_list, nrecs = _collect(key, clip=None)
                out_path = os.path.join(args.out_dir, f"{args.prefix}_{key}_{split_name}_row4.png")
                suptitle = f"Mean {key} heatmaps (raw) | {split_name}"
                plot_row_4methods(mats, titles, out_path, suptitle, vmin=args.r2_vmin, vmax=args.r2_vmax, show_numbers=show_numbers)

                if args.plot_valid_frac:
                    out_path = os.path.join(args.out_dir, f"{args.prefix}_{key}_validfrac_{split_name}_row4.png")
                    suptitle = f"Valid fraction (finite) | {key} | {split_name}"
                    plot_row_4methods(frac_valid_list, titles, out_path, suptitle,
                                      vmin=args.frac_vmin, vmax=args.frac_vmax, show_numbers=show_numbers)

                if args.plot_neg_frac and frac_neg_list[0] is not None:
                    out_path = os.path.join(args.out_dir, f"{args.prefix}_{key}_negfrac_{split_name}_row4.png")
                    suptitle = f"Negative fraction (v<0) | {key} | {split_name}"
                    plot_row_4methods(frac_neg_list, titles, out_path, suptitle,
                                      vmin=args.frac_vmin, vmax=args.frac_vmax, show_numbers=show_numbers)

        # ---- R2 clipped ----
        if args.plot_r2_clip:
            for key in ["r2_x2y", "r2_y2x"]:
                mats, titles, counts, frac_valid_list, _frac_neg_list, nrecs = _collect(key, clip=(0.0, 1.0))
                out_path = os.path.join(args.out_dir, f"{args.prefix}_{key}_clip01_{split_name}_row4.png")
                suptitle = f"Mean {key} heatmaps (clip [0,1] per-value) | {split_name}"
                plot_row_4methods(mats, titles, out_path, suptitle, vmin=args.r2clip_vmin, vmax=args.r2clip_vmax, show_numbers=show_numbers)

        return out

    # --------- per split ---------
    for sp in SPLITS:
        _plot_for_one_split(sp, records_mode="split")

    # --------- test_mean ---------
    _plot_for_one_split("test_mean", records_mode="test_mean")

if __name__ == "__main__":
    main()
