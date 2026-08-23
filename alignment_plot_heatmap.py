#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------- orders (already removed p16) ----------
SPLITS = ["test_seen", "test_similar", "test_novel"]

# 用于从 JSON 里取值的 keys（不要改）
PC_KEYS   = ["block1", "block2", "block3", "final"]
IMG_KEYS  = ["p1", "p2", "p4", "p8"]   # 不画 p16

# 论文/图中显示的 labels（你想怎么写都行）
PC_LABELS  = ["s2", "s4", "s8", "out"]
IMG_LABELS = ["p1", "p2", "p4", "p8"]

METHOD_SPECS_DEFAULT = [
    ("mmgnet_scene",        "early",  "Early"),
    ("mmgnet_scene_intermediate", "intermediate", "Hierarchical"),
    ("mmgnet_scene_concat", "concat", "Late (concat)"),
    ("mmgnet_scene_add",    "add",    "Late (add)"),
    ("mmgnet_scene_gate",   "gate",   "Late (gate)"),
]

METRIC_KEYS = ["cka", "r2_x2y", "r2_y2x"]
METRIC_DISPLAY = {
    "cka": "CKA",
    "r2_x2y": r"$R^2_{\mathrm{pc}\rightarrow\mathrm{img}}$",
    "r2_y2x": r"$R^2_{\mathrm{img}\rightarrow\mathrm{pc}}$",
}

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def build_json_path(root, method_name, fusion_type, split, interval):
    p1 = os.path.join(root, f"{method_name}_{fusion_type}_{split}_{interval}.json")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(root, f"{method_name}_{split}_{interval}.json")
    if os.path.exists(p2):
        return p2
    return p1

def mean_matrix_from_records(records, key):
    H = np.full((len(PC_KEYS), len(IMG_KEYS)), np.nan, dtype=np.float64)
    S = np.zeros_like(H)
    C = np.zeros_like(H)

    for r in records:
        m = r.get(key, {})
        for i, pl_key in enumerate(PC_KEYS):
            row = m.get(pl_key, {})
            for k, img_key in enumerate(IMG_KEYS):
                v = row.get(img_key, None)
                if v is None:
                    continue
                try:
                    v = float(v)
                except Exception:
                    continue
                if np.isfinite(v):
                    S[i, k] += v
                    C[i, k] += 1

    mask = C > 0
    H[mask] = S[mask] / C[mask]
    return H, C


def mean_matrix(j, key):
    recs = j.get("records", [])
    H, C = mean_matrix_from_records(recs, key)
    return H, C, len(recs)

def merge_records_mean(json_list, key):
    all_records = []
    for j in json_list:
        all_records.extend(j.get("records", []))
    H, C = mean_matrix_from_records(all_records, key)
    return H, C, len(all_records)
def _set_bold_ticks(ax, fontsize=26):
    ax.tick_params(axis="both", labelsize=fontsize)
    for lab in ax.get_xticklabels():
        lab.set_fontweight("bold")
    for lab in ax.get_yticklabels():
        lab.set_fontweight("bold")


import matplotlib.patheffects as pe

def annotate_with_stroke(ax, mat, fmt="{:.2f}", fontsize=16, fontweight="bold"):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                continue
            t = ax.text(
                j, i, fmt.format(v),
                ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight,
                color="black"
            )
            # 白色描边（线宽可调大一点）
            t.set_path_effects([pe.Stroke(linewidth=3.5, foreground="white"), pe.Normal()])
            
            
def plot_combo_3rows_4cols(
    mats_by_metric,          # dict: metric_key -> list of 4 mats
    col_titles,              # list of 4 fusion names
    out_path,
    suptitle,
    show_numbers=True,
    fontsize=26,
    cmap="viridis",
):
    """
    Layout:
      3 rows (CKA / pc2img / img2pc)
      4 cols (early / concat / add / gate)
      + 1 extra col for colorbar per row (so total 5 cols)
    """

    plt.rcParams.update({
        "font.size": fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # ranges per metric (keep as you used)
    ranges = {
        "cka": (0.0, 1.0),
        "r2_x2y": (-1.0, 1.0),
        "r2_y2x": (-1.0, 1.0),
    }

    fig = plt.figure(figsize=(5 * 6.2 + 1.0, 3 * 5.6))  # extra width for colorbars
    gs = gridspec.GridSpec(
        nrows=3, ncols=6,
        width_ratios=[1, 1, 1, 1, 1, 0.06],   # last column reserved for colorbar
        wspace=0.15, hspace=0.18
    )

    axes = [[None for _ in range(5)] for _ in range(3)]
    caxes = [None for _ in range(3)]

    # create axes
    for r in range(3):
        for c in range(5):
            axes[r][c] = fig.add_subplot(gs[r, c])
        caxes[r] = fig.add_subplot(gs[r, 5])

    # draw
    for r, metric_key in enumerate(METRIC_KEYS):
        vmin, vmax = ranges[metric_key]
        im_last = None

        for c in range(5):
            ax = axes[r][c]
            mat = mats_by_metric[metric_key][c]

            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap)
            im_last = im

            # ticks
            ax.set_xticks(range(len(IMG_KEYS)))
            ax.set_xticklabels(IMG_LABELS, fontsize=fontsize, fontweight="bold")

            ax.set_yticks(range(len(PC_KEYS)))
            if c == 0:
                ax.set_yticklabels(PC_LABELS, fontsize=fontsize, fontweight="bold")
            else:
                ax.set_yticklabels([])

            _set_bold_ticks(ax, fontsize=fontsize)

            # row label on the left-most subplot
            if c == 0:
                ax.set_ylabel(METRIC_DISPLAY[metric_key], fontsize=fontsize, fontweight="bold")

            # fusion labels as x-label ONLY on bottom row
            if r == 2:
                ax.set_xlabel(col_titles[c], fontsize=fontsize, fontweight="bold", labelpad=18)

            # numbers
            if show_numbers:
                annotate_with_stroke(ax, mat, fmt="{:.2f}", fontsize=14, fontweight="bold")
                # for i in range(mat.shape[0]):
                #     for j in range(mat.shape[1]):
                #         v = mat[i, j]
                #         if np.isfinite(v):
                #             ax.text(
                #                 j, i, f"{v:.2f}",
                #                 ha="center", va="center",
                #                 fontsize=max(10, fontsize - 10),
                #                 fontweight="bold",
                #                 color="black"
                #             )

        # colorbar for this row (never disappears)
        cb = fig.colorbar(im_last, cax=caxes[r])
        cb.ax.tick_params(labelsize=fontsize)
        for lab in cb.ax.get_yticklabels():
            lab.set_fontweight("bold")

    # fig.suptitle(suptitle, fontsize=fontsize + 2, fontweight="bold", y=0.98)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    fig.savefig(out_path.replace('.png', '.svg'), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_root", default="vis/feat_redun_vis", help="Folder containing json files")
    ap.add_argument("--out_dir", default="vis/feat_redun_vis", help="Output directory for figures")
    ap.add_argument("--interval", type=int, default=10, help="Interval suffix in json filename")
    ap.add_argument("--prefix", default="mmgnet", help="Figure filename prefix")
    ap.add_argument("--split", default="test_mean", choices=SPLITS + ["test_mean"])
    ap.add_argument("--no_numbers", action="store_true")
    ap.add_argument("--fontsize", type=int, default=22)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load all jsons
    data = {}
    for method_name, fusion_type, _title in METHOD_SPECS_DEFAULT:
        data[method_name] = {}
        for sp in SPLITS:
            p = build_json_path(args.json_root, method_name, fusion_type, sp, args.interval)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing json: {p}")
            data[method_name][sp] = load_json(p)
            print(f"[LOAD] {p}")

    # collect matrices for the requested split
    mats_by_metric = {"cka": [], "r2_x2y": [], "r2_y2x": []}
    col_titles = []
    for method_name, fusion_type, title in METHOD_SPECS_DEFAULT:
        col_titles.append(title)

        for metric_key in ["cka", "r2_x2y", "r2_y2x"]:
            if args.split == "test_mean":
                jlist = [data[method_name][s] for s in SPLITS]
                H, _, _ = merge_records_mean(jlist, metric_key)
            else:
                H, _, _ = mean_matrix(data[method_name][args.split], metric_key)
            mats_by_metric[metric_key].append(H)
            
    out_path = os.path.join(args.out_dir, f"{args.prefix}_combo_{args.split}.png")
    suptitle = f"Mean cross-modal alignment (CKA) and redundancy ($R^2$)"
    plot_combo_3rows_4cols(
        mats_by_metric=mats_by_metric,
        col_titles=col_titles,
        out_path=out_path,
        suptitle=suptitle,
        show_numbers=(not args.no_numbers),
        fontsize=args.fontsize,
        cmap="viridis",
    )

if __name__ == "__main__":
    main()
