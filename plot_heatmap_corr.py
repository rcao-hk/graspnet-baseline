#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# ---------- defaults (paper-friendly) ----------
PC_KEYS_DEFAULT  = ["block1", "block2", "block3", "final"]
IMG_KEYS_DEFAULT = ["p1", "p2", "p4", "p8"]   # drop p16

PC_LABEL = {"block1": "s2", "block2": "s4", "block3": "s8", "final": "out3d"}
IMG_LABEL = {"p1": "p1", "p2": "p2", "p4": "p4", "p8": "p8"}

FUSE_ORDER = ["early", "concat", "add", "gate"]
FUSE_LABEL = {
    "early":  "Early",
    "concat": "Late (Concat)",
    "add":    "Late (Add)",
    "gate":   "Late (Gate)",
}

METRIC_ORDER = ["cka", "r2_x2y", "r2_y2x"]
METRIC_LABEL = {
    "cka":    r"$\Delta$ CKA",
    "r2_x2y": r"$\Delta\;R^2$ (pc$\rightarrow$img)",
    "r2_y2x": r"$\Delta\;R^2$ (img$\rightarrow$pc)",
}

def list_json_files(root):
    out = []
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f.endswith(".json"):
                out.append(os.path.join(dp, f))
    return sorted(out)

def load_json(p):
    with open(p, "r") as f:
        return json.load(f)

def _safe_float(x):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        return None
    return None

def mean_matrix_from_records(records, key, pc_keys, img_keys):
    H = np.full((len(pc_keys), len(img_keys)), np.nan, dtype=np.float64)
    S = np.zeros_like(H)
    C = np.zeros_like(H)

    for r in records:
        m = r.get(key, {})
        for i, pk in enumerate(pc_keys):
            row = m.get(pk, {})
            for j, ik in enumerate(img_keys):
                v = _safe_float(row.get(ik, None))
                if v is None:
                    continue
                S[i, j] += v
                C[i, j] += 1

    mask = C > 0
    H[mask] = S[mask] / C[mask]
    return H, C

def matrix_for_json(j, key, pc_keys, img_keys):
    recs = j.get("records", [])
    H, C = mean_matrix_from_records(recs, key, pc_keys, img_keys)
    return H, C, len(recs)

def merge_jsons_mean(jlist, key, pc_keys, img_keys):
    all_records = []
    for j in jlist:
        all_records.extend(j.get("records", []))
    H, C = mean_matrix_from_records(all_records, key, pc_keys, img_keys)
    return H, C, len(all_records)

def infer_meta(j, path):
    """Prefer meta; fallback to filename parse when meta missing."""
    meta = j.get("meta", {}) if isinstance(j, dict) else {}
    split = str(meta.get("split", "")).strip()
    fuse  = str(meta.get("fuse_type", "")).strip().lower()
    corr  = str(meta.get("rgb_noise", "none")).strip().lower()
    sev   = int(meta.get("rgb_severity", 0) or 0)

    # fallback: parse filename like "{name}_{corr}_s{sev}_{interval}_seed{seed}.json"
    if (not fuse) or (not split):
        bn = os.path.basename(path)
        # very tolerant parsing
        # try find "_sX_" pattern
        if "_s" in bn:
            try:
                seg = bn.split("_s", 1)[1]
                sev2 = int(seg.split("_", 1)[0])
                sev = sev2
            except Exception:
                pass
        # corr is usually right before "_s"
        try:
            left = bn.split("_s", 1)[0]
            corr = left.split("_")[-1].lower()
        except Exception:
            pass

    return split, fuse, corr, sev

def annotate_text(ax, x, y, s, fontsize=9):
    t = ax.text(x, y, s, ha="center", va="center", fontsize=fontsize, color="black")
    t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
    return t

def robust_vlim(mats, q=95, min_v=1e-3):
    vals = []
    for m in mats:
        if m is None:
            continue
        vv = m[np.isfinite(m)].ravel()
        if vv.size:
            vals.append(np.abs(vv))
    if not vals:
        return 1.0
    a = np.concatenate(vals, axis=0)
    v = float(np.percentile(a, q))
    v = max(v, min_v)
    return v

def plot_delta_grid(delta_mats_by_metric, out_path, title,
                    pc_keys, img_keys, show_numbers=True,
                    cmap="RdBu_r", vlim_mode="per_row"):
    """
    delta_mats_by_metric: dict metric -> list[mat] len=4 (early/concat/add/gate)
    3 rows (metrics) x 4 cols (fuse types)
    """
    nrows = len(METRIC_ORDER)
    ncols = len(FUSE_ORDER)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 4.2*nrows), constrained_layout=True)

    # make axes 2D
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if ncols == 1:
        axes = np.expand_dims(axes, 1)

    for r, mk in enumerate(METRIC_ORDER):
        mats = delta_mats_by_metric.get(mk, [None]*ncols)

        # vlim per row (recommended)
        if vlim_mode == "per_row":
            vlim = robust_vlim(mats, q=95)
        else:
            # global across all metrics
            allm = []
            for mm in METRIC_ORDER:
                allm.extend(delta_mats_by_metric.get(mm, []))
            vlim = robust_vlim(allm, q=95)

        im0 = None
        for c, fuse in enumerate(FUSE_ORDER):
            ax = axes[r, c]
            mat = mats[c]

            if mat is None:
                ax.axis("off")
                continue

            im0 = ax.imshow(mat, vmin=-vlim, vmax=vlim, cmap=cmap)

            # titles
            if r == 0:
                ax.set_title(FUSE_LABEL.get(fuse, fuse), fontsize=12)

            # ticks/labels
            ax.set_xticks(range(len(img_keys)))
            ax.set_xticklabels([IMG_LABEL.get(k, k) for k in img_keys], fontsize=10)

            ax.set_yticks(range(len(pc_keys)))
            if c == 0:
                ax.set_yticklabels([PC_LABEL.get(k, k) for k in pc_keys], fontsize=10)
            else:
                ax.set_yticklabels([])

            if c == 0:
                ax.set_ylabel(METRIC_LABEL.get(mk, mk), fontsize=12)

            # numbers
            if show_numbers:
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        v = mat[i, j]
                        if np.isfinite(v):
                            annotate_text(ax, j, i, f"{v:+.2f}", fontsize=9)

        # per-row colorbar
        if im0 is not None:
            cbar = fig.colorbar(im0, ax=axes[r, :], fraction=0.02, pad=0.01)
            cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"[SAVE] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_root", default='vis/feat_test', help="Folder containing json files (recursive).")
    ap.add_argument("--out_dir", default='vis/feat_test', help="Output directory.")
    ap.add_argument("--corruptions", default="brightness,blur",
                    help="Comma list: brightness,blur,... (exclude none).")
    ap.add_argument("--severities", default="1,2,3,4,5",
                    help="Comma list of severities to plot (relative to clean s0).")
    ap.add_argument("--splits", default="test_mean",
                    help="Comma list: test_seen,test_similar,test_novel,test_mean")
    ap.add_argument("--pc_keys", default=",".join(PC_KEYS_DEFAULT),
                    help="Comma list of 3D keys, default block1,block2,block3,final")
    ap.add_argument("--img_keys", default=",".join(IMG_KEYS_DEFAULT),
                    help="Comma list of img keys, default p1,p2,p4,p8 (no p16)")
    ap.add_argument("--no_numbers", action="store_true", help="Disable numbers on cells.")
    ap.add_argument("--cmap", default="RdBu_r", help="Diverging colormap for deltas.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    corrs = [c.strip().lower() for c in args.corruptions.split(",") if c.strip()]
    sevs  = [int(x) for x in args.severities.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pc_keys = [x.strip() for x in args.pc_keys.split(",") if x.strip()]
    img_keys = [x.strip() for x in args.img_keys.split(",") if x.strip()]
    show_numbers = (not args.no_numbers)

    files = list_json_files(args.json_root)
    print(f"[SCAN] {len(files)} json files under {args.json_root}")

    # index json by (split, fuse, corr, sev)
    pool = {}
    for p in files:
        try:
            j = load_json(p)
        except Exception as e:
            print(f"[SKIP] load fail: {p} ({e})")
            continue

        split, fuse, corr, sev = infer_meta(j, p)
        if fuse not in FUSE_ORDER:
            continue

        key = (split, fuse, corr, int(sev))
        pool[key] = j

    def get_mean_matrix(split, fuse, corr, sev, metric_key):
        if split != "test_mean":
            j = pool.get((split, fuse, corr, sev), None)
            if j is None:
                return None
            H, _, _ = matrix_for_json(j, metric_key, pc_keys, img_keys)
            return H
        else:
            jlist = []
            for sp in ["test_seen", "test_similar", "test_novel"]:
                j = pool.get((sp, fuse, corr, sev), None)
                if j is not None:
                    jlist.append(j)
            if len(jlist) == 0:
                return None
            H, _, _ = merge_jsons_mean(jlist, metric_key, pc_keys, img_keys)
            return H

    # --------- make delta plots ----------
    for split in splits:
        for corr in corrs:
            for sev in sevs:
                # delta_mats_by_metric[metric] = [mat_early, mat_concat, mat_add, mat_gate]
                delta_mats_by_metric = {mk: [] for mk in METRIC_ORDER}
                ok_any = False

                for fuse in FUSE_ORDER:
                    # clean baseline (none, s0)
                    H_clean = {mk: get_mean_matrix(split, fuse, "none", 0, mk) for mk in METRIC_ORDER}
                    # corrupted
                    H_corr  = {mk: get_mean_matrix(split, fuse, corr, sev, mk) for mk in METRIC_ORDER}

                    for mk in METRIC_ORDER:
                        hc = H_clean[mk]
                        hn = H_corr[mk]
                        if (hc is None) or (hn is None):
                            delta = None
                        else:
                            delta = hn - hc
                            ok_any = True
                        delta_mats_by_metric[mk].append(delta)

                if not ok_any:
                    print(f"[WARN] missing data: split={split} corr={corr} s{sev}")
                    continue

                title = f"Delta vs clean (none,s0) | corr={corr} s{sev} | split={split}"
                out_path = os.path.join(args.out_dir, f"delta_vs_clean_{corr}_s{sev}_{split}.png")
                plot_delta_grid(
                    delta_mats_by_metric=delta_mats_by_metric,
                    out_path=out_path,
                    title=title,
                    pc_keys=pc_keys,
                    img_keys=img_keys,
                    show_numbers=show_numbers,
                    cmap=args.cmap,
                    vlim_mode="per_row",
                )

    print("Done.")

if __name__ == "__main__":
    main()
