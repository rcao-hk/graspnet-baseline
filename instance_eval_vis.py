#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge object-level top-k success CSV with instance-level MAE/alignment CSV,
and generate visualization analyses (without regression).

Main outputs under --output_dir:
  - merged_all.csv
  - merge_report.json
  - count_audit_steps.csv
  - count_audit.json
  - success_scene_ids.txt
  - metrics_scene_ids.txt
  - intersect_scene_ids.txt
  - per_scene_audit.csv

For a single depth metric, figures / tables are saved directly under --output_dir.
For --depth_metric all, each metric is saved into its own subfolder:
  - depth_excluding_missing/
  - depth_including_missing/
  - depth_point_l1/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, data: Dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)


def collapse_duplicates(df: pd.DataFrame, keys: Sequence[str], name: str) -> pd.DataFrame:
    if df.duplicated(list(keys)).any():
        dup_count = int(df.duplicated(list(keys)).sum())
        print(f"[{name}] Found {dup_count} duplicated key rows. Collapsing duplicates by mean/first.")
        agg = {}
        for c in df.columns:
            if c in keys:
                continue
            agg[c] = "mean" if is_numeric_series(df[c]) else "first"
        df = df.groupby(list(keys), as_index=False).agg(agg)
    return df


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def ordered_bucket_labels(values: Sequence) -> List[str]:
    uniq = [str(v) for v in pd.Series(values).dropna().astype(str).unique().tolist()]

    def _key(s: str) -> Tuple[str, int]:
        if "-Q" in s:
            prefix, tail = s.split("-Q", 1)
            try:
                return (prefix, int(tail))
            except ValueError:
                return (prefix, 999)
        return (s, 999)

    return sorted(uniq, key=_key)


def safe_qcut(series: pd.Series, q: int, prefix: str) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < 2:
        return pd.Series([np.nan] * len(series), index=series.index, dtype=object)

    q_eff = int(min(max(1, q), valid.nunique()))
    try:
        out = pd.qcut(series, q=q_eff, duplicates="drop")
    except ValueError:
        ranks = series.rank(method="average")
        out = pd.qcut(ranks, q=q_eff, duplicates="drop")

    if not pd.api.types.is_categorical_dtype(out):
        out = out.astype("category")

    cats = list(out.cat.categories)
    mapping = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(cats)}
    return out.map(mapping).astype("category")


LABELS_2 = ["low-align", "high-align"]
LABELS_3 = ["low-align", "mid-align", "high-align"]


def split_quantiles_within_group(g: pd.DataFrame, value_col: str, q: int, labels: Sequence[str]) -> pd.Series:
    out = pd.Series([np.nan] * len(g), index=g.index, dtype=object)
    valid = g[value_col].dropna()
    if valid.nunique() < 2:
        return out

    q_eff = int(min(max(1, q), valid.nunique()))
    used_labels = list(labels[:q_eff])
    try:
        qq = pd.qcut(valid, q=q_eff, labels=used_labels, duplicates="drop")
        out.loc[qq.index] = qq.astype(str)
        return out
    except ValueError:
        pass

    ranks = valid.rank(method="average")
    try:
        qq = pd.qcut(ranks, q=q_eff, labels=used_labels, duplicates="drop")
        out.loc[qq.index] = qq.astype(str)
        return out
    except ValueError:
        pass

    if q_eff <= 2:
        med = valid.median()
        out.loc[valid.index] = np.where(valid <= med, labels[0], labels[min(1, len(labels) - 1)])
    else:
        q1 = valid.quantile(1 / 3)
        q2 = valid.quantile(2 / 3)
        bins = np.where(valid <= q1, labels[0], np.where(valid <= q2, labels[1], labels[2]))
        out.loc[valid.index] = bins
    return out


def compute_correlations(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(len(sub))}
    return {
        "pearson": float(sub[x].corr(sub[y], method="pearson")),
        "spearman": float(sub[x].corr(sub[y], method="spearman")),
        "n": int(len(sub)),
    }


def maybe_log10(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    x = x.where(x > 0)
    return np.log10(x + eps)


def annotate_counts_on_curve(ax, x_positions, y_values, counts):
    for xi, yi, n in zip(x_positions, y_values, counts):
        if pd.isna(yi):
            continue
        ax.annotate(f"n={int(n)}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)


def count_stats(df: pd.DataFrame, keys: Sequence[str]) -> Dict[str, int]:
    stats = {
        "rows": int(len(df)),
        "unique_scenes": int(df["scene_id"].nunique()) if "scene_id" in df.columns else -1,
        "unique_scene_ann": -1,
        "unique_scene_ann_obj": -1,
    }
    if {"scene_id", "ann_id"}.issubset(df.columns):
        stats["unique_scene_ann"] = int(df[["scene_id", "ann_id"]].drop_duplicates().shape[0])
    if all(k in df.columns for k in keys):
        stats["unique_scene_ann_obj"] = int(df[list(keys)].drop_duplicates().shape[0])
    return stats


def append_audit_row(audit_rows: List[Dict], stage: str, df: pd.DataFrame, keys: Sequence[str], note: str = "") -> None:
    audit_rows.append({"stage": stage, **count_stats(df, keys), "note": note})


def save_scene_id_list(path: str | Path, scene_ids: Sequence[int]) -> None:
    with open(path, "w") as f:
        for sid in scene_ids:
            f.write(f"{int(sid)}\n")


def build_per_scene_audit(success_df: pd.DataFrame, metrics_df: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    def _count(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "scene_id" not in df.columns:
            return pd.DataFrame(columns=["scene_id", name])
        return df.groupby("scene_id", as_index=False).size().rename(columns={"size": name})

    succ = _count(success_df, "success_rows")
    met = _count(metrics_df, "metrics_rows")
    mer = _count(merged_df, "merged_rows")
    out = pd.merge(succ, met, on="scene_id", how="outer")
    out = pd.merge(out, mer, on="scene_id", how="outer")
    out = out.fillna(0)
    for c in ["success_rows", "metrics_rows", "merged_rows"]:
        if c in out.columns:
            out[c] = out[c].astype(int)
    out["in_success"] = out["success_rows"] > 0
    out["in_metrics"] = out["metrics_rows"] > 0
    out["in_merged"] = out["merged_rows"] > 0
    return out.sort_values("scene_id").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Alignment metric selection
# -----------------------------------------------------------------------------
PREFERRED_ALIGNMENT_COLS = [
    "r2_x2y_final_p1",
    "r2_y2x_final_p1",
    "cka_final_p1",
    "r2_x2y_block4_p8",
    "r2_y2x_block4_p8",
    "cka_block4_p8",
    "r2_x2y_final_p2",
    "r2_y2x_final_p2",
    "cka_final_p2",
]


def auto_pick_alignment_col(columns: Sequence[str]) -> str:
    cols = set(columns)
    for c in PREFERRED_ALIGNMENT_COLS:
        if c in cols:
            return c
    cands = [c for c in columns if c.startswith("r2_x2y_") or c.startswith("r2_y2x_") or c.startswith("cka_")]
    if not cands:
        raise ValueError("No alignment column found. Please pass --alignment_col explicitly.")
    return sorted(cands)[0]


def list_alignment_cols(columns: Sequence[str]) -> List[str]:
    cands = [c for c in columns if c.startswith("r2_x2y_") or c.startswith("r2_y2x_") or c.startswith("cka_")]
    ordered = [c for c in PREFERRED_ALIGNMENT_COLS if c in cands]
    remaining = [c for c in sorted(cands) if c not in ordered]
    return ordered + remaining


DEPTH_METRIC_MAP = {
    "legacy": ("inst_depth_mae_m", "depth_legacy"),
    "excluding_missing": ("inst_depth_mae_excluding_missing_m", "depth_excluding_missing"),
    "including_missing": ("inst_depth_mae_including_missing_m", "depth_including_missing"),
    "point_l1": ("inst_point_l1_mae_m", "depth_point_l1"),
}


def resolve_depth_metrics(args, metrics_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if args.depth_metric == "custom":
        if not args.mae_col:
            raise ValueError("--depth_metric custom requires --mae_col")
        return [(args.mae_col, f"custom_{args.mae_col}")]
    if args.depth_metric == "all":
        cols = [
            DEPTH_METRIC_MAP["excluding_missing"],
            DEPTH_METRIC_MAP["including_missing"],
            DEPTH_METRIC_MAP["point_l1"],
        ]
    else:
        cols = [DEPTH_METRIC_MAP[args.depth_metric]]
    for mae_col, _ in cols:
        if mae_col not in metrics_df.columns:
            raise KeyError(f"Depth metric column '{mae_col}' not found in metrics_csv")
    return cols


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_scatter_alignment_vs_success(df: pd.DataFrame, alignment_col: str, success_col: str, mae_col: str, out_path: str) -> Dict[str, float]:
    sub = df[[alignment_col, success_col, mae_col]].dropna().copy()
    corrs = compute_correlations(sub, alignment_col, success_col)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(sub[alignment_col], sub[success_col], c=sub[mae_col], s=12, alpha=0.55, linewidths=0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(mae_col)
    ax.set_xlabel(alignment_col)
    ax.set_ylabel(success_col)
    ax.set_title(f"Alignment vs success\nPearson={corrs['pearson']:.3f}, Spearman={corrs['spearman']:.3f}, N={corrs['n']}")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return corrs


def plot_scatter_mae_vs_success(df: pd.DataFrame, mae_col: str, success_col: str, alignment_col: str, out_path: str) -> Dict[str, float]:
    sub = df[[mae_col, success_col, alignment_col]].dropna().copy()
    sub = sub[sub[mae_col] > 0]
    corrs = compute_correlations(sub, mae_col, success_col)

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(sub[mae_col], sub[success_col], c=sub[alignment_col], s=12, alpha=0.55, linewidths=0)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(alignment_col)
    ax.set_xscale("log")
    ax.set_xlabel(f"{mae_col} (log scale)")
    ax.set_ylabel(success_col)
    ax.set_title(f"{mae_col} vs success\nPearson={corrs['pearson']:.3f}, Spearman={corrs['spearman']:.3f}, N={corrs['n']}")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return corrs


def plot_bucket_curve(bucket_df: pd.DataFrame, x_col: str, y_col: str, err_col: str, count_col: str, ylabel: str, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    x = np.arange(len(bucket_df))
    y = bucket_df[y_col].to_numpy(dtype=float)
    err = bucket_df[err_col].to_numpy(dtype=float)
    cnt = bucket_df[count_col].to_numpy(dtype=float)
    ax.errorbar(x, y, yerr=err, marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_df[x_col].astype(str).tolist())
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    annotate_counts_on_curve(ax, x, y, cnt)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_alignment_split_within_mae(split_df: pd.DataFrame, mae_bucket_col: str, split_col: str, out_path: str, title: str) -> None:
    piv = split_df.pivot(index=mae_bucket_col, columns=split_col, values="mean_success")
    piv_n = split_df.pivot(index=mae_bucket_col, columns=split_col, values="count")
    mae_order = ordered_bucket_labels(split_df[mae_bucket_col])
    piv = piv.reindex(mae_order)
    piv_n = piv_n.reindex(mae_order)

    split_order = [c for c in LABELS_3 if c in piv.columns] + [c for c in LABELS_2 if c in piv.columns and c not in LABELS_3]
    split_order += [c for c in piv.columns if c not in split_order]
    piv = piv.reindex(columns=split_order)
    piv_n = piv_n.reindex(columns=split_order)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(piv.index))
    ncols = max(1, len(piv.columns))
    width = min(0.8 / ncols, 0.28)
    offsets = (np.arange(ncols) - (ncols - 1) / 2.0) * width

    for j, col in enumerate(piv.columns):
        vals = piv[col].to_numpy(dtype=float)
        bars = ax.bar(x + offsets[j], vals, width=width, label=col)
        cnts = piv_n[col].to_numpy(dtype=float)
        for bar, n in zip(bars, cnts):
            h = bar.get_height()
            if not np.isfinite(h):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"n={int(n)}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in piv.index.tolist()])
    ax.set_xlabel(mae_bucket_col)
    ax.set_ylabel("mean success")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(heat_df: pd.DataFrame, mae_bucket_col: str, align_bucket_col: str, out_path: str) -> None:
    pivot = heat_df.pivot(index=mae_bucket_col, columns=align_bucket_col, values="mean_success")
    pivot_n = heat_df.pivot(index=mae_bucket_col, columns=align_bucket_col, values="count")
    row_order = ordered_bucket_labels(heat_df[mae_bucket_col])
    col_order = ordered_bucket_labels(heat_df[align_bucket_col])
    pivot = pivot.reindex(index=row_order, columns=col_order)
    pivot_n = pivot_n.reindex(index=row_order, columns=col_order)
    arr = pivot.to_numpy(dtype=float)
    arr_n = pivot_n.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    im = ax.imshow(arr, aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("mean success")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])
    ax.set_xlabel(align_bucket_col)
    ax.set_ylabel(mae_bucket_col)
    ax.set_title("Mean success across MAE/alignment buckets")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            n = arr_n[i, j]
            txt = "nan" if np.isnan(val) else f"{val:.3f}\n(n={int(n)})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_within_bucket_bin_curves(df: pd.DataFrame, mae_bucket_col: str, alignment_col: str, success_col: str, num_bins: int) -> pd.DataFrame:
    rows = []
    mae_order = ordered_bucket_labels(df[mae_bucket_col])
    for mae_bucket in mae_order:
        g = df[df[mae_bucket_col].astype(str) == mae_bucket].copy()
        g = g[[alignment_col, success_col]].dropna()
        if len(g) < 8 or g[alignment_col].nunique() < 2:
            continue
        q_eff = int(min(max(2, num_bins), g[alignment_col].nunique(), len(g)))
        try:
            bins = pd.qcut(g[alignment_col], q=q_eff, duplicates="drop")
        except ValueError:
            bins = pd.qcut(g[alignment_col].rank(method="average"), q=q_eff, duplicates="drop")
        g = g.assign(_bin=bins)
        grp = (
            g.groupby("_bin", observed=False)
            .agg(
                count=(success_col, "size"),
                mean_success=(success_col, "mean"),
                mean_alignment=(alignment_col, "mean"),
                min_alignment=(alignment_col, "min"),
                max_alignment=(alignment_col, "max"),
            )
            .reset_index(drop=True)
        )
        grp[mae_bucket_col] = mae_bucket
        grp["bin_idx"] = np.arange(1, len(grp) + 1)
        rows.append(grp)
    if not rows:
        return pd.DataFrame(columns=[mae_bucket_col, "bin_idx", "count", "mean_success", "mean_alignment", "min_alignment", "max_alignment"])
    return pd.concat(rows, ignore_index=True)


def plot_within_mae_scatter_bin_curves(df: pd.DataFrame, mae_bucket_col: str, alignment_col: str, success_col: str, num_bins: int, out_path: str) -> pd.DataFrame:
    mae_order = ordered_bucket_labels(df[mae_bucket_col])
    curve_df = compute_within_bucket_bin_curves(df, mae_bucket_col, alignment_col, success_col, num_bins)

    ncols = max(1, len(mae_order))
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6), squeeze=False)
    axes = axes[0]

    for ax, mae_bucket in zip(axes, mae_order):
        g = df[df[mae_bucket_col].astype(str) == mae_bucket].copy()
        g = g[[alignment_col, success_col]].dropna()
        corrs = compute_correlations(g, alignment_col, success_col)
        ax.scatter(g[alignment_col], g[success_col], s=10, alpha=0.35, linewidths=0)
        c = curve_df[curve_df[mae_bucket_col].astype(str) == mae_bucket].copy()
        if len(c) > 0:
            ax.plot(c["mean_alignment"], c["mean_success"], marker="o", linewidth=2)
            for _, row in c.iterrows():
                ax.annotate(f"n={int(row['count'])}", (row["mean_alignment"], row["mean_success"]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
        ax.set_title(f"{mae_bucket}\nP={corrs['pearson']:.3f}, S={corrs['spearman']:.3f}, N={corrs['n']}")
        ax.set_xlabel(alignment_col)
        ax.set_ylabel(success_col)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return curve_df


def summarize_alignment_metric(df: pd.DataFrame, metric_col: str, success_col: str, mae_col: str, mae_bucket_col: str) -> Dict[str, float]:
    sub = df[[metric_col, success_col, mae_col, mae_bucket_col]].dropna().copy()
    if len(sub) == 0:
        return {
            "alignment_col": metric_col,
            "n": 0,
            "pearson": float("nan"),
            "spearman": float("nan"),
            "high_mae_low": float("nan"),
            "high_mae_mid": float("nan"),
            "high_mae_high": float("nan"),
            "high_minus_low_in_high_mae": float("nan"),
        }

    corr = compute_correlations(sub, metric_col, success_col)
    high_mae_label = ordered_bucket_labels(sub[mae_bucket_col])[-1]
    g = sub[sub[mae_bucket_col].astype(str) == high_mae_label].copy()
    vals = {}
    if len(g) >= 3 and g[metric_col].nunique() >= 2:
        g["metric_split3"] = split_quantiles_within_group(g, metric_col, 3, LABELS_3)
        agg = g.groupby("metric_split3", observed=False)[success_col].mean()
        vals = {k: float(agg.get(k, np.nan)) for k in LABELS_3}
    else:
        vals = {k: float("nan") for k in LABELS_3}
    return {
        "alignment_col": metric_col,
        "n": int(len(sub)),
        "pearson": float(corr["pearson"]),
        "spearman": float(corr["spearman"]),
        "high_mae_low": vals["low-align"],
        "high_mae_mid": vals["mid-align"],
        "high_mae_high": vals["high-align"],
        "high_minus_low_in_high_mae": float(vals["high-align"] - vals["low-align"]) if np.isfinite(vals["high-align"]) and np.isfinite(vals["low-align"]) else float("nan"),
    }


def plot_metric_rank(metric_df: pd.DataFrame, value_col: str, title: str, out_path: str) -> None:
    show = metric_df[["alignment_col", value_col, "n"]].dropna(subset=[value_col]).sort_values(value_col, ascending=False)
    if len(show) == 0:
        return
    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(show)), 4.8))
    x = np.arange(len(show))
    bars = ax.bar(x, show[value_col].to_numpy(dtype=float))
    ax.set_xticks(x)
    ax.set_xticklabels(show["alignment_col"].tolist(), rotation=45, ha="right")
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    for bar, n in zip(bars, show["n"].to_numpy(dtype=float)):
        h = bar.get_height()
        if not np.isfinite(h):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005 * (1 if h >= 0 else -1), f"n={int(n)}", ha="center", va="bottom" if h >= 0 else "top", fontsize=7, rotation=90)
    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# One-metric analysis
# -----------------------------------------------------------------------------
def analyze_single_mae_metric(
    merged: pd.DataFrame,
    output_dir: str,
    mae_col: str,
    mae_tag: str,
    alignment_col: str,
    success_col: str,
    keys: Sequence[str],
    valid_overlap_col: Optional[str],
    sampled_points_col: Optional[str],
    num_pred_topk_col: Optional[str],
    args,
) -> Dict:
    ensure_dir(output_dir)

    filt = merged.copy()
    if valid_overlap_col is not None:
        filt = filt[filt[valid_overlap_col] >= int(args.min_valid_overlap)]
    if sampled_points_col is not None:
        filt = filt[filt[sampled_points_col] >= int(args.min_sampled_points)]
    if num_pred_topk_col is not None:
        filt = filt[filt[num_pred_topk_col] >= int(args.min_pred_topk)]
    if args.drop_missing_mae:
        filt = filt[filt[mae_col].notna()]
    filt = filt[filt[alignment_col].notna() & filt[success_col].notna()]

    merged_filtered_path = os.path.join(output_dir, "merged_filtered.csv")
    filt.to_csv(merged_filtered_path, index=False)

    if len(filt) == 0:
        raise RuntimeError(f"No rows left after filtering for mae_col={mae_col}.")

    filt["mae_bucket"] = safe_qcut(filt[mae_col], args.mae_num_bins, prefix="MAE-Q")
    filt["alignment_bucket"] = safe_qcut(filt[alignment_col], args.align_num_bins, prefix="ALIGN-Q")
    filt["alignment_split"] = filt.groupby("mae_bucket", group_keys=False).apply(
        lambda g: split_quantiles_within_group(g, alignment_col, 2, LABELS_2)
    )
    filt["alignment_split3"] = filt.groupby("mae_bucket", group_keys=False).apply(
        lambda g: split_quantiles_within_group(g, alignment_col, 3, LABELS_3)
    )

    scatter_align_corr = plot_scatter_alignment_vs_success(
        filt,
        alignment_col,
        success_col,
        mae_col,
        os.path.join(output_dir, "scatter_alignment_vs_success.png"),
    )
    scatter_mae_corr = plot_scatter_mae_vs_success(
        filt,
        mae_col,
        success_col,
        alignment_col,
        os.path.join(output_dir, "scatter_mae_vs_success.png"),
    )

    bucket_summary = (
        filt.groupby("mae_bucket", observed=False)
        .agg(
            count=(success_col, "size"),
            mean_success=(success_col, "mean"),
            std_success=(success_col, "std"),
            mean_alignment=(alignment_col, "mean"),
            std_alignment=(alignment_col, "std"),
            mean_mae=(mae_col, "mean"),
        )
        .reset_index()
    )
    bucket_summary["se_success"] = bucket_summary["std_success"] / np.sqrt(bucket_summary["count"].clip(lower=1))
    bucket_summary["se_alignment"] = bucket_summary["std_alignment"] / np.sqrt(bucket_summary["count"].clip(lower=1))
    bucket_summary.to_csv(os.path.join(output_dir, "bucket_summary.csv"), index=False)

    plot_bucket_curve(
        bucket_summary,
        x_col="mae_bucket",
        y_col="mean_success",
        err_col="se_success",
        count_col="count",
        ylabel=success_col,
        title=f"Success across MAE buckets ({mae_col})",
        out_path=os.path.join(output_dir, "mae_bucket_success.png"),
    )
    plot_bucket_curve(
        bucket_summary,
        x_col="mae_bucket",
        y_col="mean_alignment",
        err_col="se_alignment",
        count_col="count",
        ylabel=alignment_col,
        title=f"Alignment across MAE buckets ({mae_col})",
        out_path=os.path.join(output_dir, "mae_bucket_alignment.png"),
    )

    heat_df = (
        filt.groupby(["mae_bucket", "alignment_bucket"], observed=False)
        .agg(
            count=(success_col, "size"),
            mean_success=(success_col, "mean"),
            mean_mae=(mae_col, "mean"),
            mean_alignment=(alignment_col, "mean"),
        )
        .reset_index()
    )
    heat_df.to_csv(os.path.join(output_dir, "heatmap_mean_success.csv"), index=False)
    plot_heatmap(
        heat_df,
        mae_bucket_col="mae_bucket",
        align_bucket_col="alignment_bucket",
        out_path=os.path.join(output_dir, "mae_alignment_heatmap.png"),
    )

    split_df = (
        filt.dropna(subset=["mae_bucket", "alignment_split"])
        .groupby(["mae_bucket", "alignment_split"], observed=False)
        .agg(
            count=(success_col, "size"),
            mean_success=(success_col, "mean"),
            mean_alignment=(alignment_col, "mean"),
            mean_mae=(mae_col, "mean"),
        )
        .reset_index()
    )
    split_df.to_csv(os.path.join(output_dir, "alignment_split_by_mae.csv"), index=False)
    plot_alignment_split_within_mae(
        split_df,
        mae_bucket_col="mae_bucket",
        split_col="alignment_split",
        out_path=os.path.join(output_dir, "alignment_split_within_mae_bucket.png"),
        title="Success difference between low/high alignment within each MAE bucket (n shown)",
    )

    split3_df = (
        filt.dropna(subset=["mae_bucket", "alignment_split3"])
        .groupby(["mae_bucket", "alignment_split3"], observed=False)
        .agg(
            count=(success_col, "size"),
            mean_success=(success_col, "mean"),
            mean_alignment=(alignment_col, "mean"),
            mean_mae=(mae_col, "mean"),
        )
        .reset_index()
    )
    split3_df.to_csv(os.path.join(output_dir, "alignment_tertile_by_mae.csv"), index=False)
    plot_alignment_split_within_mae(
        split3_df,
        mae_bucket_col="mae_bucket",
        split_col="alignment_split3",
        out_path=os.path.join(output_dir, "alignment_tertile_within_mae_bucket.png"),
        title="Success difference between low/mid/high alignment within each MAE bucket (n shown)",
    )

    curve_df = plot_within_mae_scatter_bin_curves(
        filt.dropna(subset=["mae_bucket"]),
        mae_bucket_col="mae_bucket",
        alignment_col=alignment_col,
        success_col=success_col,
        num_bins=args.within_mae_curve_bins,
        out_path=os.path.join(output_dir, "within_mae_scatter_bin_curves.png"),
    )
    curve_df.to_csv(os.path.join(output_dir, "within_mae_scatter_bins.csv"), index=False)

    metric_summary_df = pd.DataFrame()
    metric_cols_used = [alignment_col]
    if args.analyze_all_alignment_metrics:
        metric_cols_used = list_alignment_cols(merged.columns)
        if args.max_alignment_metrics and args.max_alignment_metrics > 0:
            metric_cols_used = metric_cols_used[:args.max_alignment_metrics]
        metric_rows = []
        for metric_col in metric_cols_used:
            sub = merged.copy()
            if valid_overlap_col is not None:
                sub = sub[sub[valid_overlap_col] >= int(args.min_valid_overlap)]
            if sampled_points_col is not None:
                sub = sub[sub[sampled_points_col] >= int(args.min_sampled_points)]
            if num_pred_topk_col is not None:
                sub = sub[sub[num_pred_topk_col] >= int(args.min_pred_topk)]
            if args.drop_missing_mae:
                sub = sub[sub[mae_col].notna()]
            sub = sub[sub[metric_col].notna() & sub[success_col].notna()].copy()
            if len(sub) == 0:
                continue
            sub["mae_bucket"] = safe_qcut(sub[mae_col], args.mae_num_bins, prefix="MAE-Q")
            metric_rows.append(summarize_alignment_metric(sub, metric_col, success_col, mae_col, "mae_bucket"))

        metric_summary_df = pd.DataFrame(metric_rows)
        metric_summary_df.to_csv(os.path.join(output_dir, "multi_alignment_metric_summary.csv"), index=False)
        if len(metric_summary_df) > 0:
            plot_metric_rank(
                metric_summary_df,
                value_col="spearman",
                title=f"Alignment metrics ranked by overall Spearman with success ({mae_col})",
                out_path=os.path.join(output_dir, "alignment_metric_correlation_rank.png"),
            )
            plot_metric_rank(
                metric_summary_df,
                value_col="high_minus_low_in_high_mae",
                title=f"Alignment metrics ranked by (high-align - low-align) within highest-MAE bucket ({mae_col})",
                out_path=os.path.join(output_dir, "alignment_metric_high_mae_gap_rank.png"),
            )

    report = {
        "mae_col": mae_col,
        "mae_tag": mae_tag,
        "alignment_col": alignment_col,
        "success_col": success_col,
        "keys": list(keys),
        "thresholds": {
            "min_valid_overlap": int(args.min_valid_overlap),
            "min_pred_topk": int(args.min_pred_topk),
            "min_sampled_points": int(args.min_sampled_points),
            "drop_missing_mae": bool(args.drop_missing_mae),
        },
        "counts": {
            "filtered_rows": int(len(filt)),
            "filtered_scene_ann": int(filt[["scene_id", "ann_id"]].drop_duplicates().shape[0]),
            "filtered_scene_ann_obj": int(filt[list(keys)].drop_duplicates().shape[0]),
        },
        "correlations": {
            "alignment_vs_success": scatter_align_corr,
            "mae_vs_success": scatter_mae_corr,
        },
        "bucket_summary_preview": bucket_summary.to_dict(orient="records"),
        "alignment_metrics_analyzed": metric_cols_used,
    }
    save_json(os.path.join(output_dir, "metric_report.json"), report)
    return report


# -----------------------------------------------------------------------------
# Parser / main
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--success_csv", type=str, required=True)
    p.add_argument("--metrics_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--merge_how", type=str, default="inner", choices=["inner", "left", "right", "outer"])
    p.add_argument("--keys", type=str, default="scene_id,ann_id,object_local_id")
    p.add_argument("--alignment_col", type=str, default=None)
    p.add_argument("--success_col", type=str, default="succ_rate_topk")
    p.add_argument("--mae_col", type=str, default=None, help="Used only when --depth_metric custom")
    p.add_argument(
        "--depth_metric",
        type=str,
        default="legacy",
        choices=["legacy", "excluding_missing", "including_missing", "point_l1", "all", "custom"],
        help="Choose which depth reliability metric to visualize. Use all to save all three non-legacy depth metrics.",
    )
    p.add_argument("--min_valid_overlap", type=int, default=32)
    p.add_argument("--min_pred_topk", type=int, default=1)
    p.add_argument("--min_sampled_points", type=int, default=64)
    p.add_argument("--drop_missing_mae", action="store_true")
    p.add_argument("--mae_num_bins", type=int, default=3)
    p.add_argument("--align_num_bins", type=int, default=3)
    p.add_argument("--within_mae_curve_bins", type=int, default=5)
    p.add_argument("--analyze_all_alignment_metrics", action="store_true")
    p.add_argument("--max_alignment_metrics", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if len(keys) == 0:
        raise ValueError("--keys must contain at least one key column")

    success_df = pd.read_csv(args.success_csv)
    metrics_df = pd.read_csv(args.metrics_csv)

    missing_success = [k for k in keys if k not in success_df.columns]
    missing_metrics = [k for k in keys if k not in metrics_df.columns]
    if missing_success:
        raise KeyError(f"Missing merge keys in success_csv: {missing_success}")
    if missing_metrics:
        raise KeyError(f"Missing merge keys in metrics_csv: {missing_metrics}")

    success_df = collapse_duplicates(success_df, keys, name="success")
    metrics_df = collapse_duplicates(metrics_df, keys, name="metrics")

    success_scene_ids = sorted(success_df["scene_id"].unique().tolist()) if "scene_id" in success_df.columns else []
    metrics_scene_ids = sorted(metrics_df["scene_id"].unique().tolist()) if "scene_id" in metrics_df.columns else []
    intersect_scene_ids = sorted(set(success_scene_ids) & set(metrics_scene_ids))

    alignment_col = args.alignment_col or auto_pick_alignment_col(metrics_df.columns)
    if alignment_col not in metrics_df.columns:
        raise KeyError(f"alignment_col '{alignment_col}' not found in metrics_csv")
    if args.success_col not in success_df.columns:
        raise KeyError(f"success_col '{args.success_col}' not found in success_csv")

    size_col = first_existing(metrics_df.columns, ["inst_bbox_diag_m", "inst_size_points", "inst_mask_area"])
    valid_overlap_col = first_existing(metrics_df.columns, ["inst_num_valid_overlap"])
    sampled_points_col = first_existing(metrics_df.columns, ["inst_num_sampled_points", "inst_num_points_sampled", "inst_num_points"])
    num_pred_topk_col = first_existing(success_df.columns, ["num_pred_topk"])

    merged = pd.merge(
        success_df,
        metrics_df,
        on=keys,
        how=args.merge_how,
        indicator=True,
        suffixes=("_succ", "_met"),
    )
    if size_col is not None:
        merged["size_log10"] = maybe_log10(merged[size_col])

    merged_all_path = os.path.join(args.output_dir, "merged_all.csv")
    merged.to_csv(merged_all_path, index=False)
    save_scene_id_list(os.path.join(args.output_dir, "success_scene_ids.txt"), success_scene_ids)
    save_scene_id_list(os.path.join(args.output_dir, "metrics_scene_ids.txt"), metrics_scene_ids)
    save_scene_id_list(os.path.join(args.output_dir, "intersect_scene_ids.txt"), intersect_scene_ids)

    per_scene_audit_df = build_per_scene_audit(success_df, metrics_df, merged)
    per_scene_audit_path = os.path.join(args.output_dir, "per_scene_audit.csv")
    per_scene_audit_df.to_csv(per_scene_audit_path, index=False)

    audit_rows: List[Dict] = []
    append_audit_row(audit_rows, "success_input", success_df, keys, note="raw success csv after duplicate collapse")
    append_audit_row(audit_rows, "metrics_input", metrics_df, keys, note="raw metrics csv after duplicate collapse")
    append_audit_row(audit_rows, "merged", merged, keys, note=f"merge_how={args.merge_how}")
    tmp = merged.copy()
    if valid_overlap_col is not None:
        tmp = tmp[tmp[valid_overlap_col] >= int(args.min_valid_overlap)]
        append_audit_row(audit_rows, "after_valid_overlap", tmp, keys, note=f"{valid_overlap_col}>={int(args.min_valid_overlap)}")
    else:
        append_audit_row(audit_rows, "after_valid_overlap", tmp, keys, note="skipped (column missing)")
    if sampled_points_col is not None:
        tmp = tmp[tmp[sampled_points_col] >= int(args.min_sampled_points)]
        append_audit_row(audit_rows, "after_sampled_points", tmp, keys, note=f"{sampled_points_col}>={int(args.min_sampled_points)}")
    else:
        append_audit_row(audit_rows, "after_sampled_points", tmp, keys, note="skipped (column missing)")
    if num_pred_topk_col is not None:
        tmp = tmp[tmp[num_pred_topk_col] >= int(args.min_pred_topk)]
        append_audit_row(audit_rows, "after_num_pred_topk", tmp, keys, note=f"{num_pred_topk_col}>={int(args.min_pred_topk)}")
    else:
        append_audit_row(audit_rows, "after_num_pred_topk", tmp, keys, note="skipped (column missing)")
    append_audit_row(audit_rows, "after_drop_missing_mae", tmp, keys, note="metric-specific; see per-metric subfolders when --drop_missing_mae is enabled")
    tmp = tmp[tmp[alignment_col].notna() & tmp[args.success_col].notna()]
    append_audit_row(audit_rows, "after_notna_alignment_success", tmp, keys, note=f"require non-NaN {alignment_col} and {args.success_col}")

    pd.DataFrame(audit_rows).to_csv(os.path.join(args.output_dir, "count_audit_steps.csv"), index=False)
    save_json(os.path.join(args.output_dir, "count_audit.json"), {"steps": audit_rows})

    metric_specs = resolve_depth_metrics(args, metrics_df)
    metric_reports = []
    for mae_col, mae_tag in metric_specs:
        metric_output_dir = args.output_dir if len(metric_specs) == 1 else os.path.join(args.output_dir, mae_tag)
        report = analyze_single_mae_metric(
            merged=merged,
            output_dir=metric_output_dir,
            mae_col=mae_col,
            mae_tag=mae_tag,
            alignment_col=alignment_col,
            success_col=args.success_col,
            keys=keys,
            valid_overlap_col=valid_overlap_col,
            sampled_points_col=sampled_points_col,
            num_pred_topk_col=num_pred_topk_col,
            args=args,
        )
        metric_reports.append(report)

    merge_report = {
        "success_csv": os.path.abspath(args.success_csv),
        "metrics_csv": os.path.abspath(args.metrics_csv),
        "merged_all_csv": os.path.abspath(merged_all_path),
        "output_dir": os.path.abspath(args.output_dir),
        "keys": keys,
        "merge_how": args.merge_how,
        "alignment_col": alignment_col,
        "success_col": args.success_col,
        "depth_metric": args.depth_metric,
        "size_col": size_col,
        "valid_overlap_col": valid_overlap_col,
        "sampled_points_col": sampled_points_col,
        "num_pred_topk_col": num_pred_topk_col,
        "counts": {
            "success_rows": int(len(success_df)),
            "metrics_rows": int(len(metrics_df)),
            "merged_rows": int(len(merged)),
            "matched_rows": int((merged["_merge"] == "both").sum()),
        },
        "scene_sets": {
            "success_scene_ids": success_scene_ids,
            "metrics_scene_ids": metrics_scene_ids,
            "intersect_scene_ids": intersect_scene_ids,
        },
        "count_audit": audit_rows,
        "metric_reports": metric_reports,
    }
    save_json(os.path.join(args.output_dir, "merge_report.json"), merge_report)

    print("[done] merged_all:", merged_all_path)
    print("[done] count_audit_steps:", os.path.join(args.output_dir, "count_audit_steps.csv"))
    print("[done] success_scene_ids:", os.path.join(args.output_dir, "success_scene_ids.txt"))
    print("[done] metrics_scene_ids:", os.path.join(args.output_dir, "metrics_scene_ids.txt"))
    print("[done] intersect_scene_ids:", os.path.join(args.output_dir, "intersect_scene_ids.txt"))
    print("[done] per_scene_audit:", per_scene_audit_path)
    for _, mae_tag in metric_specs:
        metric_output_dir = args.output_dir if len(metric_specs) == 1 else os.path.join(args.output_dir, mae_tag)
        print("[done] metric_output:", metric_output_dir)


if __name__ == "__main__":
    main()
