#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPLITS = ["seen", "similar", "novel"]
METHODS = ["early", "intermediate"]


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def safe_nanmean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if len(xs) > 0 else float("nan")


def extract(fpath: str):
    """
    Compute:
      - AP@fric: overall success rate within global topK at each friction threshold
      - mAP_fric: mean(AP@fric) over friction thresholds (GraspNet-style friction-averaged)
      - per-bucket mAP_fric
    """
    j = load_json(fpath)
    fric_list = [float(x) for x in j["meta"]["fric_list"]]
    buckets = j["buckets"]  # keys: "0","1","2"

    # overall (aggregated across buckets) AP@fric
    total_n = 0
    succ_cnt = {fr: 0 for fr in fric_list}

    for b in ["0", "1", "2"]:
        nb = int(buckets[b]["n_in_topk_total"])
        total_n += nb
        for fr_str, sd in buckets[b]["success"].items():
            fr = float(fr_str)
            succ_cnt[fr] += int(sd["succ_count"])

    ap_by_fric = {
        fr: (succ_cnt[fr] / total_n) if total_n > 0 else float("nan")
        for fr in fric_list
    }
    mAP_fric = safe_nanmean([ap_by_fric[fr] for fr in fric_list])

    # per-bucket AP@fric and mAP_fric
    per_bucket_ap = {}
    per_bucket_mAP = {}
    for b in ["0", "1", "2"]:
        nb = int(buckets[b]["n_in_topk_total"])
        ap_b = {}
        for fr in fric_list:
            sd = buckets[b]["success"].get(str(fr), None)
            if sd is None:
                ap_b[fr] = float("nan")
            else:
                # succ_rate in json already = succ_count / n_in_topk_total (weighted)
                v = sd.get("succ_rate", float("nan"))
                try:
                    ap_b[fr] = float(v)
                except Exception:
                    ap_b[fr] = float("nan")
        per_bucket_ap[int(b)] = ap_b
        per_bucket_mAP[int(b)] = safe_nanmean([ap_b[fr] for fr in fric_list])

    return {
        "fric_list": fric_list,
        "total_n": total_n,
        "ap_by_fric": ap_by_fric,         # AP@μ
        "mAP_fric": mAP_fric,             # mean over μ
        "per_bucket_ap": per_bucket_ap,   # per bucket AP@μ
        "per_bucket_mAP": per_bucket_mAP  # per bucket mean over μ
    }


def line_compare(out_path, title, fric_list, y_by_method, y_label):
    plt.figure(figsize=(7.2, 4.2))
    for name, ys in y_by_method.items():
        plt.plot(fric_list, ys, marker="o", label=name)
    plt.xlabel("Friction coefficient (μ)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bar_compare(out_path, title, x_labels, series_dict, y_label):
    series_names = list(series_dict.keys())
    x = np.arange(len(x_labels))
    width = 0.36 if len(series_names) == 2 else max(0.2, 0.8 / len(series_names))

    plt.figure(figsize=(8.0, 4.2))
    for i, name in enumerate(series_names):
        vals = series_dict[name]
        plt.bar(x + (i - (len(series_names)-1)/2) * width, vals, width=width, label=name)
        for xi, v in zip(x, vals):
            if np.isfinite(v):
                plt.text(
                    xi + (i - (len(series_names)-1)/2) * width, v,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9
                )

    plt.xticks(x, x_labels)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, default="vis",
                    help="Directory containing {method}_{split}_depthnoise.json files")
    ap.add_argument("--out_dir", type=str, default="vis/depthnoise_splits_map",
                    help="Output directory for figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {m: {} for m in METHODS}

    # load all
    for m in METHODS:
        for s in SPLITS:
            fn = f"{m}_{s}_depthnoise.json"
            fp = os.path.join(args.json_dir, fn)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing file: {fp}")
            results[m][s] = extract(fp)

    # print summary (mAP over frictions)
    print("\n=== DepthNoise Conditional-on-global-topK summary (GraspNet-style friction-averaged mAP) ===")
    for s in SPLITS:
        e = results["early"][s]
        i = results["intermediate"][s]
        print(f"\n[{s.upper()}]")
        print(f"  early        : mAP_over_frics={e['mAP_fric']:.4f} | total_n={e['total_n']}")
        print(f"  intermediate : mAP_over_frics={i['mAP_fric']:.4f} | total_n={i['total_n']}")

    # macro-average across splits (equal weight)
    for m in METHODS:
        mAP_macro = float(np.mean([results[m][s]["mAP_fric"] for s in SPLITS]))
        print(f"\n[{m.upper()} macro avg over splits]")
        print(f"  mAP_over_frics_macro = {mAP_macro:.4f}")

    # plots per split:
    # (1) AP@μ curves
    # (2) per-bucket mAP bars
    for s in SPLITS:
        fric_list = results["early"][s]["fric_list"]

        y_by_method = {
            "early": [results["early"][s]["ap_by_fric"][x] for x in fric_list],
            "intermediate": [results["intermediate"][s]["ap_by_fric"][x] for x in fric_list],
        }
        line_compare(
            out_path=os.path.join(args.out_dir, f"AP_by_fric_curve_{s}.png"),
            title=f"AP@μ within global topK vs friction ({s})",
            fric_list=fric_list,
            y_by_method=y_by_method,
            y_label="AP@μ (success rate within global topK)"
        )

        # per-bucket mAP (mean over frictions)
        x_labels = ["B0", "B1", "B2"]
        bar_compare(
            out_path=os.path.join(args.out_dir, f"bucket_mAP_over_frics_{s}.png"),
            title=f"Per-bucket mAP (mean over frictions) ({s})",
            x_labels=x_labels,
            series_dict={
                "early": [results["early"][s]["per_bucket_mAP"][b] for b in [0, 1, 2]],
                "intermediate": [results["intermediate"][s]["per_bucket_mAP"][b] for b in [0, 1, 2]],
            },
            y_label="mAP_over_frics"
        )

    # cross-split mAP bar
    bar_compare(
        out_path=os.path.join(args.out_dir, "mAP_over_frics_by_split.png"),
        title="mAP (mean AP over frictions) by split",
        x_labels=[s.upper() for s in SPLITS],
        series_dict={
            "early": [results["early"][s]["mAP_fric"] for s in SPLITS],
            "intermediate": [results["intermediate"][s]["mAP_fric"] for s in SPLITS],
        },
        y_label="mAP_over_frics"
    )

    print(f"\nSaved figures to: {args.out_dir}")
    for s in SPLITS:
        print(f"  - AP_by_fric_curve_{s}.png")
        print(f"  - bucket_mAP_over_frics_{s}.png")
    print("  - mAP_over_frics_by_split.png")


if __name__ == "__main__":
    main()
