#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


SPLIT_ORDER = ["test_seen", "test_similar", "test_novel"]
SPLIT_TO_AP_SUFFIX = {
    "test_seen": "seen",
    "test_similar": "similar",
    "test_novel": "novel",
    "seen": "seen",
    "similar": "similar",
    "novel": "novel",
}


def parse_meta(path):
    meta = {}
    if not os.path.exists(path):
        return meta
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
    return meta


def try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_ap_from_text(text):
    """
    Robustly parse one AP value from an eval log.

    Supports common forms:
      AP: 0.721
      AP = 0.721
      test_seen: AP = 0.721 | CR@10 = ...

    Returns the last pure AP match. It intentionally avoids AP0.4/AP0.8-like
    keys when possible.
    """
    patterns = [
        r"(?<![A-Za-z0-9_])AP(?![A-Za-z0-9_\.])\s*[:=]\s*([0-9]*\.?[0-9]+)",
        r"\bAP\s+([0-9]*\.?[0-9]+)",
        r"\bAverage\s+Precision\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = try_float(match.group(1))
            if value is not None:
                values.append(value)
    if values:
        return values[-1]

    # Fallback: parse a bracketed list like [seen, similar, novel, mean].
    bracket_values = []
    for match in re.finditer(r"\[([^\[\]]+)\]", text):
        nums = [try_float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", match.group(1))]
        nums = [x for x in nums if x is not None]
        if nums:
            bracket_values.append(nums)
    if bracket_values and len(bracket_values[-1]) == 1:
        return bracket_values[-1][0]
    return None


def parse_eval_log(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_ap_from_text(f.read())


def split_to_ap_suffix(split):
    if split in SPLIT_TO_AP_SUFFIX:
        return SPLIT_TO_AP_SUFFIX[split]
    if split.startswith("test_"):
        return split[len("test_"):]
    return split


def find_ap_npy(split_dir, variant_dir, split, camera):
    split_suffix = split_to_ap_suffix(split)
    filename = f"ap_test_{split_suffix}_{camera}.npy"
    candidates = [
        os.path.join(split_dir, filename),
        os.path.join(variant_dir, filename),
    ]
    candidates.extend(sorted(glob.glob(os.path.join(split_dir, "**", filename), recursive=True)))
    candidates.extend(sorted(glob.glob(os.path.join(variant_dir, "**", filename), recursive=True)))

    seen = set()
    unique_candidates = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_candidates.append(path)

    for path in unique_candidates:
        if os.path.exists(path):
            return path
    return ""


def load_ap_from_npy(path, topk=50):
    """
    Recompute AP from GraspNet eval output, following the experiment_replay logic:
      AP     = mean(res[..., :topk, :])
      AP0.2  = mean(res[..., :topk, 0])
      AP0.4  = mean(res[..., :topk, 1])
      AP0.6  = mean(res[..., :topk, 2])
      AP0.8  = mean(res[..., :topk, 3])
    """
    if not path or not os.path.exists(path):
        return {}
    res = np.load(path)
    if res.ndim < 2:
        raise ValueError(f"Unexpected AP array shape in {path}: {res.shape}")

    topk_eff = min(int(topk), int(res.shape[-2]))
    top = res[..., :topk_eff, :]
    metrics = {
        "ap": float(np.mean(top)),
        "ap_topk": topk_eff,
        "ap_npy_shape": "x".join(str(x) for x in res.shape),
    }
    if res.shape[-1] >= 1:
        metrics["ap0.2"] = float(np.mean(res[..., :topk_eff, 0]))
    if res.shape[-1] >= 2:
        metrics["ap0.4"] = float(np.mean(res[..., :topk_eff, 1]))
    if res.shape[-1] >= 3:
        metrics["ap0.6"] = float(np.mean(res[..., :topk_eff, 2]))
    if res.shape[-1] >= 4:
        metrics["ap0.8"] = float(np.mean(res[..., :topk_eff, 3]))
    return metrics


def load_grouping_timer(dump_dir, grouping):
    candidates = sorted(glob.glob(os.path.join(dump_dir, "grouping_timer_*.json")))
    if not candidates:
        return {}
    with open(candidates[-1], "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get(grouping, {})


def fmt_float(x, ndigits=4):
    if x is None:
        return ""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return ""
    return f"{x:.{ndigits}f}"


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, help="Sweep run root created by run_grouping_nsample_sweep.sh")
    parser.add_argument("--baseline_nsample", type=int, default=32)
    parser.add_argument("--camera", default="realsense")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--out_prefix", default=None)
    args = parser.parse_args()

    run_root = os.path.abspath(args.run_root)
    log_root = os.path.join(run_root, "logs")
    out_prefix = args.out_prefix or os.path.join(run_root, "grouping_nsample_sweep")

    row_records = []
    variant_dirs = sorted(
        p for p in glob.glob(os.path.join(run_root, "fusion-*"))
        if os.path.isdir(p)
    )

    for variant_dir in variant_dirs:
        variant = os.path.basename(variant_dir)
        meta = parse_meta(os.path.join(variant_dir, "meta.txt"))
        grouping = meta.get("grouping_type")
        nsample = int(meta.get("grouping_nsample", "-1"))

        for split_dir in sorted(p for p in glob.glob(os.path.join(variant_dir, "test_*")) if os.path.isdir(p)):
            split = os.path.basename(split_dir)
            eval_log = os.path.join(log_root, f"{variant}_{split}_eval.log")
            infer_log = os.path.join(log_root, f"{variant}_{split}_infer.log")
            ap_npy = find_ap_npy(split_dir, variant_dir, split, args.camera)
            ap_metrics = load_ap_from_npy(ap_npy, topk=args.topk) if ap_npy else {}
            ap_from_log = parse_eval_log(eval_log)
            ap = ap_metrics.get("ap", ap_from_log)
            ap_source = "npy" if ap_metrics else ("log" if ap_from_log is not None else "")
            timer = load_grouping_timer(split_dir, grouping)

            row_records.append({
                "variant": variant,
                "grouping_type": grouping,
                "grouping_nsample": nsample,
                "split": split,
                "ap": ap,
                "ap0.2": ap_metrics.get("ap0.2"),
                "ap0.4": ap_metrics.get("ap0.4"),
                "ap0.6": ap_metrics.get("ap0.6"),
                "ap0.8": ap_metrics.get("ap0.8"),
                "ap_topk": ap_metrics.get("ap_topk"),
                "ap_source": ap_source,
                "ap_npy": ap_npy,
                "ap_npy_shape": ap_metrics.get("ap_npy_shape"),
                "ap_log_fallback": ap_from_log,
                "grouping_mean_ms": timer.get("mean_ms"),
                "grouping_total_ms": timer.get("total_ms"),
                "grouping_count": timer.get("count"),
                "grouping_warmup_count": timer.get("warmup_count"),
                "network_name": meta.get("network_name"),
                "ckpt_epoch": meta.get("ckpt_epoch"),
                "sample_interval": meta.get("sample_interval"),
                "fuse_type": meta.get("fuse_type"),
                "eval_log": eval_log if os.path.exists(eval_log) else "",
                "infer_log": infer_log if os.path.exists(infer_log) else "",
            })

    row_fields = [
        "variant", "grouping_type", "grouping_nsample", "split",
        "ap", "ap0.2", "ap0.4", "ap0.6", "ap0.8", "ap_topk",
        "ap_source", "ap_npy", "ap_npy_shape", "ap_log_fallback",
        "grouping_mean_ms", "grouping_total_ms", "grouping_count", "grouping_warmup_count",
        "network_name", "ckpt_epoch", "sample_interval", "fuse_type", "eval_log", "infer_log",
    ]
    write_csv(out_prefix + "_rows.csv", row_records, row_fields)

    grouped = defaultdict(dict)
    grouped_ap04 = defaultdict(dict)
    grouped_ap08 = defaultdict(dict)
    timer_grouped = defaultdict(list)
    meta_grouped = {}
    for row in row_records:
        key = (row["grouping_type"], row["grouping_nsample"])
        grouped[key][row["split"]] = row["ap"]
        grouped_ap04[key][row["split"]] = row.get("ap0.4")
        grouped_ap08[key][row["split"]] = row.get("ap0.8")
        if row.get("grouping_mean_ms") is not None:
            timer_grouped[key].append(float(row["grouping_mean_ms"]))
        meta_grouped[key] = {
            "network_name": row.get("network_name"),
            "ckpt_epoch": row.get("ckpt_epoch"),
            "sample_interval": row.get("sample_interval"),
            "fuse_type": row.get("fuse_type"),
        }

    base_by_group = {}
    for (grouping, nsample), split_to_ap in grouped.items():
        if nsample == args.baseline_nsample:
            vals = [split_to_ap.get(s) for s in SPLIT_ORDER]
            vals = [v for v in vals if v is not None]
            base_by_group[grouping] = sum(vals) / len(vals) if vals else None

    summary_rows = []
    for key in sorted(grouped.keys(), key=lambda x: (str(x[0]), int(x[1]))):
        grouping, nsample = key
        split_to_ap = grouped[key]
        split_to_ap04 = grouped_ap04[key]
        split_to_ap08 = grouped_ap08[key]
        vals = [split_to_ap.get(s) for s in SPLIT_ORDER]
        vals_valid = [v for v in vals if v is not None]
        mean_ap = sum(vals_valid) / len(vals_valid) if vals_valid else None
        vals_ap04 = [split_to_ap04.get(s) for s in SPLIT_ORDER]
        vals_ap04 = [v for v in vals_ap04 if v is not None]
        mean_ap04 = sum(vals_ap04) / len(vals_ap04) if vals_ap04 else None
        vals_ap08 = [split_to_ap08.get(s) for s in SPLIT_ORDER]
        vals_ap08 = [v for v in vals_ap08 if v is not None]
        mean_ap08 = sum(vals_ap08) / len(vals_ap08) if vals_ap08 else None
        timer_vals = timer_grouped.get(key, [])
        mean_grouping_ms = sum(timer_vals) / len(timer_vals) if timer_vals else None
        base = base_by_group.get(grouping)

        row = {
            "grouping_type": grouping,
            "grouping_nsample": nsample,
            "ap_seen": split_to_ap.get("test_seen"),
            "ap_similar": split_to_ap.get("test_similar"),
            "ap_novel": split_to_ap.get("test_novel"),
            "mean_ap": mean_ap,
            "mean_ap0.4": mean_ap04,
            "mean_ap0.8": mean_ap08,
            "delta_mean_vs_nsample32": (mean_ap - base) if (mean_ap is not None and base is not None) else None,
            "mean_grouping_ms": mean_grouping_ms,
        }
        row.update(meta_grouped.get(key, {}))
        summary_rows.append(row)

    summary_fields = [
        "grouping_type", "grouping_nsample",
        "ap_seen", "ap_similar", "ap_novel", "mean_ap", "mean_ap0.4", "mean_ap0.8",
        "delta_mean_vs_nsample32",
        "mean_grouping_ms", "network_name", "ckpt_epoch", "sample_interval", "fuse_type",
    ]
    write_csv(out_prefix + "_summary.csv", summary_rows, summary_fields)

    md_path = out_prefix + "_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| Grouping | nsample | Seen | Similar | Novel | Mean | Mean AP0.4 | Mean AP0.8 | Delta vs 32 | Grouping ms |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                f"| {row['grouping_type']} | {row['grouping_nsample']} | "
                f"{fmt_float(row.get('ap_seen'))} | {fmt_float(row.get('ap_similar'))} | "
                f"{fmt_float(row.get('ap_novel'))} | {fmt_float(row.get('mean_ap'))} | "
                f"{fmt_float(row.get('mean_ap0.4'))} | {fmt_float(row.get('mean_ap0.8'))} | "
                f"{fmt_float(row.get('delta_mean_vs_nsample32'))} | {fmt_float(row.get('mean_grouping_ms'), 3)} |\n"
            )

    print(f"Wrote rows:    {out_prefix}_rows.csv")
    print(f"Wrote summary: {out_prefix}_summary.csv")
    print(f"Wrote markdown:{md_path}")


if __name__ == "__main__":
    main()
