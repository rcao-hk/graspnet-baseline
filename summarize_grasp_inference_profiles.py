#!/usr/bin/env python3
"""Combine the unified baseline inference summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "method" not in data or "metrics_ms" not in data:
        raise ValueError(f"Not a unified inference summary: {path}")
    return data


def sample_key(data: Dict[str, Any]) -> List[Tuple[int, int]]:
    return [
        (int(item["scene_idx"]), int(item["anno_idx"]))
        for item in data.get("sample_order", [])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", help="Summary JSON files")
    parser.add_argument("--output", default="grasp_inference_comparison.csv")
    parser.add_argument(
        "--strict_samples",
        action="store_true",
        help="Fail if the measured scene/annotation order differs",
    )
    args = parser.parse_args()

    loaded = [(path, load(path)) for path in args.summaries]
    reference = sample_key(loaded[0][1])
    for path, data in loaded[1:]:
        current = sample_key(data)
        if current != reference:
            message = (
                f"Sample-order mismatch: {path} differs from {loaded[0][0]} "
                f"({len(current)} vs {len(reference)} measured samples)."
            )
            if args.strict_samples:
                raise RuntimeError(message)
            print(f"[WARN] {message}", file=sys.stderr)

    rows = []
    for path, data in loaded:
        metrics = data["metrics_ms"]
        rows.append(
            {
                "method": data["method"],
                "profiled_samples": data.get("profiled_samples"),
                "warmup_samples": data.get("warmup_samples"),
                "grasp_inference_mean_ms": metrics.get("grasp_inference_ms", {}).get("mean"),
                "collision_mean_ms": metrics.get("collision_ms", {}).get("mean"),
                "online_inference_mean_ms": metrics.get("online_inference_ms", {}).get("mean"),
                "throughput_fps": data.get("throughput_fps"),
                "gpu": data.get("environment", {}).get("gpu"),
                "summary_path": os.path.abspath(path),
            }
        )

    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.output}")
    for row in rows:
        print(
            f"{row['method']:<24} "
            f"grasp={row['grasp_inference_mean_ms']!s:>10} ms  "
            f"online={row['online_inference_mean_ms']!s:>10} ms"
        )


if __name__ == "__main__":
    main()
