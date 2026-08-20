#!/usr/bin/env python3
"""Aggregate MMGNet training/inference cost profiles across fusion variants.

The script scans a profile root recursively and produces one paper-ready row per
fusion type, plus long-form parameter/stage tables and protocol-consistency
checks. It accepts both the new unified schema and the legacy summaries from the
uploaded profilers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from fusion_profile_utils import PROFILE_SCHEMA_VERSION, write_json, write_rows_csv

VARIANT_ORDER = ["none", "direct", "early", "concat", "add", "gate", "intermediate"]
COMMON_FIELDS = [
    "fusion_type",
    "fusion_label",
    "grouping_type",
    "model_class",
    "image_pretrained",
    "image_pretraining_source",
    "image_backbone_frozen",
    "image_feature_dim",
    "point_feature_dim",
    "fused_feature_dim",
    "num_injections",
    "injection_stages",
    "registered_params_m",
    "active_params_m",
    "trainable_params_m",
    "image_params_m",
    "point_backbone_params_m",
    "fusion_projection_params_m",
    "prediction_head_params_m",
    "grouping_params_m",
    "architecture_warnings",
]


def _nested(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _read_first_csv_row(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return next(reader, None)


def _as_number(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return None
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return float(stripped) if any(c in stripped for c in ".eE") else int(stripped)
    except ValueError:
        return value


def _normalize_csv_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(k): _as_number(v) for k, v in row.items()}


def _phase(summary: Mapping[str, Any], path: Path) -> Optional[str]:
    phase = summary.get("phase")
    if phase in {"training", "inference"}:
        return str(phase)
    name = path.name.lower()
    if name == "training_cost_summary.json":
        return "training"
    if "inference" in name and name.endswith("_summary.json"):
        return "inference"
    return None


def _legacy_training_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    config = summary.get("config", {})
    model = summary.get("model", {})
    cost = summary.get("cost", {})
    total_params = model.get("total_params")
    trainable_params = model.get("trainable_params")
    return {
        "phase": "training",
        "run_id": summary.get("run_id"),
        "fusion_type": config.get("fuse_type", "unknown"),
        "fusion_label": config.get("fuse_type", "unknown"),
        "grouping_type": config.get("grouping_type"),
        "camera": config.get("camera"),
        "batch_size": config.get("batch_size"),
        "num_point": config.get("num_point"),
        "m_point": config.get("m_point"),
        "registered_params_m": total_params / 1e6 if isinstance(total_params, (int, float)) else None,
        "active_params_m": total_params / 1e6 if isinstance(total_params, (int, float)) else None,
        "trainable_params_m": trainable_params / 1e6 if isinstance(trainable_params, (int, float)) else None,
        "training_peak_allocated_mib": cost.get("peak_gpu_allocated_mib"),
        "training_peak_reserved_mib": cost.get("peak_gpu_reserved_mib"),
        "projected_full_train_gpu_hours": cost.get("projected_full_train_gpu_hours"),
        "projected_full_train_gpu_days": cost.get("projected_full_train_gpu_days"),
        "train_iteration_mean_ms": (
            1000.0 * cost.get("mean_train_iteration_seconds", 0.0)
            if cost.get("mean_train_iteration_seconds") is not None else None
        ),
        "protocol_fingerprint": summary.get("protocol_fingerprint"),
    }


def _legacy_inference_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    model = summary.get("model", {})
    run = summary.get("run", {})
    architecture = model.get("architecture", {})
    parameters = model.get("parameters", {})
    metrics = summary.get("metrics", {})

    def metric(name: str, stat: str = "mean") -> Any:
        return _nested(metrics, name, stat)

    total_params = model.get("total_params")
    return {
        "phase": "inference",
        "run_id": summary.get("run_id") or run.get("run_id"),
        "fusion_type": architecture.get("fusion_type", run.get("fuse_type", "unknown")),
        "fusion_label": architecture.get("fusion_label", run.get("fuse_type", "unknown")),
        "grouping_type": architecture.get("grouping_type", run.get("grouping_type")),
        "camera": run.get("camera"),
        "registered_params_m": (
            parameters.get("registered_total_params_m")
            if parameters else total_params / 1e6 if isinstance(total_params, (int, float)) else None
        ),
        "active_params_m": parameters.get("active_total_params_m") if parameters else None,
        "model_forward_mean_ms": metric("model_forward_ms"),
        "model_forward_p95_ms": metric("model_forward_ms", "p95"),
        "online_inference_mean_ms": metric("online_inference_ms"),
        "online_inference_p95_ms": metric("online_inference_ms", "p95"),
        "sensor_to_grasp_mean_ms": metric("sensor_to_grasp_ms"),
        "sensor_to_grasp_p95_ms": metric("sensor_to_grasp_ms", "p95"),
        "inference_peak_allocated_mb": metric("gpu_peak_allocated_mb", "max"),
        "inference_peak_reserved_mb": metric("gpu_peak_reserved_mb", "max"),
        "inference_peak_allocated_mib": metric("gpu_peak_allocated_mb", "max"),
        "inference_peak_reserved_mib": metric("gpu_peak_reserved_mb", "max"),
        "protocol_fingerprint": summary.get("protocol_fingerprint"),
    }


def _variant_row(summary: Mapping[str, Any], path: Path, phase: str) -> Dict[str, Any]:
    filename = "training_variant_row.csv" if phase == "training" else "inference_variant_row.csv"
    row = _read_first_csv_row(path.parent / filename)
    if row is not None:
        result = _normalize_csv_row(row)
    elif phase == "training":
        result = _legacy_training_row(summary)
    else:
        result = _legacy_inference_row(summary)
    result["phase"] = phase
    result["summary_path"] = str(path.resolve())
    result["summary_mtime"] = path.stat().st_mtime
    return result


def _discover(root: Path) -> list[tuple[Path, Dict[str, Any], str, Dict[str, Any]]]:
    records = []
    for path in root.rglob("*_summary.json"):
        try:
            summary = _read_json(path)
        except Exception:
            continue
        phase = _phase(summary, path)
        if phase is None:
            continue
        row = _variant_row(summary, path, phase)
        records.append((path, summary, phase, row))
    return records


def _select_latest(records: Iterable[tuple[Path, Dict[str, Any], str, Dict[str, Any]]], run_id: Optional[str]):
    selected: Dict[tuple[str, str], tuple[Path, Dict[str, Any], str, Dict[str, Any]]] = {}
    duplicates: list[Dict[str, Any]] = []
    for record in records:
        path, summary, phase, row = record
        if run_id is not None and row.get("run_id") != run_id:
            continue
        fusion_type = str(row.get("fusion_type", "unknown"))
        key = (fusion_type, phase)
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous[0].stat().st_mtime:
            if previous is not None:
                duplicates.append({
                    "fusion_type": fusion_type,
                    "phase": phase,
                    "kept": str(path),
                    "discarded": str(previous[0]),
                })
            selected[key] = record
        else:
            duplicates.append({
                "fusion_type": fusion_type,
                "phase": phase,
                "kept": str(previous[0]),
                "discarded": str(path),
            })
    return selected, duplicates


def _copy_common(unified: Dict[str, Any], row: Mapping[str, Any], conflicts: list[Dict[str, Any]], phase: str):
    for field in COMMON_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            continue
        if field not in unified or unified[field] in {None, ""}:
            unified[field] = value
        elif str(unified[field]) != str(value):
            conflicts.append({
                "fusion_type": row.get("fusion_type"),
                "field": field,
                "existing": unified[field],
                "incoming": value,
                "incoming_phase": phase,
            })


def _merge_phase(unified: Dict[str, Any], row: Mapping[str, Any], phase: str):
    prefix = "train_" if phase == "training" else "infer_"
    for key, value in row.items():
        if key in COMMON_FIELDS or key in {"phase", "fusion_type", "fusion_label"}:
            continue
        if key == "summary_mtime":
            continue
        unified[prefix + key] = value


def _parameter_rows(summary: Mapping[str, Any], row: Mapping[str, Any], phase: str) -> list[Dict[str, Any]]:
    parameters = summary.get("parameters")
    if not isinstance(parameters, Mapping):
        parameters = _nested(summary, "model", "parameters")
    if not isinstance(parameters, Mapping):
        return []
    result = []
    for group, stats in parameters.get("groups", {}).items():
        if not isinstance(stats, Mapping):
            continue
        result.append({
            "fusion_type": row.get("fusion_type"),
            "fusion_label": row.get("fusion_label"),
            "phase": phase,
            "parameter_group": group,
            **dict(stats),
        })
    return result


def _stage_rows(summary: Mapping[str, Any], row: Mapping[str, Any]) -> list[Dict[str, Any]]:
    if summary.get("phase") not in {None, "inference"}:
        return []
    result = []
    for metric_name, stats in summary.get("metrics", {}).items():
        if not metric_name.startswith("model_stage_") or not metric_name.endswith("_ms"):
            continue
        if not isinstance(stats, Mapping):
            continue
        result.append({
            "fusion_type": row.get("fusion_type"),
            "fusion_label": row.get("fusion_label"),
            "stage": metric_name[len("model_stage_"):-len("_ms")],
            **dict(stats),
        })
    return result


def _load_metrics_csv(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = [_normalize_csv_row(r) for r in csv.DictReader(f)]
    result = {}
    for row in rows:
        fusion_type = row.get("fusion_type")
        if fusion_type is None:
            raise ValueError("--metrics_csv must contain a fusion_type column")
        result[str(fusion_type)] = row
    return result


def _protocol_checks(
    unified_rows: list[Dict[str, Any]],
    selected: Mapping[tuple[str, str], tuple[Path, Dict[str, Any], str, Dict[str, Any]]],
    duplicates: list[Dict[str, Any]],
    conflicts: list[Dict[str, Any]],
    expected_variants: Optional[list[str]] = None,
    required_phases: Optional[list[str]] = None,
) -> Dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []

    phase_hashes: Dict[str, Dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for (fusion_type, phase), (_, summary, _, row) in selected.items():
        fingerprint = summary.get("protocol_fingerprint") or row.get("protocol_fingerprint")
        if fingerprint:
            phase_hashes[phase][str(fingerprint)].append(fusion_type)
    for phase, groups in phase_hashes.items():
        if len(groups) > 1:
            errors.append(
                f"{phase} controlled-protocol fingerprints differ across variants: {dict(groups)}"
            )

    if expected_variants:
        present = {str(r.get("fusion_type")) for r in unified_rows}
        missing = [variant for variant in expected_variants if variant not in present]
        if missing:
            errors.append(f"Expected fusion variants are missing: {missing}")
        for variant in expected_variants:
            missing_phases = [
                phase for phase in (required_phases or [])
                if (variant, phase) not in selected
            ]
            if missing_phases:
                errors.append(
                    f"{variant}: required profile phase(s) are missing: {missing_phases}."
                )

    # Hardware/software must be held fixed for measured latency and memory.
    for phase in ("training", "inference"):
        phase_rows = [r for r in unified_rows if r.get(f"{('train' if phase == 'training' else 'infer')}_summary_path")]
        prefix = "train_" if phase == "training" else "infer_"
        for field, label in (
            ("gpu_name", "GPU model"),
            ("pytorch_version", "PyTorch version"),
            ("cuda_runtime", "CUDA runtime"),
            ("precision", "precision"),
        ):
            values = {
                str(r.get(prefix + field))
                for r in phase_rows
                if r.get(prefix + field) not in {None, ""}
            }
            if len(values) > 1:
                errors.append(f"{phase} {label} differs across variants: {sorted(values)}")

    rgb_rows = [r for r in unified_rows if r.get("fusion_type") not in {"none", "direct"}]
    pretraining_pairs = {
        (str(r.get("image_pretrained")), str(r.get("image_pretraining_source")))
        for r in rgb_rows
        if r.get("image_pretrained") is not None
    }
    if len(pretraining_pairs) > 1:
        errors.append(f"RGB variants do not share image pretraining: {sorted(pretraining_pairs)}")

    frozen_states = {
        str(r.get("image_backbone_frozen"))
        for r in rgb_rows
        if r.get("image_backbone_frozen") is not None
    }
    if len(frozen_states) > 1:
        errors.append(f"RGB variants do not share image-backbone freeze state: {sorted(frozen_states)}")

    image_dims = {
        r.get("image_feature_dim") for r in rgb_rows
        if r.get("image_feature_dim") is not None
    }
    if len(image_dims) > 1:
        errors.append(f"RGB variants use different image feature channels: {sorted(image_dims)}")

    point_dims = {r.get("point_feature_dim") for r in unified_rows if r.get("point_feature_dim") is not None}
    if len(point_dims) > 1:
        errors.append(f"Point-backbone output channels differ: {sorted(point_dims)}")

    for row in unified_rows:
        point_dim = row.get("point_feature_dim")
        fused_dim = row.get("fused_feature_dim")
        if point_dim is not None and fused_dim is not None:
            row["channel_matched_downstream"] = float(point_dim) == float(fused_dim)
            if not row["channel_matched_downstream"]:
                warnings.append(
                    f"{row.get('fusion_type')}: downstream fused channels ({fused_dim}) "
                    f"differ from the point-backbone width ({point_dim})."
                )

    train_profile_counts = {
        r.get("train_profiled_iterations")
        for r in unified_rows
        if r.get("train_profiled_iterations") is not None
    }
    if len(train_profile_counts) > 1:
        warnings.append(
            f"Training variants use different numbers of steady-state iterations: {sorted(train_profile_counts)}"
        )

    inference_counts = {
        r.get("infer_profiled_samples")
        for r in unified_rows
        if r.get("infer_profiled_samples") is not None
    }
    if len(inference_counts) > 1:
        warnings.append(
            f"Inference variants use different numbers of profiled samples: {sorted(inference_counts)}"
        )

    sample_sets = {
        str(r.get("infer_sample_set_fingerprint"))
        for r in unified_rows
        if r.get("infer_sample_set_fingerprint") not in {None, ""}
    }
    sample_orders = {
        str(r.get("infer_sample_order_fingerprint"))
        for r in unified_rows
        if r.get("infer_sample_order_fingerprint") not in {None, ""}
    }
    if len(sample_sets) > 1:
        errors.append(f"Inference variants were evaluated on different sample sets: {sorted(sample_sets)}")
    if len(sample_orders) > 1:
        warnings.append(
            "Inference variants use a different profiled-sample order; "
            "order-dependent cache/warmup effects may reduce comparability."
        )
    weak_identity = [
        str(r.get("fusion_type")) for r in unified_rows
        if r.get("infer_profiled_samples") is not None
        and r.get("infer_sample_identity_is_strong") in {False, 0, "False", "false"}
    ]
    if weak_identity:
        warnings.append(
            f"Stable scene/frame identities were not available for inference variants: {weak_identity}."
        )
    intrusive = [
        str(r.get("fusion_type")) for r in unified_rows
        if r.get("infer_stage_breakdown_intrusive") in {True, 1, "True", "true"}
    ]
    if intrusive:
        warnings.append(
            "Internal stage timing was enabled for variants " + str(intrusive) +
            "; use timer-disabled runs for the primary latency comparison."
        )

    if duplicates:
        warnings.append(
            "Multiple summaries were found for at least one phase/variant; the most recently modified file was used."
        )
    if conflicts:
        errors.append(
            "Training/inference summaries disagree on common architecture fields; inspect common_field_conflicts."
        )

    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "status": "fail" if errors else "pass_with_warnings" if warnings else "pass",
        "errors": errors,
        "warnings": warnings,
        "phase_protocol_fingerprints": {
            phase: dict(groups) for phase, groups in phase_hashes.items()
        },
        "duplicates": duplicates,
        "common_field_conflicts": conflicts,
    }


def _write_markdown(path: Path, rows: list[Dict[str, Any]]) -> None:
    columns = [
        ("Fusion", "fusion_label"),
        ("Image init.", "image_pretraining_source"),
        ("Injections", "num_injections"),
        ("Params (M)", "active_params_m"),
        ("Fusion params (M)", "fusion_projection_params_m"),
        ("GFLOPs/sample", "infer_forward_gflops_per_sample_2xmac"),
        ("Inference (ms)", "infer_model_forward_mean_ms"),
        ("Infer VRAM (MiB)", "infer_inference_peak_allocated_mib"),
        ("Train iter. (ms)", "train_train_iteration_mean_ms"),
        ("Train VRAM (MiB)", "train_training_peak_allocated_mib"),
        ("Train GPU-days", "reported_train_gpu_days"),
        ("Cost source", "reported_train_cost_source"),
    ]

    def fmt(value: Any) -> str:
        if value is None or value == "":
            return "–"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = [
        "# Fusion Variant Cost Summary",
        "",
        "| " + " | ".join(title for title, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root containing variant profile directories")
    parser.add_argument("--output_dir", default=None, help="Output directory; default: <root>/aggregate")
    parser.add_argument("--run_id", default=None, help="Optional run_id filter")
    parser.add_argument("--metrics_csv", default=None, help="Optional AP/results CSV keyed by fusion_type")
    parser.add_argument(
        "--expected_variants", default=None,
        help="Comma-separated expected variants, e.g. none,early,concat,add,gate,intermediate",
    )
    parser.add_argument(
        "--required_phases", default=None,
        help="Comma-separated required phases: training,inference",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when protocol checks fail")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _discover(root)
    selected, duplicates = _select_latest(records, args.run_id)
    if not selected:
        raise SystemExit(f"No training/inference profile summaries found under {root}")

    by_variant: Dict[str, Dict[str, Any]] = {}
    conflicts: list[Dict[str, Any]] = []
    parameter_rows: list[Dict[str, Any]] = []
    stage_rows: list[Dict[str, Any]] = []

    for (fusion_type, phase), (_, summary, _, row) in selected.items():
        unified = by_variant.setdefault(
            fusion_type,
            {"fusion_type": fusion_type, "fusion_label": row.get("fusion_label", fusion_type)},
        )
        _copy_common(unified, row, conflicts, phase)
        _merge_phase(unified, row, phase)
        parameter_rows.extend(_parameter_rows(summary, row, phase))
        if phase == "inference":
            stage_rows.extend(_stage_rows(summary, row))

    rows = list(by_variant.values())
    order = {name: i for i, name in enumerate(VARIANT_ORDER)}
    rows.sort(key=lambda r: (order.get(str(r.get("fusion_type")), 999), str(r.get("fusion_type"))))

    for row in rows:
        if row.get("infer_inference_peak_allocated_mib") is None:
            row["infer_inference_peak_allocated_mib"] = row.get(
                "infer_inference_peak_allocated_mb"
            )
        if row.get("infer_inference_peak_reserved_mib") is None:
            row["infer_inference_peak_reserved_mib"] = row.get(
                "infer_inference_peak_reserved_mb"
            )
        actual_days = row.get("train_actual_completed_training_gpu_days")
        actual_hours = row.get("train_actual_completed_training_gpu_hours")
        if isinstance(actual_days, (int, float)):
            row["reported_train_gpu_days"] = actual_days
            row["reported_train_gpu_hours"] = actual_hours
            row["reported_train_cost_source"] = "actual_completed_run"
        else:
            row["reported_train_gpu_days"] = row.get("train_projected_full_train_gpu_days")
            row["reported_train_gpu_hours"] = row.get("train_projected_full_train_gpu_hours")
            row["reported_train_cost_source"] = row.get(
                "train_training_cost_source",
                "projected_from_post_warmup_optimizer_steps",
            )

    point_only = next((r for r in rows if r.get("fusion_type") == "none"), None)
    if point_only is not None:
        for row in rows:
            for field in (
                "active_params_m",
                "infer_model_forward_mean_ms",
                "infer_inference_peak_allocated_mib",
                "train_training_peak_allocated_mib",
                "reported_train_gpu_days",
            ):
                value = row.get(field)
                baseline = point_only.get(field)
                if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                    row[f"delta_{field}_vs_point_only"] = float(value) - float(baseline)

    extra_metrics = _load_metrics_csv(
        Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else None
    )
    for row in rows:
        metrics = extra_metrics.get(str(row.get("fusion_type")))
        if metrics:
            for key, value in metrics.items():
                if key != "fusion_type":
                    row[key] = value

    expected_variants = None
    if args.expected_variants:
        expected_variants = [
            value.strip() for value in args.expected_variants.split(",") if value.strip()
        ]
    required_phases = None
    if args.required_phases:
        required_phases = [
            value.strip() for value in args.required_phases.split(",") if value.strip()
        ]
        invalid = [value for value in required_phases if value not in {"training", "inference"}]
        if invalid:
            raise SystemExit(f"Invalid --required_phases values: {invalid}")
    checks = _protocol_checks(
        rows,
        selected,
        duplicates,
        conflicts,
        expected_variants=expected_variants,
        required_phases=required_phases,
    )

    unified_path = write_rows_csv(output_dir / "fusion_variant_unified.csv", rows)
    params_path = write_rows_csv(output_dir / "fusion_variant_parameter_groups.csv", parameter_rows)
    stages_path = write_rows_csv(output_dir / "fusion_variant_inference_stages.csv", stage_rows)
    checks_path = write_json(output_dir / "fusion_variant_protocol_check.json", checks)
    _write_markdown(output_dir / "fusion_variant_summary.md", rows)

    print(f"[FUSION-PROFILE] variants={len(rows)}")
    print(f"[FUSION-PROFILE] unified: {unified_path}")
    print(f"[FUSION-PROFILE] parameters: {params_path}")
    print(f"[FUSION-PROFILE] inference stages: {stages_path}")
    print(f"[FUSION-PROFILE] protocol check: {checks_path} ({checks['status']})")
    for message in checks["errors"]:
        print(f"[FUSION-PROFILE][ERROR] {message}")
    for message in checks["warnings"]:
        print(f"[FUSION-PROFILE][WARNING] {message}")

    if args.strict and checks["errors"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
