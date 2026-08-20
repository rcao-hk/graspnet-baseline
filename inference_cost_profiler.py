"""Reusable inference deployment-cost profiler for PyTorch models.

The profiler is intentionally lightweight and measures steady-state, batch-level
latency with explicit CUDA synchronization. It also records GPU allocator peaks,
process RSS samples, throughput, model footprint, and per-sample CSV rows.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, TypeVar

import numpy as np
import torch

from fusion_profile_utils import (
    PROFILE_SCHEMA_VERSION,
    RuntimeMACProfiler,
    clone_tensor_tree,
    collect_architecture_metadata,
    collect_environment,
    collect_parameter_profile,
    common_variant_identity,
    parameter_group_rows,
    protocol_fingerprint,
    save_complexity_profile,
    write_json,
    write_rows_csv,
)

T = TypeVar("T")
_MIB = 1024.0 ** 2


def add_inference_profile_args(parser: Any) -> Any:
    """Register the standard MMGNet inference-cost CLI arguments.

    The dataset-specific inference driver remains responsible for constructing
    :class:`InferenceCostProfiler`, timing preprocessing/forward/decode/collision,
    adding one row per sample, and exporting the summary. Returning ``parser``
    keeps the helper compatible with both ``argparse.ArgumentParser`` and an
    argument group.
    """
    group = parser.add_argument_group("inference cost profiling")
    group.add_argument(
        "--profile_cost",
        action="store_true",
        help="Enable synchronized inference latency/memory profiling.",
    )
    group.add_argument(
        "--profile_output_dir",
        default=None,
        help="Directory for inference_deployment_profile_* outputs.",
    )
    group.add_argument(
        "--profile_run_id",
        default=None,
        help="Shared identifier used to merge training and inference profiles.",
    )
    group.add_argument(
        "--profile_warmup",
        type=int,
        default=20,
        help="Number of complete inference samples excluded as warm-up.",
    )
    group.add_argument(
        "--profile_samples",
        type=int,
        default=100,
        help="Number of post-warm-up samples to retain; 0 profiles all samples.",
    )
    group.add_argument(
        "--profile_print_every",
        type=int,
        default=20,
        help="Print a progress line every N profiled samples.",
    )
    group.add_argument(
        "--profile_complexity",
        action="store_true",
        help="Run one separate input-dependent forward MAC/FLOP pass.",
    )
    group.add_argument(
        "--profile_stage_breakdown",
        action="store_true",
        help=(
            "Enable intrusive synchronized IGNet stage timing. Use only in a "
            "separate diagnostic run, never for the primary latency table."
        ),
    )
    return parser


def cuda_synchronize(device: torch.device) -> None:
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(
    fn: Callable[[], T],
    *,
    device: torch.device,
    sync_cuda: bool = True,
) -> Tuple[T, float]:
    """Run ``fn`` and return (result, synchronized wall-clock milliseconds)."""
    if sync_cuda:
        cuda_synchronize(device)
    start = time.perf_counter()
    result = fn()
    if sync_cuda:
        cuda_synchronize(device)
    return result, (time.perf_counter() - start) * 1000.0


def wall_time_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000.0


def get_process_rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss) / _MIB
    except Exception:
        return None


def get_process_peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB; macOS reports bytes.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return value / _MIB
    return value / 1024.0


def bytes_to_mib(value: int | float) -> float:
    return float(value) / _MIB


def collect_model_metadata(
    model: torch.nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> Dict[str, Any]:
    """Collect the same architecture/parameter schema used by training."""
    base_model = model.module if hasattr(model, "module") else model
    parameter_profile = collect_parameter_profile(base_model)
    architecture = collect_architecture_metadata(base_model)
    parameters = list(base_model.parameters())

    first_dtype = str(parameters[0].dtype).replace("torch.", "") if parameters else "unknown"
    checkpoint_bytes = (
        int(os.path.getsize(checkpoint_path))
        if checkpoint_path and os.path.isfile(checkpoint_path)
        else None
    )

    cuda_meta: Dict[str, Any] = {
        "available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        cuda_meta.update(
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_mb": bytes_to_mib(props.total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
                "cuda_version": torch.version.cuda,
            }
        )

    return {
        "model_class": type(base_model).__name__,
        # Legacy scalar fields retained for callers that already consume them.
        "total_params": parameter_profile["registered_total_params"],
        "active_params": parameter_profile["active_total_params"],
        "trainable_params": parameter_profile["trainable_total_params"],
        "parameter_size_mb": parameter_profile["registered_parameter_size_mib"],
        "buffer_size_mb": parameter_profile["buffer_size_mib"],
        "model_tensor_size_mb": parameter_profile["model_tensor_size_mib"],
        "checkpoint_path": checkpoint_path,
        "checkpoint_size_mb": bytes_to_mib(checkpoint_bytes) if checkpoint_bytes is not None else None,
        "parameter_dtype": first_dtype,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda": cuda_meta,
        "environment": collect_environment(device),
        "architecture": architecture,
        "parameters": parameter_profile,
    }


def _summary(values: Iterable[float]) -> Dict[str, float | int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "total": 0.0,
        }
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }


class InferenceCostProfiler:
    """Collect comparable per-frame deployment metrics for one fusion variant."""

    DEFAULT_METRICS = (
        "preprocess_ms",
        "model_forward_ms",
        "pred_decode_ms",
        "graspgroup_ms",
        "collision_ms",
        "online_inference_ms",
        "sensor_to_grasp_ms",
        "save_ms",
        "end_to_end_with_save_ms",
        "gpu_allocated_before_mb",
        "gpu_reserved_before_mb",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
        "gpu_incremental_peak_allocated_mb",
        "gpu_incremental_peak_reserved_mb",
        "cpu_rss_before_mb",
        "cpu_rss_after_mb",
        "num_grasps_raw",
        "num_grasps_after_collision",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        warmup: int,
        max_profiled_samples: int,
        output_dir: str,
        device: torch.device,
        print_every: int = 20,
        sync_cuda: bool = True,
        model_metadata: Optional[Dict[str, Any]] = None,
        run_metadata: Optional[Dict[str, Any]] = None,
        model: Optional[torch.nn.Module] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.warmup = max(0, int(warmup))
        self.max_profiled_samples = max(0, int(max_profiled_samples))
        self.output_dir = os.path.abspath(output_dir)
        self.device = device
        self.print_every = max(1, int(print_every))
        self.sync_cuda = bool(sync_cuda)
        self.model = model
        self.model_metadata = model_metadata or (
            collect_model_metadata(model, None, device) if model is not None else {}
        )
        self.run_metadata = run_metadata or {}
        self.run_id = run_id or self.run_metadata.get("run_id")
        architecture = self.model_metadata.get("architecture")
        if isinstance(architecture, dict):
            sampling = architecture.setdefault("sampling", {})
            if sampling.get("scene_points") is None and self.run_metadata.get("num_point") is not None:
                sampling["scene_points"] = int(self.run_metadata["num_point"])
        self.complexity: Optional[Dict[str, Any]] = None
        self.complexity_profile_error: Optional[str] = None

        self.rows: list[Dict[str, Any]] = []
        self.num_seen = 0
        self.num_profiled = 0
        self.static_gpu_allocated_mb: Optional[float] = None
        self.static_gpu_reserved_mb: Optional[float] = None
        self.rss_at_start_mb = get_process_rss_mb()

        if self.enabled:
            os.makedirs(self.output_dir, exist_ok=True)
            self.capture_static_gpu_memory()

    def capture_static_gpu_memory(self) -> None:
        if not self.enabled or self.device.type != "cuda" or not torch.cuda.is_available():
            return
        cuda_synchronize(self.device)
        self.static_gpu_allocated_mb = bytes_to_mib(torch.cuda.memory_allocated(self.device))
        self.static_gpu_reserved_mb = bytes_to_mib(torch.cuda.memory_reserved(self.device))

    def start_online_memory_tracking(self) -> Dict[str, Optional[float]]:
        state: Dict[str, Optional[float]] = {
            "gpu_allocated_before_mb": None,
            "gpu_reserved_before_mb": None,
            "cpu_rss_before_mb": get_process_rss_mb(),
        }
        if not self.enabled or self.device.type != "cuda" or not torch.cuda.is_available():
            return state

        cuda_synchronize(self.device)
        state["gpu_allocated_before_mb"] = bytes_to_mib(
            torch.cuda.memory_allocated(self.device)
        )
        state["gpu_reserved_before_mb"] = bytes_to_mib(
            torch.cuda.memory_reserved(self.device)
        )
        torch.cuda.reset_peak_memory_stats(self.device)
        return state

    def finish_online_memory_tracking(
        self,
        state: Dict[str, Optional[float]],
    ) -> Dict[str, Optional[float]]:
        result = dict(state)
        result.update(
            {
                "gpu_peak_allocated_mb": None,
                "gpu_peak_reserved_mb": None,
                "gpu_incremental_peak_allocated_mb": None,
                "gpu_incremental_peak_reserved_mb": None,
                "cpu_rss_after_mb": get_process_rss_mb(),
            }
        )
        if not self.enabled or self.device.type != "cuda" or not torch.cuda.is_available():
            return result

        cuda_synchronize(self.device)
        peak_alloc = bytes_to_mib(torch.cuda.max_memory_allocated(self.device))
        peak_reserved = bytes_to_mib(torch.cuda.max_memory_reserved(self.device))
        before_alloc = state.get("gpu_allocated_before_mb") or 0.0
        before_reserved = state.get("gpu_reserved_before_mb") or 0.0
        result.update(
            {
                "gpu_peak_allocated_mb": peak_alloc,
                "gpu_peak_reserved_mb": peak_reserved,
                "gpu_incremental_peak_allocated_mb": max(0.0, peak_alloc - before_alloc),
                "gpu_incremental_peak_reserved_mb": max(0.0, peak_reserved - before_reserved),
            }
        )
        return result

    @staticmethod
    def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model

    def _attach_model_stage_metrics(
        self,
        row: Dict[str, Any],
        model: Optional[torch.nn.Module],
    ) -> None:
        if model is None:
            return
        base = self._unwrap(model)
        getter = getattr(base, "get_last_inference_timer", None)
        if not callable(getter):
            return
        last = getter()
        if not isinstance(last, dict):
            return
        row["model_timer_internal_warmup"] = bool(last.get("is_warmup", False))
        row["model_timer_batch_size"] = int(last.get("batch_size", 0))
        for stage, value in last.get("stages_ms", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"model_stage_{stage}_ms"] = float(value)

    def add_row(
        self,
        row: Dict[str, Any],
        model: Optional[torch.nn.Module] = None,
    ) -> bool:
        """Add one frame and optionally attach IGNet's per-stage model timer."""
        if not self.enabled:
            return False

        self.num_seen += 1
        is_warmup = self.num_seen <= self.warmup
        row = dict(row)
        self._attach_model_stage_metrics(row, model or self.model)
        row["sample_index"] = self.num_seen - 1
        row["is_warmup"] = bool(is_warmup)
        self.rows.append(row)

        if not is_warmup:
            self.num_profiled += 1
            if self.num_profiled == 1 or self.num_profiled % self.print_every == 0:
                print(
                    "[DEPLOY-PROFILE] "
                    f"profiled={self.num_profiled} "
                    f"scene={int(row.get('scene_idx', -1)):04d} "
                    f"anno={int(row.get('anno_idx', -1)):04d} "
                    f"model={float(row.get('model_forward_ms', 0.0)):.3f} ms "
                    f"online={float(row.get('online_inference_ms', 0.0)):.3f} ms "
                    f"sensor-to-grasp={float(row.get('sensor_to_grasp_ms', 0.0)):.3f} ms"
                )

        return self.should_stop()

    def should_stop(self) -> bool:
        return self.max_profiled_samples > 0 and self.num_profiled >= self.max_profiled_samples

    def _profiled_rows(self) -> list[Dict[str, Any]]:
        return [r for r in self.rows if not bool(r.get("is_warmup", False))]

    def profile_forward_complexity(
        self,
        model: torch.nn.Module,
        forward_fn: Callable[[], T],
        *,
        label: str = "inference_forward_after_warmup",
    ) -> T:
        """Run one forward with input-dependent MAC hooks and return its result.

        Invoke this on a dedicated warm-up batch or a cloned ``end_points`` tree.
        IGNet mutates ``end_points`` in place, so the diagnostic and timed forwards
        must not reuse the same container. Complexity-hook overhead is kept out of
        the steady-state latency and memory distributions.
        """
        base_model = self._unwrap(model)
        timer_enabled = getattr(base_model, "_inference_timer_enabled", None)
        try:
            # Complexity hooks are diagnostic-only. Disable the intrusive internal
            # stage timer for this pass so it neither consumes timer warmup calls nor
            # adds repeated CUDA synchronizations.
            if timer_enabled is not None:
                base_model._inference_timer_enabled = False
            profiler = RuntimeMACProfiler(base_model, label=label)
            with profiler:
                result = forward_fn()
            cuda_synchronize(self.device)
            self.complexity = profiler.summary()
            batch_size = 0
            getter = getattr(self._unwrap(model), "get_last_inference_timer", None)
            if callable(getter):
                try:
                    batch_size = int(getter().get("batch_size", 0))
                except Exception:
                    batch_size = 0
            if batch_size <= 0:
                batch_size = int(self.run_metadata.get("batch_size", 1) or 1)
            self.complexity["batch_size"] = batch_size
            self.complexity["gmacs_per_sample"] = self.complexity["total_gmacs"] / batch_size
            self.complexity["gflops_per_sample"] = self.complexity["total_gflops"] / batch_size
            return result
        except Exception as exc:
            self.complexity_profile_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if timer_enabled is not None:
                base_model._inference_timer_enabled = bool(timer_enabled)

    def _metric_keys(self, rows: list[Dict[str, Any]]) -> list[str]:
        keys = list(self.DEFAULT_METRICS)
        for row in rows:
            for key, value in row.items():
                if key in keys or isinstance(value, bool):
                    continue
                if not isinstance(value, (int, float)):
                    continue
                if key.startswith("model_stage_") or key.endswith(("_ms", "_mb", "_mib")):
                    keys.append(key)
        return keys

    @staticmethod
    def _sample_identity(row: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Return a stable sample identity when the inference driver exposes one."""
        candidates = (
            ("sample_id",),
            ("scene_idx", "anno_idx", "camera"),
            ("scene_idx", "anno_idx"),
            ("scene", "frameid", "camera"),
            ("scene", "frameid"),
            ("input_path",),
        )
        for fields in candidates:
            if all(row.get(field) is not None for field in fields):
                payload = {field: row.get(field) for field in fields}
                return "+".join(fields), payload
        # This fallback checks only count/order, not dataset identity. It is marked
        # explicitly so the aggregate report does not overstate sample matching.
        return "sample_index_fallback", {"sample_index": row.get("sample_index")}

    def _sample_fingerprints(self, rows: list[Dict[str, Any]]) -> Dict[str, Any]:
        identities = [self._sample_identity(row) for row in rows]
        fields = sorted({name for name, _ in identities})
        ordered = [payload for _, payload in identities]
        ordered_payload = {"samples": ordered}
        set_payload = {
            "samples": sorted(
                (json.dumps(v, sort_keys=True, ensure_ascii=False, default=str) for v in ordered)
            )
        }
        return {
            "identity_fields": fields,
            "identity_is_strong": bool(fields and fields != ["sample_index_fallback"]),
            "order_fingerprint": protocol_fingerprint(ordered_payload),
            "set_fingerprint": protocol_fingerprint(set_payload),
            "first_samples": ordered[:5],
            "last_samples": ordered[-5:] if ordered else [],
        }

    def _automatic_controlled_protocol(self) -> Dict[str, Any]:
        """Build a variant-independent protocol from common driver metadata."""
        run = self.run_metadata

        def selected(keys: tuple[str, ...]) -> Dict[str, Any]:
            return {key: run.get(key) for key in keys if run.get(key) is not None}

        return {
            "dataset": selected((
                "dataset", "dataset_name", "dataset_root", "camera", "split",
                "scene_start", "scene_end", "annotation_start", "annotation_end",
            )),
            "input": selected((
                "batch_size", "num_point", "m_point", "voxel_size",
                "image_height", "image_width", "image_size",
            )),
            "postprocessing": selected((
                "topk", "nms", "nms_translation_thresh", "nms_rotation_thresh",
                "collision_detection", "collision_thresh", "voxel_size_cd",
            )),
            "measurement": {
                "warmup_samples": self.warmup,
                "max_profiled_samples": self.max_profiled_samples,
                "sync_cuda": self.sync_cuda,
                "batch_size": run.get("batch_size", 1),
                "precision": run.get("precision", "fp32"),
                "amp_enabled": bool(run.get("amp_enabled", False)),
                "save_time_reported_separately": True,
            },
        }

    def build_summary(self) -> Dict[str, Any]:
        rows = self._profiled_rows()
        metrics: Dict[str, Any] = {}
        for key in self._metric_keys(rows):
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
            metrics[key] = _summary(values)

        online_total_ms = metrics["online_inference_ms"]["total"]
        sensor_total_ms = metrics["sensor_to_grasp_ms"]["total"]
        online_n = int(metrics["online_inference_ms"]["count"])
        sensor_n = int(metrics["sensor_to_grasp_ms"]["count"])

        throughput = {
            "online_samples_per_s": (
                1000.0 * online_n / online_total_ms if online_total_ms > 0 else 0.0
            ),
            "sensor_to_grasp_samples_per_s": (
                1000.0 * sensor_n / sensor_total_ms if sensor_total_ms > 0 else 0.0
            ),
        }
        architecture = self.model_metadata.get("architecture", {})
        controlled_protocol = self.run_metadata.get(
            "controlled_protocol", self.run_metadata.get("protocol")
        )
        if not isinstance(controlled_protocol, dict):
            controlled_protocol = self._automatic_controlled_protocol()
        fingerprint = self.run_metadata.get("protocol_fingerprint")
        if fingerprint is None:
            fingerprint = protocol_fingerprint(controlled_protocol)
        sample_fingerprints = self._sample_fingerprints(rows)
        has_intrusive_stage_timing = any(
            key.startswith("model_stage_") and stats.get("count", 0) > 0
            for key, stats in metrics.items()
        )
        measurement_warnings = []
        if has_intrusive_stage_timing:
            measurement_warnings.append(
                "Internal per-stage timing synchronizes CUDA at stage boundaries and "
                "can inflate model_forward_ms. Use a separate timer-disabled pass for "
                "the primary end-to-end latency table; treat stage timing as diagnostic."
            )
        if not sample_fingerprints["identity_is_strong"]:
            measurement_warnings.append(
                "The driver did not expose a stable sample identity; cross-variant "
                "sample matching can verify only sample count/order indices."
            )

        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "phase": "inference",
            "run_id": self.run_id,
            "variant": common_variant_identity(
                architecture, phase="inference", run_id=self.run_id
            ),
            "num_seen": self.num_seen,
            "num_warmup": min(self.num_seen, self.warmup),
            "num_profiled": self.num_profiled,
            "sync_cuda": self.sync_cuda,
            "static_gpu_allocated_mb": self.static_gpu_allocated_mb,
            "static_gpu_reserved_mb": self.static_gpu_reserved_mb,
            "rss_at_start_mb": self.rss_at_start_mb,
            "process_peak_rss_mb": get_process_peak_rss_mb(),
            "throughput": throughput,
            "metrics": metrics,
            "environment": collect_environment(self.device),
            "controlled_protocol": controlled_protocol,
            "protocol_fingerprint": fingerprint,
            "sample_fingerprints": sample_fingerprints,
            "stage_breakdown_intrusive": has_intrusive_stage_timing,
            "measurement_warnings": measurement_warnings,
            "architecture": architecture,
            "parameters": self.model_metadata.get("parameters"),
            "complexity": self.complexity,
            "complexity_profile_error": self.complexity_profile_error,
            # Legacy nested fields retained.
            "model": self.model_metadata,
            "run": self.run_metadata,
        }

    @staticmethod
    def _nested(payload: Dict[str, Any], *keys: str, default: Any = None) -> Any:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def _variant_row(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        architecture = summary.get("architecture") or {}
        parameters = summary.get("parameters") or {}
        groups = parameters.get("groups", {})
        metrics = summary["metrics"]
        complexity = summary.get("complexity") or {}
        run = summary.get("run") or {}

        def metric(name: str, stat: str = "mean") -> Optional[float]:
            return metrics.get(name, {}).get(stat)

        def group_params(name: str) -> Optional[float]:
            return groups.get(name, {}).get("active_params_m")

        row = common_variant_identity(
            architecture, phase="inference", run_id=self.run_id
        )
        row.update(
            {
                "protocol_fingerprint": summary.get("protocol_fingerprint"),
                "sample_order_fingerprint": self._nested(
                    summary, "sample_fingerprints", "order_fingerprint"
                ),
                "sample_set_fingerprint": self._nested(
                    summary, "sample_fingerprints", "set_fingerprint"
                ),
                "sample_identity_is_strong": self._nested(
                    summary, "sample_fingerprints", "identity_is_strong", default=False
                ),
                "stage_breakdown_intrusive": summary.get("stage_breakdown_intrusive", False),
                "camera": run.get("camera"),
                "checkpoint_path": self.model_metadata.get("checkpoint_path"),
                "checkpoint_size_mb": self.model_metadata.get("checkpoint_size_mb"),
                "batch_size": run.get("batch_size", 1),
                "gpu_name": self._nested(
                    summary, "environment", "gpu", "name"
                ),
                "pytorch_version": self._nested(summary, "environment", "pytorch"),
                "cuda_runtime": self._nested(summary, "environment", "cuda_runtime"),
                "precision": run.get("precision", "fp32"),
                "num_point": run.get("num_point"),
                "m_point": run.get("m_point"),
                "image_pretrained": self._nested(
                    architecture, "image_backbone", "encoder_pretrained", default=False
                ),
                "image_pretraining_source": self._nested(
                    architecture, "image_backbone", "encoder_pretraining_source"
                ),
                "image_backbone_frozen": self._nested(
                    architecture, "image_backbone", "frozen", default=False
                ),
                "image_feature_dim": self._nested(
                    architecture, "feature_channels", "image_feature_dim", default=0
                ),
                "point_feature_dim": self._nested(
                    architecture, "feature_channels", "point_backbone_output_dim"
                ),
                "fused_feature_dim": self._nested(
                    architecture, "feature_channels", "fused_feature_dim"
                ),
                "num_injections": self._nested(
                    architecture, "fusion", "num_injections", default=0
                ),
                "injection_stages": json.dumps(
                    self._nested(
                        architecture, "fusion", "injection_stages", default=[]
                    )
                ),
                "registered_params_m": parameters.get("registered_total_params_m"),
                "active_params_m": parameters.get("active_total_params_m"),
                "trainable_params_m": parameters.get("trainable_total_params_m"),
                "image_params_m": group_params("image_backbone"),
                "point_backbone_params_m": group_params("point_backbone"),
                "fusion_projection_params_m": group_params("fusion_projection"),
                "prediction_head_params_m": group_params("prediction_heads"),
                "grouping_params_m": group_params("local_grouping"),
                "profiled_samples": summary["num_profiled"],
                "warmup_samples": summary["num_warmup"],
                "model_forward_mean_ms": metric("model_forward_ms"),
                "model_forward_p95_ms": metric("model_forward_ms", "p95"),
                "online_inference_mean_ms": metric("online_inference_ms"),
                "online_inference_p95_ms": metric("online_inference_ms", "p95"),
                "sensor_to_grasp_mean_ms": metric("sensor_to_grasp_ms"),
                "sensor_to_grasp_p95_ms": metric("sensor_to_grasp_ms", "p95"),
                "image_backbone_mean_ms": metric("model_stage_image_backbone_ms"),
                "image_to_point_mean_ms": metric("model_stage_image_to_point_ms"),
                "sparse_preprocess_mean_ms": metric("model_stage_sparse_preprocess_ms"),
                "point_backbone_mean_ms": metric("model_stage_point_backbone_ms"),
                "feature_fusion_mean_ms": metric("model_stage_feature_fusion_ms"),
                "local_grouping_mean_ms": metric("model_stage_local_grouping_ms"),
                "model_stage_total_mean_ms": metric("model_stage_model_total_ms"),
                # The legacy metric names use ``mb`` but the conversion is
                # binary MiB. Keep aliases for compatibility and expose explicit
                # ``mib`` fields for paper tables.
                "inference_peak_allocated_mb": metric("gpu_peak_allocated_mb", "max"),
                "inference_peak_reserved_mb": metric("gpu_peak_reserved_mb", "max"),
                "inference_incremental_peak_mb": metric(
                    "gpu_incremental_peak_allocated_mb", "max"
                ),
                "inference_peak_allocated_mib": metric("gpu_peak_allocated_mb", "max"),
                "inference_peak_reserved_mib": metric("gpu_peak_reserved_mb", "max"),
                "inference_incremental_peak_mib": metric(
                    "gpu_incremental_peak_allocated_mb", "max"
                ),
                "online_samples_per_s": summary["throughput"]["online_samples_per_s"],
                "forward_gmacs_per_batch": complexity.get("total_gmacs"),
                "forward_gflops_per_batch_2xmac": complexity.get("total_gflops"),
                "forward_gmacs_per_sample": complexity.get("gmacs_per_sample"),
                "forward_gflops_per_sample_2xmac": complexity.get("gflops_per_sample"),
                "sparse_count_method": complexity.get("sparse_count_method"),
                "complexity_scope": complexity.get("scope"),
                "architecture_warnings": json.dumps(
                    architecture.get("warnings", [])
                ),
            }
        )
        return row

    def export(
        self,
        *,
        basename: str = "inference_deployment_profile",
        extra_summary: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, Dict[str, Any]]:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = write_rows_csv(
            os.path.join(self.output_dir, f"{basename}_rows.csv"), self.rows
        )

        summary = self.build_summary()
        if extra_summary:
            summary.update(extra_summary)
        json_path = write_json(
            os.path.join(self.output_dir, f"{basename}_summary.json"), summary
        )

        identity = common_variant_identity(
            summary.get("architecture") or {},
            phase="inference",
            run_id=self.run_id,
        )
        parameters = summary.get("parameters")
        if isinstance(parameters, dict):
            write_rows_csv(
                os.path.join(self.output_dir, "inference_parameter_groups.csv"),
                parameter_group_rows(parameters, common=identity),
            )
        if self.complexity is not None:
            save_complexity_profile(
                self.output_dir,
                self.complexity,
                basename="inference_forward_complexity",
                common=identity,
            )
        write_rows_csv(
            os.path.join(self.output_dir, "inference_variant_row.csv"),
            [self._variant_row(summary)],
        )

        stage_rows = []
        for metric_name, stats in summary["metrics"].items():
            if not metric_name.startswith("model_stage_"):
                continue
            stage_rows.append(
                {
                    **identity,
                    "stage": metric_name[len("model_stage_"):-len("_ms")],
                    **stats,
                }
            )
        if stage_rows:
            write_rows_csv(
                os.path.join(self.output_dir, "inference_model_stage_summary.csv"),
                stage_rows,
            )

        return csv_path, json_path, summary

    @staticmethod
    def format_summary(summary: Dict[str, Any]) -> str:
        m = summary["metrics"]
        throughput = summary["throughput"]

        def line(label: str, key: str, unit: str = "ms") -> str:
            cur = m[key]
            return (
                f"{label:<28} mean={cur['mean']:>9.3f} {unit}  "
                f"median={cur['median']:>9.3f}  p95={cur['p95']:>9.3f}  "
                f"p99={cur['p99']:>9.3f}  std={cur['std']:>9.3f}"
            )

        lines = [
            "Inference deployment profile",
            f"samples={summary['num_profiled']} | warmup={summary['num_warmup']} | sync_cuda={summary['sync_cuda']}",
            "-" * 104,
            line("Preprocess", "preprocess_ms"),
            line("Model forward", "model_forward_ms"),
            line("Prediction decode", "pred_decode_ms"),
            line("GraspGroup construction", "graspgroup_ms"),
            line("Collision filtering", "collision_ms"),
            line("Online inference", "online_inference_ms"),
            line("Sensor-to-grasp", "sensor_to_grasp_ms"),
            line("Save output", "save_ms"),
            "-" * 104,
            f"Online throughput             {throughput['online_samples_per_s']:.3f} samples/s",
            f"Sensor-to-grasp throughput    {throughput['sensor_to_grasp_samples_per_s']:.3f} samples/s",
            f"Static GPU allocated          {summary.get('static_gpu_allocated_mb') or 0.0:.2f} MiB",
            f"Static GPU reserved           {summary.get('static_gpu_reserved_mb') or 0.0:.2f} MiB",
            f"Peak GPU allocated            {m['gpu_peak_allocated_mb']['max']:.2f} MiB",
            f"Peak GPU reserved             {m['gpu_peak_reserved_mb']['max']:.2f} MiB",
            f"Incremental peak allocated    {m['gpu_incremental_peak_allocated_mb']['max']:.2f} MiB",
            f"Process peak RSS              {summary['process_peak_rss_mb']:.2f} MiB",
        ]
        complexity = summary.get("complexity")
        if isinstance(complexity, dict):
            lines.append(
                "Forward complexity           "
                f"{complexity.get('total_gmacs', 0.0):.3f} GMACs / "
                f"{complexity.get('total_gflops', 0.0):.3f} GFLOPs "
                f"({complexity.get('sparse_count_method', 'unknown')})"
            )
        return "\n".join(lines)
