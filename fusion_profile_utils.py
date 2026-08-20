"""Shared profiling utilities for controlled MMGNet fusion ablations.

The utilities in this file intentionally separate four quantities that are often
conflated in ablation tables:

1. registered/trainable parameter capacity;
2. input-dependent multiply-accumulate estimates;
3. synchronized wall-clock latency; and
4. allocator/process memory.

Sparse-convolution complexity is measured from the MinkowskiEngine kernel map
when the installed version exposes it. If the kernel map cannot be queried, the
profiler falls back to an active-output-site upper-bound and records that fact in
its JSON output. This distinction is important: a dense-equivalent FLOP number
must not be presented as an exact executed sparse-convolution FLOP count.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import socket
from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np
import torch
import torch.nn as nn

PROFILE_SCHEMA_VERSION = "mmgnet-fusion-profile-v1"
_MIB = 1024.0 ** 2


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module for DataParallel/DDP wrappers."""
    return model.module if hasattr(model, "module") else model


def tensor_tree_nbytes(obj: Any) -> int:
    """Recursively count tensor storage represented by a nested object."""
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, Mapping):
        return sum(tensor_tree_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_tree_nbytes(v) for v in obj)
    return 0


def clone_tensor_tree(obj: Any) -> Any:
    """Clone nested tensor/array containers before an in-place model forward.

    IGNet appends predictions to ``end_points`` and replaces some entries (for
    example ``point_clouds``). A diagnostic complexity forward must therefore use
    a fresh container rather than reusing the object passed to the timed forward.
    """
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, Mapping):
        return type(obj)((k, clone_tensor_tree(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [clone_tensor_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_tensor_tree(v) for v in obj)
    return obj


def bytes_to_mib(value: int | float) -> float:
    return float(value) / _MIB


def json_safe(value: Any) -> Any:
    """Convert common scientific/PyTorch values to JSON-compatible objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return str(value)


def write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> str:
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False, sort_keys=True)
    return path


def write_rows_csv(
    path: str | os.PathLike[str],
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Optional[list[str]] = None,
) -> str:
    path = os.path.abspath(os.fspath(path))
    rows = [dict(r) for r in rows]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json_safe(row.get(k)) for k in fieldnames})
    return path


def flatten_dict(
    payload: Mapping[str, Any],
    *,
    prefix: str = "",
    separator: str = ".",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(flatten_dict(value, prefix=full_key, separator=separator))
        elif isinstance(value, (list, tuple)):
            result[full_key] = json.dumps(json_safe(value), ensure_ascii=False)
        else:
            result[full_key] = json_safe(value)
    return result


def summarize(values: Iterable[float]) -> Dict[str, float | int]:
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


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        json_safe(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def protocol_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return a short deterministic hash for a controlled experimental protocol."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def collect_environment(device: Optional[torch.device] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu_count": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
    }
    if device is not None:
        result["device"] = str(device)
    if torch.cuda.is_available() and device is not None and device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        result["gpu"] = {
            "index": int(index),
            "name": torch.cuda.get_device_name(index),
            "total_memory_mib": bytes_to_mib(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(index)),
        }
    return result


def _default_parameter_group(name: str) -> str:
    """Mutually-exclusive IGNet parameter grouping used in ablation tables."""
    if name.startswith("img_backbone."):
        return "image_backbone"
    if name.startswith("point_backbone.fuse_") or name.startswith("fusion_module."):
        return "fusion_projection"
    if name.startswith("point_backbone."):
        return "point_backbone"
    if name.startswith(("objectness.", "rot_head.", "depth_head.")):
        return "prediction_heads"
    if name.startswith(("crop.", "crop1.", "crop2.", "crop3.", "crop4.",
                        "multi_scale_fuse.", "multi_scale_gate.")):
        return "local_grouping"
    return "other"


def parameter_group_for_name(model: nn.Module, name: str) -> str:
    base = unwrap_model(model)
    resolver = getattr(base, "parameter_group_for_name", None)
    if callable(resolver):
        return str(resolver(name))
    return _default_parameter_group(name)


def module_group_for_name(model: nn.Module, module_name: str) -> str:
    base = unwrap_model(model)
    resolver = getattr(base, "module_group_for_name", None)
    if callable(resolver):
        return str(resolver(module_name))
    parameter_like = module_name + ".weight" if module_name else ""
    return _default_parameter_group(parameter_like)


def collect_architecture_metadata(model: nn.Module) -> Dict[str, Any]:
    base = unwrap_model(model)
    getter = getattr(base, "get_fusion_profile_metadata", None)
    if callable(getter):
        metadata = getter()
        if not isinstance(metadata, Mapping):
            raise TypeError("get_fusion_profile_metadata() must return a mapping")
        return dict(metadata)
    return {
        "model_class": type(base).__name__,
        "fusion_type": getattr(base, "fuse_type", "unknown"),
        "grouping_type": getattr(base, "grouping_type", "unknown"),
    }


def _inactive_prefixes(model: nn.Module) -> tuple[str, ...]:
    base = unwrap_model(model)
    getter = getattr(base, "get_inactive_parameter_prefixes", None)
    if callable(getter):
        return tuple(str(v) for v in getter())
    return tuple(str(v) for v in getattr(base, "inactive_parameter_prefixes", ()))


def collect_parameter_profile(model: nn.Module) -> Dict[str, Any]:
    """Collect mutually-exclusive module-wise parameter and buffer statistics."""
    base = unwrap_model(model)
    inactive_prefixes = _inactive_prefixes(base)
    groups: MutableMapping[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "registered_params": 0,
            "active_params": 0,
            "trainable_params": 0,
            "registered_parameter_bytes": 0,
            "active_parameter_bytes": 0,
            "trainable_parameter_bytes": 0,
            "buffer_elements": 0,
            "buffer_bytes": 0,
            "parameter_tensors": 0,
            "buffer_tensors": 0,
        }
    )

    seen_parameter_ids: set[int] = set()
    for name, parameter in base.named_parameters():
        if id(parameter) in seen_parameter_ids:
            continue
        seen_parameter_ids.add(id(parameter))
        group = parameter_group_for_name(base, name)
        cur = groups[group]
        count = int(parameter.numel())
        nbytes = int(count * parameter.element_size())
        inactive = any(name == p or name.startswith(p + ".") for p in inactive_prefixes)
        cur["registered_params"] += count
        cur["registered_parameter_bytes"] += nbytes
        cur["parameter_tensors"] += 1
        if not inactive:
            cur["active_params"] += count
            cur["active_parameter_bytes"] += nbytes
        if parameter.requires_grad and not inactive:
            cur["trainable_params"] += count
            cur["trainable_parameter_bytes"] += nbytes

    seen_buffer_ids: set[int] = set()
    for name, buffer in base.named_buffers():
        if id(buffer) in seen_buffer_ids:
            continue
        seen_buffer_ids.add(id(buffer))
        group = parameter_group_for_name(base, name)
        cur = groups[group]
        cur["buffer_elements"] += int(buffer.numel())
        cur["buffer_bytes"] += int(buffer.numel() * buffer.element_size())
        cur["buffer_tensors"] += 1

    ordered_names = [
        "image_backbone",
        "point_backbone",
        "fusion_projection",
        "prediction_heads",
        "local_grouping",
        "other",
    ]
    for name in list(groups):
        if name not in ordered_names:
            ordered_names.append(name)

    group_payload: Dict[str, Any] = {}
    for group_name in ordered_names:
        cur = dict(groups.get(group_name, {}))
        if not cur:
            cur = {
                "registered_params": 0,
                "active_params": 0,
                "trainable_params": 0,
                "registered_parameter_bytes": 0,
                "active_parameter_bytes": 0,
                "trainable_parameter_bytes": 0,
                "buffer_elements": 0,
                "buffer_bytes": 0,
                "parameter_tensors": 0,
                "buffer_tensors": 0,
            }
        cur["registered_params_m"] = cur["registered_params"] / 1e6
        cur["active_params_m"] = cur["active_params"] / 1e6
        cur["trainable_params_m"] = cur["trainable_params"] / 1e6
        cur["registered_parameter_size_mib"] = bytes_to_mib(cur["registered_parameter_bytes"])
        cur["active_parameter_size_mib"] = bytes_to_mib(cur["active_parameter_bytes"])
        cur["trainable_parameter_size_mib"] = bytes_to_mib(cur["trainable_parameter_bytes"])
        cur["buffer_size_mib"] = bytes_to_mib(cur["buffer_bytes"])
        group_payload[group_name] = cur

    def total(field: str) -> int:
        return int(sum(v[field] for v in group_payload.values()))

    registered = total("registered_params")
    active = total("active_params")
    trainable = total("trainable_params")
    registered_bytes = total("registered_parameter_bytes")
    active_bytes = total("active_parameter_bytes")
    trainable_bytes = total("trainable_parameter_bytes")
    buffer_bytes = total("buffer_bytes")
    return {
        "registered_total_params": registered,
        "active_total_params": active,
        "trainable_total_params": trainable,
        "registered_total_params_m": registered / 1e6,
        "active_total_params_m": active / 1e6,
        "trainable_total_params_m": trainable / 1e6,
        "registered_parameter_size_mib": bytes_to_mib(registered_bytes),
        "active_parameter_size_mib": bytes_to_mib(active_bytes),
        "trainable_parameter_size_mib": bytes_to_mib(trainable_bytes),
        "buffer_size_mib": bytes_to_mib(buffer_bytes),
        "model_tensor_size_mib": bytes_to_mib(registered_bytes + buffer_bytes),
        "inactive_parameter_prefixes": list(inactive_prefixes),
        "groups": group_payload,
    }


def parameter_group_rows(
    parameter_profile: Mapping[str, Any],
    *,
    common: Optional[Mapping[str, Any]] = None,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    common = dict(common or {})
    for group_name, stats in parameter_profile.get("groups", {}).items():
        row = dict(common)
        row["parameter_group"] = group_name
        row.update(stats)
        rows.append(row)
    return rows


def _first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "F") and torch.is_tensor(value.F):
        return value.F
    if isinstance(value, Mapping):
        for item in value.values():
            result = _first_tensor(item)
            if result is not None:
                return result
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _first_tensor(item)
            if result is not None:
                return result
    return None


def _prod(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)):
        result = 1
        for v in value:
            result *= int(v)
        return result
    if torch.is_tensor(value):
        result = 1
        for v in value.detach().cpu().view(-1).tolist():
            result *= int(v)
        return result
    return int(value)


def _kernel_map_pair_count(module: nn.Module, sparse_input: Any, sparse_output: Any) -> tuple[Optional[int], Optional[str]]:
    """Best-effort exact active input-output pair count for MinkowskiEngine."""
    try:
        coordinate_manager = sparse_input.coordinate_manager
        in_key = sparse_input.coordinate_map_key
        out_key = sparse_output.coordinate_map_key
        generator = module.kernel_generator
        kwargs = {
            "stride": getattr(generator, "kernel_stride", getattr(module, "stride", 1)),
            "kernel_size": getattr(generator, "kernel_size", getattr(module, "kernel_size", 1)),
            "dilation": getattr(generator, "kernel_dilation", getattr(module, "dilation", 1)),
            "region_type": getattr(generator, "region_type", None),
            "region_offset": getattr(generator, "region_offsets", None),
            "is_transpose": bool(getattr(module, "is_transpose", False)),
        }
        # Some versions reject explicit None values.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        kernel_map = coordinate_manager.kernel_map(in_key, out_key, **kwargs)
        pair_count = 0
        for value in kernel_map.values():
            if isinstance(value, (tuple, list)) and value:
                pair_count += int(value[0].numel())
            elif torch.is_tensor(value):
                if value.ndim >= 2 and value.shape[0] == 2:
                    pair_count += int(value.shape[1])
                else:
                    pair_count += int(value.numel())
            else:
                pair_count += len(value)
        return int(pair_count), None
    except Exception as exc:  # version-dependent private/custom API
        return None, f"{type(exc).__name__}: {exc}"


@dataclass
class _OpRecord:
    module_name: str
    module_type: str
    group: str
    calls: int = 0
    macs: float = 0.0
    flops: float = 0.0
    dense_macs: float = 0.0
    sparse_macs: float = 0.0
    sparse_pair_count: int = 0
    sparse_output_sites: int = 0
    sparse_kernel_volume: int = 0
    sparse_exact_calls: int = 0
    sparse_fallback_calls: int = 0
    notes: str = ""


class RuntimeMACProfiler(AbstractContextManager["RuntimeMACProfiler"]):
    """Input-dependent MAC/FLOP profiler based on forward hooks.

    Dense Conv/Linear MACs are exact for the observed tensor shapes. For
    Minkowski convolution, the profiler uses the exact kernel-map pair count if
    accessible; otherwise it records an active-output-site upper-bound. Element-
    wise activations, normalization, interpolation, sparse quantization/cat,
    indexing, KNN, and custom grouping kernels are not included. Measured
    latency should therefore remain the primary compute-budget metric.
    """

    _DENSE_TYPES = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.Linear,
    )

    def __init__(self, model: nn.Module, *, label: str = "forward") -> None:
        self.model = unwrap_model(model)
        self.label = str(label)
        self.handles: list[Any] = []
        self.records: Dict[str, _OpRecord] = {}
        self.errors: list[str] = []

    @staticmethod
    def _is_minkowski_convolution(module: nn.Module) -> bool:
        name = type(module).__name__
        return name.startswith("MinkowskiConvolution")

    def _record_for(self, module_name: str, module: nn.Module) -> _OpRecord:
        if module_name not in self.records:
            self.records[module_name] = _OpRecord(
                module_name=module_name,
                module_type=type(module).__name__,
                group=module_group_for_name(self.model, module_name),
            )
        return self.records[module_name]

    def _dense_hook(self, module_name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            try:
                out = _first_tensor(output)
                if out is None:
                    return
                if isinstance(module, nn.Linear):
                    macs = float(out.numel() * module.in_features)
                elif isinstance(module, (
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                )):
                    # A transposed convolution expands each input element across
                    # the kernel support. Counting from the output shape is not
                    # exact when stride/output-padding is used.
                    inp = _first_tensor(inputs)
                    if inp is None:
                        return
                    kernel_volume = _prod(module.kernel_size)
                    out_channels_per_group = int(module.out_channels // module.groups)
                    macs = float(inp.numel() * kernel_volume * out_channels_per_group)
                else:
                    kernel_volume = _prod(module.kernel_size)
                    channels_per_group = int(module.in_channels // module.groups)
                    macs = float(out.numel() * kernel_volume * channels_per_group)
                record = self._record_for(module_name, module)
                record.calls += 1
                record.macs += macs
                record.flops += 2.0 * macs
                record.dense_macs += macs
            except Exception as exc:
                self.errors.append(f"dense hook {module_name}: {type(exc).__name__}: {exc}")
        return hook

    def _sparse_hook(self, module_name: str) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            try:
                if not inputs:
                    return
                sparse_input = inputs[0]
                sparse_output = output
                input_features = _first_tensor(sparse_input)
                output_features = _first_tensor(sparse_output)
                if input_features is None or output_features is None:
                    return
                in_channels = int(getattr(module, "in_channels", input_features.shape[-1]))
                out_channels = int(getattr(module, "out_channels", output_features.shape[-1]))
                kernel = getattr(module, "kernel", None)
                if torch.is_tensor(kernel) and kernel.ndim >= 3:
                    kernel_volume = int(kernel.shape[0])
                else:
                    generator = getattr(module, "kernel_generator", None)
                    kernel_volume = int(getattr(generator, "kernel_volume", 1))
                output_sites = int(output_features.shape[0])
                pair_count, error = _kernel_map_pair_count(module, sparse_input, sparse_output)
                if pair_count is not None:
                    macs = float(pair_count * in_channels * out_channels)
                    exact = True
                else:
                    macs = float(output_sites * kernel_volume * in_channels * out_channels)
                    exact = False
                record = self._record_for(module_name, module)
                record.calls += 1
                record.macs += macs
                record.flops += 2.0 * macs
                record.sparse_macs += macs
                record.sparse_output_sites += output_sites
                record.sparse_kernel_volume = max(record.sparse_kernel_volume, kernel_volume)
                if pair_count is not None:
                    record.sparse_pair_count += int(pair_count)
                    record.sparse_exact_calls += 1
                else:
                    record.sparse_fallback_calls += 1
                    if error:
                        record.notes = error
            except Exception as exc:
                self.errors.append(f"sparse hook {module_name}: {type(exc).__name__}: {exc}")
        return hook

    def __enter__(self) -> "RuntimeMACProfiler":
        for module_name, module in self.model.named_modules():
            if not module_name:
                continue
            if self._is_minkowski_convolution(module):
                self.handles.append(module.register_forward_hook(self._sparse_hook(module_name)))
            elif isinstance(module, self._DENSE_TYPES):
                self.handles.append(module.register_forward_hook(self._dense_hook(module_name)))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Optional[bool]:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return None

    def summary(self) -> Dict[str, Any]:
        module_rows = []
        group_totals: MutableMapping[str, Dict[str, float | int]] = defaultdict(
            lambda: {
                "calls": 0,
                "macs": 0.0,
                "flops": 0.0,
                "dense_macs": 0.0,
                "sparse_macs": 0.0,
                "sparse_exact_calls": 0,
                "sparse_fallback_calls": 0,
            }
        )
        for name in sorted(self.records):
            record = self.records[name]
            row = {
                "module_name": record.module_name,
                "module_type": record.module_type,
                "group": record.group,
                "calls": record.calls,
                "macs": record.macs,
                "gmacs": record.macs / 1e9,
                "flops": record.flops,
                "gflops": record.flops / 1e9,
                "dense_macs": record.dense_macs,
                "sparse_macs": record.sparse_macs,
                "sparse_pair_count": record.sparse_pair_count,
                "sparse_output_sites": record.sparse_output_sites,
                "sparse_kernel_volume": record.sparse_kernel_volume,
                "sparse_exact_calls": record.sparse_exact_calls,
                "sparse_fallback_calls": record.sparse_fallback_calls,
                "notes": record.notes,
            }
            module_rows.append(row)
            dst = group_totals[record.group]
            for field in (
                "calls",
                "macs",
                "flops",
                "dense_macs",
                "sparse_macs",
                "sparse_exact_calls",
                "sparse_fallback_calls",
            ):
                dst[field] += row[field]

        by_group: Dict[str, Any] = {}
        for group, values in sorted(group_totals.items()):
            cur = dict(values)
            cur["gmacs"] = float(cur["macs"]) / 1e9
            cur["gflops"] = float(cur["flops"]) / 1e9
            by_group[group] = cur

        total_macs = float(sum(r["macs"] for r in module_rows))
        total_flops = float(sum(r["flops"] for r in module_rows))
        sparse_exact_calls = int(sum(r["sparse_exact_calls"] for r in module_rows))
        sparse_fallback_calls = int(sum(r["sparse_fallback_calls"] for r in module_rows))
        if sparse_fallback_calls == 0:
            sparse_method = "exact_kernel_map"
        elif sparse_exact_calls == 0:
            sparse_method = "active_output_site_upper_bound"
        else:
            sparse_method = "hybrid_exact_and_upper_bound"

        return {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "label": self.label,
            "total_macs": total_macs,
            "total_gmacs": total_macs / 1e9,
            "total_flops": total_flops,
            "total_gflops": total_flops / 1e9,
            "flop_convention": "2 FLOPs per multiply-accumulate",
            "sparse_count_method": sparse_method,
            "sparse_exact_calls": sparse_exact_calls,
            "sparse_fallback_calls": sparse_fallback_calls,
            "scope": (
                "Conv/Linear and MinkowskiConvolution only; excludes normalization, "
                "activation, interpolation, sparse quantization/cat, indexing, KNN, "
                "point grouping, pooling, and other custom operators"
            ),
            "by_group": by_group,
            "by_module": module_rows,
            "errors": list(self.errors),
        }


def save_complexity_profile(
    output_dir: str | os.PathLike[str],
    complexity: Mapping[str, Any],
    *,
    basename: str = "runtime_complexity",
    common: Optional[Mapping[str, Any]] = None,
) -> tuple[str, str]:
    output_dir = os.path.abspath(os.fspath(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    json_path = write_json(os.path.join(output_dir, f"{basename}.json"), complexity)
    rows = []
    for row in complexity.get("by_module", []):
        merged = dict(common or {})
        merged.update(row)
        rows.append(merged)
    csv_path = write_rows_csv(os.path.join(output_dir, f"{basename}_modules.csv"), rows)
    return json_path, csv_path


def common_variant_identity(
    architecture: Mapping[str, Any],
    *,
    phase: str,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "phase": phase,
        "run_id": run_id,
        "fusion_type": architecture.get("fusion_type", "unknown"),
        "fusion_label": architecture.get("fusion_label", architecture.get("fusion_type", "unknown")),
        "grouping_type": architecture.get("grouping_type", "unknown"),
        "model_class": architecture.get("model_class", "unknown"),
    }
