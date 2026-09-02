"""Dataset, runtime, and checkpoint auditing for the EcoCAR YOLO pipeline.

Nothing in this module runs at import time.  In particular, PyTorch, Pillow,
Matplotlib, and Ultralytics are imported only inside the audit functions so
unit tests can exercise label validation without initializing a model runtime.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    DatasetSpec,
    load_dataset_spec,
    resolve_weights,
    write_resolved_dataset_yaml,
)
from .utils import (
    RunContext,
    StageError,
    basic_environment,
    flatten_dict,
    import_ultralytics_yolo,
    read_json,
    relative_to,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)

LOGGER = logging.getLogger("road_detection")

EXPECTED_CLASS_NAMES: Mapping[int, str] = {
    0: "16-Wheelers",
    1: "Ad-signs",
    2: "Bike",
    3: "Bus",
    4: "Car",
    5: "Lamp-post",
    6: "MiniBus",
    7: "Motorcycle",
    8: "Person",
    9: "Pickup",
    10: "Police-car",
    11: "Traffic-light",
    12: "Traffic-sign",
    13: "Truck",
}
EXPECTED_CLASS_COUNT = len(EXPECTED_CLASS_NAMES)
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
)
SERIOUS_IMBALANCE_RATIO = 10.0


class LabelValidationError(ValueError):
    """A single nonempty YOLO detection label row is invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class YoloLabel:
    """One validated class-aware, normalized YOLO bounding box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def as_tuple(self) -> tuple[int, float, float, float, float]:
        return (self.class_id, self.x_center, self.y_center, self.width, self.height)


@dataclass
class SplitScan:
    """Internal, deterministic result of inspecting one dataset split."""

    split: str
    image_dir: Path
    label_dir: Path
    image_count: int = 0
    label_count: int = 0
    matched_label_count: int = 0
    missing_label_count: int = 0
    orphan_label_count: int = 0
    empty_label_count: int = 0
    unreadable_image_count: int = 0
    unsupported_file_count: int = 0
    object_counts: Counter[int] = field(default_factory=Counter)
    image_hashes: dict[Path, str] = field(default_factory=dict)
    file_manifest: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def object_count(self) -> int:
        return sum(self.object_counts.values())


_INTEGER_TOKEN = re.compile(r"^[+-]?\d+$")
_URL_WITH_QUERY = re.compile(r"(https?://[^\s?]+)\?[^\s]+", re.IGNORECASE)
_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|authorization)", re.IGNORECASE)


def parse_yolo_label_row(row: str | Sequence[str], nc: int) -> YoloLabel:
    """Parse and validate one nonempty YOLO detection row.

    The accepted format is exactly ``class_id x_center y_center width height``.
    ``class_id`` must be an integer in ``[0, nc - 1]``.  Coordinates must be
    finite and in ``[0, 1]``; width and height must additionally be positive.

    This function is intentionally pure so it can be unit-tested without any
    dataset or machine-learning dependencies.
    """

    if isinstance(nc, bool) or not isinstance(nc, int) or nc <= 0:
        raise ValueError("nc must be a positive integer")
    tokens = row.split() if isinstance(row, str) else [str(token) for token in row]
    if not tokens:
        raise LabelValidationError("empty_row", "label row is empty")
    if len(tokens) != 5:
        raise LabelValidationError(
            "malformed_label",
            f"expected 5 whitespace-separated values, found {len(tokens)}",
        )
    if not _INTEGER_TOKEN.fullmatch(tokens[0]):
        raise LabelValidationError("invalid_class_id", "class_id must be an integer")
    class_id = int(tokens[0])
    if not 0 <= class_id < nc:
        raise LabelValidationError(
            "invalid_class_id", f"class_id {class_id} is outside [0, {nc - 1}]"
        )
    try:
        coordinates = tuple(float(value) for value in tokens[1:])
    except ValueError as exc:
        raise LabelValidationError(
            "invalid_coordinate", "bounding-box coordinates must be numeric"
        ) from exc
    coordinate_names = ("x_center", "y_center", "width", "height")
    for name, value in zip(coordinate_names, coordinates, strict=True):
        if not math.isfinite(value):
            raise LabelValidationError("invalid_coordinate", f"{name} must be finite")
        if not 0.0 <= value <= 1.0:
            raise LabelValidationError("invalid_coordinate", f"{name}={value!r} is outside [0, 1]")
    if coordinates[2] <= 0.0:
        raise LabelValidationError("invalid_box_size", "width must be greater than zero")
    if coordinates[3] <= 0.0:
        raise LabelValidationError("invalid_box_size", "height must be greater than zero")
    return YoloLabel(class_id, *coordinates)


# Friendly aliases keep the small validation API discoverable for callers and tests.
validate_label_row = parse_yolo_label_row
parse_label_line = parse_yolo_label_row
validate_label_line = parse_yolo_label_row


def resolve_label_directory(image_directory: Path) -> Path:
    """Apply Ultralytics' conventional ``images`` -> ``labels`` path mapping."""

    parts = list(image_directory.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].casefold() == "images":
            parts[index] = "labels"
            return Path(*parts)
    return image_directory.parent / "labels"


def _safe_message(value: BaseException | str, limit: int = 1000) -> str:
    """Keep reports concise and avoid retaining query strings from download URLs."""

    message = _URL_WITH_QUERY.sub(r"\1?<redacted>", str(value)).replace("\x00", "")
    return message if len(message) <= limit else message[: limit - 3] + "..."


def _sanitize_payload(value: Any, key: str = "") -> Any:
    """Redact likely credentials and signed-URL queries from copied metadata."""

    if key and _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_payload(item, str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return _safe_message(value, limit=max(1000, len(value)))
    return value


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    split: str = "",
    path: str = "",
    line: int | str = "",
    details: Any = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "split": split,
        "path": path,
        "line": line,
        "message": _safe_message(message),
    }
    if details is not None:
        result["details"] = details
    return result


def _display_path(path: Path, spec: DatasetSpec) -> str:
    return relative_to(path, spec.root)


def _recursive_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.rglob("*") if path.is_file()),
        key=lambda path: path.as_posix().casefold(),
    )


def _relative_stem(path: Path, parent: Path) -> str:
    return path.relative_to(parent).with_suffix("").as_posix().casefold()


def _append_file_hash(
    scan: SplitScan,
    path: Path,
    kind: str,
    spec: DatasetSpec,
) -> str | None:
    try:
        digest = sha256_file(path)
        size = path.stat().st_size
    except OSError as exc:
        scan.issues.append(
            _issue(
                "critical",
                "unreadable_file",
                f"Could not hash file: {_safe_message(exc)}",
                split=scan.split,
                path=_display_path(path, spec),
            )
        )
        return None
    scan.file_manifest.append(
        {
            "split": scan.split,
            "kind": kind,
            "path": _display_path(path, spec),
            "size_bytes": size,
            "sha256": digest,
        }
    )
    return digest


def _detect_duplicate_stems(
    files: Sequence[Path],
    parent: Path,
    kind: str,
    split: str,
    spec: DatasetSpec,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    by_relative_stem: defaultdict[str, list[Path]] = defaultdict(list)
    by_bare_stem: defaultdict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_relative_stem[_relative_stem(path, parent)].append(path)
        by_bare_stem[path.stem.casefold()].append(path)
    ambiguous_paths: set[Path] = set()
    for paths in by_relative_stem.values():
        if len(paths) > 1:
            ambiguous_paths.update(paths)
            shown = [_display_path(path, spec) for path in paths]
            issues.append(
                _issue(
                    "critical",
                    f"duplicate_{kind}_stem",
                    f"Multiple {kind} files have the same relative stem: {', '.join(shown)}",
                    split=split,
                    path=shown[0],
                    details={"paths": shown},
                )
            )
    for paths in by_bare_stem.values():
        if len(paths) > 1 and not set(paths).issubset(ambiguous_paths):
            shown = [_display_path(path, spec) for path in paths]
            issues.append(
                _issue(
                    "warning",
                    f"duplicate_{kind}_basename",
                    f"Repeated {kind} basename across directories: {', '.join(shown)}",
                    split=split,
                    path=shown[0],
                    details={"paths": shown},
                )
            )
    return issues


def _validate_image(path: Path) -> str | None:
    try:
        from PIL import Image
    except ImportError as exc:  # handled once by the caller
        raise StageError("Pillow is required to verify dataset images") from exc
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError) as exc:
        return _safe_message(exc)
    return None


def _read_and_validate_label(
    path: Path,
    nc: int,
    split: str,
    spec: DatasetSpec,
) -> tuple[list[YoloLabel], bool, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        issues.append(
            _issue(
                "critical",
                "unreadable_label",
                f"Could not read label file: {_safe_message(exc)}",
                split=split,
                path=_display_path(path, spec),
            )
        )
        return [], False, issues
    rows: list[YoloLabel] = []
    nonempty = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        nonempty = True
        try:
            rows.append(parse_yolo_label_row(line, nc))
        except LabelValidationError as exc:
            issues.append(
                _issue(
                    "critical",
                    exc.code,
                    str(exc),
                    split=split,
                    path=_display_path(path, spec),
                    line=line_number,
                    details={"row": line[:500]},
                )
            )
    return rows, not nonempty, issues


def _scan_split(split: str, image_dir: Path, spec: DatasetSpec, nc: int) -> SplitScan:
    label_dir = resolve_label_directory(image_dir)
    scan = SplitScan(split=split, image_dir=image_dir, label_dir=label_dir)
    if not image_dir.is_dir():
        scan.issues.append(
            _issue(
                "critical",
                "missing_image_directory",
                "Configured split image directory does not exist or is not a directory",
                split=split,
                path=_display_path(image_dir, spec),
            )
        )
        return scan
    if not label_dir.is_dir():
        scan.issues.append(
            _issue(
                "critical",
                "missing_label_directory",
                "Expected label directory does not exist or is not a directory",
                split=split,
                path=_display_path(label_dir, spec),
            )
        )

    image_directory_files = _recursive_files(image_dir)
    image_files = [
        path for path in image_directory_files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    unsupported_images = [
        path
        for path in image_directory_files
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_directory_files = _recursive_files(label_dir)
    label_files = [path for path in label_directory_files if path.suffix.lower() == ".txt"]
    unsupported_labels = [path for path in label_directory_files if path.suffix.lower() != ".txt"]
    scan.image_count = len(image_files)
    scan.label_count = len(label_files)
    scan.unsupported_file_count = len(unsupported_images) + len(unsupported_labels)

    if not image_files:
        scan.issues.append(
            _issue(
                "critical",
                "empty_image_split",
                "Split contains no supported images",
                split=split,
                path=_display_path(image_dir, spec),
            )
        )
    if label_dir.is_dir() and not label_files:
        scan.issues.append(
            _issue(
                "critical",
                "empty_label_split",
                "Split label directory contains no .txt label files",
                split=split,
                path=_display_path(label_dir, spec),
            )
        )
    for path in unsupported_images:
        scan.issues.append(
            _issue(
                "warning",
                "unsupported_image_extension",
                f"File in image directory has unsupported extension {path.suffix or '<none>'!r}",
                split=split,
                path=_display_path(path, spec),
            )
        )
        _append_file_hash(scan, path, "unsupported_image_file", spec)
    for path in unsupported_labels:
        scan.issues.append(
            _issue(
                "warning",
                "unsupported_label_extension",
                f"File in label directory is not a .txt label: {path.suffix or '<none>'!r}",
                split=split,
                path=_display_path(path, spec),
            )
        )
        _append_file_hash(scan, path, "unsupported_label_file", spec)

    scan.issues.extend(_detect_duplicate_stems(image_files, image_dir, "image", split, spec))
    if label_dir.is_dir():
        scan.issues.extend(_detect_duplicate_stems(label_files, label_dir, "label", split, spec))

    image_keys = {_relative_stem(path, image_dir) for path in image_files}
    label_keys = {_relative_stem(path, label_dir) for path in label_files}
    image_by_key: defaultdict[str, list[Path]] = defaultdict(list)
    label_by_key: defaultdict[str, list[Path]] = defaultdict(list)
    for path in image_files:
        image_by_key[_relative_stem(path, image_dir)].append(path)
    for path in label_files:
        label_by_key[_relative_stem(path, label_dir)].append(path)

    missing_keys = sorted(image_keys - label_keys)
    orphan_keys = sorted(label_keys - image_keys)
    scan.missing_label_count = sum(len(image_by_key[key]) for key in missing_keys)
    scan.orphan_label_count = sum(len(label_by_key[key]) for key in orphan_keys)
    scan.matched_label_count = sum(
        min(len(image_by_key[key]), len(label_by_key[key])) for key in image_keys & label_keys
    )
    for key in missing_keys:
        for path in image_by_key[key]:
            scan.issues.append(
                _issue(
                    "warning",
                    "missing_label",
                    "Image has no matching label file; verify that it is an "
                    "intentional background image",
                    split=split,
                    path=_display_path(path, spec),
                )
            )
    for key in orphan_keys:
        for path in label_by_key[key]:
            scan.issues.append(
                _issue(
                    "warning",
                    "orphan_label",
                    "Label file has no matching supported image",
                    split=split,
                    path=_display_path(path, spec),
                )
            )

    pillow_available = True
    try:
        import PIL  # noqa: F401
    except ImportError:
        pillow_available = False
        scan.issues.append(
            _issue(
                "critical",
                "missing_pillow",
                "Pillow is not installed, so image readability could not be verified",
                split=split,
            )
        )
    for path in image_files:
        digest = _append_file_hash(scan, path, "image", spec)
        if digest is not None:
            scan.image_hashes[path] = digest
        if pillow_available:
            error = _validate_image(path)
            if error is not None:
                scan.unreadable_image_count += 1
                scan.issues.append(
                    _issue(
                        "critical",
                        "unreadable_image",
                        f"Image decoder could not verify the file: {error}",
                        split=split,
                        path=_display_path(path, spec),
                    )
                )

    for path in label_files:
        _append_file_hash(scan, path, "label", spec)
        records, is_empty, issues = _read_and_validate_label(path, nc, split, spec)
        scan.issues.extend(issues)
        if is_empty:
            scan.empty_label_count += 1
            scan.issues.append(
                _issue(
                    "warning",
                    "empty_label",
                    "Label file is empty; verify that it is an intentional background image",
                    split=split,
                    path=_display_path(path, spec),
                )
            )
        if _relative_stem(path, label_dir) in image_keys:
            scan.object_counts.update(record.class_id for record in records)
    return scan


def _read_cpu_memory() -> dict[str, int] | None:
    """Return Linux host memory in bytes when /proc exposes it."""

    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    values: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"MemTotal", "MemAvailable"}:
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
    }


def audit_environment(context: RunContext) -> dict[str, Any]:
    """Collect dependency, accelerator, memory, and requested-precision details."""

    environment = basic_environment()
    environment.update(
        {
            "hostname": platform.node(),
            "requested_device": context.config.training.device,
            "requested_precision": "automatic mixed precision"
            if context.config.training.amp
            else "float32",
            "cpu_memory": _read_cpu_memory(),
        }
    )
    try:
        import torch
    except ImportError as exc:
        environment["torch_runtime"] = {"available": False, "error": _safe_message(exc)}
        environment["selected_device"] = "unavailable"
        return environment

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    requested = context.config.training.device
    if requested == "auto":
        selected = "cuda:0" if cuda_available else ("mps" if mps_available else "cpu")
    elif isinstance(requested, int):
        selected = "auto-selected accelerator" if requested == -1 else f"cuda:{requested}"
    else:
        requested_text = str(requested).strip()
        selected = (
            f"cuda:{requested_text}"
            if re.fullmatch(r"\d+(?:,\d+)*", requested_text)
            else requested_text
        )

    cuda_devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            row: dict[str, Any] = {
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
            }
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                row.update(
                    {
                        "free_memory_bytes": int(free_bytes),
                        "runtime_total_memory_bytes": int(total_bytes),
                    }
                )
            except (AttributeError, RuntimeError, TypeError):
                pass
            cuda_devices.append(row)
    torch_version = getattr(torch, "__version__", None)
    torch_runtime: dict[str, Any] = {
        "available": True,
        "version": str(torch_version) if torch_version is not None else None,
        "cuda_available": cuda_available,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "rocm_version": getattr(getattr(torch, "version", None), "hip", None),
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "mps_available": mps_available,
        "cuda_devices": cuda_devices,
    }
    environment["torch_runtime"] = torch_runtime
    environment["selected_device"] = selected
    return environment


def _environment_issues(
    context: RunContext, environment: Mapping[str, Any]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    runtime = environment.get("torch_runtime")
    if not isinstance(runtime, Mapping) or not runtime.get("available"):
        issues.append(
            _issue(
                "critical",
                "torch_unavailable",
                "PyTorch is unavailable, so the configured execution device cannot be used",
            )
        )
        return issues

    requested = context.config.training.device
    requested_text = str(requested).strip().lower()
    cuda_requested = isinstance(requested, int) and requested >= 0
    cuda_indexes: list[int] = [requested] if cuda_requested else []
    if isinstance(requested, str):
        if requested_text.startswith("cuda"):
            cuda_requested = True
            suffix = requested_text.partition(":")[2]
            if suffix.isdigit():
                cuda_indexes = [int(suffix)]
        elif re.fullmatch(r"\d+(?:,\d+)*", requested_text):
            cuda_requested = True
            cuda_indexes = [int(value) for value in requested_text.split(",")]
    cuda_devices = runtime.get("cuda_devices", [])
    if cuda_requested and not runtime.get("cuda_available"):
        issues.append(
            _issue(
                "critical",
                "cuda_unavailable",
                f"training.device={requested!r} requests CUDA, but CUDA is not available",
            )
        )
    elif cuda_requested and cuda_indexes:
        available_indexes = {
            int(row["index"])
            for row in cuda_devices
            if isinstance(row, Mapping) and isinstance(row.get("index"), int)
        }
        missing = sorted(set(cuda_indexes) - available_indexes)
        if missing:
            issues.append(
                _issue(
                    "critical",
                    "cuda_device_missing",
                    f"Configured CUDA device index(es) do not exist: {missing}",
                )
            )
    if requested_text == "mps" and not runtime.get("mps_available"):
        issues.append(
            _issue(
                "critical",
                "mps_unavailable",
                "training.device='mps' was configured, but the MPS backend is not available",
            )
        )
    if context.config.training.amp and environment.get("selected_device") == "cpu":
        issues.append(
            _issue(
                "warning",
                "amp_on_cpu",
                "AMP was requested while auto device selection resolved to CPU; "
                "Ultralytics may disable AMP",
            )
        )
    return issues


def _model_task(model: Any) -> str | None:
    candidates = [getattr(model, "task", None)]
    inner = getattr(model, "model", None)
    candidates.append(getattr(inner, "task", None))
    for container_name in ("args", "yaml"):
        container = getattr(inner, container_name, None)
        if isinstance(container, Mapping):
            candidates.append(container.get("task"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate.lower()
    return None


def _model_parameter_count(model: Any) -> int | None:
    inner = getattr(model, "model", None)
    parameters = getattr(inner, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return int(sum(parameter.numel() for parameter in parameters()))
    except (AttributeError, RuntimeError, TypeError):
        return None


def audit_checkpoints(context: RunContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load configured checkpoints lazily and verify standard detection tasks."""

    checks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    try:
        yolo_class = import_ultralytics_yolo()
    except Exception as exc:
        message = _safe_message(exc)
        issues.append(_issue("critical", "ultralytics_unavailable", message))
        for candidate in context.config.models:
            checks.append(
                {
                    "model_id": candidate.id,
                    "weights": candidate.weights,
                    "status": "failed",
                    "task": None,
                    "parameter_count": None,
                    "error": message,
                }
            )
        return checks, issues

    for candidate in context.config.models:
        resolved = resolve_weights(candidate.weights, context.config.project_root)
        resolved_path = Path(resolved)
        portable_resolved = (
            relative_to(resolved_path, context.config.project_root)
            if resolved_path.is_absolute()
            else resolved
        )
        row: dict[str, Any] = {
            "model_id": candidate.id,
            "weights": candidate.weights,
            "training_batch": candidate.batch or context.config.training.batch,
            "resolved_weights": portable_resolved,
            "status": "failed",
            "task": None,
            "parameter_count": None,
            "checkpoint_path": None,
            "checkpoint_sha256": None,
            "checkpoint_size_bytes": None,
            "error": None,
        }
        model: Any = None
        try:
            model = yolo_class(resolved)
            task = _model_task(model)
            row["task"] = task
            row["parameter_count"] = _model_parameter_count(model)
            if task != "detect":
                raise StageError(
                    f"checkpoint task is {task!r}; expected standard axis-aligned object detection"
                )
            loaded_path_value = getattr(model, "ckpt_path", None)
            loaded_path = Path(str(loaded_path_value or resolved)).expanduser()
            if not loaded_path.is_absolute():
                project_candidate = (context.config.project_root / loaded_path).resolve()
                loaded_path = (
                    project_candidate if project_candidate.is_file() else loaded_path.resolve()
                )
            if loaded_path.is_file():
                row["checkpoint_path"] = relative_to(loaded_path, context.config.project_root)
                row["checkpoint_sha256"] = sha256_file(loaded_path)
                row["checkpoint_size_bytes"] = loaded_path.stat().st_size
            row["status"] = "loaded"
        except Exception as exc:  # each candidate must be reported without hiding the others
            row["error"] = _safe_message(exc)
            issues.append(
                _issue(
                    "critical",
                    "checkpoint_load_failed",
                    f"Model {candidate.id!r} could not be loaded as a detection "
                    f"checkpoint: {row['error']}",
                    path=candidate.weights,
                )
            )
        finally:
            del model
        checks.append(row)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    return checks, issues


def _cross_split_duplicates(
    scans: Sequence[SplitScan], spec: DatasetSpec
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_hash: defaultdict[str, list[tuple[str, Path]]] = defaultdict(list)
    for scan in scans:
        for path, digest in scan.image_hashes.items():
            by_hash[digest].append((scan.split, path))
    groups: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for digest, entries in sorted(by_hash.items()):
        split_names = sorted({split for split, _ in entries})
        if len(split_names) < 2:
            continue
        paths = [
            {"split": split, "path": _display_path(path, spec)}
            for split, path in sorted(entries, key=lambda item: (item[0], item[1].as_posix()))
        ]
        group = {"sha256": digest, "splits": split_names, "files": paths}
        groups.append(group)
        shown = ", ".join(f"{item['split']}:{item['path']}" for item in paths)
        issues.append(
            _issue(
                "warning",
                "cross_split_duplicate_image",
                f"DATA LEAKAGE WARNING: exact image content occurs across splits: {shown}",
                details=group,
            )
        )
    return groups, issues


def _dataset_fingerprint(
    spec: DatasetSpec,
    scans: Sequence[SplitScan],
    issues: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    records = [record for scan in scans for record in scan.file_manifest]
    try:
        yaml_hash = sha256_file(spec.yaml_path)
        records.append(
            {
                "split": "dataset",
                "kind": "yaml",
                "path": _display_path(spec.yaml_path, spec),
                "size_bytes": spec.yaml_path.stat().st_size,
                "sha256": yaml_hash,
            }
        )
    except OSError as exc:
        issues.append(
            _issue(
                "critical",
                "unreadable_dataset_yaml",
                f"Could not hash dataset YAML: {_safe_message(exc)}",
                path=_display_path(spec.yaml_path, spec),
            )
        )
    records.sort(key=lambda row: (row["split"], row["kind"], row["path"]))
    if not records:
        return None, records
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest(), records


def verify_dataset_unchanged(context: RunContext) -> str:
    """Recheck the audited file set and hashes before a data-consuming stage."""

    audit_path = context.run_dir / "data_audit" / "audit.json"
    if not audit_path.is_file():
        raise StageError("The completed audit artifact is missing; refusing to use the dataset")
    report = read_json(audit_path)
    if report.get("status") not in {"passed", "passed_with_warnings"}:
        raise StageError("The dataset audit did not pass its training gate")
    expected_records = report.get("file_manifest")
    if not isinstance(expected_records, list) or not expected_records:
        raise StageError("The audit has no reproducible dataset file manifest")

    spec = load_dataset_spec(context.dataset_yaml)
    current_files: set[Path] = {spec.yaml_path.resolve()}
    for image_dir in spec.splits.values():
        current_files.update(path.resolve() for path in _recursive_files(image_dir))
        current_files.update(
            path.resolve() for path in _recursive_files(resolve_label_directory(image_dir))
        )
    current_display = {_display_path(path, spec) for path in current_files}
    expected_display = {
        str(record.get("path"))
        for record in expected_records
        if isinstance(record, Mapping) and record.get("path")
    }
    if current_display != expected_display:
        added = sorted(current_display - expected_display)
        removed = sorted(expected_display - current_display)
        raise StageError(
            "Dataset file set changed after audit "
            f"(added={added[:5]}, removed={removed[:5]}). Start a new audit/run."
        )

    verified_records: list[dict[str, Any]] = []
    for raw_record in expected_records:
        if not isinstance(raw_record, Mapping):
            raise StageError("Audit file manifest contains an invalid record")
        record = dict(raw_record)
        declared = Path(str(record["path"]))
        path = declared if declared.is_absolute() else (spec.root / declared).resolve()
        if not path.is_file():
            raise StageError(f"Audited dataset file is missing: {path}")
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != record.get("size_bytes") or digest != record.get("sha256"):
            raise StageError(
                f"Dataset file changed after audit: {_display_path(path, spec)}. "
                "Start a new audit/run."
            )
        record["size_bytes"] = size
        record["sha256"] = digest
        verified_records.append(record)

    verified_records.sort(key=lambda row: (row["split"], row["kind"], row["path"]))
    digest = hashlib.sha256()
    for record in verified_records:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    fingerprint = digest.hexdigest()
    expected_fingerprint = report.get("dataset", {}).get("fingerprint")
    if fingerprint != expected_fingerprint:
        raise StageError("Dataset fingerprint no longer matches the completed audit")
    return fingerprint


def _prepare_runtime_dataset(context: RunContext, spec: DatasetSpec) -> Path:
    """Create a local dataset view so Ultralytics cannot cache into source data."""

    snapshot_number = 1
    while (context.run_dir / f"dataset_snapshot_{snapshot_number:03d}").exists():
        snapshot_number += 1
    snapshot_root = context.run_dir / f"dataset_snapshot_{snapshot_number:03d}"
    split_paths: dict[str, Path] = {}
    runtime_records: list[dict[str, Any]] = []
    try:
        for split in ("train", "val", "test"):
            source_images = spec.split(split)
            source_labels = resolve_label_directory(source_images)
            split_root = snapshot_root / split
            target_images = split_root / "images"
            target_labels = split_root / "labels"
            target_images.mkdir(parents=True, exist_ok=False)
            target_labels.mkdir(parents=True, exist_ok=False)
            split_paths[split] = target_images

            for source in _recursive_files(source_images):
                if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue
                target = target_images / source.relative_to(source_images)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source.resolve())
                runtime_records.append(
                    {
                        "kind": "image_symlink",
                        "path": target.relative_to(context.run_dir).as_posix(),
                        "target": str(source.resolve()),
                        "sha256": sha256_file(source),
                    }
                )
            for source in _recursive_files(source_labels):
                if source.suffix.lower() != ".txt":
                    continue
                target = target_labels / source.relative_to(source_labels)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                runtime_records.append(
                    {
                        "kind": "label_copy",
                        "path": target.relative_to(context.run_dir).as_posix(),
                        "sha256": sha256_file(target),
                    }
                )
    except OSError as exc:
        raise StageError(
            f"Could not create artifact-local runtime dataset below {snapshot_root}: {exc}"
        ) from exc

    runtime_yaml = context.run_dir / f"runtime_dataset_{snapshot_number:03d}.yaml"
    runtime_spec = DatasetSpec(
        yaml_path=runtime_yaml,
        root=snapshot_root,
        splits=split_paths,
        nc=spec.nc,
        names=spec.names,
        metadata=spec.metadata,
    )
    write_resolved_dataset_yaml(runtime_spec, runtime_yaml)
    runtime_records.append(
        {
            "kind": "descriptor",
            "path": runtime_yaml.relative_to(context.run_dir).as_posix(),
            "sha256": sha256_file(runtime_yaml),
        }
    )
    runtime_records.sort(key=lambda row: (row["kind"], row["path"]))
    runtime_manifest = context.run_dir / f"runtime_dataset_manifest_{snapshot_number:03d}.json"
    write_json(
        runtime_manifest,
        {
            "schema_version": 1,
            "runtime_yaml": runtime_yaml.relative_to(context.run_dir).as_posix(),
            "records": runtime_records,
        },
    )
    manifest = context.manifest
    dataset = dict(manifest.get("dataset", {}))
    dataset.update(
        {
            "runtime_yaml": relative_to(runtime_yaml, context.run_dir),
            "runtime_root": relative_to(snapshot_root, context.run_dir),
            "runtime_manifest": relative_to(runtime_manifest, context.run_dir),
            "runtime_manifest_sha256": sha256_file(runtime_manifest),
            "runtime_image_storage": "symlinks to audited source images",
            "runtime_label_storage": "artifact-local copies",
        }
    )
    context.update_manifest(dataset=dataset)
    return runtime_yaml


def _canonical_name(class_id: int, declared: str) -> str:
    if class_id == 5 and declared.casefold() == "lamb-post":
        return "Lamp-post"
    return declared


def _add_distribution_warnings(
    scans: Sequence[SplitScan],
    names: Mapping[int, str],
    issues: list[dict[str, Any]],
) -> None:
    distributions: list[tuple[str, Counter[int]]] = [
        (scan.split, scan.object_counts) for scan in scans
    ]
    total: Counter[int] = Counter()
    for scan in scans:
        total.update(scan.object_counts)
    distributions.append(("total", total))
    for split, counts in distributions:
        absent = [
            names[class_id] for class_id in range(EXPECTED_CLASS_COUNT) if counts[class_id] == 0
        ]
        if absent:
            issues.append(
                _issue(
                    "warning",
                    "absent_classes",
                    f"Classes absent from {split}: {', '.join(absent)}",
                    split="" if split == "total" else split,
                    details={"classes": absent},
                )
            )
        nonzero = [
            counts[class_id] for class_id in range(EXPECTED_CLASS_COUNT) if counts[class_id] > 0
        ]
        if len(nonzero) >= 2:
            ratio = max(nonzero) / min(nonzero)
            if ratio >= SERIOUS_IMBALANCE_RATIO:
                issues.append(
                    _issue(
                        "warning",
                        "serious_class_imbalance",
                        "Largest-to-smallest nonzero class count ratio in "
                        f"{split} is {ratio:.2f}:1",
                        split="" if split == "total" else split,
                        details={"ratio": ratio, "threshold": SERIOUS_IMBALANCE_RATIO},
                    )
                )


def _class_distribution_rows(
    scans: Sequence[SplitScan], names: Mapping[int, str], declared_names: Mapping[int, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    totals: Counter[int] = Counter()
    for scan in scans:
        totals.update(scan.object_counts)
    for split, counts in [*((scan.split, scan.object_counts) for scan in scans), ("total", totals)]:
        split_total = sum(counts.values())
        for class_id in range(EXPECTED_CLASS_COUNT):
            count = counts[class_id]
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "declared_class_name": declared_names.get(class_id, ""),
                    "object_count": count,
                    "share_percent": (100.0 * count / split_total) if split_total else 0.0,
                    "absent": count == 0,
                }
            )
    return rows


def _split_summary_rows(scans: Sequence[SplitScan], spec: DatasetSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scan in scans:
        counts = Counter(issue["severity"] for issue in scan.issues)
        rows.append(
            {
                "split": scan.split,
                "image_directory": _display_path(scan.image_dir, spec),
                "label_directory": _display_path(scan.label_dir, spec),
                "image_count": scan.image_count,
                "label_count": scan.label_count,
                "matched_label_count": scan.matched_label_count,
                "missing_label_count": scan.missing_label_count,
                "orphan_label_count": scan.orphan_label_count,
                "empty_label_count": scan.empty_label_count,
                "unreadable_image_count": scan.unreadable_image_count,
                "unsupported_file_count": scan.unsupported_file_count,
                "object_count": scan.object_count,
                "critical_issue_count": counts["critical"],
                "warning_count": counts["warning"],
            }
        )
    return rows


def _write_distribution_plot(
    path: Path, rows: Sequence[Mapping[str, Any]], names: Mapping[int, str]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise StageError("Matplotlib and NumPy are required to write the audit plot") from exc

    split_names = ("train", "val", "test")
    count_lookup = {
        (str(row["split"]), int(row["class_id"])): int(row["object_count"]) for row in rows
    }
    positions = np.arange(EXPECTED_CLASS_COUNT)
    width = 0.25
    figure, axis = plt.subplots(figsize=(18, 8), constrained_layout=True)
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for index, (split, color) in enumerate(zip(split_names, colors, strict=True)):
        counts = [
            count_lookup.get((split, class_id), 0) for class_id in range(EXPECTED_CLASS_COUNT)
        ]
        axis.bar(positions + (index - 1) * width, counts, width, label=split, color=color)
    axis.set_title("EcoCAR object distribution by split")
    axis.set_xlabel("Class")
    axis.set_ylabel("Labeled objects")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{class_id}: {names[class_id]}" for class_id in range(EXPECTED_CLASS_COUNT)],
        rotation=40,
        ha="right",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Split")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_markdown(
    path: Path,
    report: Mapping[str, Any],
    class_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
) -> None:
    summary = report["summary"]
    lines = [
        "# EcoCAR dataset and environment audit",
        "",
        f"- Status: **{_markdown_escape(report['status'])}**",
        f"- Generated: `{_markdown_escape(report['generated_at'])}`",
        f"- Dataset fingerprint (SHA-256): `{_markdown_escape(report['dataset']['fingerprint'])}`",
        f"- Critical issues: **{summary['critical_issue_count']}**",
        f"- Warnings: **{summary['warning_count']}**",
        f"- Exact cross-split duplicate groups: **{summary['cross_split_duplicate_group_count']}**",
        "",
        "> **Training gate:** critical issues must be resolved before training. "
        "Exact cross-split duplicates are prominent leakage warnings and must be reviewed.",
        "",
        "## Class-name normalization note",
        "",
        "Class ID 5 is reported as **Lamp-post**. If the supplied YAML used `Lamb-post`, "
        "this is a spelling-only normalization from `Lamb-post` to `Lamp-post`; the numeric "
        "class ID remains 5. The audit never edits the dataset YAML or annotations.",
        "",
        "## Split summary",
        "",
        "| Split | Images | Labels | Matched | Missing labels | Orphan labels | Empty labels | "
        "Unreadable images | Objects |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in split_rows:
        lines.append(
            f"| {row['split']} | {row['image_count']} | {row['label_count']} | "
            f"{row['matched_label_count']} | {row['missing_label_count']} | "
            f"{row['orphan_label_count']} | {row['empty_label_count']} | "
            f"{row['unreadable_image_count']} | {row['object_count']} |"
        )

    total_rows = [row for row in class_rows if row["split"] == "total"]
    lines.extend(
        [
            "",
            "## Overall class distribution",
            "",
            "| ID | Class | Declared name | Objects | Share |",
            "|---:|---|---|---:|---:|",
        ]
    )
    for row in total_rows:
        lines.append(
            f"| {row['class_id']} | {_markdown_escape(row['class_name'])} | "
            f"{_markdown_escape(row['declared_class_name'])} | {row['object_count']} | "
            f"{float(row['share_percent']):.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Checkpoint compatibility",
            "",
            "Checkpoint loading occurs only when the user invokes this audit. A `loaded` row "
            "confirms that Ultralytics identified the checkpoint as a standard detection model.",
            "",
            "| Model | Weights | Status | Task | Parameters | Error |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for row in report["checkpoint_checks"]:
        lines.append(
            f"| {_markdown_escape(row['model_id'])} | {_markdown_escape(row['weights'])} | "
            f"{_markdown_escape(row['status'])} | {_markdown_escape(row.get('task') or '')} | "
            f"{_markdown_escape(row.get('parameter_count') or '')} | "
            f"{_markdown_escape(row.get('error') or '')} |"
        )

    duplicates = report["cross_split_duplicates"]
    lines.extend(["", "## Cross-split duplicate-content check", ""])
    if duplicates:
        lines.extend(
            [
                "**DATA LEAKAGE WARNING:** the following files have byte-identical image content "
                "in more than one split.",
                "",
                "| SHA-256 | Splits | Files |",
                "|---|---|---|",
            ]
        )
        for group in duplicates:
            shown_files = ", ".join(f"{entry['split']}:{entry['path']}" for entry in group["files"])
            lines.append(
                f"| `{group['sha256']}` | {_markdown_escape(', '.join(group['splits']))} | "
                f"{_markdown_escape(shown_files)} |"
            )
    else:
        lines.append(
            "No exact image-content duplicates were found across train, validation, and test."
        )

    lines.extend(
        [
            "",
            "## Runtime environment",
            "",
            "```json",
            json.dumps(report["environment"], indent=2, sort_keys=True),
            "```",
            "",
            "## Issues",
            "",
        ]
    )
    issues = report["issues"]
    if issues:
        lines.extend(
            [
                "| Severity | Code | Split | Path | Line | Message |",
                "|---|---|---|---|---:|---|",
            ]
        )
        for item in issues:
            lines.append(
                f"| {_markdown_escape(item['severity'])} | {_markdown_escape(item['code'])} | "
                f"{_markdown_escape(item['split'])} | {_markdown_escape(item['path'])} | "
                f"{_markdown_escape(item['line'])} | {_markdown_escape(item['message'])} |"
            )
    else:
        lines.append("No issues found.")
    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            "- `audit.json`: full machine-readable audit, environment, fingerprint, "
            "and file manifest",
            "- `class_distribution.csv` and `class_distribution.png`: per-class counts by split",
            "- `split_summary.csv`: split-level image, label, issue, and object counts",
            "- `issues.csv`: actionable issue list",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _dataset_structure_issues(spec: DatasetSpec) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if spec.nc != EXPECTED_CLASS_COUNT:
        issues.append(
            _issue(
                "critical",
                "unexpected_class_count",
                f"EcoCAR requires {EXPECTED_CLASS_COUNT} classes (IDs 0-13), "
                f"but dataset YAML declares {spec.nc}",
                path=_display_path(spec.yaml_path, spec),
            )
        )
    for class_id, expected in EXPECTED_CLASS_NAMES.items():
        declared = spec.names.get(class_id)
        if declared is None:
            continue
        if class_id == 5 and declared.casefold() == "lamb-post":
            issues.append(
                _issue(
                    "warning",
                    "class_name_spelling",
                    "Class ID 5 is declared as 'Lamb-post'; reports use the proposed "
                    "spelling 'Lamp-post' without changing the ID or source YAML",
                    path=_display_path(spec.yaml_path, spec),
                )
            )
        elif declared != expected:
            issues.append(
                _issue(
                    "critical",
                    "unexpected_class_name",
                    f"Class ID {class_id} is named {declared!r}; expected {expected!r}",
                    path=_display_path(spec.yaml_path, spec),
                )
            )
    return issues


def run_audit(context: RunContext) -> dict[str, str]:
    """Execute the complete audit and write all required artifacts.

    This is intended to be called through :func:`road_detection.utils.run_stage`.
    It writes the reports and updates provenance even when critical problems are
    found, then raises :class:`StageError` so subsequent pipeline stages cannot
    train on invalid data.
    """

    output_dir = context.run_dir / "data_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = load_dataset_spec(context.dataset_yaml)
    environment = audit_environment(context)
    checkpoint_checks, checkpoint_issues = audit_checkpoints(context)
    issues = (
        _dataset_structure_issues(spec)
        + _environment_issues(context, environment)
        + checkpoint_issues
    )

    # The project contract fixes IDs to 0-13.  Continue with that domain even if
    # nc itself is wrong so the audit can still produce a complete report.
    scan_nc = EXPECTED_CLASS_COUNT
    scans = [
        _scan_split(split, spec.split(split), spec, scan_nc) for split in ("train", "val", "test")
    ]
    for scan in scans:
        issues.extend(scan.issues)

    display_names = {
        class_id: _canonical_name(class_id, spec.names.get(class_id, expected))
        for class_id, expected in EXPECTED_CLASS_NAMES.items()
    }
    duplicates, duplicate_issues = _cross_split_duplicates(scans, spec)
    issues.extend(duplicate_issues)
    _add_distribution_warnings(scans, display_names, issues)
    fingerprint, file_manifest = _dataset_fingerprint(spec, scans, issues)

    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    issues.sort(
        key=lambda item: (
            severity_rank.get(str(item["severity"]), 99),
            str(item["split"]),
            str(item["path"]),
            str(item["line"]),
            str(item["code"]),
        )
    )
    class_rows = _class_distribution_rows(scans, display_names, spec.names)
    split_rows = _split_summary_rows(scans, spec)

    class_csv = output_dir / "class_distribution.csv"
    split_csv = output_dir / "split_summary.csv"
    issues_csv = output_dir / "issues.csv"
    plot_path = output_dir / "class_distribution.png"
    audit_json = output_dir / "audit.json"
    audit_markdown = output_dir / "audit.md"
    write_csv(
        class_csv,
        class_rows,
        fieldnames=(
            "split",
            "class_id",
            "class_name",
            "declared_class_name",
            "object_count",
            "share_percent",
            "absent",
        ),
    )
    write_csv(
        split_csv,
        split_rows,
        fieldnames=(
            "split",
            "image_directory",
            "label_directory",
            "image_count",
            "label_count",
            "matched_label_count",
            "missing_label_count",
            "orphan_label_count",
            "empty_label_count",
            "unreadable_image_count",
            "unsupported_file_count",
            "object_count",
            "critical_issue_count",
            "warning_count",
        ),
    )
    try:
        _write_distribution_plot(plot_path, class_rows, display_names)
    except StageError as exc:
        issues.append(_issue("critical", "plot_generation_failed", str(exc), path=plot_path.name))

    counts = Counter(item["severity"] for item in issues)
    status = (
        "failed"
        if counts["critical"]
        else ("passed_with_warnings" if counts["warning"] else "passed")
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": status,
        "dataset": {
            "yaml": _display_path(spec.yaml_path, spec),
            "root": _display_path(spec.root, spec),
            "declared_nc": spec.nc,
            "declared_names": dict(spec.names),
            "report_names": display_names,
            "splits": {name: _display_path(path, spec) for name, path in spec.splits.items()},
            "metadata": _sanitize_payload(spec.metadata),
            "fingerprint_algorithm": "sha256 over sorted file-manifest records",
            "fingerprint": fingerprint,
        },
        "class_name_note": {
            "class_id": 5,
            "supplied_name": spec.names.get(5),
            "report_name": "Lamp-post",
            "source_modified": False,
            "explanation": "Lamb-post -> Lamp-post is spelling-only; numeric class ID remains 5.",
        },
        "environment": environment,
        "checkpoint_checks": checkpoint_checks,
        "summary": {
            "critical_issue_count": counts["critical"],
            "warning_count": counts["warning"],
            "info_count": counts["info"],
            "cross_split_duplicate_group_count": len(duplicates),
            "image_count": sum(scan.image_count for scan in scans),
            "label_count": sum(scan.label_count for scan in scans),
            "object_count": sum(scan.object_count for scan in scans),
        },
        "split_summary": split_rows,
        "class_distribution": class_rows,
        "cross_split_duplicates": duplicates,
        "issues": issues,
        "file_manifest": file_manifest,
    }
    write_json(audit_json, report)
    write_csv(
        issues_csv,
        issues,
        fieldnames=("severity", "code", "split", "path", "line", "message"),
    )
    _write_markdown(audit_markdown, report, class_rows, split_rows)

    manifest = context.manifest
    dataset_manifest = dict(manifest.get("dataset", {}))
    dataset_manifest.update(
        {
            "fingerprint": fingerprint,
            "fingerprint_algorithm": "sha256 over sorted file-manifest records",
            "file_count": len(file_manifest),
            "audited_at": report["generated_at"],
            "audit_report": relative_to(audit_json, context.run_dir),
            "split_summary": {
                row["split"]: {
                    "images": row["image_count"],
                    "labels": row["label_count"],
                    "objects": row["object_count"],
                }
                for row in split_rows
            },
            "cross_split_duplicate_group_count": len(duplicates),
        }
    )
    context.update_manifest(
        environment=environment,
        dataset=dataset_manifest,
        models=checkpoint_checks,
    )
    environment_lines = [f"{name}: {value}" for name, value in flatten_dict("", environment)]
    (context.run_dir / "environment.txt").write_text(
        "\n".join(environment_lines) + "\n", encoding="utf-8"
    )

    outputs = {
        "audit_json": relative_to(audit_json, context.run_dir),
        "audit_markdown": relative_to(audit_markdown, context.run_dir),
        "class_distribution_csv": relative_to(class_csv, context.run_dir),
        "class_distribution_plot": relative_to(plot_path, context.run_dir),
        "split_summary_csv": relative_to(split_csv, context.run_dir),
        "issues_csv": relative_to(issues_csv, context.run_dir),
    }
    if counts["critical"]:
        raise StageError(
            f"Audit found {counts['critical']} critical issue(s); see "
            f"{relative_to(audit_markdown, context.run_dir)} before training"
        )
    verify_dataset_unchanged(context)
    runtime_yaml = _prepare_runtime_dataset(context, spec)
    LOGGER.info(
        "Audit completed with %d warning(s); dataset fingerprint %s; runtime dataset %s",
        counts["warning"],
        fingerprint,
        relative_to(runtime_yaml, context.run_dir),
    )
    return outputs


# A short alias reads naturally from orchestration code.
audit_dataset = run_audit


__all__ = [
    "LabelValidationError",
    "YoloLabel",
    "audit_checkpoints",
    "audit_dataset",
    "audit_environment",
    "parse_label_line",
    "parse_yolo_label_row",
    "resolve_label_directory",
    "run_audit",
    "verify_dataset_unchanged",
    "validate_label_line",
    "validate_label_row",
]
