"""Shared run management, provenance, logging, and serialization helpers."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .config import PipelineConfig, load_dataset_spec, write_resolved_dataset_yaml

LOGGER = logging.getLogger("road_detection")


class PipelineError(RuntimeError):
    """Base exception for expected pipeline failures."""


class StageError(PipelineError):
    """Raised when a pipeline stage cannot complete safely."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def json_safe(value: Any) -> Any:
    """Convert common scientific/path values to strict JSON-compatible data."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StageError(f"Expected a JSON object in {path}")
    return payload


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        ordered: list[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fieldnames = ordered
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(row.get(key)) for key in fieldnames})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def setup_logging(run_dir: Path, verbose: bool = False) -> None:
    run_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("road_detection")
    logger.handlers.clear()
    logger.setLevel(level)
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "logs" / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)


def git_provenance(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(project_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("ultralytics", "torch", "numpy", "PyYAML", "Pillow", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def basic_environment() -> dict[str, Any]:
    return {
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "dependencies": dependency_versions(),
    }


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed available random-number generators without requiring torch at import time."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            LOGGER.warning("Could not fully enable deterministic torch algorithms")


def ultralytics_device(device: str | int) -> str | int | None:
    return None if device == "auto" else device


def inference_half_enabled(device: str | int, amp_requested: bool) -> bool:
    """Use FP16 inference only when the selected runtime can safely support it."""

    if not amp_requested:
        return False
    normalized = str(device).strip().lower()
    if normalized in {"cpu", "mps"}:
        return False
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@dataclass(frozen=True)
class RunContext:
    config: PipelineConfig
    run_id: str
    run_dir: Path
    manifest_path: Path
    dataset_yaml: Path

    @property
    def manifest(self) -> dict[str, Any]:
        return read_json(self.manifest_path)

    def update_manifest(self, **updates: Any) -> None:
        payload = self.manifest
        payload.update(updates)
        payload["updated_at"] = utc_now()
        write_json(self.manifest_path, payload)

    def stage_status(self, stage: str) -> str | None:
        return self.manifest.get("stages", {}).get(stage, {}).get("status")

    def begin_stage(self, stage: str, *, force: bool = False) -> bool:
        payload = self.manifest
        stages = payload.setdefault("stages", {})
        current = stages.get(stage, {})
        if current.get("status") == "completed" and not force:
            LOGGER.info("Stage %s already completed; skipping (use --force to rerun)", stage)
            return False
        stages[stage] = {"status": "running", "started_at": utc_now(), "error": None}
        payload["status"] = "running"
        payload["updated_at"] = utc_now()
        write_json(self.manifest_path, payload)
        return True

    def complete_stage(self, stage: str, outputs: Mapping[str, Any] | None = None) -> None:
        payload = self.manifest
        record = payload.setdefault("stages", {}).setdefault(stage, {})
        record.update({"status": "completed", "completed_at": utc_now(), "error": None})
        if outputs:
            record["outputs"] = json_safe(outputs)
            payload.setdefault("results", {}).update(json_safe(outputs))
        payload["updated_at"] = utc_now()
        write_json(self.manifest_path, payload)

    def fail_stage(self, stage: str, error: BaseException | str) -> None:
        payload = self.manifest
        record = payload.setdefault("stages", {}).setdefault(stage, {})
        record.update({"status": "failed", "completed_at": utc_now(), "error": str(error)})
        payload["status"] = "failed"
        payload["updated_at"] = utc_now()
        write_json(self.manifest_path, payload)

    def finalize(self) -> None:
        payload = self.manifest
        failed = [name for name, row in payload.get("stages", {}).items() if row.get("status") == "failed"]
        payload["status"] = "failed" if failed else "completed"
        payload["completed_at"] = utc_now()
        payload["updated_at"] = utc_now()
        write_json(self.manifest_path, payload)


def _experiment_dir(config: PipelineConfig) -> Path:
    return config.experiment.output_root / config.experiment.name


def list_run_ids(config: PipelineConfig) -> list[str]:
    root = _experiment_dir(config)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file())


def create_run(config: PipelineConfig, run_id: str | None = None) -> RunContext:
    run_id = run_id or new_run_id()
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in run_id):
        raise StageError("run id may contain only letters, numbers, dots, dashes, and underscores")
    run_dir = _experiment_dir(config) / run_id
    if run_dir.exists():
        raise StageError(f"Run already exists: {run_dir}. Resume it with --run-id {run_id}")
    for directory in ("logs", "data_audit", "smoke_test", "models", "comparison", "test", "predictions"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    spec = load_dataset_spec(config.dataset.yaml)
    resolved_dataset = write_resolved_dataset_yaml(spec, run_dir / "resolved_dataset.yaml")
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "experiment_name": config.experiment.name,
        "run_id": run_id,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "completed_at": None,
        "status": "created",
        "git": git_provenance(config.project_root),
        "dataset": {
            "yaml": relative_to(config.dataset.yaml, config.project_root),
            "resolved_yaml": relative_to(resolved_dataset, run_dir),
            "fingerprint": None,
        },
        "models": [
            {"id": model.id, "weights": model.weights, "batch": model.batch or config.training.batch}
            for model in config.models
        ],
        "seed": config.experiment.seed,
        "environment": basic_environment(),
        "resolved_config": "resolved_config.yaml",
        "stages": {},
        "results": {},
    }
    write_json(manifest_path, manifest)
    environment_lines = [
        f"{key}: {value}" for key, value in basic_environment().items() if key != "dependencies"
    ]
    environment_lines.extend(
        f"{package}: {version}" for package, version in basic_environment()["dependencies"].items()
    )
    (run_dir / "environment.txt").write_text("\n".join(environment_lines) + "\n", encoding="utf-8")
    return RunContext(config, run_id, run_dir, manifest_path, resolved_dataset)


def resume_run(config: PipelineConfig, run_id: str | None = None) -> RunContext:
    available = list_run_ids(config)
    selected = run_id or (available[-1] if available else None)
    if selected is None:
        raise StageError("No existing run found. Start with the audit command or pass --run-id")
    run_dir = _experiment_dir(config) / selected
    manifest_path = run_dir / "manifest.json"
    dataset_yaml = run_dir / "resolved_dataset.yaml"
    if not manifest_path.is_file() or not dataset_yaml.is_file():
        raise StageError(f"Run {selected!r} is incomplete or not a pipeline run")
    manifest = read_json(manifest_path)
    if manifest.get("experiment_name") != config.experiment.name:
        raise StageError(f"Run {selected!r} belongs to a different experiment")
    return RunContext(config, selected, run_dir, manifest_path, dataset_yaml)


def require_completed(context: RunContext, *stages: str) -> None:
    missing = [stage for stage in stages if context.stage_status(stage) != "completed"]
    if missing:
        raise StageError(f"Required stage(s) not completed: {', '.join(missing)}")


def run_stage(
    context: RunContext,
    name: str,
    function: Any,
    *,
    force: bool = False,
    **kwargs: Any,
) -> Any:
    """Execute a stage with consistent manifest state transitions."""

    if not context.begin_stage(name, force=force):
        return None
    try:
        result = function(context, **kwargs)
    except BaseException as exc:
        context.fail_stage(name, exc)
        raise
    outputs = result if isinstance(result, Mapping) else None
    context.complete_stage(name, outputs)
    return result


def import_ultralytics_yolo() -> Any:
    """Import YOLO lazily so configuration/tests never initialize model runtimes."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise StageError(
            "Ultralytics is not installed in this environment. Install the project dependencies first."
        ) from exc
    return YOLO


def supported_images(directory: Path) -> list[Path]:
    extensions = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def flatten_dict(prefix: str, payload: Mapping[str, Any]) -> Iterable[tuple[str, Any]]:
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from flatten_dict(name, value)
        else:
            yield name, value
