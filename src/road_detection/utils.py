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
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import PipelineConfig, load_dataset_spec, write_resolved_dataset_yaml

LOGGER = logging.getLogger("road_detection")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class PipelineError(RuntimeError):
    """Base exception for expected pipeline failures."""


class StageError(PipelineError):
    """Raised when a pipeline stage cannot complete safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def recorded_prediction_source(context: RunContext) -> Path | None:
    """Return a previously persisted CLI prediction source, when present."""

    raw = context.manifest.get("cli_overrides", {}).get("prediction.source")
    if not isinstance(raw, str) or not raw:
        return None
    value = Path(raw).expanduser()
    return (
        (context.config.project_root / value).resolve()
        if not value.is_absolute()
        else value.resolve()
    )


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


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None
) -> None:
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
        failed = [
            name for name, row in payload.get("stages", {}).items() if row.get("status") == "failed"
        ]
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
    return sorted(
        path.name for path in root.iterdir() if path.is_dir() and (path / "manifest.json").is_file()
    )


def create_run(config: PipelineConfig, run_id: str | None = None) -> RunContext:
    run_id = run_id or new_run_id()
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise StageError(
            "run id must start with a letter/number and contain only letters, "
            "numbers, dots, dashes, and underscores"
        )
    run_dir = _experiment_dir(config) / run_id
    if run_dir.exists():
        raise StageError(f"Run already exists: {run_dir}. Resume it with --run-id {run_id}")
    for directory in (
        "logs",
        "data_audit",
        "smoke_test",
        "models",
        "comparison",
        "test",
        "predictions",
    ):
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
        "configuration": {
            "source": relative_to(config.config_path, config.project_root),
            "source_sha256": sha256_file(config.config_path),
            "resolved": "resolved_config.yaml",
            "resolved_sha256": sha256_file(resolved_config_path),
        },
        "git": git_provenance(config.project_root),
        "dataset": {
            "yaml": relative_to(config.dataset.yaml, config.project_root),
            "source_descriptor_sha256": sha256_file(config.dataset.yaml),
            "resolved_yaml": relative_to(resolved_dataset, run_dir),
            "fingerprint": None,
        },
        "models": [
            {
                "id": model.id,
                "weights": model.weights,
                "batch": model.batch or config.training.batch,
            }
            for model in config.models
        ],
        "seed": config.experiment.seed,
        "environment": basic_environment(),
        "resolved_config": "resolved_config.yaml",
        "cli_overrides": {},
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
    configuration = manifest.get("configuration", {})
    expected_config_hash = configuration.get("source_sha256")
    current_config_hash = sha256_file(config.config_path)
    if expected_config_hash and expected_config_hash != current_config_hash:
        raise StageError(
            "Configuration drift detected: the experiment YAML changed after this run "
            "was created. Start a new audit/run instead of mixing artifacts."
        )
    expected_dataset_hash = manifest.get("dataset", {}).get("source_descriptor_sha256")
    current_dataset_hash = sha256_file(config.dataset.yaml)
    if expected_dataset_hash and expected_dataset_hash != current_dataset_hash:
        raise StageError(
            "Dataset descriptor drift detected after run creation. Start a new audit/run."
        )

    resolved_config_path = run_dir / "resolved_config.yaml"
    expected_resolved_hash = configuration.get("resolved_sha256")
    if (
        not isinstance(expected_resolved_hash, str)
        or not resolved_config_path.is_file()
        or sha256_file(resolved_config_path) != expected_resolved_hash
    ):
        raise StageError("Frozen resolved configuration changed after run creation")
    try:
        saved = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageError(f"Could not validate frozen resolved configuration: {exc}") from exc
    current = config.to_dict()
    if isinstance(saved, dict):
        # The source-YAML hashes above protect the configured checkpoint names.
        # Normalize these fields because a bare registry checkpoint can become a
        # project-local path after audit downloads it.
        for index, model in enumerate(config.models):
            for payload in (saved, current):
                rows = payload.get("models")
                if isinstance(rows, list) and index < len(rows) and isinstance(rows[index], dict):
                    rows[index]["weights"] = model.weights
        if manifest.get("cli_overrides", {}).get("prediction.source") is not None:
            saved.setdefault("prediction", {})["source"] = current.get("prediction", {}).get(
                "source"
            )
    if saved != current:
        raise StageError(
            "Resolved configuration drift detected. Start a new audit/run rather than "
            "combining incompatible settings."
        )
    return RunContext(config, selected, run_dir, manifest_path, dataset_yaml)


def record_prediction_source_override(context: RunContext, source: str | Path) -> Path:
    """Persist the exact CLI prediction-source override in run provenance."""

    value = Path(source).expanduser()
    resolved = (
        (context.config.project_root / value).resolve()
        if not value.is_absolute()
        else value.resolve()
    )
    resolved_config_path = context.run_dir / "resolved_config.yaml"
    try:
        payload = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise StageError(f"Could not update resolved configuration: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("prediction"), dict):
        raise StageError("Frozen resolved configuration has no prediction section")
    payload["prediction"]["source"] = str(resolved)
    temporary = resolved_config_path.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(resolved_config_path)

    manifest = context.manifest
    manifest.setdefault("cli_overrides", {})["prediction.source"] = str(resolved)
    manifest.setdefault("configuration", {})["resolved_sha256"] = sha256_file(resolved_config_path)
    manifest["updated_at"] = utc_now()
    write_json(context.manifest_path, manifest)
    return resolved


def require_completed(context: RunContext, *stages: str) -> None:
    missing = [stage for stage in stages if context.stage_status(stage) != "completed"]
    if missing:
        raise StageError(f"Required stage(s) not completed: {', '.join(missing)}")


def audited_source_weights(context: RunContext, model_id: str, configured: str) -> str:
    """Return the audit-frozen source checkpoint and verify its hash when local."""

    model_rows = context.manifest.get("models", [])
    record = next(
        (
            row
            for row in model_rows
            if isinstance(row, Mapping) and row.get("model_id", row.get("id")) == model_id
        ),
        None,
    )
    if not isinstance(record, Mapping) or record.get("status") != "loaded":
        raise StageError(f"Model {model_id!r} has no successful checkpoint audit record")
    checkpoint_value = record.get("checkpoint_path")
    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (context.config.project_root / checkpoint).resolve()
        if not checkpoint.is_file():
            raise StageError(f"Audited source checkpoint is missing: {checkpoint}")
        expected_hash = record.get("checkpoint_sha256")
        if expected_hash and sha256_file(checkpoint) != expected_hash:
            raise StageError(
                f"Audited source checkpoint changed after audit: {checkpoint}. "
                "Start a new audit/run."
            )
        return str(checkpoint)
    from .config import resolve_weights

    return resolve_weights(configured, context.config.project_root)


def runtime_dataset_yaml(context: RunContext) -> Path:
    """Return the audit-created artifact-local dataset descriptor."""

    dataset = context.manifest.get("dataset", {})
    value = dataset.get("runtime_yaml")
    if not isinstance(value, str) or not value:
        raise StageError(
            "The audit did not create an artifact-local runtime dataset. Rerun audit in a new run."
        )
    path = Path(value)
    path = path if path.is_absolute() else context.run_dir / path
    if not path.is_file():
        raise StageError(f"Artifact-local runtime dataset descriptor is missing: {path}")

    manifest_value = dataset.get("runtime_manifest")
    expected_manifest_hash = dataset.get("runtime_manifest_sha256")
    if not isinstance(manifest_value, str) or not isinstance(expected_manifest_hash, str):
        raise StageError("Artifact-local runtime dataset integrity record is missing")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = context.run_dir / manifest_path
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_hash:
        raise StageError("Artifact-local runtime dataset manifest changed after audit")
    integrity = read_json(manifest_path)
    records = integrity.get("records")
    if not isinstance(records, list) or not records:
        raise StageError("Artifact-local runtime dataset manifest has no file records")
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise StageError("Artifact-local runtime dataset manifest has an invalid record")
        declared = Path(str(raw_record.get("path", "")))
        if declared.is_absolute() or ".." in declared.parts:
            raise StageError("Artifact-local runtime dataset manifest has an unsafe path")
        artifact = context.run_dir / declared
        kind = raw_record.get("kind")
        if kind == "image_symlink":
            expected_target = Path(str(raw_record.get("target", "")))
            if not artifact.is_symlink() or artifact.resolve() != expected_target.resolve():
                raise StageError(f"Runtime image link changed after audit: {artifact}")
            expected_hash = raw_record.get("sha256")
            try:
                actual_hash = sha256_file(artifact)
            except OSError as exc:
                raise StageError(f"Runtime image link is unreadable: {artifact}") from exc
            if not isinstance(expected_hash, str) or actual_hash != expected_hash:
                raise StageError(f"Runtime image content changed after audit: {artifact}")
        elif kind in {"label_copy", "descriptor"}:
            expected_hash = raw_record.get("sha256")
            if (
                artifact.is_symlink()
                or not artifact.is_file()
                or not isinstance(expected_hash, str)
                or sha256_file(artifact) != expected_hash
            ):
                raise StageError(f"Runtime dataset file changed after audit: {artifact}")
        else:
            raise StageError(f"Unknown runtime dataset record kind: {kind!r}")
    return path.resolve()


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
            "Ultralytics is not installed in this environment. Install the project "
            "dependencies first."
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
