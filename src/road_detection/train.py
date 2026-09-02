"""Ultralytics smoke testing and reproducible candidate training.

Ultralytics is imported only inside the public stage functions.  Importing this
module is therefore safe for configuration validation and unit tests: it never
loads a checkpoint, inspects the real dataset, or initializes an accelerator.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import ModelSettings, load_dataset_spec
from .data_audit import verify_dataset_unchanged
from .utils import (
    LOGGER,
    RunContext,
    StageError,
    audited_source_weights,
    import_ultralytics_yolo,
    read_json,
    relative_to,
    require_completed,
    runtime_dataset_yaml,
    seed_everything,
    sha256_file,
    ultralytics_device,
    utc_now,
    write_json,
)

_IMAGE_EXTENSIONS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _portable_path(path: Path, context: RunContext) -> str:
    """Prefer a run-relative artifact path while retaining external paths."""

    return relative_to(path, context.run_dir)


def _record_path(value: Any, context: RunContext) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else context.run_dir / path


def _next_attempt(root: Path, prefix: str) -> tuple[str, Path]:
    """Return an unused deterministic attempt name without creating it."""

    root.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10_000):
        name = f"{prefix}_{number:03d}"
        destination = root / name
        if not destination.exists():
            return name, destination
    raise StageError(f"Could not allocate a new {prefix} output directory below {root}")


def _task_from_model(model: Any) -> str | None:
    task = getattr(model, "task", None)
    if isinstance(task, str) and task:
        return task

    overrides = getattr(model, "overrides", None)
    if isinstance(overrides, Mapping):
        task = overrides.get("task")
        if isinstance(task, str) and task:
            return task

    inner_model = getattr(model, "model", None)
    task = getattr(inner_model, "task", None)
    if isinstance(task, str) and task:
        return task
    args = getattr(inner_model, "args", None)
    if isinstance(args, Mapping):
        task = args.get("task")
        if isinstance(task, str) and task:
            return task
    return None


def _require_detection_model(model: Any, weights: str) -> None:
    task = _task_from_model(model)
    if task != "detect":
        shown = repr(task) if task is not None else "unknown"
        raise StageError(
            f"Checkpoint {weights!r} reports task {shown}; only standard axis-aligned "
            "YOLO detection checkpoints are supported"
        )


def _training_arguments(
    context: RunContext,
    candidate: ModelSettings,
    *,
    project: Path,
    name: str,
    epochs: int,
    fraction: float | None = None,
) -> dict[str, Any]:
    training = context.config.training
    arguments: dict[str, Any] = {
        "data": str(runtime_dataset_yaml(context)),
        "epochs": epochs,
        "imgsz": training.imgsz,
        "batch": candidate.batch or training.batch,
        "device": ultralytics_device(training.device),
        "workers": training.workers,
        "patience": training.patience,
        "pretrained": training.pretrained,
        "deterministic": training.deterministic,
        "seed": context.config.experiment.seed,
        "amp": training.amp,
        "cache": training.cache,
        "resume": training.resume,
        "val": True,
        "save": True,
        "plots": True,
        "project": str(project),
        "name": name,
        # Every invocation has a fresh attempt name.  Keeping exist_ok false is
        # a second guard against silently replacing an earlier Ultralytics run.
        "exist_ok": False,
    }
    if fraction is not None:
        arguments["fraction"] = fraction
    return arguments


def _as_plain_value(value: Any, *, depth: int = 0) -> Any:
    """Convert trainer arguments/metrics without retaining runtime objects."""

    if depth > 8:
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _as_plain_value(item, depth=depth + 1)
            for key, item in value.items()
            if not any(token in str(key).lower() for token in ("password", "secret", "token"))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_as_plain_value(item, depth=depth + 1) for item in value]
    if hasattr(value, "item"):
        try:
            return _as_plain_value(value.item(), depth=depth + 1)
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _trainer_arguments(trainer: Any, fallback: Mapping[str, Any]) -> dict[str, Any]:
    args = getattr(trainer, "args", None)
    if isinstance(args, Mapping):
        values = dict(args)
    elif args is not None and hasattr(args, "__dict__"):
        values = vars(args)
    else:
        values = dict(fallback)
    return _as_plain_value(values)


def _read_result_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error) as exc:
        raise StageError(f"Could not read Ultralytics results CSV {path}: {exc}") from exc
    if not rows:
        raise StageError(f"Ultralytics results CSV contains no epoch rows: {path}")
    return rows


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _best_epoch(rows: Sequence[Mapping[str, Any]], trainer: Any) -> int | None:
    stopper = getattr(trainer, "stopper", None)
    trainer_best = getattr(trainer, "best_epoch", None)
    stopper_best = getattr(stopper, "best_epoch", None)
    for value in (trainer_best, stopper_best):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            # Ultralytics trainer/stopper epochs are zero-based internally;
            # reports use a human-readable one-based epoch number.
            return value + 1

    if not rows:
        return None
    map_key = next(
        (key for key in rows[0] if "map50-95" in key.lower().replace(" ", "")),
        None,
    )
    if map_key is not None:
        scored = [(_float_or_none(row.get(map_key)), index) for index, row in enumerate(rows)]
        finite = [(score, index) for score, index in scored if score is not None]
        if finite:
            row = rows[max(finite, key=lambda item: item[0])[1]]
            epoch = _float_or_none(row.get("epoch"))
            if epoch is not None:
                first_epoch = _float_or_none(rows[0].get("epoch"))
                return int(epoch) + 1 if first_epoch == 0 else int(epoch)
            return max(finite, key=lambda item: item[0])[1] + 1

    epoch = _float_or_none(rows[-1].get("epoch"))
    if epoch is not None:
        first_epoch = _float_or_none(rows[0].get("epoch"))
        return int(epoch) + 1 if first_epoch == 0 else int(epoch)
    return len(rows)


def _metrics_from_training(result: Any, trainer: Any) -> dict[str, Any]:
    sources = (
        result if isinstance(result, Mapping) else None,
        getattr(result, "results_dict", None),
        getattr(trainer, "metrics", None),
    )
    for source in sources:
        if isinstance(source, Mapping):
            return _as_plain_value(source)
    return {}


def _trainer_artifacts(
    model: Any,
    expected_run_dir: Path,
) -> tuple[Path, Path, Path, Path, Any]:
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise StageError("Ultralytics did not expose a trainer after training")

    save_dir_raw = getattr(trainer, "save_dir", None)
    run_dir = Path(save_dir_raw).resolve() if save_dir_raw else expected_run_dir.resolve()
    if run_dir != expected_run_dir.resolve():
        raise StageError(
            f"Ultralytics wrote to unexpected directory {run_dir}; expected "
            f"the isolated attempt directory {expected_run_dir.resolve()}"
        )
    best_raw = getattr(trainer, "best", None)
    last_raw = getattr(trainer, "last", None)
    best = Path(best_raw).resolve() if best_raw else run_dir / "weights" / "best.pt"
    last = Path(last_raw).resolve() if last_raw else run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"

    missing = [
        label
        for label, path in (("best.pt", best), ("last.pt", last), ("results.csv", results_csv))
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise StageError(
            f"Ultralytics training did not produce usable {', '.join(missing)} below {run_dir}"
        )
    return run_dir, best, last, results_csv, trainer


def _attempt_history(existing: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not existing:
        return []
    attempts = existing.get("attempts")
    if isinstance(attempts, list):
        return [dict(row) for row in attempts if isinstance(row, Mapping)]
    previous = {key: value for key, value in existing.items() if key != "attempts"}
    return [previous]


def _write_summary(
    summary_path: Path,
    attempt_path: Path,
    record: Mapping[str, Any],
    existing: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attempt_record = dict(record)
    write_json(attempt_path, attempt_record)
    summary = dict(attempt_record)
    summary["attempts"] = [*_attempt_history(existing), attempt_record]
    write_json(summary_path, summary)
    return summary


def _find_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in _IMAGE_EXTENSIONS:
        return [source]
    if not source.is_dir():
        return []
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _prediction_directory(model: Any, fallback: Path) -> Path:
    predictor = getattr(model, "predictor", None)
    save_dir = getattr(predictor, "save_dir", None)
    return Path(save_dir).resolve() if save_dir else fallback.resolve()


def _completed_candidate_record(
    context: RunContext,
    summary_path: Path,
) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    record = read_json(summary_path)
    if record.get("status") != "completed":
        return None
    best = _record_path(record.get("best_checkpoint"), context)
    results = _record_path(record.get("results_csv"), context)
    if best is None or not best.is_file() or results is None or not results.is_file():
        raise StageError(
            f"Completed training record has missing artifacts: {summary_path}. "
            "Start a new run instead of silently replacing it."
        )
    expected_hash = record.get("best_checkpoint_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise StageError(f"Completed training record has no frozen best.pt hash: {summary_path}")
    if sha256_file(best) != expected_hash:
        raise StageError(f"Completed best.pt changed after training: {best}. Start a new run.")
    return record


def _smoke_candidate(context: RunContext) -> tuple[ModelSettings, str]:
    """Choose the smallest audited checkpoint by parameters, size, then config order."""

    audit_rows = {
        str(row.get("model_id", row.get("id"))): row
        for row in context.manifest.get("models", [])
        if isinstance(row, Mapping)
    }

    def positive_number(value: Any) -> float | None:
        parsed = _float_or_none(value)
        return parsed if parsed is not None and parsed > 0 else None

    ranked: list[tuple[tuple[Any, ...], ModelSettings]] = []
    for index, candidate in enumerate(context.config.models):
        row = audit_rows.get(candidate.id, {})
        parameters = positive_number(row.get("parameter_count"))
        size = positive_number(row.get("checkpoint_size_bytes"))
        rank = (
            parameters is None,
            parameters if parameters is not None else math.inf,
            size is None,
            size if size is not None else math.inf,
            index,
        )
        ranked.append((rank, candidate))
    _, winner = min(ranked, key=lambda item: item[0])
    winner_row = audit_rows.get(winner.id, {})
    reason = (
        "smallest audited candidate by parameter count, then checkpoint bytes, "
        "then configuration order"
    )
    if positive_number(winner_row.get("parameter_count")) is None:
        reason += " (parameter count unavailable for at least the selected candidate)"
    return winner, reason


def run_smoke_test(context: RunContext) -> dict[str, Any]:
    """Run a one-epoch smoke test with the smallest audited candidate.

    Audit-measured parameter count chooses the smallest model, with checkpoint
    bytes and configuration order as deterministic fallbacks. This function is
    intentionally called only by an explicit runtime CLI stage.
    """

    require_completed(context, "audit")
    verify_dataset_unchanged(context)
    config = context.config
    candidate, selection_reason = _smoke_candidate(context)
    smoke_root = context.run_dir / "smoke_test"
    summary_path = smoke_root / "smoke_test.json"
    existing = read_json(summary_path) if summary_path.is_file() else None
    attempt_name, expected_run_dir = _next_attempt(smoke_root, "smoke")
    started_at = utc_now()
    started = time.perf_counter()
    attempt_record_path = expected_run_dir / "smoke_test.json"
    requested: dict[str, Any] = {}

    try:
        seed_everything(config.experiment.seed, config.training.deterministic)
        YOLO = import_ultralytics_yolo()
        source_weights = audited_source_weights(context, candidate.id, candidate.weights)
        model = YOLO(source_weights)
        _require_detection_model(model, source_weights)

        requested = _training_arguments(
            context,
            candidate,
            project=smoke_root,
            name=attempt_name,
            epochs=config.smoke_test.epochs,
            fraction=config.smoke_test.fraction,
        )
        training_started = time.perf_counter()
        result = model.train(**requested)
        training_seconds = time.perf_counter() - training_started
        run_dir, best, last, results_csv, trainer = _trainer_artifacts(model, expected_run_dir)
        rows = _read_result_rows(results_csv)

        dataset = load_dataset_spec(runtime_dataset_yaml(context))
        sources = _find_images(dataset.split("val"))[: config.smoke_test.max_predictions]
        if not sources:
            raise StageError(
                "Smoke test could not find a supported validation image for prediction"
            )

        prediction_model = YOLO(str(best))
        _require_detection_model(prediction_model, str(best))
        prediction_name = "prediction"
        prediction_fallback = run_dir / prediction_name
        prediction_source: str | list[str]
        prediction_source = (
            str(sources[0]) if len(sources) == 1 else [str(path) for path in sources]
        )
        prediction_model.predict(
            source=prediction_source,
            imgsz=config.training.imgsz,
            conf=config.prediction.confidence,
            iou=config.evaluation.prediction_iou,
            max_det=config.evaluation.max_detections,
            device=ultralytics_device(config.training.device),
            save=True,
            project=str(run_dir),
            name=prediction_name,
            exist_ok=False,
            verbose=False,
        )
        prediction_dir = _prediction_directory(prediction_model, prediction_fallback)
        saved_predictions = _find_images(prediction_dir)
        if not saved_predictions:
            raise StageError(
                f"Smoke inference returned without saving an annotated image below {prediction_dir}"
            )

        epochs_completed = len(rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "model_id": candidate.id,
            "selection_reason": selection_reason,
            "attempt": attempt_name,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "training_duration_seconds": training_seconds,
            "task": "detect",
            "source_weights": candidate.weights,
            "resolved_source_weights": source_weights,
            "requested_settings": _as_plain_value(requested),
            "actual_settings": _trainer_arguments(trainer, requested),
            "fraction": config.smoke_test.fraction,
            "epochs_completed": epochs_completed,
            "best_epoch": _best_epoch(rows, trainer),
            "best_epoch_basis": "one-based epoch number",
            "early_stopping": epochs_completed < config.smoke_test.epochs,
            "ultralytics_run_dir": _portable_path(run_dir, context),
            "results_csv": _portable_path(results_csv, context),
            "best_checkpoint": _portable_path(best, context),
            "best_checkpoint_sha256": sha256_file(best),
            "best_checkpoint_size_bytes": best.stat().st_size,
            "last_checkpoint": _portable_path(last, context),
            "last_checkpoint_sha256": sha256_file(last),
            "metrics": _metrics_from_training(result, trainer),
            "prediction_sources": [str(path.resolve()) for path in sources],
            "prediction_dir": _portable_path(prediction_dir, context),
            "saved_predictions": [_portable_path(path, context) for path in saved_predictions],
            "error": None,
        }
        summary = _write_summary(
            summary_path,
            run_dir / "smoke_test.json",
            record,
            existing,
        )
        LOGGER.info("Smoke test completed with %s", candidate.id)
        return {
            "smoke_model": candidate.id,
            "smoke_test_record": _portable_path(summary_path, context),
            "smoke_best_checkpoint": summary["best_checkpoint"],
            "smoke_prediction_dir": summary["prediction_dir"],
        }
    except Exception as exc:
        failed_run_dir = expected_run_dir
        failed_run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "status": "failed",
            "model_id": candidate.id,
            "selection_reason": selection_reason,
            "attempt": attempt_name,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "task": "detect",
            "source_weights": candidate.weights,
            "requested_settings": _as_plain_value(requested),
            "ultralytics_run_dir": _portable_path(failed_run_dir, context),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_summary(summary_path, attempt_record_path, record, existing)
        LOGGER.exception("Smoke test failed for %s", candidate.id)
        if isinstance(exc, StageError):
            raise
        raise StageError(f"Smoke test failed for {candidate.id}: {exc}") from exc


def _train_one_candidate(
    context: RunContext,
    candidate: ModelSettings,
    YOLO: Any,
    *,
    retrain_completed: bool,
) -> tuple[dict[str, Any], bool]:
    model_root = context.run_dir / "models" / candidate.id
    model_root.mkdir(parents=True, exist_ok=True)
    summary_path = model_root / "training.json"
    existing = read_json(summary_path) if summary_path.is_file() else None
    if not retrain_completed:
        completed = _completed_candidate_record(context, summary_path)
        if completed is not None:
            LOGGER.info("Candidate %s is already complete; reusing best.pt", candidate.id)
            reused = dict(completed)
            reused["reused"] = True
            return reused, True

    attempt_name, expected_run_dir = _next_attempt(model_root, "train")
    attempt_record_path = expected_run_dir / "training.json"
    started_at = utc_now()
    started = time.perf_counter()
    requested: dict[str, Any] = {}
    source_weights = audited_source_weights(context, candidate.id, candidate.weights)

    try:
        seed_everything(
            context.config.experiment.seed,
            context.config.training.deterministic,
        )
        model = YOLO(source_weights)
        _require_detection_model(model, source_weights)
        requested = _training_arguments(
            context,
            candidate,
            project=model_root,
            name=attempt_name,
            epochs=context.config.training.epochs,
        )
        training_started = time.perf_counter()
        result = model.train(**requested)
        training_seconds = time.perf_counter() - training_started
        run_dir, best, last, results_csv, trainer = _trainer_artifacts(model, expected_run_dir)
        rows = _read_result_rows(results_csv)
        epochs_completed = len(rows)
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": "completed",
            "model_id": candidate.id,
            "attempt": attempt_name,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "training_duration_seconds": training_seconds,
            "task": "detect",
            "source_weights": candidate.weights,
            "resolved_source_weights": source_weights,
            "batch_size": candidate.batch or context.config.training.batch,
            "batch_override": candidate.batch is not None,
            "requested_settings": _as_plain_value(requested),
            "actual_settings": _trainer_arguments(trainer, requested),
            "requested_epochs": context.config.training.epochs,
            "epochs_completed": epochs_completed,
            "best_epoch": _best_epoch(rows, trainer),
            "best_epoch_basis": "one-based epoch number",
            "early_stopping": epochs_completed < context.config.training.epochs,
            "ultralytics_run_dir": _portable_path(run_dir, context),
            "results_csv": _portable_path(results_csv, context),
            "best_checkpoint": _portable_path(best, context),
            "best_checkpoint_sha256": sha256_file(best),
            "best_checkpoint_size_bytes": best.stat().st_size,
            "last_checkpoint": _portable_path(last, context),
            "last_checkpoint_sha256": sha256_file(last),
            "metrics": _metrics_from_training(result, trainer),
            "error": None,
        }
        summary = _write_summary(
            summary_path,
            run_dir / "training.json",
            record,
            existing,
        )
        LOGGER.info(
            "Training completed for %s; downstream stages will use %s",
            candidate.id,
            summary["best_checkpoint"],
        )
        return summary, False
    except Exception as exc:
        failed_run_dir = expected_run_dir
        failed_run_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "status": "failed",
            "model_id": candidate.id,
            "attempt": attempt_name,
            "started_at": started_at,
            "completed_at": utc_now(),
            "duration_seconds": time.perf_counter() - started,
            "task": "detect",
            "source_weights": candidate.weights,
            "resolved_source_weights": source_weights,
            "batch_size": candidate.batch or context.config.training.batch,
            "batch_override": candidate.batch is not None,
            "requested_settings": _as_plain_value(requested),
            "requested_epochs": context.config.training.epochs,
            "ultralytics_run_dir": _portable_path(failed_run_dir, context),
            "error": f"{type(exc).__name__}: {exc}",
        }
        summary = _write_summary(summary_path, attempt_record_path, record, existing)
        LOGGER.exception("Training failed for candidate %s; continuing", candidate.id)
        return summary, False


def train_candidates(
    context: RunContext,
    *,
    retrain_completed: bool = False,
) -> dict[str, Any]:
    """Train all candidates independently and retain each candidate's best.pt.

    Completed candidates are reused by default, which makes a resumed pipeline
    safe after another candidate failed.  An orchestrator may set
    ``retrain_completed=True`` only for an explicit force/retrain request; a new
    numbered output directory is still allocated so prior artifacts survive.
    """

    require_completed(context, "audit", "smoke_test")
    verify_dataset_unchanged(context)
    YOLO = import_ultralytics_yolo()
    records: dict[str, dict[str, Any]] = {}
    reused_models: list[str] = []
    failed_models: list[str] = []

    for candidate in context.config.models:
        record, reused = _train_one_candidate(
            context,
            candidate,
            YOLO,
            retrain_completed=retrain_completed,
        )
        records[candidate.id] = record
        if reused:
            reused_models.append(candidate.id)
        if record.get("status") != "completed":
            failed_models.append(candidate.id)

    training_records = {
        model_id: _portable_path(
            context.run_dir / "models" / model_id / "training.json",
            context,
        )
        for model_id in records
    }
    best_checkpoints = {
        model_id: record["best_checkpoint"]
        for model_id, record in records.items()
        if record.get("status") == "completed" and record.get("best_checkpoint")
    }
    if failed_models:
        details = "; ".join(
            f"{model_id}: {records[model_id].get('error', 'unknown error')}"
            for model_id in failed_models
        )
        raise StageError(
            "One or more candidate trainings failed after all independent candidates "
            f"were attempted ({details}). Inspect: {training_records}"
        )

    return {
        "trained_models": list(records),
        "reused_models": reused_models,
        "failed_models": [],
        "training_records": training_records,
        # Evaluation and comparison must always consume these best checkpoints,
        # never the corresponding last.pt files.
        "best_checkpoints": best_checkpoints,
    }
