"""Validation and frozen-winner test evaluation for trained YOLO detectors."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_dataset_spec
from .data_audit import verify_dataset_unchanged
from .metrics import DetectionFrame, aggregate_detection_coverage
from .utils import (
    RunContext,
    StageError,
    basic_environment,
    import_ultralytics_yolo,
    inference_half_enabled,
    json_safe,
    read_json,
    relative_to,
    require_completed,
    runtime_dataset_yaml,
    sha256_file,
    supported_images,
    ultralytics_device,
    write_csv,
    write_json,
)

LOGGER = logging.getLogger("road_detection")


def _as_array(value: Any, *, dtype: Any = np.float64) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _attribute(obj: Any, dotted_name: str) -> Any:
    current = obj
    for name in dotted_name.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
    return current


def _first_float(obj: Any, *names: str) -> float | None:
    for name in names:
        result = _as_float(_attribute(obj, name))
        if result is not None:
            return result
    return None


def _result_dict(metrics: Any) -> Mapping[str, Any]:
    result = getattr(metrics, "results_dict", None)
    return result if isinstance(result, Mapping) else {}


def _metric_from_result_dict(result: Mapping[str, Any], *tokens: str) -> float | None:
    normalized_tokens = tuple(token.lower().replace(" ", "") for token in tokens)
    normalized_items = [(str(key).lower().replace(" ", ""), value) for key, value in result.items()]
    for token in normalized_tokens:
        for normalized_key, value in normalized_items:
            if normalized_key == token:
                parsed = _as_float(value)
                if parsed is not None:
                    return parsed
    return None


def _extract_metric_rows(
    metrics: Any, class_names: Mapping[int, str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract metrics across Ultralytics releases without relying on one API shape."""

    box = getattr(metrics, "box", None)
    result = _result_dict(metrics)

    def coalesce(primary: float | None, fallback: float | None) -> float | None:
        return primary if primary is not None else fallback

    precision = coalesce(
        _first_float(box, "mp"),
        _metric_from_result_dict(result, "metrics/precision(b)", "metrics/precision"),
    )
    recall = coalesce(
        _first_float(box, "mr"),
        _metric_from_result_dict(result, "metrics/recall(b)", "metrics/recall"),
    )
    map50 = coalesce(
        _first_float(box, "map50"),
        _metric_from_result_dict(result, "metrics/map50(b)", "metrics/map50"),
    )
    map50_95 = coalesce(
        _first_float(box, "map"),
        _metric_from_result_dict(result, "metrics/map50-95(b)", "metrics/map50-95"),
    )

    ap = _as_array(getattr(box, "ap", None))
    if ap.ndim == 1 and ap.size:
        ap = ap.reshape(1, -1)
    map75 = _first_float(box, "map75")
    if map75 is None and ap.ndim == 2 and ap.shape[1] > 5:
        map75 = _as_float(np.mean(ap[:, 5]))
    if map75 is None:
        map75 = _metric_from_result_dict(
            result, "metrics/map75(b)", "metrics/map75", "map75(b)", "map75"
        )

    speed_source = getattr(metrics, "speed", {})
    speed = speed_source if isinstance(speed_source, Mapping) else {}
    aggregate = {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "preprocess_ms_per_image": _as_float(speed.get("preprocess")),
        "inference_ms_per_image": _as_float(speed.get("inference")),
        "postprocess_ms_per_image": _as_float(speed.get("postprocess")),
        "validation_loss_ms_per_image": _as_float(speed.get("loss")),
    }

    per_precision = _as_array(getattr(box, "p", None))
    per_recall = _as_array(getattr(box, "r", None))
    per_ap50 = _as_array(getattr(box, "ap50", None))
    full_class_indexed_map = False
    if ap.ndim == 2 and ap.size:
        per_map = np.mean(ap, axis=1)
        per_map75 = ap[:, 5] if ap.shape[1] > 5 else np.asarray([])
    else:
        raw_maps = _as_array(getattr(box, "maps", None))
        per_map = raw_maps
        per_map75 = np.asarray([])
        full_class_indexed_map = len(raw_maps) == len(class_names)

    present = _as_array(getattr(box, "ap_class_index", None), dtype=np.int64)
    row_count = max(
        len(per_precision), len(per_recall), len(per_ap50), len(per_map), len(per_map75)
    )
    if not len(present) and row_count:
        present = np.arange(row_count, dtype=np.int64)
    index_by_class = {int(class_id): index for index, class_id in enumerate(present.tolist())}

    def at(values: np.ndarray, index: int | None) -> float | None:
        if index is None or index >= len(values):
            return None
        return _as_float(values[index])

    per_class: list[dict[str, Any]] = []
    for class_id, class_name in sorted(class_names.items()):
        index = index_by_class.get(class_id)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": at(per_precision, index),
                "recall": at(per_recall, index),
                "map50": at(per_ap50, index),
                "map75": at(per_map75, index),
                "map50_95": at(per_map, class_id if full_class_indexed_map else index),
            }
        )
    return aggregate, per_class


def _resolve_artifact_path(value: str | Path, context: RunContext) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (context.run_dir / path).resolve()


def read_training_provenance(context: RunContext, model_id: str) -> dict[str, Any]:
    path = context.run_dir / "models" / model_id / "training.json"
    if not path.is_file():
        return {}
    payload = read_json(path)
    payload["artifact"] = relative_to(path, context.run_dir)
    return payload


def find_best_checkpoint(context: RunContext, model_id: str) -> Path:
    """Resolve the successful training run's ``best.pt`` without loading it."""

    provenance = read_training_provenance(context, model_id)
    if not provenance:
        raise StageError(f"Training provenance is missing for {model_id}")
    status = str(provenance.get("status", "")).lower()
    if status not in {"completed", "success", "succeeded"}:
        raise StageError(f"Training for {model_id} is not completed (status={status!r})")

    configured = provenance.get("best_checkpoint")
    expected_hash = provenance.get("best_checkpoint_sha256")
    if not isinstance(configured, str) or not configured:
        raise StageError(f"Training provenance has no frozen best.pt path for {model_id}")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise StageError(f"Training provenance has no frozen best.pt hash for {model_id}")
    checkpoint = _resolve_artifact_path(configured, context)
    if not checkpoint.is_file() or checkpoint.name != "best.pt":
        raise StageError(f"Frozen best.pt is missing for {model_id}: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    if actual_hash != expected_hash:
        raise StageError(
            f"Frozen best.pt changed after training for {model_id}: {checkpoint}. Start a new run."
        )
    return checkpoint


def _training_value(provenance: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = _attribute(provenance, key)
        if value is not None:
            return value
    return None


def _model_efficiency(model: Any, checkpoint: Path) -> dict[str, Any]:
    parameters = None
    gradients = None
    module = getattr(model, "model", None)
    if module is not None and hasattr(module, "parameters"):
        try:
            parameter_list = list(module.parameters())
            parameters = int(sum(parameter.numel() for parameter in parameter_list))
            gradients = int(
                sum(parameter.numel() for parameter in parameter_list if parameter.requires_grad)
            )
        except (AttributeError, TypeError):
            pass

    info = None
    for target in (module, model):
        method = getattr(target, "info", None)
        if not callable(method):
            continue
        try:
            info = method(verbose=False)
            break
        except (TypeError, ValueError, RuntimeError):
            continue
    flops_g = None
    if isinstance(info, (tuple, list)) and len(info) >= 4:
        flops_g = _as_float(info[3])
        if parameters is None:
            with suppress(TypeError, ValueError):
                parameters = int(info[1])
    elif isinstance(info, Mapping):
        flops_g = _first_float(info, "flops", "gflops", "GFLOPs")

    return {
        "parameters": parameters,
        "trainable_parameters": gradients,
        "flops_g": flops_g,
        "checkpoint_size_mib": checkpoint.stat().st_size / (1024.0 * 1024.0),
    }


def _xywhn_to_xyxy(boxes: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(boxes, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    output = np.empty_like(array)
    output[:, 0] = array[:, 0] - array[:, 2] / 2.0
    output[:, 1] = array[:, 1] - array[:, 3] / 2.0
    output[:, 2] = array[:, 0] + array[:, 2] / 2.0
    output[:, 3] = array[:, 1] + array[:, 3] / 2.0
    return output


def _ground_truth(image: Path, images_root: Path) -> tuple[np.ndarray, np.ndarray]:
    labels_root = images_root.with_name("labels")
    try:
        relative = image.relative_to(images_root)
    except ValueError as exc:
        raise StageError(f"Image {image} is outside split directory {images_root}") from exc
    label_path = labels_root / relative.with_suffix(".txt")
    if not label_path.is_file():
        return np.empty((0, 4), dtype=np.float64), np.empty((0,), dtype=np.int64)

    boxes: list[list[float]] = []
    classes: list[int] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise StageError(f"Malformed label row at {label_path}:{line_number}")
        try:
            class_value = float(fields[0])
            coordinates = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise StageError(f"Non-numeric label row at {label_path}:{line_number}") from exc
        if not class_value.is_integer() or not np.isfinite(coordinates).all():
            raise StageError(f"Invalid label row at {label_path}:{line_number}")
        classes.append(int(class_value))
        boxes.append(coordinates)
    return _xywhn_to_xyxy(boxes), np.asarray(classes, dtype=np.int64)


def _prediction_arrays(result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )
    normalized = _as_array(getattr(boxes, "xyxyn", None))
    if normalized.size == 0:
        normalized = _as_array(getattr(boxes, "xyxy", None))
        shape = getattr(result, "orig_shape", None)
        if normalized.size and shape and len(shape) >= 2:
            height, width = float(shape[0]), float(shape[1])
            normalized = normalized / np.asarray([width, height, width, height])
    normalized = normalized.reshape(-1, 4)
    classes = _as_array(getattr(boxes, "cls", None), dtype=np.int64).reshape(-1)
    confidences = _as_array(getattr(boxes, "conf", None)).reshape(-1)
    if not (len(normalized) == len(classes) == len(confidences)):
        raise StageError("Ultralytics returned inconsistent prediction box arrays")
    return normalized, classes, confidences


def _coverage_predictions(
    model: Any,
    context: RunContext,
    split: str,
    class_names: Mapping[int, str],
    *,
    save_predictions: bool,
    output_dir: Path,
) -> dict[str, Any]:
    spec = load_dataset_spec(runtime_dataset_yaml(context))
    images_root = spec.split(split)
    images = supported_images(images_root)
    if not images:
        raise StageError(f"No supported images found in {split} split: {images_root}")

    settings = context.config.evaluation
    predict_kwargs: dict[str, Any] = {
        "source": [str(path) for path in images],
        "stream": True,
        "imgsz": context.config.training.imgsz,
        "batch": settings.batch,
        "device": ultralytics_device(context.config.training.device),
        "conf": settings.confidence,
        "iou": settings.prediction_iou,
        "max_det": settings.max_detections,
        "half": inference_half_enabled(context.config.training.device, context.config.training.amp),
        "save": save_predictions,
        "save_txt": save_predictions,
        "save_conf": save_predictions,
        "verbose": False,
    }
    if save_predictions:
        predict_kwargs.update({"project": str(output_dir), "name": "predictions", "exist_ok": True})
    results: Iterable[Any] = model.predict(**predict_kwargs)
    frames: list[DetectionFrame] = []
    sentinel = object()
    for image, result in itertools.zip_longest(images, results, fillvalue=sentinel):
        if image is sentinel or result is sentinel:
            raise StageError("Ultralytics prediction count did not match the input image count")
        gt_boxes, gt_classes = _ground_truth(image, images_root)
        pred_boxes, pred_classes, confidences = _prediction_arrays(result)
        keep = confidences >= settings.confidence
        frames.append(
            DetectionFrame(
                ground_truth_boxes=gt_boxes,
                ground_truth_classes=gt_classes,
                prediction_boxes=pred_boxes[keep],
                prediction_classes=pred_classes[keep],
                prediction_confidences=confidences[keep],
            )
        )
    coverage = aggregate_detection_coverage(
        frames, class_names, iou_threshold=settings.matching_iou
    )
    # Also expose aggregate values directly under ``coverage`` for simple
    # machine consumers while preserving the explicit nested form.
    coverage.update(coverage["aggregate"])
    coverage["confidence"] = settings.confidence
    coverage["prediction_iou"] = settings.prediction_iou
    coverage["max_detections"] = settings.max_detections
    return coverage


def _merge_per_class(
    metric_rows: Sequence[Mapping[str, Any]], coverage_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    coverage = {int(row["class_id"]): row for row in coverage_rows}
    merged: list[dict[str, Any]] = []
    for metric in metric_rows:
        class_id = int(metric["class_id"])
        # Conventional precision/recall remain Ultralytics validation metrics;
        # fixed-threshold values use the detection_* names.
        merged.append({**dict(coverage.get(class_id, {})), **dict(metric)})
    return merged


def _evaluation_settings(context: RunContext, split: str) -> dict[str, Any]:
    settings = context.config.evaluation
    return {
        "split": split,
        "imgsz": context.config.training.imgsz,
        "batch": settings.batch,
        "device": context.config.training.device,
        "half_requested": context.config.training.amp,
        "half_enabled": inference_half_enabled(
            context.config.training.device, context.config.training.amp
        ),
        "confidence": settings.confidence,
        "prediction_iou": settings.prediction_iou,
        "matching_iou": settings.matching_iou,
        "max_detections": settings.max_detections,
    }


def _run_evaluation(
    context: RunContext,
    model_id: str,
    checkpoint: Path,
    split: str,
    output_dir: Path,
    *,
    save_predictions: bool,
) -> dict[str, Any]:
    YOLO = import_ultralytics_yolo()
    model = YOLO(str(checkpoint), task="detect")
    settings = context.config.evaluation
    val_kwargs: dict[str, Any] = {
        "data": str(runtime_dataset_yaml(context)),
        "split": split,
        "imgsz": context.config.training.imgsz,
        "batch": settings.batch,
        "device": ultralytics_device(context.config.training.device),
        "conf": settings.confidence,
        "iou": settings.prediction_iou,
        "max_det": settings.max_detections,
        "half": inference_half_enabled(context.config.training.device, context.config.training.amp),
        "plots": True,
        "save_json": True,
        "save_txt": save_predictions,
        "save_conf": save_predictions,
        "project": str(output_dir),
        "name": "ultralytics",
        "exist_ok": True,
        "verbose": False,
    }
    validation_metrics = model.val(**val_kwargs)
    spec = load_dataset_spec(runtime_dataset_yaml(context))
    aggregate, metric_rows = _extract_metric_rows(validation_metrics, spec.names)
    coverage = _coverage_predictions(
        model,
        context,
        split,
        spec.names,
        save_predictions=save_predictions,
        output_dir=output_dir,
    )
    per_class = _merge_per_class(metric_rows, coverage["per_class"])
    efficiency = _model_efficiency(model, checkpoint)
    provenance = read_training_provenance(context, model_id)
    best_epoch = _training_value(
        provenance, "best_epoch", "summary.best_epoch", "metrics.best_epoch"
    )
    training_duration = _training_value(
        provenance,
        "training_duration_seconds",
        "duration_seconds",
        "wall_clock_seconds",
        "summary.training_duration_seconds",
    )
    save_dir = getattr(validation_metrics, "save_dir", None) or output_dir / "ultralytics"
    speed = {
        "preprocess_ms": aggregate["preprocess_ms_per_image"],
        "inference_ms": aggregate["inference_ms_per_image"],
        "postprocess_ms": aggregate["postprocess_ms_per_image"],
    }
    flat = {
        **coverage["aggregate"],
        **aggregate,
        **efficiency,
        "preprocess_ms": speed["preprocess_ms"],
        "inference_ms": speed["inference_ms"],
        "postprocess_ms": speed["postprocess_ms"],
        "best_epoch": best_epoch,
        "training_duration_seconds": training_duration,
    }
    return {
        "status": "completed",
        "model_id": model_id,
        "checkpoint": relative_to(checkpoint, context.run_dir),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split": split,
        **flat,
        "aggregate": aggregate,
        "coverage": coverage,
        "speed": speed,
        "efficiency": efficiency,
        "per_class": per_class,
        "training": json_safe(provenance),
        "settings": _evaluation_settings(context, split),
        "ultralytics_output": relative_to(Path(save_dir), context.run_dir),
    }


def _aggregate_csv_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "model_id",
        "split",
        "checkpoint",
        "precision",
        "recall",
        "map50",
        "map75",
        "map50_95",
        "ground_truth_count",
        "prediction_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "parameters",
        "trainable_parameters",
        "flops_g",
        "checkpoint_size_mib",
        "best_epoch",
        "training_duration_seconds",
        "preprocess_ms_per_image",
        "inference_ms_per_image",
        "postprocess_ms_per_image",
    )
    return {key: payload.get(key) for key in keys}


def _write_candidate_evaluation(output_dir: Path, payload: Mapping[str, Any]) -> None:
    write_json(output_dir / "metrics.json", payload)
    write_csv(output_dir / "aggregate_metrics.csv", [_aggregate_csv_row(payload)])
    write_csv(output_dir / "per_class_metrics.csv", payload.get("per_class", []))


def evaluate_candidates(context: RunContext) -> dict[str, Any]:
    """Evaluate every configured candidate's trained ``best.pt`` on validation."""

    require_completed(context, "train")
    verify_dataset_unchanged(context)
    split = context.config.evaluation.comparison_split
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in context.config.models:
        output_dir = context.run_dir / "models" / candidate.id / "evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Evaluating %s on %s", candidate.id, split)
        try:
            checkpoint = find_best_checkpoint(context, candidate.id)
            payload = _run_evaluation(
                context,
                candidate.id,
                checkpoint,
                split,
                output_dir,
                save_predictions=False,
            )
            _write_candidate_evaluation(output_dir, payload)
            successes.append(payload)
        except Exception as exc:
            LOGGER.exception("Evaluation failed for %s", candidate.id)
            failure = {"model_id": candidate.id, "status": "failed", "error": str(exc)}
            write_json(output_dir / "metrics.json", failure)
            failures.append(failure)

    summary = {
        "status": "failed" if failures else "completed",
        "split": split,
        "successful_models": [row["model_id"] for row in successes],
        "failed_models": failures,
        "models": [
            {
                "model_id": row["model_id"],
                "status": row["status"],
                "metrics": relative_to(
                    context.run_dir / "models" / row["model_id"] / "evaluation" / "metrics.json",
                    context.run_dir,
                ),
            }
            for row in successes
        ],
    }
    summary_path = context.run_dir / "comparison" / "evaluation_summary.json"
    write_json(summary_path, summary)
    if failures:
        failed_ids = ", ".join(row["model_id"] for row in failures)
        raise StageError(
            f"Validation evaluation was incomplete; failed models: {failed_ids}. "
            "Successful model results were preserved."
        )
    return {"evaluation_summary": relative_to(summary_path, context.run_dir)}


def _selected_model_id(selection: Mapping[str, Any]) -> str:
    for key in ("winner_model_id", "selected_model_id", "model_id"):
        value = selection.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("winner", "selected_model", "selection"):
        value = selection.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping):
            try:
                return _selected_model_id(value)
            except StageError:
                pass
    raise StageError("Frozen selection.json does not identify a winning model")


def _test_summary_markdown(payload: Mapping[str, Any]) -> str:
    def value(key: str, digits: int = 4) -> str:
        raw = payload.get(key)
        if raw is None:
            return "not available"
        if isinstance(raw, (int, np.integer)):
            return str(raw)
        try:
            return f"{float(raw):.{digits}f}"
        except (TypeError, ValueError):
            return str(raw)

    return "\n".join(
        [
            "# Frozen winner test evaluation",
            "",
            f"- Winner: `{payload['model_id']}`",
            f"- Checkpoint: `{payload['checkpoint']}`",
            "- Selection was frozen from validation results before this test evaluation.",
            "- The test result was not used to revise the winner or tune thresholds.",
            f"- Precision: {value('precision')}",
            f"- Recall: {value('recall')}",
            f"- mAP@0.50: {value('map50')}",
            f"- mAP@0.75: {value('map75')}",
            f"- mAP@0.50:0.95: {value('map50_95')}",
            f"- Correctly matched detections: {value('true_positives', 0)}",
            f"- Fixed-threshold detection recall: {value('detection_recall')}",
            "",
        ]
    )


def evaluate_winner_test(context: RunContext) -> dict[str, Any]:
    """Evaluate only the validation-selected winner on the untouched test split."""

    require_completed(context, "compare")
    verify_dataset_unchanged(context)
    if not context.config.test.evaluate_winner_only:
        raise StageError("Winner-only test evaluation guard is disabled; refusing to run")
    selection_path = context.run_dir / "comparison" / "selection.json"
    if not selection_path.is_file():
        raise StageError("Frozen comparison/selection.json is required before test evaluation")
    selection = read_json(selection_path)
    frozen_selection = context.manifest.get("selection", {})
    expected_selection_hash = frozen_selection.get("sha256")
    if (
        not isinstance(expected_selection_hash, str)
        or sha256_file(selection_path) != expected_selection_hash
    ):
        raise StageError("Frozen selection.json changed after validation comparison")
    if str(selection.get("status", "")).lower() != "frozen":
        raise StageError("selection.json is not a completed frozen selection")
    winner_id = _selected_model_id(selection)
    configured_ids = {candidate.id for candidate in context.config.models}
    if winner_id not in configured_ids:
        raise StageError(f"Selected winner {winner_id!r} is not a configured model")

    checkpoint = find_best_checkpoint(context, winner_id)
    expected_checkpoint_hash = selection.get("checkpoint_sha256")
    if (
        not isinstance(expected_checkpoint_hash, str)
        or sha256_file(checkpoint) != expected_checkpoint_hash
    ):
        raise StageError("Selected winner checkpoint does not match the frozen selection")
    output_dir = context.run_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Evaluating frozen winner %s on the test split", winner_id)
    payload = _run_evaluation(
        context,
        winner_id,
        checkpoint,
        "test",
        output_dir,
        save_predictions=True,
    )
    if payload.get("checkpoint_sha256") != expected_checkpoint_hash:
        raise StageError("Winner checkpoint changed during test evaluation")
    payload["selection"] = {
        "artifact": relative_to(selection_path, context.run_dir),
        "sha256": sha256_file(selection_path),
        "winner_frozen_before_test": True,
    }
    payload["environment"] = context.manifest.get("environment", basic_environment())

    metrics_path = output_dir / "winner_test_metrics.json"
    aggregate_path = output_dir / "winner_test_metrics.csv"
    per_class_path = output_dir / "winner_per_class_metrics.csv"
    summary_path = output_dir / "test_summary.md"
    write_json(metrics_path, payload)
    write_csv(aggregate_path, [_aggregate_csv_row(payload)])
    write_csv(per_class_path, payload["per_class"])
    summary_path.write_text(_test_summary_markdown(payload), encoding="utf-8")
    return {
        "winner_test_metrics": relative_to(metrics_path, context.run_dir),
        "winner_test_metrics_csv": relative_to(aggregate_path, context.run_dir),
        "winner_per_class_metrics": relative_to(per_class_path, context.run_dir),
        "test_summary": relative_to(summary_path, context.run_dir),
    }


# Backward-friendly singular name for integrations that treat evaluation as one stage.
evaluate = evaluate_candidates
