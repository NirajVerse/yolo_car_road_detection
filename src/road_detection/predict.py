"""Frozen-winner demonstration-image prediction and structured output."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_dataset_spec, resolve_weights
from .data_audit import verify_dataset_unchanged
from .evaluate import find_best_checkpoint
from .utils import (
    RunContext,
    StageError,
    import_ultralytics_yolo,
    inference_half_enabled,
    read_json,
    relative_to,
    require_completed,
    sha256_file,
    supported_images,
    write_csv,
    write_json,
)


def _choose_source(context: RunContext, source: str | Path | None) -> Path:
    if source is not None:
        value = Path(source).expanduser()
        path = (
            (context.config.project_root / value).resolve()
            if not value.is_absolute()
            else value.resolve()
        )
    elif context.config.prediction.source is not None:
        path = context.config.prediction.source.resolve()
    else:
        spec = load_dataset_spec(context.dataset_yaml)
        candidates = supported_images(spec.split("test"))
        if not candidates:
            raise StageError(
                "No supported test image is available for deterministic demonstration selection"
            )
        # Lexicographic selection is deterministic and does not inspect winner output.
        path = candidates[0]
    if not path.is_file():
        raise StageError(f"Prediction source does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError) as exc:
        raise StageError(f"Prediction source is not a readable image: {path}") from exc
    return path


def _detection_rows(result: Any, class_names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    xywhn = boxes.xywhn.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for index, (pixel_box, normalized_box, class_id, confidence) in enumerate(
        zip(xyxy, xywhn, classes, confidences, strict=True), start=1
    ):
        x1, y1, x2, y2 = (float(value) for value in pixel_box)
        x_center, y_center, width, height = (float(value) for value in normalized_box)
        rows.append(
            {
                "detection_id": index,
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "confidence": float(confidence),
                "x1_px": x1,
                "y1_px": y1,
                "x2_px": x2,
                "y2_px": y2,
                "x_center_normalized": x_center,
                "y_center_normalized": y_center,
                "width_normalized": width,
                "height_normalized": height,
                "xyxy_pixels": [x1, y1, x2, y2],
                "yolo_normalized": [x_center, y_center, width, height],
            }
        )
    return rows


def predict_winner(context: RunContext, source: str | Path | None = None) -> dict[str, str]:
    """Run the frozen winner on one image and save image/JSON/CSV artifacts."""

    require_completed(context, "compare", "test_winner")
    verify_dataset_unchanged(context)
    selection_path = context.run_dir / "comparison" / "selection.json"
    selection = read_json(selection_path)
    if selection.get("status") != "frozen" or not selection.get("winner"):
        raise StageError("A frozen validation winner is required before prediction")
    winner = str(selection["winner"])
    frozen_selection = context.manifest.get("selection", {})
    expected_selection_hash = frozen_selection.get("sha256")
    if (
        not isinstance(expected_selection_hash, str)
        or sha256_file(selection_path) != expected_selection_hash
    ):
        raise StageError("Frozen selection.json changed after validation comparison")
    checkpoint = find_best_checkpoint(context, winner)
    expected_checkpoint_hash = selection.get("checkpoint_sha256")
    if (
        not isinstance(expected_checkpoint_hash, str)
        or sha256_file(checkpoint) != expected_checkpoint_hash
    ):
        raise StageError("Prediction checkpoint does not match the frozen selection")
    source_path = _choose_source(context, source)
    output_dir = context.run_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = source_path.suffix.lower() or ".jpg"
    input_copy = output_dir / f"input_image{extension}"
    shutil.copy2(source_path, input_copy)

    YOLO = import_ultralytics_yolo()
    model = YOLO(resolve_weights(str(checkpoint), context.config.project_root), task="detect")
    task = getattr(model, "task", "detect")
    if task != "detect":
        raise StageError(f"Frozen winner checkpoint has task={task!r}; expected 'detect'")
    results = model.predict(
        source=str(source_path),
        conf=context.config.prediction.confidence,
        iou=context.config.evaluation.prediction_iou,
        imgsz=context.config.training.imgsz,
        device=None if context.config.training.device == "auto" else context.config.training.device,
        max_det=context.config.evaluation.max_detections,
        half=inference_half_enabled(context.config.training.device, context.config.training.amp),
        save=False,
        verbose=False,
    )
    if not results:
        raise StageError("Ultralytics returned no result object for the prediction image")
    if sha256_file(checkpoint) != expected_checkpoint_hash:
        raise StageError("Winner checkpoint changed during prediction")
    result = results[0]
    names_raw = getattr(result, "names", None) or getattr(model, "names", {})
    if isinstance(names_raw, list):
        class_names = {index: str(name) for index, name in enumerate(names_raw)}
    else:
        class_names = {int(key): str(value) for key, value in dict(names_raw).items()}
    detections = _detection_rows(result, class_names)

    annotated = result.plot(labels=True, conf=True, line_width=2)
    if not isinstance(annotated, np.ndarray):
        raise StageError("Ultralytics did not return an annotated image array")
    annotated_path = output_dir / "annotated_image.jpg"
    # Ultralytics plot() returns BGR; Pillow expects RGB.
    Image.fromarray(annotated[..., ::-1]).save(annotated_path, quality=95)

    detections_json = output_dir / "detections.json"
    write_json(
        detections_json,
        {
            "winner": winner,
            "checkpoint": relative_to(checkpoint, context.run_dir),
            "source": str(source_path),
            "confidence_threshold": context.config.prediction.confidence,
            "prediction_iou": context.config.evaluation.prediction_iou,
            "half_enabled": inference_half_enabled(
                context.config.training.device, context.config.training.amp
            ),
            "image_width": int(result.orig_shape[1]),
            "image_height": int(result.orig_shape[0]),
            "detection_count": len(detections),
            "detections": detections,
        },
    )
    detections_csv = output_dir / "detections.csv"
    csv_fields = [
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1_px",
        "y1_px",
        "x2_px",
        "y2_px",
        "x_center_normalized",
        "y_center_normalized",
        "width_normalized",
        "height_normalized",
    ]
    write_csv(detections_csv, detections, csv_fields)
    return {
        "prediction_input": relative_to(input_copy, context.run_dir),
        "annotated_image": relative_to(annotated_path, context.run_dir),
        "detections_json": relative_to(detections_json, context.run_dir),
        "detections_csv": relative_to(detections_csv, context.run_dir),
    }
