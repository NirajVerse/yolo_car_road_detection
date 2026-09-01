"""Pure object-detection matching and coverage metrics.

The matching functions in this module intentionally have no dependency on
Ultralytics or PyTorch.  Boxes use ``xyxy`` coordinates in any common scale
(pixels or normalized coordinates).  Predictions are considered in descending
confidence order and may match at most one, as-yet-unmatched ground-truth box
of the same class.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectionMatch:
    """One class-aware prediction/ground-truth assignment."""

    prediction_index: int
    ground_truth_index: int
    class_id: int
    confidence: float
    iou: float


@dataclass(frozen=True)
class MatchResult:
    """Result of matching all predictions in one image."""

    matches: tuple[DetectionMatch, ...]
    unmatched_prediction_indices: tuple[int, ...]
    unmatched_ground_truth_indices: tuple[int, ...]
    prediction_order: tuple[int, ...]

    @property
    def true_positives(self) -> int:
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        return len(self.unmatched_prediction_indices)

    @property
    def false_negatives(self) -> int:
        return len(self.unmatched_ground_truth_indices)

    @property
    def tp(self) -> int:
        return self.true_positives

    @property
    def fp(self) -> int:
        return self.false_positives

    @property
    def fn(self) -> int:
        return self.false_negatives


@dataclass(frozen=True)
class DetectionFrame:
    """Ground truth and predictions for one image."""

    ground_truth_boxes: Sequence[Sequence[float]] | np.ndarray
    ground_truth_classes: Sequence[int] | np.ndarray
    prediction_boxes: Sequence[Sequence[float]] | np.ndarray
    prediction_classes: Sequence[int] | np.ndarray
    prediction_confidences: Sequence[float] | np.ndarray


def _boxes(value: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    if np.any(array[:, 2] < array[:, 0]) or np.any(array[:, 3] < array[:, 1]):
        raise ValueError(f"{name} contains a box with x2 < x1 or y2 < y1")
    return array


def _classes(value: Sequence[int] | np.ndarray, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        array = np.empty((0,), dtype=np.int64)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    try:
        numeric = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer class ids") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer class ids")
    return numeric.astype(np.int64)


def _confidences(value: Sequence[float] | np.ndarray, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        array = np.empty((0,), dtype=np.float64)
    if array.ndim != 1 or len(array) != length:
        raise ValueError(
            f"prediction_confidences must have shape ({length},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("prediction_confidences contains non-finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("prediction_confidences must be in [0, 1]")
    return array


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Return intersection-over-union for two ``xyxy`` boxes."""

    first = _boxes(np.asarray(box_a).reshape(1, -1), "box_a")[0]
    second = _boxes(np.asarray(box_b).reshape(1, -1), "box_b")[0]
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    area_a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def pairwise_iou(
    boxes_a: Sequence[Sequence[float]] | np.ndarray,
    boxes_b: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return the pairwise IoU matrix for two collections of ``xyxy`` boxes."""

    first = _boxes(boxes_a, "boxes_a")
    second = _boxes(boxes_b, "boxes_b")
    if not len(first) or not len(second):
        return np.zeros((len(first), len(second)), dtype=np.float64)

    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    first_area = np.prod(np.clip(first[:, 2:] - first[:, :2], 0.0, None), axis=1)
    second_area = np.prod(np.clip(second[:, 2:] - second[:, :2], 0.0, None), axis=1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0.0,
    )


def _iou_threshold(value: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("iou_threshold must be numeric")
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    return threshold


def match_detections(
    ground_truth_boxes: Sequence[Sequence[float]] | np.ndarray,
    ground_truth_classes: Sequence[int] | np.ndarray,
    prediction_boxes: Sequence[Sequence[float]] | np.ndarray,
    prediction_classes: Sequence[int] | np.ndarray,
    prediction_confidences: Sequence[float] | np.ndarray,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Greedily perform class-aware, confidence-ordered one-to-one matching.

    For each prediction, the unmatched ground-truth box of the same class with
    the largest IoU is selected.  Stable sorting and lowest-index tie handling
    make the result deterministic.
    """

    threshold = _iou_threshold(iou_threshold)

    gt_boxes = _boxes(ground_truth_boxes, "ground_truth_boxes")
    pred_boxes = _boxes(prediction_boxes, "prediction_boxes")
    gt_classes = _classes(ground_truth_classes, len(gt_boxes), "ground_truth_classes")
    pred_classes = _classes(prediction_classes, len(pred_boxes), "prediction_classes")
    confidences = _confidences(prediction_confidences, len(pred_boxes))
    ious = pairwise_iou(pred_boxes, gt_boxes)

    order = np.argsort(-confidences, kind="stable").astype(int).tolist()
    unmatched_ground_truth = set(range(len(gt_boxes)))
    matched_predictions: set[int] = set()
    matches: list[DetectionMatch] = []

    for prediction_index in order:
        candidates = [
            index
            for index in sorted(unmatched_ground_truth)
            if gt_classes[index] == pred_classes[prediction_index]
        ]
        if not candidates:
            continue
        candidate_ious = ious[prediction_index, candidates]
        best_offset = int(np.argmax(candidate_ious))
        ground_truth_index = candidates[best_offset]
        match_iou = float(candidate_ious[best_offset])
        if match_iou < threshold:
            continue
        unmatched_ground_truth.remove(ground_truth_index)
        matched_predictions.add(prediction_index)
        matches.append(
            DetectionMatch(
                prediction_index=prediction_index,
                ground_truth_index=ground_truth_index,
                class_id=int(pred_classes[prediction_index]),
                confidence=float(confidences[prediction_index]),
                iou=match_iou,
            )
        )

    return MatchResult(
        matches=tuple(matches),
        unmatched_prediction_indices=tuple(
            index for index in order if index not in matched_predictions
        ),
        unmatched_ground_truth_indices=tuple(sorted(unmatched_ground_truth)),
        prediction_order=tuple(order),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _coverage_row(
    ground_truth_count: int,
    prediction_count: int,
    true_positives: int,
) -> dict[str, int | float | None]:
    false_positives = prediction_count - true_positives
    false_negatives = ground_truth_count - true_positives
    precision = _ratio(true_positives, prediction_count)
    recall = _ratio(true_positives, ground_truth_count)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0.0:
        f1 = 2.0 * precision * recall / (precision + recall)
    return {
        "ground_truth_count": int(ground_truth_count),
        "prediction_count": int(prediction_count),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        # Concise aliases keep machine consumers simple while the descriptive
        # names make reports unambiguous.
        "gt_count": int(ground_truth_count),
        "pred_count": int(prediction_count),
        "tp": int(true_positives),
        "fp": int(false_positives),
        "fn": int(false_negatives),
        "precision": precision,
        "recall": recall,
    }


def aggregate_detection_coverage(
    frames: Iterable[DetectionFrame],
    class_names: Mapping[int, str] | Sequence[str],
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Aggregate TP/FP/FN and fixed-threshold coverage over multiple images."""

    threshold = _iou_threshold(iou_threshold)
    names = (
        {int(key): str(value) for key, value in class_names.items()}
        if isinstance(class_names, Mapping)
        else {index: str(value) for index, value in enumerate(class_names)}
    )
    class_ids = sorted(names)
    gt_counts = {class_id: 0 for class_id in class_ids}
    pred_counts = {class_id: 0 for class_id in class_ids}
    tp_counts = {class_id: 0 for class_id in class_ids}
    image_count = 0

    for frame in frames:
        gt_boxes = _boxes(frame.ground_truth_boxes, "ground_truth_boxes")
        pred_boxes = _boxes(frame.prediction_boxes, "prediction_boxes")
        gt_classes = _classes(
            frame.ground_truth_classes, len(gt_boxes), "ground_truth_classes"
        )
        pred_classes = _classes(
            frame.prediction_classes, len(pred_boxes), "prediction_classes"
        )
        unknown = sorted(
            (set(gt_classes.tolist()) | set(pred_classes.tolist())) - set(class_ids)
        )
        if unknown:
            raise ValueError(f"Detections contain unknown class ids: {unknown}")
        result = match_detections(
            gt_boxes,
            gt_classes,
            pred_boxes,
            pred_classes,
            frame.prediction_confidences,
            threshold,
        )
        for class_id in class_ids:
            gt_counts[class_id] += int(np.count_nonzero(gt_classes == class_id))
            pred_counts[class_id] += int(np.count_nonzero(pred_classes == class_id))
        for match in result.matches:
            tp_counts[match.class_id] += 1
        image_count += 1

    per_class: list[dict[str, object]] = []
    for class_id in class_ids:
        row: dict[str, object] = {
            "class_id": class_id,
            "class_name": names[class_id],
            **_coverage_row(
                gt_counts[class_id], pred_counts[class_id], tp_counts[class_id]
            ),
        }
        per_class.append(row)

    aggregate = _coverage_row(
        sum(gt_counts.values()), sum(pred_counts.values()), sum(tp_counts.values())
    )
    return {
        "image_count": image_count,
        "matching_iou": threshold,
        "aggregate": aggregate,
        "per_class": per_class,
    }


# Descriptive alias retained for callers that prefer an explicit verb.
calculate_detection_coverage = aggregate_detection_coverage
aggregate_coverage = aggregate_detection_coverage
class_aware_match = match_detections
compute_iou = box_iou
