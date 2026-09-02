from __future__ import annotations

import numpy as np
import pytest

from road_detection.metrics import (
    DetectionFrame,
    aggregate_detection_coverage,
    box_iou,
    match_detections,
    pairwise_iou,
)


def test_box_and_pairwise_iou() -> None:
    assert box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)
    matrix = pairwise_iou([[0, 0, 1, 1], [0, 0, 2, 2]], [[0, 0, 1, 1]])
    np.testing.assert_allclose(matrix, [[1.0], [0.25]])


def test_matching_is_class_aware_one_to_one_and_confidence_ordered() -> None:
    result = match_detections(
        ground_truth_boxes=[[0, 0, 1, 1], [2, 2, 3, 3]],
        ground_truth_classes=[0, 1],
        prediction_boxes=[
            [0, 0, 1, 1],  # lower-confidence duplicate for class 0
            [0, 0, 0.9, 0.9],  # higher-confidence class 0; gets the match
            [2, 2, 3, 3],  # correct geometry, wrong class
            [2, 2, 3, 3],  # class 1 match
        ],
        prediction_classes=[0, 0, 0, 1],
        prediction_confidences=[0.6, 0.9, 0.8, 0.7],
        iou_threshold=0.5,
    )

    assert result.prediction_order == (1, 2, 3, 0)
    assert [(match.prediction_index, match.ground_truth_index) for match in result.matches] == [
        (1, 0),
        (3, 1),
    ]
    assert result.true_positives == 2
    assert result.false_positives == 2
    assert result.false_negatives == 0


def test_coverage_aggregation_includes_all_classes() -> None:
    frames = [
        DetectionFrame(
            ground_truth_boxes=[[0, 0, 1, 1], [2, 2, 3, 3]],
            ground_truth_classes=[0, 1],
            prediction_boxes=[[0, 0, 1, 1], [5, 5, 6, 6]],
            prediction_classes=[0, 1],
            prediction_confidences=[0.9, 0.8],
        ),
        DetectionFrame(
            ground_truth_boxes=[[0, 0, 1, 1]],
            ground_truth_classes=[0],
            prediction_boxes=[],
            prediction_classes=[],
            prediction_confidences=[],
        ),
    ]
    coverage = aggregate_detection_coverage(frames, {0: "Person", 1: "Bike", 2: "Sign"}, 0.5)

    assert coverage["image_count"] == 2
    assert coverage["aggregate"] == {
        "ground_truth_count": 3,
        "prediction_count": 2,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 2,
        "detection_precision": 0.5,
        "detection_recall": pytest.approx(1 / 3),
        "detection_f1": pytest.approx(0.4),
        "gt_count": 3,
        "pred_count": 2,
        "tp": 1,
        "fp": 1,
        "fn": 2,
        "precision": 0.5,
        "recall": pytest.approx(1 / 3),
    }
    per_class = {row["class_id"]: row for row in coverage["per_class"]}
    assert per_class[0]["true_positives"] == 1
    assert per_class[0]["false_negatives"] == 1
    assert per_class[1]["false_positives"] == 1
    assert per_class[2]["ground_truth_count"] == 0
    assert per_class[2]["detection_recall"] is None


@pytest.mark.parametrize("threshold", [-0.1, 1.1, "0.5"])
def test_invalid_matching_threshold_is_rejected(threshold) -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        match_detections([], [], [], [], [], threshold)


def test_unknown_class_is_rejected_during_aggregation() -> None:
    frame = DetectionFrame([[0, 0, 1, 1]], [99], [], [], [])
    with pytest.raises(ValueError, match="unknown class"):
        aggregate_detection_coverage([frame], {0: "known"})
