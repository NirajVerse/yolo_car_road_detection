from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from road_detection.benchmark import _latency_statistics
from road_detection.evaluate import _extract_metric_rows, _ground_truth
from road_detection.predict import _detection_rows


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    def __init__(self):
        self.xyxy = _Array([[1.0, 2.0, 5.0, 8.0]])
        self.xywhn = _Array([[0.3, 0.4, 0.2, 0.3]])
        self.cls = _Array([1])
        self.conf = _Array([0.9])

    def __len__(self):
        return 1


def test_ultralytics_metric_extraction_preserves_zero_and_absent_class() -> None:
    ap = np.asarray(
        [
            np.linspace(0.9, 0.4, 10),
            np.linspace(0.7, 0.2, 10),
        ]
    )
    box = SimpleNamespace(
        mp=0.0,
        mr=0.5,
        map50=0.8,
        map=0.6,
        ap=ap,
        p=np.asarray([0.0, 0.7]),
        r=np.asarray([0.4, 0.8]),
        ap50=np.asarray([0.9, 0.7]),
        ap_class_index=np.asarray([0, 2]),
    )
    metrics = SimpleNamespace(
        box=box,
        speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
        results_dict={},
    )

    aggregate, rows = _extract_metric_rows(metrics, {0: "A", 1: "B", 2: "C"})

    assert aggregate["precision"] == 0.0
    assert aggregate["map75"] == pytest.approx(float(np.mean(ap[:, 5])))
    assert rows[0]["precision"] == 0.0
    assert rows[1]["precision"] is None
    assert rows[1]["map50_95"] is None
    assert rows[2]["recall"] == pytest.approx(0.8)


def test_latency_statistics_have_required_percentiles_and_fps() -> None:
    stats = _latency_statistics([10.0, 20.0, 30.0])

    assert stats["latency_mean_ms"] == pytest.approx(20.0)
    assert stats["latency_median_ms"] == pytest.approx(20.0)
    assert stats["latency_p90_ms"] == pytest.approx(28.0)
    assert stats["latency_p95_ms"] == pytest.approx(29.0)
    assert stats["fps"] == pytest.approx(50.0)


def test_prediction_rows_include_pixel_and_normalized_coordinates() -> None:
    rows = _detection_rows(SimpleNamespace(boxes=_Boxes()), {1: "Bike"})

    assert rows == [
        {
            "detection_id": 1,
            "class_id": 1,
            "class_name": "Bike",
            "confidence": pytest.approx(0.9),
            "x1_px": 1.0,
            "y1_px": 2.0,
            "x2_px": 5.0,
            "y2_px": 8.0,
            "x_center_normalized": pytest.approx(0.3),
            "y_center_normalized": pytest.approx(0.4),
            "width_normalized": pytest.approx(0.2),
            "height_normalized": pytest.approx(0.3),
            "xyxy_pixels": [1.0, 2.0, 5.0, 8.0],
            "yolo_normalized": [
                pytest.approx(0.3),
                pytest.approx(0.4),
                pytest.approx(0.2),
                pytest.approx(0.3),
            ],
        }
    ]


def test_ground_truth_reader_accepts_a_utf8_bom(tmp_path: Path) -> None:
    images = tmp_path / "valid/images"
    labels = tmp_path / "valid/labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "sample.jpg"
    image.write_bytes(b"synthetic image placeholder")
    (labels / "sample.txt").write_text("\ufeff2 0.5 0.5 0.4 0.2\n", encoding="utf-8")

    boxes, classes = _ground_truth(image, images)

    assert classes.tolist() == [2]
    assert boxes[0].tolist() == pytest.approx([0.3, 0.4, 0.7, 0.6])
