from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., tuple[Path, dict[str, Any]]]:
    def factory(*, overrides: dict[str, Any] | None = None) -> tuple[Path, dict[str, Any]]:
        project = tmp_path / "project"
        configs = project / "configs"
        dataset_root = project / "dataset"
        configs.mkdir(parents=True, exist_ok=True)
        for split in ("train", "valid", "test"):
            (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
            (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)
        dataset = {
            "path": "dataset",
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 3,
            "names": {0: "Person", 1: "Bike", 2: "Traffic-sign"},
            "roboflow": {"version": 1},
        }
        (project / "data.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
        raw: dict[str, Any] = {
            "experiment": {"name": "synthetic", "seed": 0, "output_root": "artifacts"},
            "dataset": {"yaml": "data.yaml"},
            "models": [
                {"id": "tiny", "weights": "tiny.pt"},
                {"id": "small", "weights": "small.pt"},
            ],
            "training": {
                "epochs": 2,
                "imgsz": 64,
                "batch": 2,
                "device": "cpu",
                "workers": 0,
                "patience": 1,
                "pretrained": True,
                "deterministic": True,
                "amp": False,
                "cache": False,
                "resume": False,
            },
            "smoke_test": {"epochs": 1, "fraction": 0.5, "max_predictions": 1},
            "evaluation": {
                "comparison_split": "val",
                "batch": 1,
                "confidence": 0.25,
                "prediction_iou": 0.7,
                "matching_iou": 0.5,
                "max_detections": 10,
            },
            "benchmark": {
                "batch": 1,
                "warmup_iterations": 1,
                "measured_iterations": 2,
                "sample_count": 1,
            },
            "selection": {
                "primary_metric": "map50_95",
                "minimum_fps": None,
                "map_tie_tolerance": 0.005,
                "safety_classes": ["Person", "Bike"],
                "tie_breaker": "latency_p95_ms",
            },
            "test": {"evaluate_winner_only": True},
            "prediction": {"source": None, "confidence": 0.25},
        }
        if overrides:
            _deep_update(raw, deepcopy(overrides))
        config_path = configs / "experiment.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        return config_path, raw

    return factory


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
