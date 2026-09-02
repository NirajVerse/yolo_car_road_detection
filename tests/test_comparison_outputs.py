from __future__ import annotations

import pytest

from road_detection.compare import compare_models
from road_detection.config import load_config
from road_detection.utils import StageError, create_run, read_json, write_json


def _mark_completed(context, *stages: str) -> None:
    manifest = context.manifest
    for stage in stages:
        manifest.setdefault("stages", {})[stage] = {"status": "completed"}
    write_json(context.manifest_path, manifest)


def _write_candidate(
    context, model_id: str, map_value: float, safety_recall: float, latency: float
) -> None:
    model_dir = context.run_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = "a" * 64
    write_json(
        model_dir / "training.json",
        {
            "status": "completed",
            "model_id": model_id,
            "best_epoch": 2,
            "training_duration_seconds": 10.0,
            "batch_size": 2,
            "early_stopping": False,
            "best_checkpoint": f"models/{model_id}/train_001/weights/best.pt",
            "best_checkpoint_sha256": checkpoint_hash,
        },
    )
    evaluation = model_dir / "evaluation"
    evaluation.mkdir(exist_ok=True)
    per_class = [
        {
            "class_id": 0,
            "class_name": "Person",
            "precision": 0.7,
            "recall": 0.7,
            "map50": 0.75,
            "map50_95": map_value,
            "detection_recall": safety_recall,
        },
        {
            "class_id": 1,
            "class_name": "Bike",
            "precision": 0.6,
            "recall": 0.6,
            "map50": 0.7,
            "map50_95": map_value - 0.1,
            "detection_recall": safety_recall,
        },
        {
            "class_id": 2,
            "class_name": "Traffic-sign",
            "precision": 0.5,
            "recall": 0.5,
            "map50": 0.6,
            "map50_95": map_value - 0.2,
            "detection_recall": 0.5,
        },
    ]
    write_json(
        evaluation / "metrics.json",
        {
            "status": "completed",
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_hash,
            "precision": 0.7,
            "recall": 0.65,
            "map50": map_value + 0.1,
            "map75": map_value - 0.05,
            "map50_95": map_value,
            "parameters": 1000,
            "flops_g": 1.2,
            "checkpoint_size_mib": 5.0,
            "speed": {"preprocess_ms": 1.0, "inference_ms": 2.0, "postprocess_ms": 0.5},
            "coverage": {
                "ground_truth_count": 10,
                "prediction_count": 10,
                "true_positives": 7,
                "false_positives": 3,
                "false_negatives": 3,
                "recall": 0.7,
            },
            "per_class": per_class,
        },
    )
    write_json(
        model_dir / "benchmark.json",
        {
            "status": "completed",
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_hash,
            "latency_mean_ms": latency,
            "latency_median_ms": latency,
            "latency_std_ms": 0.2,
            "latency_p90_ms": latency + 0.5,
            "latency_p95_ms": latency + 1.0,
            "fps": 1000.0 / latency,
            "device_name": "synthetic CPU",
            "precision": "float32",
        },
    )


def test_comparison_writes_all_outputs_and_freezes_validation_winner(config_factory) -> None:
    path, _ = config_factory()
    context = create_run(load_config(path), "comparison-run")
    _mark_completed(context, "evaluate", "benchmark")
    _write_candidate(context, "tiny", 0.700, 0.60, 10.0)
    _write_candidate(context, "small", 0.698, 0.80, 14.0)

    outputs = compare_models(context)

    assert set(outputs) == {
        "model_comparison_csv",
        "model_comparison_json",
        "summary",
        "selection",
    }
    comparison = context.run_dir / "comparison"
    required_plots = {
        "accuracy_comparison.png",
        "latency_comparison.png",
        "precision_recall_comparison.png",
        "model_size_comparison.png",
        "accuracy_latency_tradeoff.png",
        "per_class_map_heatmap.png",
        "per_class_recall_heatmap.png",
    }
    assert all((comparison / name).stat().st_size > 0 for name in required_plots)
    selection = read_json(comparison / "selection.json")
    assert selection["status"] == "frozen"
    assert selection["winner"] == "small"
    assert selection["checkpoint_sha256"] == "a" * 64
    assert context.manifest["selection"]["sha256"]
    summary = (comparison / "summary.md").read_text(encoding="utf-8")
    assert "Accuracy–latency tradeoff" in summary
    assert "Consistently weakest class" in summary


def test_comparison_records_failure_and_blocks_selection(config_factory) -> None:
    path, _ = config_factory()
    context = create_run(load_config(path), "failed-comparison-run")
    _mark_completed(context, "evaluate", "benchmark")
    _write_candidate(context, "tiny", 0.700, 0.60, 10.0)
    failed_dir = context.run_dir / "models" / "small"
    failed_dir.mkdir(parents=True)
    write_json(
        failed_dir / "training.json",
        {"status": "failed", "model_id": "small", "error": "synthetic OOM"},
    )

    with pytest.raises(StageError, match="blocked"):
        compare_models(context)

    selection = read_json(context.run_dir / "comparison" / "selection.json")
    assert selection["status"] == "blocked_incomplete"
    assert selection["winner"] is None
    comparison = read_json(context.run_dir / "comparison" / "model_comparison.json")
    assert [row["status"] for row in comparison["models"]] == ["completed", "failed"]


def test_comparison_blocks_mixed_checkpoint_identities(config_factory) -> None:
    path, _ = config_factory()
    context = create_run(load_config(path), "identity-mismatch-run")
    _mark_completed(context, "evaluate", "benchmark")
    _write_candidate(context, "tiny", 0.700, 0.60, 10.0)
    _write_candidate(context, "small", 0.698, 0.80, 14.0)
    benchmark_path = context.run_dir / "models" / "small" / "benchmark.json"
    benchmark = read_json(benchmark_path)
    benchmark["checkpoint_sha256"] = "b" * 64
    write_json(benchmark_path, benchmark)

    with pytest.raises(StageError, match="blocked"):
        compare_models(context)

    rows = read_json(context.run_dir / "comparison" / "model_comparison.json")["models"]
    small = next(row for row in rows if row["model_id"] == "small")
    assert small["status"] == "failed"
    assert "different checkpoint bytes" in small["failure_message"]
