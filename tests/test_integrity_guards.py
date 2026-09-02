from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from road_detection.cli import _single_stage
from road_detection.config import load_config
from road_detection.evaluate import find_best_checkpoint
from road_detection.train import _best_epoch, _trainer_artifacts
from road_detection.utils import (
    PipelineError,
    StageError,
    audited_source_weights,
    create_run,
    record_prediction_source_override,
    recorded_prediction_source,
    resume_run,
    sha256_file,
    write_json,
)


def test_resume_rejects_experiment_yaml_drift(config_factory) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    create_run(config, "drift-run")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["training"]["epochs"] += 1
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(StageError, match="Configuration drift"):
        resume_run(load_config(config_path), "drift-run")


def test_resume_rejects_dataset_descriptor_drift(config_factory) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    create_run(config, "dataset-drift-run")
    descriptor = config.project_root / "data.yaml"
    payload = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    payload["note"] = "changed"
    descriptor.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(StageError, match="Dataset descriptor drift"):
        resume_run(load_config(config_path), "dataset-drift-run")


def test_resume_rejects_resolved_configuration_drift(config_factory) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    context = create_run(config, "resolved-drift-run")
    frozen = context.run_dir / "resolved_config.yaml"
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["training"]["epochs"] += 1
    frozen.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(StageError, match="Frozen resolved configuration changed"):
        resume_run(config, "resolved-drift-run")


def test_recorded_source_override_does_not_look_like_config_drift(
    config_factory,
) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    context = create_run(config, "override-run")
    source = config.project_root / "demo.jpg"
    source.write_bytes(b"synthetic input placeholder")

    resolved = record_prediction_source_override(context, source)
    resumed = resume_run(load_config(config_path), "override-run")

    assert resolved == source.resolve()
    assert resumed.manifest["cli_overrides"]["prediction.source"] == str(source.resolve())
    assert recorded_prediction_source(resumed) == source.resolve()
    frozen = yaml.safe_load((resumed.run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert frozen["prediction"]["source"] == str(source.resolve())


def test_audited_source_checkpoint_hash_is_enforced(config_factory) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    context = create_run(config, "checkpoint-run")
    checkpoint = config.project_root / "tiny.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = context.manifest
    manifest["models"] = [
        {
            "model_id": "tiny",
            "status": "loaded",
            "checkpoint_path": "tiny.pt",
            "checkpoint_sha256": sha256_file(checkpoint),
        }
    ]
    write_json(context.manifest_path, manifest)

    assert audited_source_weights(context, "tiny", "tiny.pt") == str(checkpoint)
    checkpoint.write_bytes(b"mutated checkpoint")
    with pytest.raises(StageError, match="changed after audit"):
        audited_source_weights(context, "tiny", "tiny.pt")


def test_trained_best_checkpoint_hash_is_enforced(config_factory) -> None:
    config_path, _ = config_factory()
    config = load_config(config_path)
    context = create_run(config, "trained-checkpoint-run")
    checkpoint = context.run_dir / "models/tiny/train_001/weights/best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic trained checkpoint")
    write_json(
        context.run_dir / "models/tiny/training.json",
        {
            "status": "completed",
            "model_id": "tiny",
            "best_checkpoint": "models/tiny/train_001/weights/best.pt",
            "best_checkpoint_sha256": sha256_file(checkpoint),
        },
    )

    assert find_best_checkpoint(context, "tiny") == checkpoint.resolve()
    checkpoint.write_bytes(b"changed trained checkpoint")
    with pytest.raises(StageError, match="changed after training"):
        find_best_checkpoint(context, "tiny")


def test_trainer_output_must_stay_in_allocated_attempt(tmp_path: Path) -> None:
    expected = tmp_path / "expected"
    model = SimpleNamespace(trainer=SimpleNamespace(save_dir=tmp_path / "elsewhere"))

    with pytest.raises(StageError, match="unexpected directory"):
        _trainer_artifacts(model, expected)


def test_best_epoch_is_reported_one_based() -> None:
    rows = [{"epoch": "0", "metrics/mAP50-95(B)": "0.7"}]
    trainer = SimpleNamespace(best_epoch=None, stopper=SimpleNamespace(best_epoch=0))

    assert _best_epoch(rows, trainer) == 1


def test_force_refuses_to_invalidate_completed_downstream_stage(config_factory) -> None:
    config_path, _ = config_factory()
    context = create_run(load_config(config_path), "force-run")
    manifest = context.manifest
    manifest["stages"] = {"evaluate": {"status": "completed"}}
    write_json(context.manifest_path, manifest)
    called = False

    def should_not_run(_context):
        nonlocal called
        called = True

    with pytest.raises(PipelineError, match="downstream stage"):
        _single_stage(
            context,
            "train",
            should_not_run,
            Namespace(force=True),
        )
    assert called is False
