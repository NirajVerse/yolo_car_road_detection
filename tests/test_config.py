from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from road_detection.config import (
    ConfigError,
    load_config,
    load_dataset_spec,
    resolve_weights,
    write_resolved_dataset_yaml,
)
from road_detection.utils import StageError, create_run, resume_run


def test_load_config_resolves_project_paths(config_factory) -> None:
    path, _ = config_factory()
    config = load_config(path)

    assert config.project_root == path.parent.parent.resolve()
    assert config.dataset.yaml == config.project_root / "data.yaml"
    assert config.experiment.output_root == config.project_root / "artifacts"
    assert [model.id for model in config.models] == ["tiny", "small"]
    assert config.training.workers == 0


def test_device_alias_is_normalized(config_factory) -> None:
    path, _ = config_factory(overrides={"training": {"device": "CUDA:0"}})

    assert load_config(path).training.device == "cuda:0"


def test_dataset_paths_are_relative_to_dataset_yaml(config_factory) -> None:
    path, _ = config_factory()
    config = load_config(path)
    spec = load_dataset_spec(config.dataset.yaml)

    assert spec.root == config.project_root / "dataset"
    assert spec.split("train") == config.project_root / "dataset/train/images"
    assert spec.split("val") == config.project_root / "dataset/valid/images"
    assert spec.split("test") == config.project_root / "dataset/test/images"
    assert spec.names == {0: "Person", 1: "Bike", 2: "Traffic-sign"}
    assert spec.metadata["roboflow"]["version"] == 1


def test_resolved_dataset_yaml_uses_absolute_paths(config_factory, tmp_path: Path) -> None:
    path, _ = config_factory()
    spec = load_dataset_spec(load_config(path).dataset.yaml)
    destination = write_resolved_dataset_yaml(spec, tmp_path / "resolved.yaml")
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))

    assert Path(payload["path"]).is_absolute()
    assert Path(payload["train"]).is_absolute()
    assert payload["names"][1] == "Bike"
    assert payload["roboflow"] == {"version": 1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"models": [{"id": "same", "weights": "a.pt"}, {"id": "same", "weights": "b.pt"}]},
            "Duplicate model id",
        ),
        ({"training": {"batch": 0}}, "training.batch must be positive"),
        ({"evaluation": {"confidence": 1.1}}, "evaluation.confidence must be in"),
        ({"evaluation": {"comparison_split": "test"}}, "comparison_split must be 'val'"),
        ({"benchmark": {"batch": 2}}, "benchmark.batch must be 1"),
        ({"models": [{"id": "bad", "weights": "yolo11n-seg.pt"}]}, "not a standard axis-aligned"),
        ({"models": [{"id": "bad", "weights": "yolo26n-depth.pt"}]}, "not a standard axis-aligned"),
        ({"models": [{"id": "has whitespace", "weights": "model.pt"}]}, "must start with"),
        ({"experiment": {"name": "../unsafe"}}, "must start with"),
        ({"training": {"device": "gpu"}}, "training.device must be"),
        ({"training": {"device": -2}}, "training.device integer"),
        ({"training": {"device": -1}}, "training.device integer"),
        ({"training": {"device": "0,1"}}, "training.device must be"),
        ({"training": {"cache": "somewhere"}}, "training.cache string"),
        ({"training": {"resume": True}}, "training.resume=true is not supported"),
        ({"smoke_test": {"epochs": 2}}, "smoke_test.epochs must be exactly 1"),
        ({"selection": {"minimum_fps": float("nan")}}, "minimum_fps must be positive"),
        ({"test": {"evaluate_winner_only": False}}, "must remain true"),
        ({"selection": {"safety_classes": ["Unknown"]}}, "Unknown selection.safety_classes"),
    ],
)
def test_invalid_configuration_is_rejected(config_factory, overrides, message) -> None:
    path, _ = config_factory(overrides=overrides)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_missing_dataset_yaml_is_rejected(config_factory) -> None:
    path, _ = config_factory(overrides={"dataset": {"yaml": "missing.yaml"}})
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(path)


def test_create_and_resume_synthetic_run(config_factory) -> None:
    path, _ = config_factory()
    config = load_config(path)
    context = create_run(config, "unit-run")

    assert context.manifest["status"] == "created"
    assert context.dataset_yaml.is_file()
    assert resume_run(config, "unit-run").run_dir == context.run_dir
    with pytest.raises(StageError, match="already exists"):
        create_run(config, "unit-run")
    with pytest.raises(StageError, match="must start"):
        create_run(config, ".")


def test_resolved_dataset_redacts_sensitive_metadata(config_factory, tmp_path: Path) -> None:
    path, _ = config_factory()
    dataset_path = path.parent.parent / "data.yaml"
    payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    payload["access_token"] = "top-secret"
    payload["download_url"] = "https://example.test/data.zip?signature=secret"
    dataset_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    spec = load_dataset_spec(dataset_path)
    resolved = write_resolved_dataset_yaml(spec, tmp_path / "resolved.yaml")
    copied = yaml.safe_load(resolved.read_text(encoding="utf-8"))

    assert copied["access_token"] == "<redacted>"
    assert copied["download_url"] == "https://example.test/data.zip"
    assert "top-secret" not in resolved.read_text(encoding="utf-8")


def test_local_basename_checkpoint_resolves_from_project_root(config_factory) -> None:
    path, _ = config_factory()
    project_root = path.parent.parent
    checkpoint = project_root / "local.pt"
    checkpoint.write_bytes(b"synthetic checkpoint placeholder")

    assert resolve_weights("local.pt", project_root) == str(checkpoint.resolve())
    assert resolve_weights("registry-name.pt", project_root) == "registry-name.pt"
