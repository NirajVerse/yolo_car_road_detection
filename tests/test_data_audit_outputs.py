from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from PIL import Image

from road_detection.config import DatasetSpec, load_config, load_dataset_spec
from road_detection.data_audit import (
    EXPECTED_CLASS_NAMES,
    _dataset_structure_issues,
    run_audit,
    verify_dataset_unchanged,
)
from road_detection.utils import StageError, create_run, runtime_dataset_yaml


def _populate_synthetic_dataset(config_path: Path) -> None:
    project = config_path.parent.parent
    descriptor = project / "data.yaml"
    payload = yaml.safe_load(descriptor.read_text(encoding="utf-8"))
    payload["nc"] = 14
    payload["names"] = dict(EXPECTED_CLASS_NAMES)
    descriptor.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for index, split in enumerate(("train", "valid", "test"), start=1):
        image_path = project / "dataset" / split / "images" / f"sample-{index}.png"
        label_path = project / "dataset" / split / "labels" / f"sample-{index}.txt"
        Image.new("RGB", (8, 8), color=(index * 30, index * 20, index * 10)).save(image_path)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")


def test_synthetic_audit_writes_schema_and_dataset_guard(config_factory, monkeypatch) -> None:
    config_path, _ = config_factory()
    _populate_synthetic_dataset(config_path)
    project = config_path.parent.parent
    nested_image = project / "dataset/train/images/nested/example.png"
    nested_label = project / "dataset/train/labels/nested/example.txt"
    nested_image.parent.mkdir(parents=True)
    nested_label.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(nested_image)
    nested_label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    context = create_run(load_config(config_path), "audit-run")

    monkeypatch.setattr(
        "road_detection.data_audit.audit_environment",
        lambda _context: {
            "python": "synthetic",
            "selected_device": "cpu",
            "torch_runtime": {
                "available": True,
                "cuda_available": False,
                "mps_available": False,
                "cuda_devices": [],
            },
        },
    )
    monkeypatch.setattr(
        "road_detection.data_audit.audit_checkpoints",
        lambda _context: (
            [
                {
                    "model_id": model.id,
                    "weights": model.weights,
                    "status": "loaded",
                    "task": "detect",
                    "parameter_count": 1,
                    "checkpoint_path": None,
                    "checkpoint_sha256": None,
                    "error": None,
                }
                for model in _context.config.models
            ],
            [],
        ),
    )

    outputs = run_audit(context)

    assert set(outputs) == {
        "audit_json",
        "audit_markdown",
        "class_distribution_csv",
        "class_distribution_plot",
        "split_summary_csv",
        "issues_csv",
    }
    for relative_path in outputs.values():
        assert (context.run_dir / relative_path).stat().st_size > 0
    fingerprint = verify_dataset_unchanged(context)
    assert len(fingerprint) == 64
    runtime_spec = load_dataset_spec(runtime_dataset_yaml(context))
    linked_image = runtime_spec.split("train") / "nested/example.png"
    copied_label = runtime_spec.split("train").with_name("labels") / "nested/example.txt"
    assert linked_image.is_symlink()
    assert linked_image.resolve() == nested_image.resolve()
    assert copied_label.is_file() and not copied_label.is_symlink()
    assert copied_label.read_text(encoding="utf-8") == nested_label.read_text(encoding="utf-8")
    copied_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(StageError, match="Runtime dataset file changed"):
        runtime_dataset_yaml(context)

    changed_label = context.config.project_root / "dataset/train/labels/sample-1.txt"
    changed_label.write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    with pytest.raises(StageError, match="changed after audit"):
        verify_dataset_unchanged(context)


def test_changed_class_semantics_are_critical_but_lamb_spelling_is_not(
    tmp_path: Path,
) -> None:
    names = dict(EXPECTED_CLASS_NAMES)
    names[0] = "Wrong semantic class"
    spec = DatasetSpec(
        yaml_path=tmp_path / "data.yaml",
        root=tmp_path,
        splits={name: tmp_path / name for name in ("train", "val", "test")},
        nc=14,
        names=names,
        metadata={},
    )
    assert any(
        issue["severity"] == "critical" and issue["code"] == "unexpected_class_name"
        for issue in _dataset_structure_issues(spec)
    )

    spelling_names = dict(EXPECTED_CLASS_NAMES)
    spelling_names[5] = "Lamb-post"
    spelling_spec = DatasetSpec(
        yaml_path=tmp_path / "data.yaml",
        root=tmp_path,
        splits=spec.splits,
        nc=14,
        names=spelling_names,
        metadata={},
    )
    spelling_issues = _dataset_structure_issues(spelling_spec)
    assert any(issue["code"] == "class_name_spelling" for issue in spelling_issues)
    assert not any(issue["severity"] == "critical" for issue in spelling_issues)
