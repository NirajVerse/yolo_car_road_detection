from __future__ import annotations

import pytest

from road_detection.data_audit import (
    LabelValidationError,
    parse_yolo_label_row,
    resolve_label_directory,
)


def test_valid_yolo_row() -> None:
    label = parse_yolo_label_row("13 0.5 0.25 0.2 1.0", 14)
    assert label.as_tuple() == (13, 0.5, 0.25, 0.2, 1.0)


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ("", "empty_row"),
        ("1 0.5 0.5 0.2", "malformed_label"),
        ("1.0 0.5 0.5 0.2 0.2", "invalid_class_id"),
        ("14 0.5 0.5 0.2 0.2", "invalid_class_id"),
        ("-1 0.5 0.5 0.2 0.2", "invalid_class_id"),
        ("1 nope 0.5 0.2 0.2", "invalid_coordinate"),
        ("1 nan 0.5 0.2 0.2", "invalid_coordinate"),
        ("1 1.01 0.5 0.2 0.2", "invalid_coordinate"),
        ("1 0.5 -0.1 0.2 0.2", "invalid_coordinate"),
        ("1 0.5 0.5 0 0.2", "invalid_box_size"),
        ("1 0.5 0.5 0.2 0", "invalid_box_size"),
    ],
)
def test_invalid_yolo_rows(row: str, code: str) -> None:
    with pytest.raises(LabelValidationError) as raised:
        parse_yolo_label_row(row, 14)
    assert raised.value.code == code


def test_label_directory_preserves_nested_prefix(tmp_path) -> None:
    images = tmp_path / "valid" / "images"
    assert resolve_label_directory(images) == tmp_path / "valid" / "labels"
