from __future__ import annotations

import pytest

from road_detection.compare import select_winner
from road_detection.config import SelectionSettings
from road_detection.utils import StageError


def _settings(minimum_fps=None, tolerance=0.005) -> SelectionSettings:
    return SelectionSettings(
        primary_metric="map50_95",
        minimum_fps=minimum_fps,
        map_tie_tolerance=tolerance,
        safety_classes=("Person", "Bike"),
        tie_breaker="latency_p95_ms",
    )


def _row(model_id, map_value, safety, p95, size, fps=50.0, status="completed"):
    return {
        "model_id": model_id,
        "status": status,
        "map50_95": map_value,
        "safety_class_mean_recall": safety,
        "latency_p95_ms": p95,
        "checkpoint_size_mib": size,
        "fps": fps,
    }


def test_highest_map_wins_outside_tolerance() -> None:
    decision = select_winner(
        [_row("accurate", 0.70, 0.5, 20, 10), _row("safe", 0.69, 0.9, 10, 5)],
        _settings(tolerance=0.005),
    )
    assert decision.winner == "accurate"


def test_safety_recall_breaks_accuracy_tie() -> None:
    decision = select_winner(
        [_row("a", 0.700, 0.7, 20, 10), _row("b", 0.698, 0.8, 30, 20)],
        _settings(),
    )
    assert decision.winner == "b"
    assert decision.accuracy_tie_models == ("a", "b")


def test_latency_then_size_then_id_break_exact_ties() -> None:
    rows = [
        _row("z", 0.7, 0.8, 10, 4),
        _row("b", 0.7, 0.8, 9, 5),
        _row("a", 0.7, 0.8, 9, 5),
    ]
    assert select_winner(rows, _settings()).winner == "a"


def test_failed_models_are_excluded() -> None:
    rows = [_row("failed", 0.99, 0.99, 1, 1, status="failed"), _row("good", 0.6, 0.6, 20, 5)]
    assert select_winner(rows, _settings()).winner == "good"


def test_minimum_fps_eliminates_slow_model() -> None:
    rows = [_row("slow", 0.8, 0.9, 100, 10, fps=20), _row("fast", 0.7, 0.7, 10, 5, fps=40)]
    decision = select_winner(rows, _settings(minimum_fps=30))
    assert decision.winner == "fast"
    assert decision.constraint_satisfied is True


def test_no_model_meeting_fps_uses_declared_fallback() -> None:
    rows = [_row("accurate", 0.8, 0.8, 20, 10, fps=20), _row("balanced", 0.7, 0.7, 15, 8, fps=28)]
    decision = select_winner(rows, _settings(minimum_fps=40))
    assert decision.winner == "balanced"
    assert decision.constraint_satisfied is False
    assert decision.fallback_score == pytest.approx(0.49)


def test_no_successful_row_raises() -> None:
    with pytest.raises(StageError, match="No successfully evaluated"):
        select_winner([_row("bad", 0.9, 0.9, 1, 1, status="failed")], _settings())
