"""Validation comparison, plots, and deterministic winner selection."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SelectionSettings
from .utils import (
    RunContext,
    StageError,
    read_json,
    relative_to,
    require_completed,
    write_csv,
    write_json,
)


@dataclass(frozen=True)
class WinnerDecision:
    winner: str
    rule: str
    constraint_satisfied: bool
    considered_models: tuple[str, ...]
    accuracy_tie_models: tuple[str, ...]
    metrics: Mapping[str, Any]
    fallback_score: float | None = None


def _finite_number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _best_by_smallest(rows: Sequence[Mapping[str, Any]], key: str) -> list[Mapping[str, Any]]:
    finite = [row for row in rows if math.isfinite(_finite_number(row.get(key)))]
    if not finite:
        return list(rows)
    best = min(_finite_number(row.get(key)) for row in finite)
    return [row for row in finite if math.isclose(_finite_number(row.get(key)), best, abs_tol=1e-12)]


def select_winner(
    rows: Sequence[Mapping[str, Any]], settings: SelectionSettings
) -> WinnerDecision:
    """Apply the documented validation-only selection rule.

    Rows must already include ``safety_class_mean_recall`` derived from the
    fixed-threshold per-class validation metrics. Failed rows are excluded.
    """

    eligible = [
        row
        for row in rows
        if row.get("status") == "completed"
        and math.isfinite(_finite_number(row.get(settings.primary_metric)))
        and math.isfinite(_finite_number(row.get("fps")))
        and math.isfinite(_finite_number(row.get("latency_p95_ms")))
    ]
    if not eligible:
        raise StageError("No successfully evaluated and benchmarked model is eligible for selection")

    fallback_score: float | None = None
    constraint_satisfied = True
    if settings.minimum_fps is None:
        pool = eligible
        rule_prefix = "No minimum FPS constraint was configured."
    else:
        passing = [row for row in eligible if _finite_number(row.get("fps")) >= settings.minimum_fps]
        if passing:
            pool = passing
            rule_prefix = f"Models below {settings.minimum_fps:g} FPS were eliminated."
        else:
            constraint_satisfied = False
            scored: list[tuple[float, Mapping[str, Any]]] = []
            for row in eligible:
                attainment = min(1.0, _finite_number(row.get("fps"), 0.0) / settings.minimum_fps)
                score = _finite_number(row.get(settings.primary_metric), 0.0) * attainment
                scored.append((score, row))
            fallback_score = max(score for score, _ in scored)
            pool = [row for score, row in scored if math.isclose(score, fallback_score, abs_tol=1e-12)]
            rule_prefix = (
                f"No model met {settings.minimum_fps:g} FPS; used the declared fallback "
                "score mAP50-95 × min(1, FPS/minimum_FPS)."
            )

    if constraint_satisfied:
        best_map = max(_finite_number(row.get(settings.primary_metric)) for row in pool)
        accuracy_ties = [
            row
            for row in pool
            if best_map - _finite_number(row.get(settings.primary_metric))
            <= settings.map_tie_tolerance + 1e-12
        ]
    else:
        accuracy_ties = list(pool)

    finalists = accuracy_ties
    safety_values = [
        _finite_number(row.get("safety_class_mean_recall")) for row in finalists
    ]
    finite_safety = [value for value in safety_values if math.isfinite(value)]
    if finite_safety:
        best_safety = max(finite_safety)
        finalists = [
            row
            for row in finalists
            if math.isclose(
                _finite_number(row.get("safety_class_mean_recall")), best_safety, abs_tol=1e-12
            )
        ]
    finalists = _best_by_smallest(finalists, "latency_p95_ms")
    finalists = _best_by_smallest(finalists, "checkpoint_size_mib")
    winner_row = min(finalists, key=lambda row: str(row.get("model_id")))
    winner = str(winner_row["model_id"])

    rule = (
        f"{rule_prefix} Highest validation mAP50-95 wins; models within "
        f"{settings.map_tie_tolerance:g} use safety-class mean recall, then p95 latency, "
        "then checkpoint size, then model id for exact reproducibility."
    )
    return WinnerDecision(
        winner=winner,
        rule=rule,
        constraint_satisfied=constraint_satisfied,
        considered_models=tuple(str(row["model_id"]) for row in pool),
        accuracy_tie_models=tuple(str(row["model_id"]) for row in accuracy_ties),
        metrics={
            "map50_95": _finite_number(winner_row.get("map50_95")),
            "safety_class_mean_recall": _finite_number(
                winner_row.get("safety_class_mean_recall")
            ),
            "latency_p95_ms": _finite_number(winner_row.get("latency_p95_ms")),
            "fps": _finite_number(winner_row.get("fps")),
            "checkpoint_size_mib": _finite_number(winner_row.get("checkpoint_size_mib")),
        },
        fallback_score=fallback_score,
    )


def _load_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except StageError:
        return {}


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value: Any = mapping
        for part in key.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return default


def _resolve_artifact_path(context: RunContext, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else context.run_dir / path


def _training_overfit_signal(context: RunContext, training: Mapping[str, Any]) -> str:
    csv_path = _resolve_artifact_path(context, training.get("results_csv"))
    if csv_path is None or not csv_path.is_file():
        return "not assessable (epoch metrics unavailable)"
    try:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return "not assessable (epoch metrics unreadable)"
    if len(rows) < 5:
        return "not assessable (too few epochs)"
    headers = rows[0].keys()
    train_key = next((key for key in headers if "train/box_loss" in key), None)
    val_key = next((key for key in headers if "val/box_loss" in key), None)
    if train_key is None or val_key is None:
        return "not assessable (box-loss columns unavailable)"
    train = [_finite_number(row.get(train_key)) for row in rows]
    val = [_finite_number(row.get(val_key)) for row in rows]
    if not all(math.isfinite(item) for item in train + val):
        return "not assessable (non-finite epoch metrics)"
    rising_val = val[-1] > min(val) * 1.10
    falling_train = train[-1] < train[max(0, len(train) // 2)]
    return "possible late-epoch overfitting" if rising_val and falling_train else "no clear signal"


def _per_class_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = metrics.get("per_class", [])
    if isinstance(raw, Mapping):
        rows: list[dict[str, Any]] = []
        for name, values in raw.items():
            row = dict(values) if isinstance(values, Mapping) else {}
            row.setdefault("class_name", str(name))
            rows.append(row)
        return rows
    if isinstance(raw, list):
        return [dict(row) for row in raw if isinstance(row, Mapping)]
    return []


def _comparison_row(context: RunContext, model_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_dir = context.run_dir / "models" / model_id
    training = _load_optional(model_dir / "training.json")
    metrics = _load_optional(model_dir / "evaluation" / "metrics.json")
    if not metrics:
        metrics = _load_optional(model_dir / "evaluation.json")
    benchmark = _load_optional(model_dir / "benchmark.json")
    if not benchmark:
        benchmark = _load_optional(model_dir / "benchmark" / "benchmark.json")
    per_class = _per_class_rows(metrics)
    safety_values = [
        _finite_number(
            _first(row, "detection_recall", "coverage.recall", "recall")
        )
        for row in per_class
        if row.get("class_name") in context.config.selection.safety_classes
        and math.isfinite(
            _finite_number(
                _first(row, "detection_recall", "coverage.recall", "recall")
            )
        )
    ]
    statuses = [training.get("status"), metrics.get("status"), benchmark.get("status")]
    status = "completed" if all(value == "completed" for value in statuses) else "failed"
    errors = [
        str(_first(payload, "error", "failure_message"))
        for payload in (training, metrics, benchmark)
        if _first(payload, "error", "failure_message")
    ]
    row = {
        "model_id": model_id,
        "status": status,
        "failure_message": "; ".join(errors) if errors else (None if status == "completed" else "missing or incomplete artifact"),
        "precision": _first(metrics, "precision", "aggregate.precision"),
        "recall": _first(metrics, "recall", "aggregate.recall"),
        "map50": _first(metrics, "map50", "aggregate.map50"),
        "map75": _first(metrics, "map75", "aggregate.map75"),
        "map50_95": _first(metrics, "map50_95", "aggregate.map50_95"),
        "ground_truth_objects": _first(metrics, "coverage.ground_truth_count", "coverage.gt"),
        "prediction_count": _first(metrics, "coverage.prediction_count", "coverage.predictions"),
        "true_positives": _first(metrics, "coverage.true_positives", "coverage.tp"),
        "false_positives": _first(metrics, "coverage.false_positives", "coverage.fp"),
        "false_negatives": _first(metrics, "coverage.false_negatives", "coverage.fn"),
        "detection_recall": _first(metrics, "coverage.recall", "coverage.detection_recall"),
        "safety_class_mean_recall": float(np.mean(safety_values)) if safety_values else None,
        "parameters": _first(metrics, "parameters", "efficiency.parameters"),
        "flops_g": _first(metrics, "flops_g", "efficiency.flops_g"),
        "checkpoint_size_mib": _first(metrics, "checkpoint_size_mib", "efficiency.checkpoint_size_mib"),
        "best_epoch": _first(training, "best_epoch"),
        "training_duration_seconds": _first(
            training, "training_duration_seconds", "duration_seconds"
        ),
        "training_batch": _first(
            training, "batch_size", "actual_settings.batch", "batch"
        ),
        "early_stopping": _first(training, "early_stopping", "early_stopped"),
        "preprocess_ms": _first(metrics, "speed.preprocess_ms", "preprocess_ms"),
        "inference_ms_ultralytics": _first(metrics, "speed.inference_ms", "inference_ms"),
        "postprocess_ms": _first(metrics, "speed.postprocess_ms", "postprocess_ms"),
        "latency_mean_ms": _first(benchmark, "latency_mean_ms", "mean_ms"),
        "latency_median_ms": _first(benchmark, "latency_median_ms", "median_ms"),
        "latency_std_ms": _first(benchmark, "latency_std_ms", "std_ms"),
        "latency_p90_ms": _first(benchmark, "latency_p90_ms", "p90_ms"),
        "latency_p95_ms": _first(benchmark, "latency_p95_ms", "p95_ms"),
        "fps": _first(benchmark, "fps"),
        "device": _first(benchmark, "device.name", "device_name", "device"),
        "precision_mode": _first(benchmark, "precision", "precision_mode"),
        "overfitting_evidence": _training_overfit_signal(context, training),
    }
    return row, per_class


def _style_axis(axis: Any, title: str, ylabel: str = "") -> None:
    axis.set_title(title, fontweight="bold")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _grouped_bar(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[tuple[str, str]],
    title: str,
    ylabel: str,
) -> None:
    labels = [str(row["model_id"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(fields))
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.8))
    for index, (field, display) in enumerate(fields):
        values = [_finite_number(row.get(field), 0.0) for row in rows]
        axis.bar(x + (index - (len(fields) - 1) / 2) * width, values, width, label=display)
    axis.set_xticks(x, labels)
    axis.legend(frameon=False)
    _style_axis(axis, title, ylabel)
    _save_figure(figure, path)


def _single_bar(path: Path, rows: Sequence[Mapping[str, Any]], field: str, title: str, ylabel: str) -> None:
    labels = [str(row["model_id"]) for row in rows]
    values = [_finite_number(row.get(field), 0.0) for row in rows]
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.6), 4.8))
    bars = axis.bar(labels, values, color="#377eb8")
    axis.bar_label(bars, fmt="%.2f", padding=3)
    _style_axis(axis, title, ylabel)
    _save_figure(figure, path)


def _tradeoff_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for row in rows:
        x = _finite_number(row.get("latency_mean_ms"))
        y = _finite_number(row.get("map50_95"))
        if math.isfinite(x) and math.isfinite(y):
            axis.scatter(x, y, s=85)
            axis.annotate(str(row["model_id"]), (x, y), xytext=(6, 6), textcoords="offset points")
    axis.set_xlabel("Mean local latency (ms/image, batch 1)")
    _style_axis(axis, "Validation accuracy–latency tradeoff", "mAP50–95")
    _save_figure(figure, path)


def _heatmap(
    path: Path,
    model_ids: Sequence[str],
    class_names: Sequence[str],
    values: np.ndarray,
    title: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(max(7, len(model_ids) * 1.5), max(5, len(class_names) * 0.38))
    )
    image = axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_xticks(np.arange(len(model_ids)), model_ids)
    axis.set_yticks(np.arange(len(class_names)), class_names)
    axis.set_title(title, fontweight="bold")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            text = "—" if not math.isfinite(float(value)) else f"{value:.2f}"
            axis.text(column_index, row_index, text, ha="center", va="center", fontsize=7, color="white" if math.isfinite(float(value)) and value < 0.55 else "black")
    figure.colorbar(image, ax=axis, label="score")
    _save_figure(figure, path)


def _plots(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    per_class_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    class_names: Sequence[str],
) -> None:
    successful = [row for row in rows if row.get("status") == "completed"]
    if not successful:
        return
    _grouped_bar(
        output_dir / "accuracy_comparison.png",
        successful,
        (("map50", "mAP50"), ("map50_95", "mAP50–95")),
        "Validation accuracy",
        "score",
    )
    _grouped_bar(
        output_dir / "latency_comparison.png",
        successful,
        (("latency_mean_ms", "mean"), ("latency_p95_ms", "p95")),
        "Controlled local latency (batch 1)",
        "milliseconds/image",
    )
    _grouped_bar(
        output_dir / "precision_recall_comparison.png",
        successful,
        (("precision", "precision"), ("recall", "recall")),
        "Validation precision and recall",
        "score",
    )
    _single_bar(
        output_dir / "model_size_comparison.png",
        successful,
        "checkpoint_size_mib",
        "Best-checkpoint size",
        "MiB",
    )
    _tradeoff_plot(output_dir / "accuracy_latency_tradeoff.png", successful)
    model_ids = [str(row["model_id"]) for row in successful]
    for metric, filename, title in (
        ("map50_95", "per_class_map_heatmap.png", "Per-class validation mAP50–95"),
        ("recall", "per_class_recall_heatmap.png", "Per-class validation recall"),
    ):
        matrix = np.full((len(class_names), len(model_ids)), np.nan)
        for column, model_id in enumerate(model_ids):
            lookup = {str(row.get("class_name")): row for row in per_class_by_model.get(model_id, [])}
            for row_index, class_name in enumerate(class_names):
                matrix[row_index, column] = _finite_number(lookup.get(class_name, {}).get(metric))
        _heatmap(output_dir / filename, model_ids, class_names, matrix, title)


def _summary_markdown(
    context: RunContext,
    rows: Sequence[Mapping[str, Any]],
    decision: WinnerDecision | None,
    per_class_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    config = context.config
    successful = [row for row in rows if row.get("status") == "completed"]
    lines = [
        "# Model comparison summary",
        "",
        "## Controlled comparison",
        "",
        (
            f"All candidates used the same dataset split, seed ({config.experiment.seed}), image size "
            f"({config.training.imgsz}), epoch budget ({config.training.epochs}), validation confidence "
            f"({config.evaluation.confidence}), matching IoU ({config.evaluation.matching_iou}), and local "
            "batch-1 benchmark protocol. Candidate checkpoint architecture/size differed; any configured "
            "per-model batch override is disclosed below."
        ),
        "",
        "| Model | Status | mAP50–95 | Recall | Correct detections | Mean latency (ms) | p95 (ms) | FPS | Size (MiB) | Batch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        number = lambda key: "—" if not math.isfinite(_finite_number(row.get(key))) else f"{_finite_number(row.get(key)):.4g}"
        lines.append(
            f"| {row['model_id']} | {row['status']} | {number('map50_95')} | {number('recall')} | "
            f"{number('true_positives')} | {number('latency_mean_ms')} | {number('latency_p95_ms')} | "
            f"{number('fps')} | {number('checkpoint_size_mib')} | {number('training_batch')} |"
        )
    lines.extend(["", "## Findings", ""])
    if successful:
        best_accuracy = max(successful, key=lambda row: _finite_number(row.get("map50_95"), -math.inf))
        best_recall = max(successful, key=lambda row: _finite_number(row.get("recall"), -math.inf))
        most_correct = max(successful, key=lambda row: _finite_number(row.get("true_positives"), -math.inf))
        fastest = min(successful, key=lambda row: _finite_number(row.get("latency_mean_ms"), math.inf))
        class_scores: dict[str, dict[str, list[float]]] = {}
        for row in successful:
            model_id = str(row["model_id"])
            for class_row in per_class_by_model.get(model_id, []):
                class_name = str(class_row.get("class_name", class_row.get("class_id", "unknown")))
                bucket = class_scores.setdefault(class_name, {"map": [], "recall": []})
                class_map = _finite_number(class_row.get("map50_95"))
                class_recall = _finite_number(
                    _first(class_row, "detection_recall", "recall")
                )
                if math.isfinite(class_map):
                    bucket["map"].append(class_map)
                if math.isfinite(class_recall):
                    bucket["recall"].append(class_recall)
        weakest_text = "not assessable from available per-class metrics"
        comparable_classes = [
            (name, float(np.mean(values["map"])), float(np.mean(values["recall"])))
            for name, values in class_scores.items()
            if values["map"] and values["recall"]
        ]
        if comparable_classes:
            weakest_name, weakest_map, weakest_recall = min(
                comparable_classes, key=lambda item: (item[1], item[2], item[0])
            )
            weakest_text = (
                f"**{weakest_name}** (mean mAP50–95={weakest_map:.4f}, fixed-threshold "
                f"mean recall={weakest_recall:.4f} across completed candidates)"
            )
        accuracy_latency_text = (
            f"The accuracy leader {best_accuracy['model_id']} recorded "
            f"{_finite_number(best_accuracy.get('map50_95')):.4f} mAP50–95 at "
            f"{_finite_number(best_accuracy.get('latency_mean_ms')):.3f} ms mean latency; "
            f"the fastest model {fastest['model_id']} recorded "
            f"{_finite_number(fastest.get('map50_95')):.4f} mAP50–95 at "
            f"{_finite_number(fastest.get('latency_mean_ms')):.3f} ms."
        )
        lines.extend(
            [
                f"- Highest validation mAP50–95: **{best_accuracy['model_id']}** ({_finite_number(best_accuracy.get('map50_95')):.4f}).",
                f"- Best aggregate recall: **{best_recall['model_id']}** ({_finite_number(best_recall.get('recall')):.4f}).",
                f"- Most correctly matched detections: **{most_correct['model_id']}** ({int(_finite_number(most_correct.get('true_positives'), 0))} true positives). Raw prediction count is not used as an accuracy metric.",
                f"- Lowest controlled mean latency: **{fastest['model_id']}** ({_finite_number(fastest.get('latency_mean_ms')):.3f} ms/image; {_finite_number(fastest.get('fps')):.2f} FPS).",
                f"- Consistently weakest class by mean per-class mAP: {weakest_text}.",
                f"- Accuracy–latency tradeoff: {accuracy_latency_text}",
                "- Overfitting heuristic: " + "; ".join(
                    f"{row['model_id']} — {row['overfitting_evidence']}" for row in successful
                ) + ".",
                "- Small mAP differences near the configured tie tolerance should not be overstated on this relatively small dataset.",
            ]
        )
    failures = [row for row in rows if row.get("status") != "completed"]
    if failures:
        lines.extend(
            [
                "",
                "## Incomplete candidates",
                "",
                *[f"- **{row['model_id']}**: {row['failure_message']}" for row in failures],
                "",
                "Winner selection is blocked until every configured candidate completes successfully.",
            ]
        )
    elif decision is not None:
        lines.extend(
            [
                "",
                "## Frozen validation winner",
                "",
                f"**{decision.winner}** is selected. {decision.rule}",
                "",
                (
                    f"Winner metrics: mAP50–95={decision.metrics['map50_95']:.4f}, safety-class mean "
                    f"recall={decision.metrics['safety_class_mean_recall']:.4f}, p95 latency="
                    f"{decision.metrics['latency_p95_ms']:.3f} ms, FPS={decision.metrics['fps']:.2f}, "
                    f"checkpoint={decision.metrics['checkpoint_size_mib']:.2f} MiB."
                ),
                "",
                "This decision is frozen before the untouched test split is evaluated.",
            ]
        )
    return "\n".join(lines) + "\n"


def compare_models(context: RunContext) -> dict[str, str]:
    """Aggregate candidate artifacts, create plots, and freeze a winner."""

    require_completed(context, "evaluate", "benchmark")
    output_dir = context.run_dir / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    per_class: dict[str, list[dict[str, Any]]] = {}
    for model in context.config.models:
        row, class_rows = _comparison_row(context, model.id)
        rows.append(row)
        per_class[model.id] = class_rows

    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "comparison_split": context.config.evaluation.comparison_split,
            "model_order": [model.id for model in context.config.models],
            "models": rows,
        },
    )
    class_names = list(load_class_names(context))
    _plots(output_dir, rows, per_class, class_names)

    failures = [row for row in rows if row["status"] != "completed"]
    decision: WinnerDecision | None = None
    selection_path = output_dir / "selection.json"
    selection_md_path = output_dir / "selection.md"
    if failures:
        blocked = {
            "status": "blocked_incomplete",
            "winner": None,
            "failed_models": [row["model_id"] for row in failures],
            "reason": "Every configured candidate must finish training, validation, and benchmarking before selection.",
        }
        write_json(selection_path, blocked)
        selection_md_path.write_text(
            "# Model selection\n\nSelection is blocked because these configured candidates are incomplete: "
            + ", ".join(str(row["model_id"]) for row in failures)
            + ". No test evaluation is permitted yet.\n",
            encoding="utf-8",
        )
    else:
        decision = select_winner(rows, context.config.selection)
        decision_payload = {"status": "frozen", **asdict(decision)}
        write_json(selection_path, decision_payload)
        fallback_line = (
            f"Fallback compromise score: {decision.fallback_score:.6f}\n\n"
            if decision.fallback_score is not None
            else ""
        )
        selection_md_path.write_text(
            "# Frozen model selection\n\n"
            f"Winner: **{decision.winner}**\n\n"
            f"Rule: {decision.rule}\n\n"
            f"Constraint satisfied: {decision.constraint_satisfied}\n\n"
            f"{fallback_line}"
            f"Validation metrics: mAP50–95={decision.metrics['map50_95']:.4f}, "
            f"safety-class mean recall={decision.metrics['safety_class_mean_recall']:.4f}, "
            f"p95 latency={decision.metrics['latency_p95_ms']:.3f} ms, "
            f"FPS={decision.metrics['fps']:.2f}, checkpoint={decision.metrics['checkpoint_size_mib']:.2f} MiB.\n\n"
            "The winner is frozen before test evaluation and will not be revised from test results.\n",
            encoding="utf-8",
        )

    summary_path = output_dir / "summary.md"
    summary_path.write_text(
        _summary_markdown(context, rows, decision, per_class), encoding="utf-8"
    )
    outputs = {
        "model_comparison_csv": relative_to(csv_path, context.run_dir),
        "model_comparison_json": relative_to(json_path, context.run_dir),
        "summary": relative_to(summary_path, context.run_dir),
        "selection": relative_to(selection_path, context.run_dir),
    }
    if failures:
        raise StageError(
            "Comparison artifacts were written, but selection is blocked by failed/incomplete models: "
            + ", ".join(str(row["model_id"]) for row in failures)
        )
    return outputs


def load_class_names(context: RunContext) -> Iterable[str]:
    from .config import load_dataset_spec

    spec = load_dataset_spec(context.config.dataset.yaml)
    return (spec.names[index] for index in range(spec.nc))
