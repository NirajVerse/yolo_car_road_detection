"""Controlled, same-input batch-one latency benchmarking for trained candidates."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import load_dataset_spec
from .evaluate import find_best_checkpoint
from .utils import (
    RunContext,
    StageError,
    basic_environment,
    import_ultralytics_yolo,
    inference_half_enabled,
    relative_to,
    require_completed,
    sha256_file,
    supported_images,
    ultralytics_device,
    write_json,
)

LOGGER = logging.getLogger("road_detection")


def _representative_images(images: Sequence[Path], sample_count: int) -> list[Path]:
    """Choose deterministic, evenly spaced images from a sorted split."""

    if not images:
        return []
    count = min(sample_count, len(images))
    if count == len(images):
        return list(images)
    indices = np.linspace(0, len(images) - 1, num=count, dtype=int)
    return [images[int(index)] for index in indices]


def _preload_images(paths: Sequence[Path]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise StageError(f"Could not preload benchmark image {path}: {exc}") from exc
        # Ultralytics treats numpy input as OpenCV-style BGR.
        arrays.append(np.ascontiguousarray(rgb[:, :, ::-1]))
    return arrays


def _torch_runtime(model: Any) -> tuple[dict[str, Any], Callable[[], None]]:
    """Describe the actual runtime and return a device synchronization callback."""

    try:
        import torch
    except ImportError as exc:
        raise StageError("PyTorch is required for latency benchmarking") from exc

    predictor = getattr(model, "predictor", None)
    module = getattr(predictor, "model", None)
    if module is None:
        module = getattr(model, "model", None)
    device = "unknown"
    precision = "unknown"
    try:
        parameter = next(module.parameters())
        device = str(parameter.device)
        precision = str(parameter.dtype).replace("torch.", "")
    except (AttributeError, StopIteration, TypeError):
        pass

    def synchronize_cpu() -> None:
        pass

    synchronize: Callable[[], None] = synchronize_cpu
    device_name = device
    if device.startswith("cuda") and torch.cuda.is_available():
        device_object = torch.device(device)
        device_name = torch.cuda.get_device_name(device_object)

        def synchronize_cuda() -> None:
            torch.cuda.synchronize(device_object)

        synchronize = synchronize_cuda
    runtime = {
        "device": device,
        "device_name": device_name,
        "precision": precision,
        "torch_version": str(torch.__version__),
        "cuda_version": getattr(torch.version, "cuda", None),
        "rocm_version": getattr(torch.version, "hip", None),
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }
    return runtime, synchronize


def _latency_statistics(latencies_ms: Sequence[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("At least one latency measurement is required")
    values = np.asarray(latencies_ms, dtype=np.float64)
    mean = float(np.mean(values))
    median = float(np.median(values))
    standard_deviation = float(np.std(values, ddof=0))
    return {
        "latency_mean_ms": mean,
        "latency_median_ms": median,
        "latency_std_ms": standard_deviation,
        # Verb-first aliases are kept for human-readable downstream exports.
        "mean_latency_ms": mean,
        "median_latency_ms": median,
        "std_latency_ms": standard_deviation,
        "latency_p90_ms": float(np.percentile(values, 90)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "fps": float(1000.0 / mean) if mean > 0.0 else float("inf"),
    }


def _predict_once(model: Any, image: np.ndarray, context: RunContext) -> None:
    settings = context.config.evaluation
    model.predict(
        source=image,
        imgsz=context.config.training.imgsz,
        batch=1,
        device=ultralytics_device(context.config.training.device),
        conf=settings.confidence,
        iou=settings.prediction_iou,
        max_det=settings.max_detections,
        half=inference_half_enabled(
            context.config.training.device, context.config.training.amp
        ),
        save=False,
        stream=False,
        verbose=False,
    )


def _benchmark_model(
    context: RunContext,
    model_id: str,
    checkpoint: Path,
    paths: Sequence[Path],
    images: Sequence[np.ndarray],
) -> dict[str, Any]:
    YOLO = import_ultralytics_yolo()
    # Checkpoint loading is deliberately outside every timed region.
    model = YOLO(str(checkpoint), task="detect")
    settings = context.config.benchmark

    # The first call initializes the predictor. It is not part of warmup or timing.
    _predict_once(model, images[0], context)
    torch_runtime, synchronize = _torch_runtime(model)

    for iteration in range(settings.warmup_iterations):
        _predict_once(model, images[iteration % len(images)], context)
    synchronize()

    latencies_ms: list[float] = []
    for iteration in range(settings.measured_iterations):
        image = images[iteration % len(images)]
        synchronize()
        start_ns = time.perf_counter_ns()
        _predict_once(model, image, context)
        synchronize()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        latencies_ms.append(elapsed_ms)

    runtime = basic_environment()
    result = {
        "status": "completed",
        "model_id": model_id,
        "checkpoint": relative_to(checkpoint, context.run_dir),
        "checkpoint_sha256": sha256_file(checkpoint),
        **_latency_statistics(latencies_ms),
        "latencies_ms": latencies_ms,
        "device": torch_runtime["device"],
        "device_name": torch_runtime["device_name"],
        "precision": torch_runtime["precision"],
        "framework": "PyTorch",
        "framework_versions": runtime.get("dependencies", {}),
        "runtime": torch_runtime,
        "batch": 1,
        "imgsz": context.config.training.imgsz,
        "warmup_iterations": settings.warmup_iterations,
        "measured_iterations": settings.measured_iterations,
        "representative_image_count": len(images),
        "configured_sample_count": settings.sample_count,
        "input_images": [relative_to(path, context.config.project_root) for path in paths],
        "timed_scope": "preloaded ndarray preprocessing, inference, and postprocessing",
        "excluded_from_timing": [
            "checkpoint loading",
            "predictor initialization",
            "image disk reads",
            "warmup",
            "plotting",
            "file saving",
        ],
        "settings": {
            "confidence": context.config.evaluation.confidence,
            "prediction_iou": context.config.evaluation.prediction_iou,
            "max_detections": context.config.evaluation.max_detections,
            "half_requested": context.config.training.amp,
            "half_enabled": inference_half_enabled(
                context.config.training.device, context.config.training.amp
            ),
        },
    }
    return result


def benchmark_candidates(context: RunContext) -> dict[str, Any]:
    """Benchmark all trained candidates using one preloaded validation input set."""

    require_completed(context, "evaluate")
    spec = load_dataset_spec(context.dataset_yaml)
    available = supported_images(spec.split(context.config.evaluation.comparison_split))
    paths = _representative_images(available, context.config.benchmark.sample_count)
    if not paths:
        raise StageError("No validation images are available for the controlled benchmark")
    # Disk reads happen exactly once, before any model-specific timing starts.
    images = _preload_images(paths)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for candidate in context.config.models:
        model_dir = context.run_dir / "models" / candidate.id
        model_dir.mkdir(parents=True, exist_ok=True)
        output_path = model_dir / "benchmark.json"
        LOGGER.info("Benchmarking %s on %d preloaded images", candidate.id, len(images))
        try:
            checkpoint = find_best_checkpoint(context, candidate.id)
            payload = _benchmark_model(
                context, candidate.id, checkpoint, paths=paths, images=images
            )
            write_json(output_path, payload)
            successes.append(payload)
        except Exception as exc:
            LOGGER.exception("Benchmark failed for %s", candidate.id)
            failure = {"model_id": candidate.id, "status": "failed", "error": str(exc)}
            write_json(output_path, failure)
            failures.append(failure)

    summary = {
        "status": "failed" if failures else "completed",
        "input_images": [
            relative_to(path, context.config.project_root) for path in paths
        ],
        "successful_models": [row["model_id"] for row in successes],
        "failed_models": failures,
        "models": [
            {
                "model_id": row["model_id"],
                "status": row["status"],
                "benchmark": relative_to(
                    context.run_dir / "models" / row["model_id"] / "benchmark.json",
                    context.run_dir,
                ),
            }
            for row in successes
        ],
    }
    summary_path = context.run_dir / "comparison" / "benchmark_summary.json"
    write_json(summary_path, summary)
    if failures:
        failed_ids = ", ".join(row["model_id"] for row in failures)
        raise StageError(
            f"Latency benchmark was incomplete; failed models: {failed_ids}. "
            "Successful measurements were preserved."
        )
    return {"benchmark_summary": relative_to(summary_path, context.run_dir)}


# Short alias used by callers that expose the stage as ``benchmark``.
benchmark = benchmark_candidates
