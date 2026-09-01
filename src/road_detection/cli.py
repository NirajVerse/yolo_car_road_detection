"""Command-line entry point for the staged EcoCAR YOLO pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from .config import ConfigError, load_config
from .utils import (
    PipelineError,
    RunContext,
    create_run,
    list_run_ids,
    resume_run,
    run_stage,
    setup_logging,
)

LOGGER = logging.getLogger("road_detection")

STAGE_ORDER = (
    "audit",
    "smoke_test",
    "train",
    "evaluate",
    "benchmark",
    "compare",
    "test_winner",
    "predict",
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiment.yaml"),
        help="Experiment YAML (default: configs/experiment.yaml)",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Existing run to resume. audit/run create this id if it does not exist; "
            "other commands default to the latest run."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly rerun a completed stage within this run and replace its stage outputs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="road-detection",
        description="Reproducible staged Ultralytics YOLO road-object pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "audit": "Validate environment, configured checkpoints, and dataset; create a new run.",
        "smoke-test": "Run the isolated one-epoch smallest-model smoke test.",
        "train": "Train every configured candidate independently.",
        "evaluate": "Evaluate each candidate best.pt on validation with fixed settings.",
        "benchmark": "Measure controlled local batch-1 latency for every candidate.",
        "compare": "Aggregate validation/latency results, plot them, and freeze the winner.",
        "test-winner": "Evaluate the frozen winner once on the untouched test split.",
        "predict": "Run the frozen winner on one demonstration image.",
        "run": "Run all stages in the safe order, resuming completed stages.",
    }
    for name, description in help_text.items():
        command_parser = subparsers.add_parser(name, help=description, description=description)
        _add_common_arguments(command_parser)
        if name in {"predict", "run"}:
            command_parser.add_argument(
                "--source",
                type=Path,
                help="Demonstration image; otherwise prediction.source or a deterministic test image is used.",
            )
    return parser


def _resolve_context(args: argparse.Namespace) -> RunContext:
    config = load_config(args.config)
    starting_command = args.command in {"audit", "run"}
    if starting_command:
        available = set(list_run_ids(config))
        if args.run_id and args.run_id in available:
            context = resume_run(config, args.run_id)
        else:
            context = create_run(config, args.run_id)
    else:
        context = resume_run(config, args.run_id)
    setup_logging(context.run_dir, args.verbose)
    LOGGER.info("Experiment: %s | run: %s", config.experiment.name, context.run_id)
    return context


def _single_stage(
    context: RunContext,
    stage_name: str,
    function: Callable[..., Any],
    args: argparse.Namespace,
    **kwargs: Any,
) -> None:
    force = bool(args.force)
    if force:
        stage_index = STAGE_ORDER.index(stage_name)
        completed_downstream = [
            stage
            for stage in STAGE_ORDER[stage_index + 1 :]
            if context.stage_status(stage) == "completed"
        ]
        if completed_downstream:
            raise PipelineError(
                f"Cannot force {stage_name} after downstream stage(s) completed: "
                f"{', '.join(completed_downstream)}. Start a new audit/run instead."
            )
        if stage_name == "test_winner" and context.stage_status(stage_name) == "completed":
            raise PipelineError(
                "The frozen winner has already been evaluated on test in this run; "
                "start a new run rather than evaluating test twice."
            )
    result = run_stage(context, stage_name, function, force=force, **kwargs)
    if result is not None:
        LOGGER.info("Completed stage %s", stage_name)


def _dispatch_single(context: RunContext, args: argparse.Namespace) -> None:
    if args.command == "audit":
        from .data_audit import run_audit

        _single_stage(context, "audit", run_audit, args)
    elif args.command == "smoke-test":
        from .train import run_smoke_test

        _single_stage(context, "smoke_test", run_smoke_test, args)
    elif args.command == "train":
        from .train import train_candidates

        _single_stage(
            context,
            "train",
            train_candidates,
            args,
            retrain_completed=args.force,
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate_candidates

        _single_stage(context, "evaluate", evaluate_candidates, args)
    elif args.command == "benchmark":
        from .benchmark import benchmark_candidates

        _single_stage(context, "benchmark", benchmark_candidates, args)
    elif args.command == "compare":
        from .compare import compare_models

        _single_stage(context, "compare", compare_models, args)
    elif args.command == "test-winner":
        from .evaluate import evaluate_winner_test

        _single_stage(context, "test_winner", evaluate_winner_test, args)
    elif args.command == "predict":
        from .predict import predict_winner

        _single_stage(context, "predict", predict_winner, args, source=args.source)
        context.finalize()
    else:
        raise PipelineError(f"Unsupported command: {args.command}")


def _run_all(context: RunContext, args: argparse.Namespace) -> None:
    from .benchmark import benchmark_candidates
    from .compare import compare_models
    from .data_audit import run_audit
    from .evaluate import evaluate_candidates, evaluate_winner_test
    from .predict import predict_winner
    from .train import run_smoke_test, train_candidates

    stages: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
        ("audit", run_audit, {}),
        ("smoke_test", run_smoke_test, {}),
        ("train", train_candidates, {"retrain_completed": args.force}),
        ("evaluate", evaluate_candidates, {}),
        ("benchmark", benchmark_candidates, {}),
        ("compare", compare_models, {}),
        ("test_winner", evaluate_winner_test, {}),
        ("predict", predict_winner, {"source": args.source}),
    ]
    for name, function, kwargs in stages:
        _single_stage(context, name, function, args, **kwargs)
    context.finalize()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        context = _resolve_context(args)
        if args.command == "run":
            _run_all(context, args)
        else:
            _dispatch_single(context, args)
    except (ConfigError, PipelineError) as exc:
        LOGGER.error("%s", exc)
        if not LOGGER.handlers:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        return 130
    except Exception as exc:  # Preserve a useful traceback in verbose/log output.
        LOGGER.exception("Unexpected pipeline failure: %s", exc)
        return 1
    print(f"Run ID: {context.run_id}")
    print(f"Artifacts: {context.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
