"""Configuration parsing, validation, and dataset path resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment or dataset configuration is invalid."""


@dataclass(frozen=True)
class ExperimentSettings:
    name: str
    seed: int
    output_root: Path


@dataclass(frozen=True)
class DatasetSettings:
    yaml: Path


@dataclass(frozen=True)
class ModelSettings:
    id: str
    weights: str
    batch: int | None = None


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    imgsz: int
    batch: int
    device: str | int
    workers: int
    patience: int
    pretrained: bool
    deterministic: bool
    amp: bool
    cache: bool | str
    resume: bool


@dataclass(frozen=True)
class SmokeTestSettings:
    epochs: int = 1
    fraction: float = 0.05
    max_predictions: int = 1


@dataclass(frozen=True)
class EvaluationSettings:
    comparison_split: str
    batch: int
    confidence: float
    prediction_iou: float
    matching_iou: float
    max_detections: int


@dataclass(frozen=True)
class BenchmarkSettings:
    batch: int
    warmup_iterations: int
    measured_iterations: int
    sample_count: int = 25


@dataclass(frozen=True)
class SelectionSettings:
    primary_metric: str
    minimum_fps: float | None
    map_tie_tolerance: float
    safety_classes: tuple[str, ...]
    tie_breaker: str


@dataclass(frozen=True)
class TestSettings:
    evaluate_winner_only: bool


@dataclass(frozen=True)
class PredictionSettings:
    source: Path | None
    confidence: float


@dataclass(frozen=True)
class PipelineConfig:
    """Fully validated experiment configuration with resolved filesystem paths."""

    experiment: ExperimentSettings
    dataset: DatasetSettings
    models: tuple[ModelSettings, ...]
    training: TrainingSettings
    smoke_test: SmokeTestSettings
    evaluation: EvaluationSettings
    benchmark: BenchmarkSettings
    selection: SelectionSettings
    test: TestSettings
    prediction: PredictionSettings
    config_path: Path
    project_root: Path

    def model(self, model_id: str) -> ModelSettings:
        for candidate in self.models:
            if candidate.id == model_id:
                return candidate
        raise ConfigError(f"Unknown model id: {model_id}")

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML/JSON-safe resolved configuration."""

        payload = asdict(self)
        payload.pop("config_path")
        payload.pop("project_root")
        payload["experiment"]["output_root"] = str(self.experiment.output_root)
        payload["dataset"]["yaml"] = str(self.dataset.yaml)
        payload["models"] = [asdict(model) for model in self.models]
        payload["selection"]["safety_classes"] = list(self.selection.safety_classes)
        payload["prediction"]["source"] = (
            str(self.prediction.source) if self.prediction.source else None
        )
        return payload


@dataclass(frozen=True)
class DatasetSpec:
    """Normalized YOLO dataset description."""

    yaml_path: Path
    root: Path
    splits: Mapping[str, Path]
    nc: int
    names: Mapping[int, str]
    metadata: Mapping[str, Any]

    def split(self, name: str) -> Path:
        try:
            return self.splits[name]
        except KeyError as exc:
            raise ConfigError(f"Dataset has no {name!r} split") from exc


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _required(section: Mapping[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required field: {section_name}.{key}")
    return section[key]


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be {comparator}")
    return value


def _probability(value: Any, name: str, *, zero_allowed: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if zero_allowed else result > 0.0
    if not lower_ok or result > 1.0:
        interval = "[0, 1]" if zero_allowed else "(0, 1]"
        raise ConfigError(f"{name} must be in {interval}")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value


def _resolve_project_path(value: Any, project_root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_checkpoint_name(weights: str, name: str) -> None:
    if not weights.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    stem = Path(weights).stem.lower()
    forbidden = ("-cls", "-seg", "-pose", "-obb")
    if any(token in stem for token in forbidden):
        raise ConfigError(f"{name} is not a standard axis-aligned detection checkpoint: {weights}")


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a configuration.

    Relative paths are resolved from the project root. For the conventional
    ``configs/experiment.yaml`` layout, the project root is the parent of
    ``configs``; otherwise it is the configuration file's parent directory.
    """

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    root = _mapping(raw, "configuration")
    project_root = (
        config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    ).resolve()

    exp = _mapping(_required(root, "experiment", "configuration"), "experiment")
    name = _required(exp, "name", "experiment")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("experiment.name must be a non-empty string")
    if any(char in name for char in "/\\") or name in {".", ".."}:
        raise ConfigError("experiment.name must be a safe directory name")
    experiment = ExperimentSettings(
        name=name,
        seed=_positive_int(_required(exp, "seed", "experiment"), "experiment.seed", allow_zero=True),
        output_root=_resolve_project_path(
            _required(exp, "output_root", "experiment"), project_root, "experiment.output_root"
        ),
    )

    dataset_raw = _mapping(_required(root, "dataset", "configuration"), "dataset")
    dataset_path = _resolve_project_path(
        _required(dataset_raw, "yaml", "dataset"), project_root, "dataset.yaml"
    )
    if not dataset_path.is_file():
        raise ConfigError(f"dataset.yaml does not exist: {dataset_path}")
    dataset = DatasetSettings(yaml=dataset_path)

    model_rows = _required(root, "models", "configuration")
    if not isinstance(model_rows, list) or not model_rows:
        raise ConfigError("models must be a non-empty list")
    models: list[ModelSettings] = []
    ids: set[str] = set()
    for index, item in enumerate(model_rows):
        row = _mapping(item, f"models[{index}]")
        model_id = _required(row, "id", f"models[{index}]")
        weights = _required(row, "weights", f"models[{index}]")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ConfigError(f"models[{index}].id must be a non-empty string")
        if any(char in model_id for char in "/\\") or model_id in {".", ".."}:
            raise ConfigError(f"models[{index}].id must be a safe directory name")
        if model_id in ids:
            raise ConfigError(f"Duplicate model id: {model_id}")
        if not isinstance(weights, str):
            raise ConfigError(f"models[{index}].weights must be a string")
        _validate_checkpoint_name(weights, f"models[{index}].weights")
        batch_override = row.get("batch")
        if batch_override is not None:
            batch_override = _positive_int(batch_override, f"models[{index}].batch")
        models.append(ModelSettings(id=model_id, weights=weights, batch=batch_override))
        ids.add(model_id)

    train = _mapping(_required(root, "training", "configuration"), "training")
    device = _required(train, "device", "training")
    if not isinstance(device, (str, int)) or isinstance(device, bool):
        raise ConfigError("training.device must be a string or integer")
    cache = _required(train, "cache", "training")
    if not isinstance(cache, (bool, str)):
        raise ConfigError("training.cache must be a boolean or Ultralytics cache mode")
    training = TrainingSettings(
        epochs=_positive_int(_required(train, "epochs", "training"), "training.epochs"),
        imgsz=_positive_int(_required(train, "imgsz", "training"), "training.imgsz"),
        batch=_positive_int(_required(train, "batch", "training"), "training.batch"),
        device=device,
        workers=_positive_int(
            _required(train, "workers", "training"), "training.workers", allow_zero=True
        ),
        patience=_positive_int(
            _required(train, "patience", "training"), "training.patience", allow_zero=True
        ),
        pretrained=_bool(_required(train, "pretrained", "training"), "training.pretrained"),
        deterministic=_bool(
            _required(train, "deterministic", "training"), "training.deterministic"
        ),
        amp=_bool(_required(train, "amp", "training"), "training.amp"),
        cache=cache,
        resume=_bool(_required(train, "resume", "training"), "training.resume"),
    )

    smoke_raw = _mapping(root.get("smoke_test", {}), "smoke_test")
    smoke_test = SmokeTestSettings(
        epochs=_positive_int(smoke_raw.get("epochs", 1), "smoke_test.epochs"),
        fraction=_probability(smoke_raw.get("fraction", 0.05), "smoke_test.fraction", zero_allowed=False),
        max_predictions=_positive_int(
            smoke_raw.get("max_predictions", 1), "smoke_test.max_predictions"
        ),
    )

    eval_raw = _mapping(_required(root, "evaluation", "configuration"), "evaluation")
    split = _required(eval_raw, "comparison_split", "evaluation")
    if split not in {"val"}:
        raise ConfigError("evaluation.comparison_split must be 'val'")
    evaluation = EvaluationSettings(
        comparison_split=split,
        batch=_positive_int(_required(eval_raw, "batch", "evaluation"), "evaluation.batch"),
        confidence=_probability(
            _required(eval_raw, "confidence", "evaluation"), "evaluation.confidence"
        ),
        prediction_iou=_probability(
            _required(eval_raw, "prediction_iou", "evaluation"), "evaluation.prediction_iou"
        ),
        matching_iou=_probability(
            _required(eval_raw, "matching_iou", "evaluation"), "evaluation.matching_iou"
        ),
        max_detections=_positive_int(
            _required(eval_raw, "max_detections", "evaluation"), "evaluation.max_detections"
        ),
    )

    bench_raw = _mapping(_required(root, "benchmark", "configuration"), "benchmark")
    benchmark = BenchmarkSettings(
        batch=_positive_int(_required(bench_raw, "batch", "benchmark"), "benchmark.batch"),
        warmup_iterations=_positive_int(
            _required(bench_raw, "warmup_iterations", "benchmark"),
            "benchmark.warmup_iterations",
            allow_zero=True,
        ),
        measured_iterations=_positive_int(
            _required(bench_raw, "measured_iterations", "benchmark"),
            "benchmark.measured_iterations",
        ),
        sample_count=_positive_int(bench_raw.get("sample_count", 25), "benchmark.sample_count"),
    )
    if benchmark.batch != 1:
        raise ConfigError("benchmark.batch must be 1 for comparable single-frame latency")

    selection_raw = _mapping(_required(root, "selection", "configuration"), "selection")
    primary = _required(selection_raw, "primary_metric", "selection")
    if primary != "map50_95":
        raise ConfigError("selection.primary_metric must be 'map50_95'")
    minimum_fps_raw = _required(selection_raw, "minimum_fps", "selection")
    if minimum_fps_raw is not None:
        if isinstance(minimum_fps_raw, bool) or not isinstance(minimum_fps_raw, (int, float)):
            raise ConfigError("selection.minimum_fps must be null or numeric")
        minimum_fps_raw = float(minimum_fps_raw)
        if minimum_fps_raw <= 0:
            raise ConfigError("selection.minimum_fps must be positive")
    tolerance = _required(selection_raw, "map_tie_tolerance", "selection")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ConfigError("selection.map_tie_tolerance must be numeric")
    tolerance = float(tolerance)
    if not 0 <= tolerance <= 1:
        raise ConfigError("selection.map_tie_tolerance must be in [0, 1]")
    safety = _required(selection_raw, "safety_classes", "selection")
    if not isinstance(safety, list) or not safety or not all(isinstance(x, str) and x for x in safety):
        raise ConfigError("selection.safety_classes must be a non-empty list of names")
    if len(set(safety)) != len(safety):
        raise ConfigError("selection.safety_classes contains duplicates")
    tie_breaker = _required(selection_raw, "tie_breaker", "selection")
    if tie_breaker != "latency_p95_ms":
        raise ConfigError("selection.tie_breaker must be 'latency_p95_ms'")
    selection = SelectionSettings(
        primary_metric=primary,
        minimum_fps=minimum_fps_raw,
        map_tie_tolerance=tolerance,
        safety_classes=tuple(safety),
        tie_breaker=tie_breaker,
    )

    test_raw = _mapping(_required(root, "test", "configuration"), "test")
    evaluate_winner_only = _bool(
        _required(test_raw, "evaluate_winner_only", "test"), "test.evaluate_winner_only"
    )
    if not evaluate_winner_only:
        raise ConfigError(
            "test.evaluate_winner_only must remain true unless an explicit post-selection study is added"
        )
    test = TestSettings(evaluate_winner_only=evaluate_winner_only)

    prediction_raw = _mapping(_required(root, "prediction", "configuration"), "prediction")
    source_raw = _required(prediction_raw, "source", "prediction")
    source = None
    if source_raw is not None:
        source = _resolve_project_path(source_raw, project_root, "prediction.source")
    prediction = PredictionSettings(
        source=source,
        confidence=_probability(
            _required(prediction_raw, "confidence", "prediction"), "prediction.confidence"
        ),
    )

    config = PipelineConfig(
        experiment=experiment,
        dataset=dataset,
        models=tuple(models),
        training=training,
        smoke_test=smoke_test,
        evaluation=evaluation,
        benchmark=benchmark,
        selection=selection,
        test=test,
        prediction=prediction,
        config_path=config_path,
        project_root=project_root,
    )
    spec = load_dataset_spec(dataset.yaml)
    missing_safety = sorted(set(config.selection.safety_classes) - set(spec.names.values()))
    if missing_safety:
        raise ConfigError(f"Unknown selection.safety_classes: {', '.join(missing_safety)}")
    return config


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Resolve dataset paths relative to the dataset YAML file itself."""

    yaml_path = Path(path).expanduser().resolve()
    if not yaml_path.is_file():
        raise ConfigError(f"Dataset YAML does not exist: {yaml_path}")
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid dataset YAML in {yaml_path}: {exc}") from exc
    data = _mapping(raw, "dataset YAML")
    nc = _positive_int(_required(data, "nc", "dataset YAML"), "dataset nc")
    names_raw = _required(data, "names", "dataset YAML")
    if isinstance(names_raw, list):
        names = {index: str(value) for index, value in enumerate(names_raw)}
    elif isinstance(names_raw, Mapping):
        try:
            names = {int(key): str(value) for key, value in names_raw.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigError("Dataset class-name keys must be integers") from exc
    else:
        raise ConfigError("Dataset names must be a list or mapping")
    if set(names) != set(range(nc)):
        raise ConfigError(f"Dataset names must define exactly class ids 0 through {nc - 1}")
    if any(not value.strip() for value in names.values()):
        raise ConfigError("Dataset class names must be non-empty")
    if len(set(names.values())) != len(names):
        raise ConfigError("Dataset class names must be unique")

    root_value = data.get("path", ".")
    if not isinstance(root_value, str):
        raise ConfigError("Dataset path must be a string")
    root = Path(root_value).expanduser()
    root = (yaml_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    splits: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        value = _required(data, split, "dataset YAML")
        if not isinstance(value, str) or not value:
            raise ConfigError(f"Dataset {split} path must be a non-empty string")
        split_path = Path(value).expanduser()
        splits[split] = (root / split_path).resolve() if not split_path.is_absolute() else split_path.resolve()

    metadata = {key: value for key, value in data.items() if key not in {"path", "train", "val", "test", "nc", "names"}}
    return DatasetSpec(
        yaml_path=yaml_path,
        root=root,
        splits=splits,
        nc=nc,
        names=names,
        metadata=metadata,
    )


def write_resolved_dataset_yaml(spec: DatasetSpec, destination: Path) -> Path:
    """Write an unambiguous absolute-path descriptor for Ultralytics calls."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "path": str(spec.root),
        "train": str(spec.splits["train"]),
        "val": str(spec.splits["val"]),
        "test": str(spec.splits["test"]),
        "nc": spec.nc,
        "names": dict(spec.names),
        **dict(spec.metadata),
    }
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def resolve_weights(weights: str, project_root: Path) -> str:
    """Resolve explicit local paths while preserving registry checkpoint names."""

    value = Path(weights).expanduser()
    if value.is_absolute():
        return str(value.resolve())
    if value.parent != Path("."):
        return str((project_root / value).resolve())
    return weights

