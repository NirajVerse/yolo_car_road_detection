# EcoCAR YOLO road-object detection

This repository implements a configurable, validation-driven Ultralytics YOLO object-detection
pipeline for the EcoCAR AI Lead interview challenge. It audits the dataset, smoke-tests the
runtime, fine-tunes multiple detection checkpoints, evaluates and benchmarks every candidate,
freezes a winner from validation evidence, evaluates that winner once on test, and produces one
annotated demonstration image.

The repository contains implementation only. No audit, checkpoint load/download, training,
validation, benchmark, test inference, or prediction is performed during setup or unit testing.

## Repository and dataset layout

The Git repository is the `model/` directory. The supplied read-only dataset is stored beside it:

```text
eco_car/
├── model/                       # this Git repository
│   ├── configs/experiment.yaml
│   ├── data.yaml                # path: ..
│   ├── src/road_detection/
│   └── tests/
├── train/{images,labels}/
├── valid/{images,labels}/
└── test/{images,labels}/
```

`road_detection` resolves every path relative to the file that declares it. Experiment paths are
relative to the repository root; dataset split paths are relative to `data.yaml` and its `path`
field. At run creation it writes `resolved_config.yaml` and an absolute-path
`resolved_dataset.yaml`. After a successful audit it creates an artifact-local runtime dataset:
images are read-only symlinks to the audited source, while labels are copied locally. Ultralytics
therefore writes label or image caches only inside the run directory, never into the supplied
dataset.

The tracked descriptor preserves the Roboflow metadata and numeric class IDs. Class 5 is named
`Lamp-post`; the audit explicitly records that this normalizes the supplied `Lamb-post` spelling
without changing the class ID.

## Environment setup

Use the existing virtual environment, or create a project-local one if it is unavailable:

```bash
cd /home/niraj-gupta/Documents/projects/eco_car/model
source ../.venv/bin/activate
python -m pip install -e ".[dev]"
```

This installs the declared Python dependencies, including PyTorch, Ultralytics 8.4 or newer,
PyYAML, NumPy, Pillow, and Matplotlib. Installing the package does not run the pipeline. The
`audit` command later verifies the exact installed versions, device capabilities, and whether each
configured checkpoint can be loaded. A bare checkpoint name such as `yolo26n.pt` may cause
Ultralytics to download it when the user runs `audit`; use an existing local path in the
configuration for an offline workflow.

Check the non-executing interface and unit tests with:

```bash
python -m road_detection.cli --help
pytest
```

The tests use only temporary synthetic files, arrays, and mocks. They never import or instantiate
an Ultralytics model.

## Configure an experiment

Edit only [`configs/experiment.yaml`](configs/experiment.yaml) for experiment settings. The model
list is data-driven:

```yaml
models:
  - id: yolo26n
    weights: yolo26n.pt
  - id: yolo26s
    weights: yolo26s.pt
  - id: yolo26m
    weights: yolo26m.pt
```

Another standard axis-aligned Ultralytics detection checkpoint—such as `yolov8n.pt` or
`yolo11n.pt`—works without a Python change. Classification, segmentation, pose, and OBB checkpoint
names are rejected. An optional `batch` on one model documents a necessary memory-driven override;
otherwise every model uses `training.batch`.

Configuration validation rejects missing fields/files, duplicate or unsafe model IDs, unsupported
comparison splits, non-positive batch sizes, invalid probabilities, non-detection task suffixes,
unknown safety classes, multi-device benchmark settings, and an unsafe test policy. Use one runtime
device (`auto`, `cpu`, `mps`, or one CUDA index) so batch-1 latency remains controlled. Validation
comparison is fixed to `val`, and `test.evaluate_winner_only` must remain true. Set
`training.resume: false`; pipeline continuation uses `--run-id` and intentionally does not expose
Ultralytics' checkpoint-resume behavior.

## Manual runbook

Run the following from the repository root. `audit` creates a unique run and prints its run ID.
Each later command resumes the latest run by default; supplying the printed ID is safer when more
than one experiment is active.

```bash
python -m road_detection.cli audit --config configs/experiment.yaml
python -m road_detection.cli smoke-test --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli train --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli evaluate --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli benchmark --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli compare --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli test-winner --config configs/experiment.yaml --run-id RUN_ID
python -m road_detection.cli predict --config configs/experiment.yaml --run-id RUN_ID --source /absolute/path/to/image.jpg
```

If `--source` and `prediction.source` are both unset, prediction deterministically uses the first
supported test image. That choice is not based on model output and is not used for tuning.

The complete sequence can be run in one process:

```bash
python -m road_detection.cli run --config configs/experiment.yaml --source /absolute/path/to/image.jpg
```

The full run performs audit → smoke test → training → validation evaluation → local benchmark →
comparison/selection → winner-only test → prediction. It stops on critical audit problems or an
incomplete candidate and never silently hides a failure.

### Safe resume behavior

Each run has a unique directory and manifest. Completed stages are skipped, successful candidates
are preserved on partial retry, and training uses numbered attempt directories rather than
overwriting a previous attempt. Resume explicitly with `--run-id RUN_ID`. Use `--force` only when
you intentionally want to repeat a completed stage inside that run; it is rejected when doing so
would invalidate a completed downstream stage. The winner's completed test evaluation cannot be
forced a second time. Normal continuation does not retrain completed candidates.

Resume verifies hashes of the source and resolved experiment configuration, dataset descriptor,
audited dataset file manifest, and locally resolved source checkpoints. Any drift requires a new
audit/run. A prediction image supplied with `--source` is persisted in the resolved configuration
and reused if that prediction stage must be resumed.

Do not change the candidate list or evaluation thresholds partway through a run. Start a new audit
to create a new run whenever the resolved experimental design changes.

## What each stage produces

- `audit` checks dependencies, device/runtime, checkpoint compatibility, split paths, readable
  images, YOLO label rows, missing/orphan/empty labels, duplicate stems, class distributions,
  imbalance, and exact cross-split image hashes. Critical errors block training.
- `smoke-test` uses the smallest audited candidate for one epoch on a configured fraction. The
  candidate is chosen from audit-measured parameter count, with checkpoint bytes and configuration
  order as deterministic fallbacks. It verifies train/validation, weights, metrics, and one saved
  prediction in an isolated directory that is excluded from comparison.
- `train` runs every candidate with the same seed, image size, budget, patience, and comparable
  settings. It records actual settings, duration, best epoch, early stopping, results CSV, and
  `best.pt`/`last.pt`. Only `best.pt` is evaluated.
- `evaluate` runs a fixed validation protocol and stores aggregate/per-class accuracy, Ultralytics
  timing, model efficiency, and class-aware object-coverage counts.
- `benchmark` preloads the same representative validation images, warms each model, synchronizes
  CUDA around timed calls, and records controlled batch-1 latency without checkpoint loading,
  image disk reads, plotting, or saving in the timed region.
- `compare` records every configured candidate (including failures), creates seven consistent plots,
  documents the tradeoff, and atomically freezes the validation winner.
- `test-winner` loads only that frozen winner and evaluates it once on the untouched test split.
- `predict` copies the input, saves a readable annotated JPEG, and writes JSON/CSV detections with
  class ID/name, confidence, pixel `xyxy`, and normalized YOLO center/width/height coordinates.

## Metrics and matching

- **Precision** is `TP / (TP + FP)`: the share of reported detections that are correct.
- **Recall** is `TP / (TP + FN)`: the share of labeled objects that are found.
- **mAP50** averages per-class average precision at IoU 0.50.
- **mAP50–95** averages AP across IoU thresholds 0.50 through 0.95 and is the primary accuracy
  metric. mAP75 is also retained when the installed Ultralytics version exposes it.
- **Latency** is local wall-clock milliseconds per preloaded image at batch 1 after warmup. The
  comparison reports mean, median, standard deviation, p90, p95, and `FPS = 1000 / mean_ms`.

“Most objects detected” is not measured by raw box count. At one shared confidence and matching-IoU
threshold, predictions are sorted by confidence and greedily matched one-to-one to an unmatched
ground-truth box of the same class with the highest eligible IoU. The reports include GT objects,
predictions, correctly matched detections/TP, FP, FN, precision, and detection recall overall and by
class. This prevents duplicate or false boxes from being presented as better coverage.

## Winner rule and test isolation

Selection uses validation only:

1. If `minimum_fps` is set, eliminate candidates below it.
2. Choose the highest validation mAP50–95.
3. Within `map_tie_tolerance`, choose higher mean recall across the configured safety classes.
4. If still tied, choose lower local p95 latency.
5. If still tied, choose the smaller checkpoint; an exact final tie uses model ID for determinism.

If no candidate meets `minimum_fps`, the selection file says the constraint failed and uses the
explicit compromise score `mAP50–95 × min(1, FPS / minimum_fps)`; it never claims the constraint
passed. If any candidate is failed/incomplete, selection is blocked rather than silently choosing
from a reduced field.

The test split is reserved for one post-selection estimate. Test metrics never feed back into
training, thresholds, candidate choice, or the frozen winner.

## Artifacts and provenance

Every run writes beneath:

```text
artifacts/<experiment_name>/<run_id>/
├── resolved_config.yaml
├── resolved_dataset.yaml
├── runtime_dataset_<attempt>.yaml
├── runtime_dataset_manifest_<attempt>.json
├── dataset_snapshot_<attempt>/
├── manifest.json
├── environment.txt
├── logs/
├── data_audit/
├── smoke_test/
├── models/<model_id>/
├── comparison/
├── test/
└── predictions/
```

The manifest tracks timestamps, stage state/failures, Git revision and dirty state, dataset
fingerprint, configured checkpoints and resolved settings, seed, dependency/device information,
and principal outputs. Source, trained, evaluated, benchmarked, selected, tested, and predicted
checkpoint identities are tied together with SHA-256 hashes so mixed or changed weights block the
pipeline. Reports use run-relative artifact paths when practical. Runtime artifacts, datasets,
downloaded/trained checkpoints, caches, and virtual environments are Git-ignored.

## Limitations

The roughly 840-image, 14-class dataset is small for modern detection. Rare-class AP/recall can be
high variance; a difference near the tie tolerance may not be meaningful. Exact duplicate detection
finds byte-identical leakage but not near-duplicates or adjacent video frames. One validation split
does not estimate cross-split uncertainty, the lightweight overfitting check is a heuristic, and
latency applies only to the recorded machine/runtime/precision—not to unrelated COCO benchmark
hardware or a future embedded target. These results should be described with the recorded numeric
evidence and without overstating small differences.
