# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-09

First update since the 2023 release. Restructures the repository around an
installable package, adds reference implementations of the components the paper
describes, and documents what the committed artefacts actually contain.

### Security

- **Removed live Roboflow credentials from five notebooks.** An account API key
  and two dataset download keys. They remain in git history —
  **rotate the key**. See [`docs/SECURITY.md`](docs/SECURITY.md).
- Added `gitleaks` as a pre-commit hook and a CI job.

### Added

- `src/deepseanet/`, an installable package:
  - `models/biskfpn.py` — BiFPN and the proposed BiSkFPN neck, from the paper's
    equation, with the deconvolution and skip branches.
  - `models/backbone.py` — MBConv + squeeze-excitation backbone with compound
    scaling, emitting a five-level pyramid.
  - `models/head.py` — shared class and box subnets, anchor generation.
  - `models/activations.py` — Swish with a learnable `β`.
  - `losses.py` — focal, smooth-L1, L2, combined as `L = L_cls + αL_box + βL_reg`.
  - `attacks/uap.py` — the paper's closed-form UAP, a PGD-style optimised UAP,
    and the curriculum schedule for adversarial training.
  - `explain/gradcampp.py` — true GradCAM++, plus EigenCAM for comparison.
  - `data/` — Brackish preprocessing, format conversion, splitting.
  - `cli.py` — `deepseanet prepare | convert | split | model | report | verify`.
- `tests/` — 87 tests covering shapes, gradients, numerical behaviour, format
  round-trips and split invariants.
- `docs/` — reproducibility record, results, dataset, architecture, model card,
  migration guide, security notes.
- `configs/` — YAML configs including `deepseanet_bifpn.yaml`, the ablation
  baseline that isolates the neck's contribution.
- `results/checkpoints.json` — MD5 checksums for every checkpoint, with notes on
  the mislabelling.
- `grouped_split` — splits by source video, so near-duplicate frames from
  continuous footage cannot land on both sides.
- CI: lint, tests on Python 3.9/3.11/3.12, secret scan.
- Packaging: `pyproject.toml`, `Makefile`, `CITATION.cff`, MIT `LICENSE`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and PR templates.

### Fixed

- **`yolo_to_coco` coordinate bug.** The original converter divided every box
  coordinate by 1000, collapsing essentially every box to zero area on 960×540
  frames. Fixed, with a regression test.
- Detector now raises a clear error when the input size is not divisible by the
  coarsest pyramid stride, instead of silently misaligning anchors.

### Changed

- Experiment directories flattened: `0_Dataset/` … `5_GradCAM++/` became
  `notebooks/`, `results/` and `assets/`. Full mapping in
  [`docs/MIGRATION.md`](docs/MIGRATION.md).
- Notebooks renamed with a numeric prefix and a descriptive name. The four CAM
  notebooks are now named after the **checkpoint** they load rather than the
  architecture, because the original names were wrong.

### Removed

- 2,508 prediction JPEGs (14 representative samples kept in `assets/samples/`).
- Four `.pt` checkpoints, ~110 MB — fetch with `scripts/fetch_weights.sh`.
- TensorBoard event files, superseded by the `results.csv` logs.

Repository size: 338 MB → ~53 MB.

### Documented

- The two checkpoints in `5_GradCAM++/` were **mislabelled**: MD5s show
  `best_efficientDet.pt` is the YOLOv8 checkpoint and `best_yolov8.pt` is the
  YOLOv5 one. Neither is an EfficientDet model.
- The CAM notebooks run **EigenCAM**, not the GradCAM++ the paper describes.
- The committed YOLO runs used **100 epochs**, not the 350 in Table 4.
- Tables 5 and 6 do not report a single consistent metric.
- Random splits leak near-duplicate frames and inflate every published score on
  this dataset.

## [1.0.0] — 2023-06

Original release accompanying the paper.
