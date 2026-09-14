# What changed in v2.0.0

A map from the 2023 layout to this one, for anyone who has the old repository
checked out or has linked to a path in it.

## Directory mapping

| Old path | New path |
|---|---|
| `0_Dataset/Dataset.ipynb` | `notebooks/00_dataset_preparation.ipynb` |
| `0_Dataset/dataset.py` | superseded by `src/deepseanet/data/` |
| `1_YOLO5_Experiment/1_YOLO5.ipynb` | `notebooks/01_yolov5s.ipynb` |
| `1_YOLO5_Experiment/content/.../exp/` | `results/yolov5s/` |
| `2_YOLO8_Experiment/2_YOLO8.ipynb` | `notebooks/02_yolov8s.ipynb` |
| `2_YOLO8_Experiment/runs/detect/train/` | `results/yolov8s/` |
| `3_EfficientDet_Experiment/main_350epochs_AWS/*.ipynb` | `notebooks/03_efficientdet_lite0_350ep.ipynb` |
| `3_EfficientDet_Experiment/.../model.tflite` | `results/efficientdet_lite0/model.tflite` |
| `3_EfficientDet_Experiment/older_versions/*.ipynb` | `notebooks/legacy/` |
| `3_EfficientDet_Experiment/older_versions/main.py` | `notebooks/legacy/yolo_to_coco_original.py` |
| `4_Detectron2/4_Detectron2.ipynb` | `notebooks/04_detectron2_faster_rcnn.ipynb` |
| `5_GradCAM++/*.ipynb` | `notebooks/05_`…`08_cam_*.ipynb` |
| `5_GradCAM++/*.pt` | not committed — see below |

## What was removed, and why

**2,508 prediction JPEGs** (`1_YOLO5_Experiment/content 2/.../exp/`, 1,508
files; `2_YOLO8_Experiment/runs/detect/predict2/`, 1,000 files). These were the
bulk of a 338 MB repository. Fourteen representative frames are kept in
`assets/samples/`. Regenerate the full set in minutes:

```bash
yolo task=detect mode=predict model=weights/yolov8s_best.pt \
     source=data/brackish/images/test conf=0.25 save=True
```

**Four `.pt` checkpoints** (110 MB). Fetch them with
`bash scripts/fetch_weights.sh`, then verify with `deepseanet verify`.
`.gitattributes` is configured for Git LFS if you would rather commit them —
run `git lfs install` first.

**TensorBoard event files.** Superseded by the `results.csv` logs, which are
kept and are what `deepseanet report` reads.

Nothing was deleted from your archives; the originals remain in
`efficientdet-advml-main.zip` and on
[archive.org](https://archive.org/download/deepseanet/deepseanet.zip).

## What was fixed

1. **Credentials removed** from five notebooks. See [`SECURITY.md`](SECURITY.md)
   — the keys are still in git history and the Roboflow key needs rotating.
2. **`yolo_to_coco` coordinate bug.** The original divided every coordinate by
   1000. Fixed and regression-tested.
3. **Checkpoint mislabelling documented.** The two files in `5_GradCAM++/` are
   a YOLOv8 and a YOLOv5 checkpoint with swapped names. Checksums in
   `results/checkpoints.json`.

## What was added

| | |
|---|---|
| `src/deepseanet/` | Installable package: BiSkFPN, Swish, MBConv backbone, head, losses, UAP attacks, GradCAM++, data pipeline, CLI |
| `tests/` | 87 tests |
| `docs/` | Reproducibility record, results, dataset, architecture, model card |
| `configs/` | YAML configs, including the ablation baseline |
| `.github/workflows/ci.yml` | Lint, test on 3.9/3.11/3.12, secret scan |
| `results/checkpoints.json` | Checksums for every checkpoint |

## Running the old notebooks today

The YOLO notebooks still run on Colab. Supply your key via the environment
rather than editing the cell:

```python
import os
os.environ["ROBOFLOW_API_KEY"] = "..."   # or use Colab secrets
```

`tflite-model-maker` (the EfficientDet notebook) pins old TensorFlow versions
and no longer resolves on current Python. Options, in order of effort:

1. Python 3.9 in a container with the pinned versions.
2. Port to `tensorflow-lite-model-maker`'s successor, or to
   [`effdet`](https://github.com/rwightman/efficientdet-pytorch) in PyTorch.
3. Use `src/deepseanet/` with a pretrained backbone, which is what it is for.

## Upgrading an existing clone

The history is unchanged, so a normal pull works. If you want the credentials
gone from history you need a rewrite, which invalidates every existing clone:

```bash
pip install git-filter-repo
git filter-repo --replace-text secrets.txt   # one `literal:KEY==>REDACTED` per line
git push --force
```

**Rotate the Roboflow key first.** A rewrite hides the string; it does not
invalidate the credential.
