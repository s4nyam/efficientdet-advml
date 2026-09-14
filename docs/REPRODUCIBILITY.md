# Reproducibility notes

What the committed artefacts actually contain, where they differ from the
published tables, and what a reader should know before building on any of it.

This document is written against the repository as it was published in 2023 and
audited in 2026. Nothing here retracts the paper; it records the gap between a
paper and the code that produced it, which is normal and worth writing down.

---

## 1. Scope of the repository

The paper reports five repetitions per detector and a mean with a standard
deviation. **The repository contains one run per detector.** The repeated runs
behind Table 5 were not committed, so the `mean ± std` columns cannot be
reproduced from what is here.

| Component in the paper | In the original repository |
|---|---|
| BiSkFPN neck | Not implemented. Added in `src/deepseanet/models/biskfpn.py` for this release. |
| Swish in the backbone | Not implemented separately; the TFLite spec's own activations were used. Added in `src/deepseanet/models/activations.py`. |
| Multi-focal-loss head | Not implemented. Added in `src/deepseanet/losses.py`. |
| UAP generation | Not implemented. Added in `src/deepseanet/attacks/uap.py`. |
| Adversarial (curriculum) training | Not implemented. Schedule added in `src/deepseanet/attacks/uap.py`. |
| GradCAM++ | Notebooks use **EigenCAM**, a different method. True GradCAM++ added in `src/deepseanet/explain/gradcampp.py`. |
| Five-fold repetition | Single run per detector. |

The `notebooks/03_efficientdet_lite0_350ep.ipynb` notebook trains the **stock
`efficientdet_lite0` specification** from TFLite Model Maker. It is a faithful
EfficientDet-Lite0, but it is not a modified one: no skip connections, no neck
changes, no adversarial component.

> The modules added in `src/` are **reference implementations written from the
> paper's equations**. They are tested for shape, gradient flow and numerical
> behaviour. They did **not** produce the numbers in the paper, and running them
> will not reproduce those numbers.

---

## 2. What the committed runs actually logged

Parsed directly from `results/` with `deepseanet report`:

| Run | Toolkit | Input | Schedule | Split evaluated | AP@0.5 | AP@[.5:.95] |
|---|---|---|---|---|---|---|
| EfficientDet-Lite0 | TFLite Model Maker | 320² | 350 ep, batch 64 | test (1,000) | 0.898 | 0.601 |
| EfficientDet-Lite0 (TFLite export) | TFLite Model Maker | 320² | — | test (1,000) | 0.863 | 0.561 |
| YOLOv5s | Ultralytics YOLOv5 | 416 | 100 ep, batch 16 | val (1,506) | 0.976 | 0.748 |
| YOLOv8s | Ultralytics 8.0.20 | 800 | 100 ep, batch 16 | val (2,000) | 0.988 | 0.836 |
| Faster R-CNN X101-FPN | Detectron2 | default | 300 iterations | test | 0.433 | 0.204 |

Reproduce the YOLO rows from the logs in this repository:

```bash
deepseanet report --results results
```

### Side by side with the paper

| Detector | Paper, Table 5 (mean ± std) | Committed run, AP@0.5 |
|---|---|---|
| Proposed EfficientDet | 98.6 ± 1.0 | 89.8 (test), 86.3 as TFLite |
| YOLOv8 | 98.2 ± 0.17 | 98.8 (val) |
| YOLOv5 | 97.6 ± 0.61 | 97.6 (val) |
| Detectron2 | 95.2 ± 1.4 | 43.3 (test) |

The YOLOv5 row matches. The others do not, for reasons worth separating:

- **EfficientDet** — the committed run is Lite0 at 320², evaluated on the test
  split. A larger variant, a different split, or the modified architecture
  described in the paper would all move this number.
- **Detectron2** — the committed run is 300 iterations, which is a smoke test,
  not training. The gap is a schedule difference, not a model difference.
- **Epochs** — the paper's Table 4 lists 350 epochs for every model.
  `results/yolov5s/opt.yaml` and `results/yolov8s/args.yaml` both record
  **100**. The EfficientDet notebook does use 350.

### Table 5 and Table 6 do not report the same metric

The Detectron2 row of Table 6 (28.1, 14.5, 8.6, 3.8, 26.1, 40.6) matches the
per-class COCO AP@[.5:.95] logged in the Detectron2 notebook (28.8, 14.6, 8.6,
3.9, 25.7, 40.7) almost exactly. Table 5 reports 95.2 for the same model. One
table is a strict averaged-IoU metric and the other is not; they should not be
read as two views of one number.

### Class naming

Tables 3 and 6 name classes after the **video folders** the clips were filed
under (`fish-big`, `fish-school`, `fish-small`, …). The **boxes** carry six
object classes: `fish`, `small_fish`, `crab`, `shrimp`, `jellyfish`,
`starfish`. `configs/brackish.yaml` and `deepseanet.BRACKISH_CLASSES` use the
box classes.

---

## 3. Issues found in the 2026 audit

### 3.1 Exposed credentials (fixed here, still in git history)

Five notebooks contained live Roboflow credentials:

| File | Secret |
|---|---|
| `2_YOLO8.ipynb` | Roboflow account API key, passed to `Roboflow(api_key=...)` |
| `3_EfficientDet_*` (×3) | Dataset download key in a `universe.roboflow.com/ds/...?key=...` URL |
| `4_Detectron2.ipynb` | A second dataset download key |

All are replaced in this release with `os.environ["ROBOFLOW_API_KEY"]` and
`${ROBOFLOW_KEY}` placeholders, and `gitleaks` now runs in CI and as a
pre-commit hook.

**Scrubbing the working tree does not remove them from git history.** If you are
pushing to the existing repository, rotate the Roboflow key first, and consider
`git filter-repo` or a fresh history. See [`SECURITY.md`](SECURITY.md).

### 3.2 Two checkpoints are mislabelled

`5_GradCAM++/` contained two files whose names do not match their contents.
MD5 checksums prove it:

| Committed name | MD5 | Byte-identical to | Actually is |
|---|---|---|---|
| `best_efficientDet.pt` | `f7be0694…` | `2_YOLO8_Experiment/.../best.pt` | **YOLOv8s** |
| `best_yolov8.pt` | `cece6489…` | `1_YOLO5_Experiment/.../best.pt` | **YOLOv5s** |

Neither is an EfficientDet model, and the names are swapped relative to what
they hold. This is consistent with the CAM notebooks, which load both through
`torch.hub.load('ultralytics/yolov5', 'custom', ...)` — a loader that cannot
open an EfficientDet checkpoint at all.

Consequence: the four CAM notebooks differ in **which YOLO checkpoint they load
and at what confidence threshold**, not in detector architecture. The figure
comparing "YOLOv8 against EfficientDet" heatmaps is comparing two YOLO models.

The only genuine EfficientDet artefact from the project is
`results/efficientdet_lite0/model.tflite` (4.4 MB, committed here).

Checksums are recorded in `results/checkpoints.json`. Verify any copy with:

```bash
deepseanet verify --manifest results/checkpoints.json
```

### 3.3 Coordinate bug in the YOLO→COCO converter

`notebooks/legacy/yolo_to_coco_original.py` divides every box coordinate by
1000 before writing:

```python
create_annotation_from_yolo_format(
    int(min_x / 1000), int(min_y / 1000),
    int(width / 1000), int(height / 1000), ...)
```

On 960×540 frames this truncates essentially every box to `(0, 0, 0, 0)`. Any
COCO file produced by that script is unusable. The file is kept for provenance
only. `deepseanet.data.convert.yolo_to_coco` is the corrected implementation,
and `tests/test_data.py::test_yolo_to_coco_produces_absolute_pixel_boxes` is a
regression test against exactly this.

---

## 4. Limitations that apply to any result on this data

These are properties of the dataset and the protocol, not mistakes.

**Neighbouring frames leak.** Frames are cut from continuous video, so
consecutive frames are nearly identical. A random split puts near-duplicates
into both training and test and inflates every score. This affects the paper's
numbers and every published baseline on Brackish. Use
`deepseanet split --strategy grouped`, which assigns whole source clips to one
side, for anything you intend to report.

**AP@0.5 is near its ceiling.** Several detectors pass 0.97, where differences
shrink into noise. AP@[.5:.95] separates models far better, and the spread
between 0.60 (EfficientDet-Lite0) and 0.84 (YOLOv8s) is much more informative
than the 0.9 percentage points between their AP@0.5 scores.

**One site, one camera.** Every frame comes from the same fixed rig in
Limfjorden. Nothing here demonstrates transfer to other water, depths, lighting
or hardware.

**Classes are imbalanced.** Shrimp and jellyfish together make up 3–4% of the
validation boxes. A single mean hides performance on exactly the animals that
are hardest to monitor — report per-class numbers.

---

## 5. Reproducing what can be reproduced

```bash
# 1. Summarise the committed logs. No training, no GPU.
deepseanet report --results results

# 2. Confirm the architecture claims: BiSkFPN adds parameters over BiFPN.
deepseanet model --neck bifpn   --image-size 512
deepseanet model --neck biskfpn --image-size 512

# 3. Rebuild the splits from the raw frames.
deepseanet prepare --images data/images --labels data/labels --normalize
deepseanet split --images data/images --labels data/labels \
                 --strategy grouped --output data/brackish

# 4. Retrain a baseline with the settings the committed run used.
#    configs/yolov5s.yaml and configs/yolov8s.yaml record them verbatim.
```

The YOLO notebooks re-run top to bottom on Colab once you supply your own
Roboflow key via `ROBOFLOW_API_KEY`. The EfficientDet notebook needs
`tflite-model-maker`, which pins old TensorFlow versions and no longer installs
cleanly on current Python — see [`MIGRATION.md`](MIGRATION.md) for options.

---

## 6. If you are citing or extending this work

Cite the paper for the idea and this repository for the code. If you report new
numbers on Brackish:

1. Use a clip-grouped split and say so.
2. Report AP@[.5:.95] alongside AP@0.5.
3. Report per-class numbers, especially shrimp and small fish.
4. State the input resolution and epoch count — they dominate the comparison.
5. Name which split you evaluated on. Half the confusion above comes from
   mixing validation and test numbers in one table.
