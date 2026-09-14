# Notebooks

The original experiments, renamed for ordering and with credentials removed.
Each runs top to bottom on Google Colab or AWS SageMaker.

| Notebook | What it does | Environment |
|---|---|---|
| `00_dataset_preparation.ipynb` | Download, extract frames, clean, normalise, split | Colab |
| `01_yolov5s.ipynb` | Train YOLOv5s, 100 epochs at 416 px | Colab, GPU |
| `02_yolov8s.ipynb` | Train YOLOv8s, 100 epochs at 800 px | SageMaker, GPU |
| `03_efficientdet_lite0_350ep.ipynb` | Train EfficientDet-Lite0, 350 epochs, export TFLite | SageMaker |
| `04_detectron2_faster_rcnn.ipynb` | Faster R-CNN X101-FPN, 300 iterations | Colab, GPU |
| `05_cam_checkpointA_crab.ipynb` | EigenCAM on the crab frame, checkpoint A, conf ≥ 0.8 | Colab |
| `06_cam_checkpointA_fish_school.ipynb` | Same, fish-school frame | Colab |
| `07_cam_checkpointB_crab.ipynb` | EigenCAM on the crab frame, checkpoint B, conf ≥ 0.2 | Colab |
| `08_cam_checkpointB_fish_school.ipynb` | Same, fish-school frame | Colab |

## Two things to know before running these

**The CAM notebooks are named after checkpoints, not architectures.** They were
originally named `GradCAM++_EfficientDet_*` and `GradCAM++_YOLOv8_*`. Both
names were wrong twice over: the checkpoints they load are a YOLOv8 and a
YOLOv5 model (verified by MD5), and the method they run is EigenCAM, not
GradCAM++. They differ in **which checkpoint and which confidence threshold**,
not in detector architecture. See
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

For actual GradCAM++, use `deepseanet.explain.GradCAMPlusPlus`.

**Credentials come from the environment now.** Cells that contained a Roboflow
key have been replaced with placeholders:

```python
import os
os.environ["ROBOFLOW_API_KEY"] = "..."   # or google.colab.userdata
```

Set `ROBOFLOW_API_KEY` before running, and never paste a key back into a cell —
notebook *outputs* capture URLs too.

## `legacy/`

Earlier EfficientDet attempts and the original YOLO→COCO converter, kept for
provenance. **`yolo_to_coco_original.py` contains a coordinate bug** that
divides every box by 1000; use `deepseanet.data.convert.yolo_to_coco` instead.

## Environment notes

`tflite-model-maker` (notebook 03) pins old TensorFlow versions and no longer
resolves on current Python. Options are in
[`../docs/MIGRATION.md`](../docs/MIGRATION.md).
