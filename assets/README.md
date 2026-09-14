# Sample assets

A small, representative selection. The original repository committed 2,508
prediction JPEGs; those made up most of a 338 MB repository and are trivially
regenerated.

| File pattern | What it is |
|---|---|
| `samples/frame_crab.jpg`, `samples/frame_fish_school.jpg` | Source frames used by the CAM notebooks |
| `samples/yolov5s_pred_*.jpg` | YOLOv5s predictions on held-out test frames |
| `samples/yolov8s_pred_*.jpg` | YOLOv8s predictions on held-out test frames |
| `samples/efficientdet_lite0_pred_*.jpg` | EfficientDet-Lite0 predictions |

Regenerate the full set:

```bash
yolo task=detect mode=predict model=weights/yolov8s_best.pt \
     source=data/brackish/images/test conf=0.25 save=True
```

Training curves and confusion matrices live in `results/`, not here.
