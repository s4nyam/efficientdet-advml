# Committed training runs

One run per detector, exactly as the trainers wrote it. Summarise them without
re-running anything:

```bash
deepseanet report --results results
```

| Directory | Contents |
|---|---|
| `yolov5s/` | `results.csv` (100 epochs), `hyp.yaml`, `opt.yaml`, PR/F1/P/R curves, confusion matrix |
| `yolov8s/` | `results.csv` (100 epochs), `args.yaml`, curves, confusion matrix, validation batches |
| `efficientdet_lite0/` | `model.tflite` — the exported EfficientDet-Lite0, 350 epochs |
| `detectron2/` | Metrics are inline in the notebook; see `notebooks/04_detectron2_faster_rcnn.ipynb` |
| `checkpoints.json` | MD5 checksums for every checkpoint, including the mislabelled ones |

## Caveats

- The epoch counts here (**100** for both YOLO runs) do not match the paper's
  Table 4, which reports 350 for every model.
- YOLOv5 was evaluated on its own split at 416 px; YOLOv8 on the Roboflow split
  at 800 px. The numbers are not directly comparable.
- The five repetitions behind the paper's Table 5 are not in this repository.

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

## Verifying checkpoints

```bash
bash scripts/fetch_weights.sh
deepseanet verify --manifest results/checkpoints.json
```
