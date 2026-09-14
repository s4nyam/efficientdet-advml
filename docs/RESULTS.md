# Results

Two kinds of numbers live here and they must not be mixed: what the **paper
reports**, and what the **committed runs logged**. Section 2 of
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) explains where they diverge.

---

## 1. Reported in the paper

### Table 5 — mAP over five repetitions

| Model | mAP₁ | mAP₂ | mAP₃ | mAP₄ | mAP₅ | Mean ± Std |
|---|---|---|---|---|---|---|
| YOLOv3 | 31.9 | 30.2 | 29.5 | 32.5 | 31.7 | 31.1 ± 1.1 |
| YOLOv4 | 84.6 | 83.8 | 84.2 | 85.2 | 80.9 | 83.7 ± 1.4 |
| YOLOv5 | 96.7 | 98.0 | 97.5 | 98.5 | 97.3 | 97.6 ± 0.61 |
| YOLOv8 | 98.0 | 98.5 | 98.3 | 98.1 | 98.2 | 98.2 ± 0.17 |
| Detectron2 | 94.5 | 93.4 | 94.8 | 95.7 | 97.8 | 95.2 ± 1.4 |
| **Proposed EfficientDet** | 99.5 | 98.7 | 98.0 | 97.0 | 99.8 | **98.6 ± 1.0** |

### Table 6 — class-wise mAP

Class names follow the paper, which labels classes by the **video category**
the clip was filed under rather than by the box class.

| Model | crab | fish-big | fish-school | fish-small | shrimp | jellyfish |
|---|---|---|---|---|---|---|
| YOLOv3 | 92.7 | 89.9 | 84.0 | 62.3 | 76.6 | 82.0 |
| YOLOv4 | 93.1 | 78.9 | 88.2 | 59.2 | 73.2 | 83.2 |
| YOLOv5 | 81.8 | 56.3 | 80.9 | 66.9 | 69.6 | 93.3 |
| YOLOv8 | 82.8 | 63.2 | 85.7 | 69.5 | 65.0 | 97.4 |
| Detectron2 | 28.1 | 14.5 | 8.6 | 3.8 | 26.1 | 40.6 |
| **Proposed EfficientDet** | 89.5 | **94.6** | 87.2 | **82.1** | **79.9** | 95.2 |

The proposed model's margin is largest on **fish-small** (82.1 against 69.5 for
the next best), which is the class the BiSkFPN skip path is meant to help:
small, fast-moving targets whose evidence lives in the finest feature map.

> Table 5 and Table 6 do not report a single consistent metric — see
> [`REPRODUCIBILITY.md` §2](REPRODUCIBILITY.md).

### Adversarial learning

| Setting | mAP |
|---|---|
| EfficientDet | 98.56 |
| EfficientDet + adversarial learning (UAP) | **98.63** |
| YOLOv5 + adversarial learning (UAP) | 98.04 |

IoU of feature maps, five-fold: **88.54%**.

---

## 2. Logged by the committed runs

Parsed from `results/` with `deepseanet report --results results`. No training
required.

| Run | Epochs | AP@0.5 | AP@[.5:.95] | Precision | Recall |
|---|---|---|---|---|---|
| YOLOv5s | 100 | 0.9758 | 0.7483 | 0.9647 | 0.9554 |
| YOLOv8s | 100 | 0.9879 | 0.8357 | 0.9929 | 0.9770 |
| EfficientDet-Lite0 | 350 | 0.898 (test) | 0.601 | — | — |
| EfficientDet-Lite0, TFLite export | 350 | 0.863 (test) | 0.561 | — | — |
| Faster R-CNN X101-FPN | 300 iters | 0.433 (test) | 0.204 | — | — |

Splits and input sizes differ between runs (YOLOv5 at 416 px on its own split,
YOLOv8 at 800 px on the Roboflow split, EfficientDet at 320²). Compare orders
of magnitude, not third decimals.

Diagnostic plots saved by the trainers are in `results/yolov5s/` and
`results/yolov8s/`: `PR_curve.png`, `F1_curve.png`, `confusion_matrix.png`,
`results.png`.

---

## 3. Architecture cost

Measured with `deepseanet model`:

| Neck | Backbone | Neck | Head | Total |
|---|---|---|---|---|
| BiFPN | 8,575,168 | 187,065 | 39,258 | **8,801,491** |
| BiSkFPN | 8,575,168 | 1,259,001 | 39,258 | **9,873,427** |

BiSkFPN costs about **1.07 M extra parameters** at `phi=0`, a 12% increase on
the whole model, concentrated entirely in the neck. That is the price of the
transposed-convolution up-sampling and the strided skip projections; whether it
buys accuracy on your data is an ablation worth running, and
`configs/deepseanet_bifpn.yaml` exists so you can run it with one variable
changed.

Reproduce:

```bash
deepseanet model --neck bifpn   --image-size 512
deepseanet model --neck biskfpn --image-size 512
```
