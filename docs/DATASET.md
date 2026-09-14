# The Brackish dataset

Three cameras bolted to a pillar of the Limfjord bridge between Aalborg and
Nørresundby, nine metres below the surface, recorded and annotated frame by
frame by Aalborg University.

> Pedersen, M., Bruslund Haurum, J., Gade, R., & Moeslund, T. B. (2019).
> *Detection of Marine Animals in a New Underwater Dataset with Varying
> Visibility.* CVPR Workshops, 18–26.

- Kaggle: <https://www.kaggle.com/datasets/aalborguniversity/brackish-dataset>
- Roboflow Universe: <https://universe.roboflow.com/brackish/brackish-2fdzd>

**The dataset is not in this repository** and must not be committed to it —
`.gitignore` blocks `data/`, `images/`, `labels/` and video files.

---

## Classes

Boxes carry six object classes:

| id | name | notes |
|---|---|---|
| 0 | `fish` | large fish |
| 1 | `small_fish` | hardest class: a few dozen pixels, moves between frames |
| 2 | `crab` | often stationary on the seabed |
| 3 | `shrimp` | rare, ~2% of boxes |
| 4 | `jellyfish` | drifts slowly, easy to localise |
| 5 | `starfish` | easiest class: essentially stationary |

The paper's Tables 3 and 6 name classes after the **video folders** clips were
filed under (`fish-big`, `fish-school`, `fish-small`, …), which is a different
grouping. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Box counts reported in the paper's Table 3, after preprocessing:

| Category | Examples |
|---|---|
| Crab | 1,751 |
| Fish-big | 2,992 |
| Fish-school | 927 |
| Fish-small | 2,268 |
| Shrimp | 824 |
| Jellyfish | 1,237 |

---

## Preprocessing pipeline

Each step has a function in `deepseanet.data` and a CLI equivalent.

1. **Extract frames.** Decode each `.avi` at 960×540 with bicubic scaling —
   the resolution the shipped annotations were drawn against. Changing it
   without rescaling the labels silently misaligns every box.

   ```python
   from deepseanet.data import extract_frames
   extract_frames("videos/", "data/images", width=960, height=540)
   ```

2. **Pair frames with labels.** Match by filename stem; report orphans.

3. **Drop frames with no boxes.** The original run discarded roughly 3,700
   unlabelled frames and 1,807 zero-visibility frames, leaving ~10,000 frames
   with ~10,000 label files.

4. **Normalise coordinates.** The shipped label files use the YOLO *field
   order* but **pixel values**. Divide by the frame dimensions to get `[0, 1]`.

5. **Split.** See below.

6. **Export per toolkit.** YOLO text stays as is; Detectron2 wants COCO JSON;
   TFLite Model Maker wants Pascal VOC XML.

All of steps 2–5 in one go:

```bash
deepseanet prepare --images data/images --labels data/labels --normalize
deepseanet split   --images data/images --labels data/labels \
                   --strategy grouped --output data/brackish
deepseanet convert --to coco --images data/brackish/images/train \
                   --labels data/brackish/labels/train \
                   --output data/brackish/train_coco.json
```

---

## Splitting: why `grouped` is the default

Frames come from continuous video. Frame 41 and frame 42 of the same clip are
nearly the same image. A random split scatters those near-duplicates across
train and test, so the model is evaluated on frames it has effectively already
seen, and **every score goes up**.

`grouped_split` assigns whole source clips to one side:

```python
from deepseanet.data import grouped_split, video_id_of
split = grouped_split(frames, video_id_of, (0.7, 0.2, 0.1), seed=0)
```

`random_split` is kept because it reproduces what the original notebooks and
the published 70:20:10 split did. Use it to reproduce; use `grouped` to report.

Expect grouped-split numbers to be **lower**. That is the point — they are
closer to what the model would do on footage it has never seen.

---

## Annotation formats

| Format | Coordinates | Used by |
|---|---|---|
| YOLO `.txt` | `class cx cy w h`, normalised `[0,1]` | YOLOv5, YOLOv8 |
| COCO JSON | `[x, y, w, h]`, absolute pixels, category ids **one-based** | Detectron2 |
| Pascal VOC XML | `xmin ymin xmax ymax`, absolute pixels | TFLite Model Maker |

Conversion is round-trip tested. Note the one-based COCO category ids:
`coco_to_yolo` applies `class_offset=-1` by default to undo the shift.
