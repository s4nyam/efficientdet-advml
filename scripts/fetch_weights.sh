#!/usr/bin/env bash
# Download the trained checkpoints that are too large to keep at HEAD.
#
# The EfficientDet-Lite0 model is committed directly at
# results/efficientdet_lite0/model.tflite (4.4 MB) -- nothing to fetch for it.
#
# Usage:  bash scripts/fetch_weights.sh [output_dir]
# Then:   deepseanet verify --manifest results/checkpoints.json --root .

set -euo pipefail

OUT="${1:-weights}"

# The checkpoints were committed to this repository before the v2.0.0
# restructure. They are no longer at HEAD, but git history keeps them, so we
# fetch them from a pinned commit. Pinning the SHA -- rather than a branch --
# means this script keeps working and always returns the same bytes.
PIN="1827c48ee2c8e42b4fc99fc7c918d1e8de7c5d1f"
BASE="https://raw.githubusercontent.com/s4nyam/efficientdet-advml/${PIN}"

mkdir -p "$OUT"

fetch() {
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then
    echo "exists, skipping: $dest"
    return
  fi
  echo "downloading: $(basename "$dest")"
  curl -fL --retry 3 --progress-bar -o "$dest" "$url"
}

# NOTE ON NAMES
# Two checkpoints in the 2023 layout were mislabelled. Verified by MD5:
#   5_GradCAM++/best_efficientDet.pt  ->  f7be0694...  is the YOLOv8s model
#   5_GradCAM++/best_yolov8.pt        ->  cece6489...  is the YOLOv5s model
# Neither is an EfficientDet model. We therefore fetch from the canonical
# training-run paths, which were named correctly, and save them under names
# that match what they actually are.
YOLOV5_RUN="1_YOLO5_Experiment/content/yolov5/runs/train/exp/weights"
YOLOV8_RUN="2_YOLO8_Experiment/runs/detect/train/weights"

fetch "${BASE}/${YOLOV5_RUN}/best.pt" "$OUT/yolov5s_best.pt"
fetch "${BASE}/${YOLOV5_RUN}/last.pt" "$OUT/yolov5s_last.pt"
fetch "${BASE}/${YOLOV8_RUN}/best.pt" "$OUT/yolov8s_best.pt"
fetch "${BASE}/${YOLOV8_RUN}/last.pt" "$OUT/yolov8s_last.pt"

cat <<'MSG'

Done. Verify the checksums with:

    deepseanet verify --manifest results/checkpoints.json --root .

If a checksum does not match the manifest, the file you received is not the one
this project was evaluated on -- open an issue rather than training on it.

Full supporting material for the paper (slides, figures, source documents, 1 GB):
    https://archive.org/download/deepseanet/deepseanet.zip
MSG
