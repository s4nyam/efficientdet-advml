"""Annotation format conversion: YOLO text, COCO JSON and Pascal VOC XML.

Every toolkit used in this project wants a different format for the same boxes:

===============  ==================================================
Toolkit          Format
===============  ==================================================
YOLOv5, YOLOv8   ``<class> <cx> <cy> <w> <h>`` per line, normalised
Detectron2       COCO JSON, ``[x, y, w, h]`` in absolute pixels
TFLite Model     Pascal VOC XML, ``xmin/ymin/xmax/ymax`` in pixels
Maker
===============  ==================================================

.. warning::
   The converter shipped in the original repository
   (``notebooks/legacy/yolo_to_coco_original.py``) divided every box
   coordinate by 1000 before writing it out::

       create_annotation_from_yolo_format(int(min_x/1000), int(min_y/1000), ...)

   With 960x540 frames that truncates essentially every box to ``(0, 0, 0, 0)``.
   Any COCO file produced by that script is unusable. The implementation here
   does not do that. See ``docs/REPRODUCIBILITY.md``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BoxRecord",
    "read_yolo_labels",
    "write_yolo_labels",
    "yolo_to_coco",
    "yolo_to_voc",
    "coco_to_yolo",
    "normalize_yolo_file",
]

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class BoxRecord:
    """One annotated box in normalised YOLO form.

    Attributes:
        class_id: Zero-based class index.
        cx: Centre x, normalised to ``[0, 1]``.
        cy: Centre y, normalised to ``[0, 1]``.
        w: Width, normalised to ``[0, 1]``.
        h: Height, normalised to ``[0, 1]``.
    """

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Absolute ``(x1, y1, x2, y2)`` corners for an image of this size."""
        bw, bh = self.w * width, self.h * height
        x1 = self.cx * width - bw / 2
        y1 = self.cy * height - bh / 2
        return x1, y1, x1 + bw, y1 + bh

    def to_xywh(self, width: int, height: int) -> tuple[float, float, float, float]:
        """Absolute COCO ``(x, y, w, h)`` for an image of this size."""
        x1, y1, x2, y2 = self.to_xyxy(width, height)
        return x1, y1, x2 - x1, y2 - y1


def read_yolo_labels(path: str | Path) -> list[BoxRecord]:
    """Parse a YOLO ``.txt`` label file. Missing or empty files give ``[]``."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[BoxRecord] = []
    for lineno, line in enumerate(p.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(f"{p}:{lineno}: expected 5 fields, got {len(parts)}")
        cid, cx, cy, w, h = parts[:5]
        records.append(BoxRecord(int(float(cid)), float(cx), float(cy), float(w), float(h)))
    return records


def write_yolo_labels(
    path: str | Path, records: Iterable[BoxRecord], *, precision: int = 6
) -> None:
    """Write records back out in YOLO text format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{r.class_id} {r.cx:.{precision}f} {r.cy:.{precision}f} {r.w:.{precision}f} {r.h:.{precision}f}"
        for r in records
    ]
    p.write_text("\n".join(lines) + ("\n" if lines else ""))


def normalize_yolo_file(
    path: str | Path, image_width: int = 960, image_height: int = 540, *, class_offset: int = 0
) -> int:
    """Divide pixel coordinates by the image size, in place.

    The Brackish release ships label files whose coordinates are in *pixels*
    despite using the YOLO field order. This rescales them to ``[0, 1]``.

    Args:
        path: The label file to rewrite.
        image_width: Frame width the coordinates were written against.
        image_height: Frame height the coordinates were written against.
        class_offset: Added to every class id. Use ``-1`` to shift one-based
            ids down to the zero-based ids YOLO expects.

    Returns:
        The number of boxes rewritten.
    """
    records = read_yolo_labels(path)
    scaled = [
        BoxRecord(
            r.class_id + class_offset,
            r.cx / image_width,
            r.cy / image_height,
            r.w / image_width,
            r.h / image_height,
        )
        for r in records
    ]
    write_yolo_labels(path, scaled)
    return len(scaled)


def _image_size(path: Path) -> tuple[int, int]:
    """Read ``(width, height)`` without decoding the whole image where possible."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise ImportError("reading image sizes needs Pillow") from exc


def _paired(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
    return pairs


def yolo_to_coco(
    images_dir: str | Path,
    labels_dir: str | Path,
    output: str | Path,
    class_names: Sequence[str],
    *,
    supercategory: str = "marine_animal",
) -> dict:
    """Convert a YOLO directory pair into a single COCO JSON file.

    Args:
        images_dir: Directory of images.
        labels_dir: Directory of matching ``.txt`` label files.
        output: Path of the JSON file to write.
        class_names: Class names in label-id order; COCO category ids are
            one-based, so ``class_names[0]`` becomes category ``1``.
        supercategory: Value written into every category entry.

    Returns:
        The COCO dictionary that was written.
    """
    images_dir, labels_dir, output = Path(images_dir), Path(labels_dir), Path(output)
    coco: dict = {
        "info": {"description": "Brackish dataset converted from YOLO format"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": supercategory}
            for i, name in enumerate(class_names)
        ],
    }
    ann_id = 1
    for image_id, (img_path, lbl_path) in enumerate(_paired(images_dir, labels_dir), start=1):
        width, height = _image_size(img_path)
        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )
        for rec in read_yolo_labels(lbl_path):
            x, y, w, h = rec.to_xywh(width, height)
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": rec.class_id + 1,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(coco, indent=2))
    return coco


def yolo_to_voc(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    class_names: Sequence[str],
) -> int:
    """Convert a YOLO directory pair into one Pascal VOC XML file per image.

    TFLite Model Maker's ``DataLoader.from_pascal_voc`` expects this layout.

    Returns:
        The number of XML files written.
    """
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path, lbl_path in _paired(images_dir, labels_dir):
        width, height = _image_size(img_path)
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = images_dir.name
        ET.SubElement(root, "filename").text = img_path.name
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"
        for rec in read_yolo_labels(lbl_path):
            x1, y1, x2, y2 = rec.to_xyxy(width, height)
            obj = ET.SubElement(root, "object")
            name = (
                class_names[rec.class_id] if rec.class_id < len(class_names) else str(rec.class_id)
            )
            ET.SubElement(obj, "name").text = name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            box = ET.SubElement(obj, "bndbox")
            ET.SubElement(box, "xmin").text = str(max(0, int(round(x1))))
            ET.SubElement(box, "ymin").text = str(max(0, int(round(y1))))
            ET.SubElement(box, "xmax").text = str(min(width, int(round(x2))))
            ET.SubElement(box, "ymax").text = str(min(height, int(round(y2))))
        ET.ElementTree(root).write(output_dir / f"{img_path.stem}.xml", encoding="utf-8")
        count += 1
    return count


def coco_to_yolo(coco_json: str | Path, output_dir: str | Path, *, class_offset: int = -1) -> int:
    """Convert a COCO JSON file back into per-image YOLO label files.

    Args:
        coco_json: The COCO annotation file.
        output_dir: Directory the ``.txt`` files are written to.
        class_offset: Added to COCO's one-based category ids. The default of
            ``-1`` produces the zero-based ids YOLO expects.

    Returns:
        The number of label files written.
    """
    coco = json.loads(Path(coco_json).read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sizes = {img["id"]: (img["width"], img["height"], img["file_name"]) for img in coco["images"]}
    grouped: dict[int, list[BoxRecord]] = {img_id: [] for img_id in sizes}
    for ann in coco["annotations"]:
        width, height, _ = sizes[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        grouped[ann["image_id"]].append(
            BoxRecord(
                ann["category_id"] + class_offset,
                (x + w / 2) / width,
                (y + h / 2) / height,
                w / width,
                h / height,
            )
        )
    for img_id, records in grouped.items():
        _, _, file_name = sizes[img_id]
        write_yolo_labels(output_dir / f"{Path(file_name).stem}.txt", records)
    return len(grouped)
