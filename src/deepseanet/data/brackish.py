"""Brackish dataset preparation: video -> frames -> cleaned, paired label files.

The pipeline mirrors ``notebooks/00_dataset_preparation.ipynb`` but as a
testable module with explicit paths instead of hard-coded ``/content``
directories, so it runs the same on Colab, SageMaker or a laptop.

Steps, in order:

1. :func:`extract_frames` - decode every ``.avi`` to JPEG at 960x540 with
   ffmpeg, matching the resolution the official annotations were drawn at.
2. :func:`pair_images_and_labels` - keep only frames that have a label file.
3. :func:`drop_empty_labels` - remove frames whose label file has no boxes.
   In the original run this discarded roughly 3,700 unlabelled frames and
   1,807 zero-visibility frames.
4. :func:`normalize_labels` - rescale pixel coordinates to ``[0, 1]``.
5. :func:`class_distribution` - count boxes per class, for the dataset table.

Source: Pedersen, Haurum, Gade and Moeslund, "Detection of Marine Animals in a
New Underwater Dataset with Varying Visibility", CVPR Workshops 2019.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .convert import IMAGE_SUFFIXES, normalize_yolo_file, read_yolo_labels

__all__ = [
    "FramePair",
    "PreparationReport",
    "extract_frames",
    "pair_images_and_labels",
    "drop_empty_labels",
    "normalize_labels",
    "class_distribution",
    "video_id_of",
]

DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 540


@dataclass(frozen=True)
class FramePair:
    """An image and its label file."""

    image: Path
    label: Path

    @property
    def stem(self) -> str:
        return self.image.stem


@dataclass
class PreparationReport:
    """Counts produced while preparing the dataset, for the README table."""

    frames_extracted: int = 0
    pairs_found: int = 0
    dropped_no_label: int = 0
    dropped_empty_label: int = 0
    labels_normalized: int = 0

    @property
    def frames_kept(self) -> int:
        return self.pairs_found - self.dropped_empty_label

    def as_dict(self) -> dict[str, int]:
        return {
            "frames_extracted": self.frames_extracted,
            "pairs_found": self.pairs_found,
            "dropped_no_label": self.dropped_no_label,
            "dropped_empty_label": self.dropped_empty_label,
            "frames_kept": self.frames_kept,
            "labels_normalized": self.labels_normalized,
        }


def _require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it (apt install ffmpeg / brew install ffmpeg) "
            "or use the pre-extracted frames from the Kaggle or Roboflow release."
        )
    return exe


def extract_frames(
    videos_dir: str | Path,
    output_dir: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    pattern: str = "*.avi",
    quality: int = 2,
) -> int:
    """Decode videos to JPEG frames with ffmpeg.

    Args:
        videos_dir: Directory containing the Brackish ``.avi`` files.
        output_dir: Where frames are written, named ``<video>-%04d.jpg``.
        width: Output width. Keep at 960 to match the shipped annotations.
        height: Output height. Keep at 540 for the same reason.
        pattern: Glob used to find videos.
        quality: ffmpeg ``-q:v`` value; 2 is visually lossless, 31 is worst.

    Returns:
        The number of frames written.

    Raises:
        RuntimeError: If ffmpeg is not installed or a decode fails.
    """
    ffmpeg = _require_ffmpeg()
    videos_dir, output_dir = Path(videos_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before = sum(1 for _ in output_dir.glob("*.jpg"))
    for video in sorted(videos_dir.rglob(pattern)):
        cmd = [
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"scale={width}:{height}",
            "-sws_flags",
            "bicubic",
            "-q:v",
            str(quality),
            str(output_dir / f"{video.stem}-%04d.jpg"),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {video.name}: {result.stderr.strip()}")
    return sum(1 for _ in output_dir.glob("*.jpg")) - before


def pair_images_and_labels(
    images_dir: str | Path, labels_dir: str | Path
) -> tuple[list[FramePair], list[Path]]:
    """Match images to label files by stem.

    Returns:
        ``(pairs, unmatched_images)``. Anything in ``unmatched_images`` has no
        annotation and should not be used for training or evaluation.
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    pairs: list[FramePair] = []
    unmatched: list[Path] = []
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append(FramePair(img, lbl))
        else:
            unmatched.append(img)
    return pairs, unmatched


def drop_empty_labels(pairs: Sequence[FramePair], *, delete: bool = False) -> list[FramePair]:
    """Filter out frames whose label file contains no boxes.

    Args:
        pairs: Candidate frames.
        delete: If ``True``, also unlink the discarded files from disk. Off by
            default so a mistake is recoverable.

    Returns:
        The frames that carry at least one box.
    """
    kept: list[FramePair] = []
    for pair in pairs:
        if read_yolo_labels(pair.label):
            kept.append(pair)
        elif delete:
            pair.image.unlink(missing_ok=True)
            pair.label.unlink(missing_ok=True)
    return kept


def normalize_labels(
    labels_dir: str | Path,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    class_offset: int = 0,
) -> int:
    """Rescale every label file in a directory to normalised coordinates.

    Args:
        labels_dir: Directory of ``.txt`` files.
        width: Frame width the coordinates were written against.
        height: Frame height.
        class_offset: Added to every class id; use ``-1`` for one-based ids.

    Returns:
        The number of files rewritten.
    """
    labels_dir = Path(labels_dir)
    count = 0
    for lbl in sorted(labels_dir.glob("*.txt")):
        normalize_yolo_file(lbl, width, height, class_offset=class_offset)
        count += 1
    return count


def class_distribution(
    labels_dir: str | Path, class_names: Sequence[str] | None = None
) -> dict[str, int]:
    """Count boxes per class across a directory of label files.

    Args:
        labels_dir: Directory of ``.txt`` files.
        class_names: Names in label-id order. Ids without a name are reported
            as ``"class_<id>"``.

    Returns:
        Mapping from class name to box count, ordered by ``class_names``.
    """
    counter: Counter[int] = Counter()
    for lbl in Path(labels_dir).glob("*.txt"):
        for rec in read_yolo_labels(lbl):
            counter[rec.class_id] += 1
    if class_names is None:
        return {f"class_{cid}": n for cid, n in sorted(counter.items())}
    out = {name: counter.get(i, 0) for i, name in enumerate(class_names)}
    for cid, n in sorted(counter.items()):
        if cid >= len(class_names):
            out[f"class_{cid}"] = n
    return out


def video_id_of(frame: str | Path) -> str:
    """Recover the source video name from a frame filename.

    ``extract_frames`` writes ``<video>-0001.jpg``, so stripping the trailing
    ``-NNNN`` recovers the clip. This is what
    :func:`~deepseanet.data.split.grouped_split` groups on, so near-identical
    neighbouring frames cannot land on both sides of a split.

    Example:
        >>> video_id_of("2019-02-22_22-31-28to2019-02-22_22-31-38_1-0042.jpg")
        '2019-02-22_22-31-28to2019-02-22_22-31-38_1'
    """
    stem = Path(frame).stem
    head, sep, tail = stem.rpartition("-")
    if sep and tail.isdigit():
        return head
    return stem


def iter_pairs(images_dir: str | Path, labels_dir: str | Path) -> Iterator[FramePair]:
    """Yield matched frames lazily, for datasets too large to list at once."""
    pairs, _ = pair_images_and_labels(images_dir, labels_dir)
    yield from pairs
