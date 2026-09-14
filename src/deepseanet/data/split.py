"""Train / validation / test splits.

Two strategies are provided, and which one you pick changes the numbers a lot.

:func:`random_split`
    Shuffle every frame and cut. This reproduces what the committed notebooks
    did, and what the 70:20:10 split in the paper refers to.

:func:`grouped_split`
    Shuffle *videos*, then assign all of a video's frames to one side. Brackish
    frames come from continuous footage: neighbouring frames are nearly
    identical, so a random split puts near-duplicates in both training and test
    and inflates every score. Grouping by source clip removes that leak and
    gives a harder, more honest estimate. Use this for any new claim.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

__all__ = ["Split", "random_split", "grouped_split", "materialize_split", "write_data_yaml"]


@dataclass
class Split:
    """Three disjoint lists of items."""

    train: list
    val: list
    test: list

    def __post_init__(self) -> None:
        self.train, self.val, self.test = list(self.train), list(self.val), list(self.test)

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def _check_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    if len(ratios) != 3:
        raise ValueError("expected three ratios: (train, val, test)")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {total}")
    return tuple(float(r) for r in ratios)  # type: ignore[return-value]


def random_split(
    items: Sequence, ratios: Sequence[float] = (0.7, 0.2, 0.1), *, seed: int = 0
) -> Split:
    """Shuffle and cut into three parts.

    Args:
        items: Anything indexable - paths, ``FramePair`` objects, ids.
        ratios: ``(train, val, test)`` fractions, summing to 1.
        seed: RNG seed, so the split is reproducible.

    Returns:
        A :class:`Split`.

    .. note::
       On frame-sequence data this leaks near-duplicates across splits. Prefer
       :func:`grouped_split` for anything you intend to report.
    """
    train_r, val_r, _ = _check_ratios(ratios)
    pool = list(items)
    random.Random(seed).shuffle(pool)
    n = len(pool)
    n_train = round(n * train_r)
    n_val = round(n * val_r)
    return Split(pool[:n_train], pool[n_train : n_train + n_val], pool[n_train + n_val :])


def grouped_split(
    items: Sequence,
    key: Callable[[object], str],
    ratios: Sequence[float] = (0.7, 0.2, 0.1),
    *,
    seed: int = 0,
) -> Split:
    """Split by group so no group spans two sides.

    Groups are shuffled and then assigned greedily to whichever split is
    furthest below its target share, which keeps the sizes close to ``ratios``
    even when clips differ wildly in length.

    Args:
        items: The frames to split.
        key: Maps an item to its group id. For Brackish frames use
            :func:`~deepseanet.data.brackish.video_id_of`.
        ratios: ``(train, val, test)`` fractions.
        seed: RNG seed.

    Returns:
        A :class:`Split` in which every group appears in exactly one part.

    Example:
        >>> from deepseanet.data import grouped_split, video_id_of
        >>> frames = [f"clipA-{i:04d}.jpg" for i in range(10)] + \
                     [f"clipB-{i:04d}.jpg" for i in range(10)]
        >>> s = grouped_split(frames, video_id_of, (0.5, 0.5, 0.0), seed=1)
        >>> {video_id_of(f) for f in s.train} & {video_id_of(f) for f in s.val}
        set()
    """
    train_r, val_r, test_r = _check_ratios(ratios)
    groups: dict[str, list] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)

    names = sorted(groups)
    random.Random(seed).shuffle(names)

    targets = {"train": train_r, "val": val_r, "test": test_r}
    buckets: dict[str, list] = {"train": [], "val": [], "test": []}
    total = sum(len(v) for v in groups.values())
    for name in names:
        deficits = {
            split: targets[split] - (len(buckets[split]) / total if total else 0.0)
            for split in buckets
            if targets[split] > 0
        }
        if not deficits:
            deficits = {"train": 1.0}
        chosen = max(deficits, key=lambda s: deficits[s])
        buckets[chosen].extend(groups[name])
    return Split(buckets["train"], buckets["val"], buckets["test"])


def materialize_split(
    split: Split,
    output_dir: str | Path,
    *,
    images_subdir: str = "images",
    labels_subdir: str = "labels",
    copy: bool = True,
) -> dict[str, int]:
    """Write a split to disk in the ``images/<split>`` + ``labels/<split>`` layout.

    Args:
        split: The split to write. Items must be
            :class:`~deepseanet.data.brackish.FramePair` objects.
        output_dir: Root directory to create the layout under.
        images_subdir: Name of the images directory.
        labels_subdir: Name of the labels directory.
        copy: Copy files if ``True``, hard-link if ``False``. Linking is
            instant and costs no extra disk, but edits propagate.

    Returns:
        Number of frames written per split.
    """
    import shutil

    output_dir = Path(output_dir)
    written: dict[str, int] = {}
    for name in ("train", "val", "test"):
        pairs = getattr(split, name)
        img_dir = output_dir / images_subdir / name
        lbl_dir = output_dir / labels_subdir / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for pair in pairs:
            for src, dst_dir in ((pair.image, img_dir), (pair.label, lbl_dir)):
                dst = dst_dir / Path(src).name
                if dst.exists():
                    continue
                if copy:
                    shutil.copy2(src, dst)
                else:
                    Path(dst).hardlink_to(src)
        written[name] = len(pairs)
    return written


def write_data_yaml(
    output: str | Path,
    class_names: Sequence[str],
    *,
    root: str = ".",
    images_subdir: str = "images",
) -> Path:
    """Write the ``data.yaml`` file the Ultralytics trainers expect.

    Args:
        output: Path of the YAML file.
        class_names: Class names in label-id order.
        root: Dataset root, written as the ``path`` key.
        images_subdir: Directory holding the per-split image folders.

    Returns:
        The path written.
    """
    output = Path(output)
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(class_names))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"path: {root}\n"
        f"train: {images_subdir}/train\n"
        f"val: {images_subdir}/val\n"
        f"test: {images_subdir}/test\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names}\n"
    )
    return output
