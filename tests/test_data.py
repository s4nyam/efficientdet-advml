"""Dataset preparation, annotation conversion and splitting.

These tests do not need PyTorch, but conftest imports it for the model suite;
they are cheap and run in the same pass.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from deepseanet.data import (
    BoxRecord,
    class_distribution,
    coco_to_yolo,
    drop_empty_labels,
    grouped_split,
    normalize_yolo_file,
    pair_images_and_labels,
    random_split,
    read_yolo_labels,
    video_id_of,
    write_data_yaml,
    write_yolo_labels,
    yolo_to_coco,
    yolo_to_voc,
)


@pytest.fixture
def dataset(tmp_path):
    """Four frames from two clips; one frame has no boxes."""
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    specs = {
        "clipA-0001": [BoxRecord(0, 0.5, 0.5, 0.2, 0.4)],
        "clipA-0002": [BoxRecord(1, 0.25, 0.25, 0.1, 0.1), BoxRecord(0, 0.75, 0.75, 0.2, 0.2)],
        "clipB-0001": [BoxRecord(2, 0.5, 0.5, 0.5, 0.5)],
        "clipB-0002": [],
    }
    for stem, recs in specs.items():
        Image.new("RGB", (960, 540), "black").save(images / f"{stem}.jpg")
        write_yolo_labels(labels / f"{stem}.txt", recs)
    return images, labels


def test_pairing_reports_images_without_labels(dataset, tmp_path):
    images, labels = dataset
    Image.new("RGB", (960, 540)).save(images / "orphan.jpg")
    pairs, unmatched = pair_images_and_labels(images, labels)
    assert len(pairs) == 4
    assert [p.name for p in unmatched] == ["orphan.jpg"]


def test_drop_empty_labels_removes_frames_with_no_boxes(dataset):
    images, labels = dataset
    pairs, _ = pair_images_and_labels(images, labels)
    kept = drop_empty_labels(pairs)
    assert len(kept) == 3
    assert "clipB-0002" not in {p.stem for p in kept}


def test_drop_empty_labels_can_delete_from_disk(dataset):
    images, labels = dataset
    pairs, _ = pair_images_and_labels(images, labels)
    drop_empty_labels(pairs, delete=True)
    assert not (images / "clipB-0002.jpg").exists()
    assert not (labels / "clipB-0002.txt").exists()


def test_normalize_rescales_pixel_coordinates(tmp_path):
    path = tmp_path / "a.txt"
    write_yolo_labels(path, [BoxRecord(0, 480.0, 270.0, 96.0, 54.0)])
    normalize_yolo_file(path, 960, 540)
    rec = read_yolo_labels(path)[0]
    assert rec.cx == pytest.approx(0.5)
    assert rec.cy == pytest.approx(0.5)
    assert rec.w == pytest.approx(0.1)
    assert rec.h == pytest.approx(0.1)


def test_normalize_can_shift_one_based_class_ids(tmp_path):
    path = tmp_path / "a.txt"
    write_yolo_labels(path, [BoxRecord(1, 480.0, 270.0, 96.0, 54.0)])
    normalize_yolo_file(path, 960, 540, class_offset=-1)
    assert read_yolo_labels(path)[0].class_id == 0


def test_read_yolo_labels_on_missing_or_empty_file(tmp_path):
    assert read_yolo_labels(tmp_path / "nope.txt") == []
    (tmp_path / "empty.txt").write_text("")
    assert read_yolo_labels(tmp_path / "empty.txt") == []


def test_read_yolo_labels_rejects_malformed_lines(tmp_path):
    (tmp_path / "bad.txt").write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError, match="expected 5 fields"):
        read_yolo_labels(tmp_path / "bad.txt")


def test_yolo_to_coco_produces_absolute_pixel_boxes(dataset, tmp_path):
    """Regression test for the /1000 bug in the original converter."""
    images, labels = dataset
    out = tmp_path / "train.json"
    coco = yolo_to_coco(images, labels, out, ["a", "b", "c"])

    assert len(coco["images"]) == 4
    assert len(coco["annotations"]) == 4
    assert coco["categories"][0]["id"] == 1, "COCO category ids are one-based"

    box = next(a for a in coco["annotations"] if a["image_id"] == 1)["bbox"]
    assert box == [
        pytest.approx(384.0),
        pytest.approx(162.0),
        pytest.approx(192.0),
        pytest.approx(216.0),
    ]
    assert all(a["area"] > 0 for a in coco["annotations"]), "no box may collapse to zero area"
    assert json.loads(out.read_text())["images"]


def test_yolo_to_voc_writes_one_xml_per_image(dataset, tmp_path):
    import xml.etree.ElementTree as ET

    images, labels = dataset
    out = tmp_path / "voc"
    assert yolo_to_voc(images, labels, out, ["a", "b", "c"]) == 4
    root = ET.parse(out / "clipA-0001.xml").getroot()
    assert root.findtext("size/width") == "960"
    assert root.findtext("object/name") == "a"
    assert int(root.findtext("object/bndbox/xmin")) == 384


def test_coco_to_yolo_round_trips(dataset, tmp_path):
    images, labels = dataset
    yolo_to_coco(images, labels, tmp_path / "c.json", ["a", "b", "c"])
    coco_to_yolo(tmp_path / "c.json", tmp_path / "back")
    original = read_yolo_labels(labels / "clipA-0001.txt")[0]
    restored = read_yolo_labels(tmp_path / "back" / "clipA-0001.txt")[0]
    assert restored.class_id == original.class_id
    assert restored.cx == pytest.approx(original.cx, abs=1e-3)
    assert restored.w == pytest.approx(original.w, abs=1e-3)


def test_class_distribution_counts_boxes_not_files(dataset):
    _, labels = dataset
    dist = class_distribution(labels, ["a", "b", "c"])
    assert dist == {"a": 2, "b": 1, "c": 1}


def test_video_id_strips_the_frame_number():
    assert video_id_of("clipA-0042.jpg") == "clipA"
    assert video_id_of("2019-02-22_22-31-28to2019-02-22_22-31-38_1-0042.jpg") == (
        "2019-02-22_22-31-28to2019-02-22_22-31-38_1"
    )
    assert video_id_of("no_frame_number.jpg") == "no_frame_number"


def test_random_split_partitions_everything_exactly_once():
    items = list(range(100))
    split = random_split(items, (0.7, 0.2, 0.1), seed=3)
    assert split.sizes == {"train": 70, "val": 20, "test": 10}
    assert sorted(split.train + split.val + split.test) == items


def test_random_split_is_deterministic_for_a_seed():
    a = random_split(list(range(50)), seed=11)
    b = random_split(list(range(50)), seed=11)
    assert a.train == b.train


def test_grouped_split_never_puts_a_clip_on_two_sides():
    frames = [f"clip{c}-{i:04d}.jpg" for c in "ABCDEFGHIJ" for i in range(20)]
    split = grouped_split(frames, video_id_of, (0.6, 0.2, 0.2), seed=5)
    groups = {
        name: {video_id_of(f) for f in getattr(split, name)} for name in ("train", "val", "test")
    }
    assert not groups["train"] & groups["val"]
    assert not groups["train"] & groups["test"]
    assert not groups["val"] & groups["test"]
    assert len(split) == len(frames)


def test_grouped_split_approximates_the_requested_ratios():
    frames = [f"clip{c}-{i:04d}.jpg" for c in "ABCDEFGHIJKLMNOPQRST" for i in range(10)]
    split = grouped_split(frames, video_id_of, (0.7, 0.2, 0.1), seed=0)
    assert 0.55 <= len(split.train) / len(frames) <= 0.85


def test_split_rejects_ratios_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        random_split([1, 2, 3], (0.5, 0.2, 0.2))


def test_write_data_yaml_lists_classes_in_id_order(tmp_path):
    path = write_data_yaml(tmp_path / "data.yaml", ["fish", "crab"])
    text = path.read_text()
    assert "nc: 2" in text
    assert "0: fish" in text and "1: crab" in text
