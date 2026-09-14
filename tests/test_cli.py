"""The command-line interface."""

from __future__ import annotations

import pytest
from PIL import Image

from deepseanet.cli import main
from deepseanet.data import BoxRecord, write_yolo_labels


@pytest.fixture
def dataset(tmp_path):
    images, labels = tmp_path / "images", tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for clip in ("clipA", "clipB"):
        for i in range(3):
            stem = f"{clip}-{i:04d}"
            Image.new("RGB", (960, 540)).save(images / f"{stem}.jpg")
            write_yolo_labels(labels / f"{stem}.txt", [BoxRecord(0, 0.5, 0.5, 0.1, 0.1)])
    return images, labels


def test_version_flag_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "deepseanet" in capsys.readouterr().out


def test_missing_command_is_an_error():
    with pytest.raises(SystemExit):
        main([])


def test_prepare_reports_counts(dataset, capsys):
    images, labels = dataset
    assert main(["prepare", "--images", str(images), "--labels", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "paired frames        : 6" in out
    assert "boxes per class" in out


def test_split_grouped_keeps_clips_intact(dataset, tmp_path, capsys):
    images, labels = dataset
    code = main(
        [
            "split",
            "--images",
            str(images),
            "--labels",
            str(labels),
            "--strategy",
            "grouped",
            "--train",
            "0.5",
            "--val",
            "0.5",
            "--test",
            "0.0",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "strategy: grouped" in out
    assert (tmp_path / "out" / "data.yaml").exists()
    assert (tmp_path / "out" / "images" / "train").is_dir()


def test_convert_to_coco_writes_a_file(dataset, tmp_path, capsys):
    images, labels = dataset
    out = tmp_path / "c.json"
    assert (
        main(
            [
                "convert",
                "--to",
                "coco",
                "--images",
                str(images),
                "--labels",
                str(labels),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert out.exists()
    assert "6 images" in capsys.readouterr().out


def test_model_command_prints_a_parameter_breakdown(capsys):
    pytest.importorskip("torch")
    assert main(["model", "--neck", "biskfpn", "--image-size", "128"]) == 0
    out = capsys.readouterr().out
    assert "backbone" in out and "neck" in out and "total" in out


def test_report_command_on_the_committed_logs(tmp_path, capsys):
    run = tmp_path / "yolov5s"
    run.mkdir()
    run.write_text if False else (run / "results.csv").write_text(
        "epoch,metrics/mAP_0.5\n0,0.5\n1,0.9\n"
    )
    assert main(["report", "--results", str(tmp_path)]) == 0
    assert "yolov5s" in capsys.readouterr().out
