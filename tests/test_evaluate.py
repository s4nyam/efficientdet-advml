"""Parsing the committed training logs."""

from __future__ import annotations

import pytest

from deepseanet.evaluate import (
    collect_runs,
    format_table,
    mean_std,
    read_results_csv,
    summarize_run,
)

YOLOV5_CSV = """\
               epoch,      train/box_loss,    metrics/precision,       metrics/recall,      metrics/mAP_0.5, metrics/mAP_0.5:0.95
                   0,             0.10000,              0.50000,              0.40000,              0.30000,              0.10000
                   1,             0.05000,              0.80000,              0.70000,              0.90000,              0.60000
                   2,             0.04000,              0.85000,              0.75000,              0.88000,              0.65000
"""

YOLOV8_CSV = """\
               epoch, metrics/precision(B),    metrics/recall(B),     metrics/mAP50(B),  metrics/mAP50-95(B)
                   0,              0.60000,              0.50000,              0.40000,              0.20000
                   1,              0.99000,              0.97000,              0.98000,              0.83000
"""


@pytest.fixture
def results_dir(tmp_path):
    (tmp_path / "yolov5s").mkdir()
    (tmp_path / "yolov8s").mkdir()
    (tmp_path / "yolov5s" / "results.csv").write_text(YOLOV5_CSV)
    (tmp_path / "yolov8s" / "results.csv").write_text(YOLOV8_CSV)
    (tmp_path / "no_log").mkdir()
    return tmp_path


def test_read_results_csv_strips_padded_headers(results_dir):
    rows = read_results_csv(results_dir / "yolov5s" / "results.csv")
    assert len(rows) == 3
    assert "metrics/mAP_0.5" in rows[0]


def test_summarize_picks_final_and_best_separately(results_dir):
    summary = summarize_run(results_dir / "yolov5s")
    assert summary.epochs == 3
    assert summary.final["map50"] == pytest.approx(0.88)
    assert summary.best["map50"] == (pytest.approx(0.90), 1)


def test_summarize_handles_the_yolov8_column_names(results_dir):
    summary = summarize_run(results_dir / "yolov8s")
    assert summary.final["map50"] == pytest.approx(0.98)
    assert summary.final["map"] == pytest.approx(0.83)


def test_summarize_raises_when_there_is_no_log(results_dir):
    with pytest.raises(FileNotFoundError, match="no results.csv"):
        summarize_run(results_dir / "no_log")


def test_collect_runs_skips_directories_without_logs(results_dir):
    names = [s.name for s in collect_runs(results_dir)]
    assert names == ["yolov5s", "yolov8s"]


def test_mean_std_matches_the_papers_table_5_row():
    """YOLOv8: 98.0, 98.5, 98.3, 98.1, 98.2 -> reported as 98.2 +/- 0.17."""
    mean, std = mean_std([98.0, 98.5, 98.3, 98.1, 98.2])
    assert mean == pytest.approx(98.22, abs=0.01)
    assert std == pytest.approx(0.19, abs=0.03)


def test_mean_std_of_a_single_value_has_zero_spread():
    assert mean_std([5.0]) == (5.0, 0.0)


def test_mean_std_population_denominator():
    _, std = mean_std([1.0, 3.0], sample=False)
    assert std == pytest.approx(1.0)


def test_mean_std_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one value"):
        mean_std([])


def test_format_table_renders_markdown(results_dir):
    table = format_table(collect_runs(results_dir))
    lines = table.splitlines()
    assert lines[0].startswith("| run")
    assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}
    assert len(lines) == 4


def test_format_table_on_no_runs():
    assert "no runs" in format_table([])
