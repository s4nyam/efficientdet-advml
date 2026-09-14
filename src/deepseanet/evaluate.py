"""Read the committed training logs and summarise them.

The ``results/`` directory keeps one training log per detector, exactly as the
Ultralytics trainers wrote it. This module turns those CSVs into the summary
table in ``docs/RESULTS.md`` without re-running anything, and provides the
mean-and-standard-deviation helper used for the paper's five-repetition tables.

Only the standard library is needed; ``pandas`` is used if it happens to be
installed but is never required.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "RunSummary",
    "read_results_csv",
    "summarize_run",
    "mean_std",
    "format_table",
    "collect_runs",
]

# Column aliases: YOLOv5 and YOLOv8 name the same metrics differently.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "map50": ("metrics/mAP_0.5", "metrics/mAP50(B)", "metrics/mAP50"),
    "map": ("metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    "precision": ("metrics/precision", "metrics/precision(B)"),
    "recall": ("metrics/recall", "metrics/recall(B)"),
}


@dataclass
class RunSummary:
    """Final and best metrics of one training run.

    Attributes:
        name: Run identifier, normally the directory name.
        epochs: Number of epochs logged.
        final: Metrics at the last epoch.
        best: Best value each metric reached, and the epoch it happened at.
    """

    name: str
    epochs: int
    final: dict[str, float]
    best: dict[str, tuple[float, int]]

    def as_row(self) -> dict[str, object]:
        """Flatten to a single dict, for tabulating several runs together."""
        row: dict[str, object] = {"run": self.name, "epochs": self.epochs}
        row.update({k: round(v, 4) for k, v in self.final.items()})
        row.update({f"best_{k}": round(v, 4) for k, (v, _) in self.best.items()})
        return row


def read_results_csv(path: str | Path) -> list[dict[str, float]]:
    """Parse an Ultralytics ``results.csv`` into a list of per-epoch dicts.

    Column names are stripped of the padding whitespace the trainers write, and
    every value is coerced to ``float`` where possible.
    """
    rows: list[dict[str, float]] = []
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                key = key.strip()
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    continue
            if row:
                rows.append(row)
    return rows


def _resolve(rows: Sequence[dict[str, float]], metric: str) -> str | None:
    if not rows:
        return None
    columns = rows[0]
    for alias in _METRIC_ALIASES.get(metric, (metric,)):
        if alias in columns:
            return alias
    return None


def summarize_run(path: str | Path, name: str | None = None) -> RunSummary:
    """Summarise one ``results.csv``.

    Args:
        path: Path to the CSV, or to the directory containing it.
        name: Display name. Defaults to the parent directory name.

    Returns:
        A :class:`RunSummary`.

    Raises:
        FileNotFoundError: If no ``results.csv`` is found.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "results.csv"
    if not p.exists():
        raise FileNotFoundError(f"no results.csv at {p}")

    rows = read_results_csv(p)
    final: dict[str, float] = {}
    best: dict[str, tuple[float, int]] = {}
    for metric in _METRIC_ALIASES:
        column = _resolve(rows, metric)
        if column is None:
            continue
        values = [r[column] for r in rows if column in r]
        if not values:
            continue
        final[metric] = values[-1]
        top = max(values)
        best[metric] = (top, values.index(top))
    return RunSummary(name or p.parent.name, len(rows), final, best)


def mean_std(values: Iterable[float], *, sample: bool = True) -> tuple[float, float]:
    """Mean and standard deviation, as reported in the paper's Table 5.

    Args:
        values: The repeated measurements.
        sample: Use the sample standard deviation (``n-1`` denominator), which
            is the right choice for a handful of repeated runs. Set ``False``
            for the population formula.

    Returns:
        ``(mean, std)``. With fewer than two values the std is ``0.0``.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("mean_std needs at least one value")
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    denom = len(vals) - 1 if sample else len(vals)
    var = sum((v - mean) ** 2 for v in vals) / denom
    return mean, math.sqrt(var)


def collect_runs(results_dir: str | Path) -> list[RunSummary]:
    """Summarise every subdirectory of ``results/`` that holds a ``results.csv``."""
    out: list[RunSummary] = []
    for child in sorted(Path(results_dir).iterdir()):
        if child.is_dir() and (child / "results.csv").exists():
            out.append(summarize_run(child))
    return out


def format_table(summaries: Sequence[RunSummary], *, markdown: bool = True) -> str:
    """Render summaries as a Markdown (or plain) table.

    Args:
        summaries: Runs to tabulate.
        markdown: Emit a Markdown table with a separator row.

    Returns:
        The rendered table.
    """
    if not summaries:
        return "(no runs found)"
    rows = [s.as_row() for s in summaries]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in columns}
    lines = ["| " + " | ".join(c.ljust(widths[c]) for c in columns) + " |"]
    if markdown:
        lines.append("| " + " | ".join("-" * widths[c] for c in columns) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(c, "-")).ljust(widths[c]) for c in columns) + " |"
        )
    return "\n".join(lines)
