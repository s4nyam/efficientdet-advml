"""Shared fixtures. Torch-dependent tests skip cleanly when torch is absent."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="PyTorch is needed for model tests")


@pytest.fixture(scope="session")
def rng() -> torch.Generator:
    g = torch.Generator().manual_seed(1234)
    return g


@pytest.fixture
def pyramid() -> list[torch.Tensor]:
    """A five-level feature pyramid with halving spatial sizes."""
    channels = [40, 112, 320, 320, 320]
    sizes = [32, 16, 8, 4, 2]
    return [torch.randn(2, c, s, s) for c, s in zip(channels, sizes)]
