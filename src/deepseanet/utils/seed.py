"""Seeding helpers.

The paper reports five repetitions per model as ``mean +/- std``. Those runs
are only meaningful if each one is individually reproducible, so seed
everything and record which seed produced which number.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "worker_init_fn"]


def seed_everything(seed: int = 0, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: The seed value.
        deterministic: Also force deterministic cuDNN kernels. This makes runs
            bit-reproducible on the same hardware at the cost of throughput,
            and raises if an op has no deterministic implementation.

    Returns:
        The seed, so it can be logged alongside the results.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding, so augmentation differs per worker but repeats."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
