"""Seeding for the runners.

Torch is imported lazily so the CPU-only tests can use this without it installed.
The reference subsets are drawn from a local default_rng(seed) in the runners, so
this only pins the global streams that torch, NumPy and random draw from.
"""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed random, NumPy and torch.

    With `deterministic`, cuDNN goes into its slower exact mode as well.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 the global stream is the one being pinned

    try:
        import torch  # noqa: PLC0415 torch is optional for the CPU-only tests
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
