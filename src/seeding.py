"""Seeding for the runners.

Torch is imported lazily so the CPU-only tests can use this without it installed.
The reference subsets are drawn from a local default_rng(seed) in the runners, so
this only pins the global streams that torch, NumPy and random draw from.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed random, NumPy and torch, and with `deterministic` put cuDNN in its slower exact mode."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch isn't installed, so only random and NumPy are seeded")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN is in deterministic mode")


def _self_test() -> None:
    """The same seed has to give the same draws, and a different one different draws."""
    set_seed(123)
    a = np.random.rand(4).tolist()
    set_seed(123)
    b = np.random.rand(4).tolist()
    assert a == b, "set_seed(123) gave different NumPy draws the second time"

    set_seed(456)
    c = np.random.rand(4).tolist()
    assert c != a, "set_seed(456) gave the same draws as set_seed(123)"
    print("set_seed ok")


if __name__ == "__main__":
    _self_test()
