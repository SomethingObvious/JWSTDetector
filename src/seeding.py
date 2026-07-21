"""Reproducibility helper shared by the runners.

Seeds Python's ``random``, NumPy's global RNG, and (when installed) PyTorch, so a
run tagged with ``--seed N`` can be repeated. Torch is imported lazily and its
absence is tolerated, which keeps this module importable on a CPU-only box for
testing without pulling in the GPU stack.

Note on scope: the runners already pick their reference subsets with a *local*
``numpy.random.default_rng(seed)``, so calling :func:`set_seed` does not change
which tiles get chosen. It only pins the previously unseeded torch / ``random`` /
global-NumPy streams, which is why it is safe to add to an existing pipeline.
"""

from __future__ import annotations

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG this project touches.

    Args:
        seed: The seed applied to ``random``, NumPy, and torch.
        deterministic: When true, also force cuDNN into deterministic mode
            (``cudnn.deterministic=True``, ``cudnn.benchmark=False``). This can
            slow the GPU path and is opt-in for exactly that reason; the default
            leaves cuDNN untouched so timing and numerics match a normal run.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; seeded random + numpy only")
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN set to deterministic mode")


def _self_test() -> None:
    """Framework-free check: the same seed reproduces the same draws."""
    set_seed(123)
    a = np.random.rand(4).tolist()
    set_seed(123)
    b = np.random.rand(4).tolist()
    assert a == b, "set_seed did not reproduce identical numpy draws"

    set_seed(456)
    c = np.random.rand(4).tolist()
    assert c != a, "different seeds unexpectedly produced identical draws"
    print("seeding self-test passed")


if __name__ == "__main__":
    _self_test()
