"""Turning one tile's patch distances into one anomaly score.

This stays free of torch so the scoring rules can be checked on any machine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

SCORE_MODES = ("top1p", "topk_mean", "max", "quantile", "peak_minus_median", "peak_z")


def aggregate_score(
    dists: npt.NDArray[Any],
    mode: str = "top1p",
    *,
    top_frac: float = 0.01,
    quantile: float = 0.9995,
    eps: float = 1e-8,
) -> float:
    """Collapse the patch distances of one tile into a single anomaly score.

    top1p              mean of the top 1% of patches, the AnomalyDINO default
    topk_mean          the same with top_frac as the fraction
    max                the single worst patch, the most sensitive to a small
                       needle and the noisiest
    quantile           a high quantile, nearly as sensitive as max but steadier
    peak_minus_median  how far the worst patch stands above the tile's own median,
                       so a tile that is odd all over doesn't outrank one strange
                       object
    peak_z             the same peak in standard deviations of the tile

    Returns:
        The score, or NaN when none of the distances is finite.

    Raises:
        ValueError: If `mode` isn't one of SCORE_MODES.
    """
    d: npt.NDArray[np.float32] = np.asarray(dists, dtype=np.float32).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        # No patch could be compared with anything, which isn't the same as normal.
        return float("nan")

    if mode == "top1p":
        top_frac, mode = 0.01, "topk_mean"

    def topk_mean() -> float:
        m = max(1, round(d.size * float(np.clip(top_frac, 1e-6, 1.0))))
        return float(np.mean(np.partition(d, -m)[-m:]))

    reducers: dict[str, Callable[[], Any]] = {
        "topk_mean": topk_mean,
        "max": lambda: np.max(d),
        "quantile": lambda: np.quantile(d, float(np.clip(quantile, 0.0, 1.0))),
        "peak_minus_median": lambda: np.max(d) - np.median(d),
        "peak_z": lambda: (np.max(d) - np.mean(d)) / (np.std(d) + eps),
    }
    if mode not in reducers:
        msg = f"Unknown score_mode {mode!r}, expected one of {SCORE_MODES}"
        raise ValueError(msg)
    return float(reducers[mode]())
