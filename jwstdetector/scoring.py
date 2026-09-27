"""Turning one tile's patch distances into one anomaly score.

This stays free of torch so the scoring rules can be checked on any machine.
"""

from __future__ import annotations

import numpy as np

SCORE_MODES = ("top1p", "topk_mean", "max", "quantile", "peak_minus_median", "peak_z")


def aggregate_score(
    dists: np.ndarray,
    mode: str = "top1p",
    *,
    top_frac: float = 0.01,
    quantile: float = 0.9995,
    eps: float = 1e-8,
) -> float:
    """Collapse the patch distances of one tile into a single anomaly score.

    top1p              mean of the top 1% of patches, the AnomalyDINO default
    topk_mean          the same with top_frac as the fraction
    max                the single worst patch, most sensitive to a small needle and noisiest
    quantile           a high quantile, nearly as sensitive as max but steadier
    peak_minus_median  how far the worst patch stands above the tile's own median, so a
                       tile that is odd all over doesn't outrank one strange object
    peak_z             the same peak in standard deviations of the tile
    """
    d = np.asarray(dists, dtype=np.float32).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0

    if mode == "top1p":
        top_frac, mode = 0.01, "topk_mean"

    if mode == "topk_mean":
        m = max(1, round(d.size * float(np.clip(top_frac, 1e-6, 1.0))))
        return float(np.mean(np.partition(d, -m)[-m:]))
    if mode == "max":
        return float(np.max(d))
    if mode == "quantile":
        return float(np.quantile(d, float(np.clip(quantile, 0.0, 1.0))))
    if mode == "peak_minus_median":
        return float(np.max(d) - np.median(d))
    if mode == "peak_z":
        return float((np.max(d) - np.mean(d)) / (np.std(d) + eps))

    raise ValueError(f"Unknown score_mode {mode!r}, expected one of {SCORE_MODES}")
