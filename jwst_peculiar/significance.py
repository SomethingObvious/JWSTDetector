"""Conformal p-values and Benjamini-Hochberg q-values for tile scores.

A tile's p-value is the share of calibration tiles that score at least as high, with
the usual plus one on top and bottom (Bates et al. 2023). It is valid when the
calibration tiles are exchangeable with the ordinary tiles being tested, which is
why `jwst-peculiar run --reference_roots` scores each control field against the other
control fields only. The target is scored against all of them, and while the kNN
bank holds every patch, that bigger reference only lowers its scores. Real anomalies
among the calibration tiles only raise theirs, so both push p-values up.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_REPORTED_Q = 0.1


def conformal_p(
    scores: npt.NDArray[np.float64], calibration: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """(1 + calibration scores at least as high) / (n + 1) for each score.

    A NaN score gets 1, as nothing was there to compare it with. NaN calibration
    scores are dropped.

    Returns:
        One p-value per score, in the same order.

    Raises:
        ValueError: If none of the calibration scores is finite.
    """
    cal = np.sort(calibration[np.isfinite(calibration)])
    if cal.size == 0:
        msg = "There are no finite calibration scores to compare with."
        raise ValueError(msg)
    at_least = cal.size - np.searchsorted(cal, scores, side="left")
    p = (1 + at_least) / (cal.size + 1)
    return np.where(np.isfinite(scores), p, 1.0)


def bh_q(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Benjamini-Hochberg adjusted p-values.

    Returns:
        The q-values, so q <= alpha picks the set BH picks at alpha.
    """
    m = p.size
    order = np.argsort(p, kind="stable")
    scaled = p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def write_significance(
    path: Path, scores: dict[str, float], calibration: npt.NDArray[np.float64]
) -> int:
    """Write each tile's key, p and q.

    Returns:
        How many tiles have q at most 0.1.
    """
    keys = sorted(scores)
    s = np.array([scores[k] for k in keys], dtype=np.float64)
    p = conformal_p(s, calibration)
    q = bh_q(p)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sample", "PValue", "QValue"])
        for k, pk, qk in zip(keys, p, q, strict=True):
            w.writerow([k, f"{pk:.6g}", f"{qk:.6g}"])
    return int((q <= _REPORTED_Q).sum())


def read_significance(path: Path) -> dict[str, tuple[float, float]]:
    """Read back what `write_significance` wrote.

    Returns:
        (p, q) for each tile key.
    """
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {
            r["Sample"]: (float(r["PValue"]), float(r["QValue"]))
            for r in csv.DictReader(f)
        }
