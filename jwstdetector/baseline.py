"""An isolation forest on hand-built tile features, the way Astronomaly scores images.

It's here as the standard baseline for the benchmark. Each tile becomes a short vector
of brightness percentiles per channel, colour ratios, the shape of the bright regions
as second moments at a few thresholds, and the radially averaged power spectrum, and
the forest ranks the tiles by how quickly random splits isolate them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from jwstdetector.data import relative_key

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_PERCENTILES = (1, 10, 50, 90, 99, 99.9)
_THRESHOLDS = (90, 99, 99.9)
_SPECTRUM_BINS = 8
# Fewer pixels than this and there's no ellipse to fit.
_MIN_PIXELS = 3


def _moments(mask: npt.NDArray[np.bool_]) -> list[float]:
    """Area, ellipticity and centre offset of the pixels in `mask`.

    Returns:
        The three numbers, as an ellipse fit would give them.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < _MIN_PIXELS:
        return [0.0, 0.0, 0.0]
    cov = np.cov(np.stack([xs, ys]).astype(np.float64))
    lo, hi = np.linalg.eigvalsh(cov)
    ellipticity = 1.0 - np.sqrt(max(lo, 0.0) / hi) if hi > 0 else 0.0
    h, w = mask.shape
    offset = np.hypot(xs.mean() - w / 2, ys.mean() - h / 2) / max(h, w)
    return [len(xs) / mask.size, float(ellipticity), float(offset)]


def _spectrum(grey: npt.NDArray[np.float64]) -> list[float]:
    """Log power in `_SPECTRUM_BINS` log-spaced radial frequency bands.

    Returns:
        One value per band, 0 for a band no frequency falls in.
    """
    power = np.abs(np.fft.fftshift(np.fft.fft2(grey - grey.mean()))) ** 2
    h, w = grey.shape
    y, x = np.indices((h, w))
    r = np.hypot(x - w // 2, y - h // 2)
    edges = np.geomspace(1, min(h, w) / 2, _SPECTRUM_BINS + 1)
    band = np.digitize(r, edges) - 1
    out: list[float] = []
    for b in range(_SPECTRUM_BINS):
        sel = band == b
        out.append(float(np.log10(power[sel].mean() + 1e-12)) if sel.any() else 0.0)
    return out


def tile_features(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """The baseline's feature vector for one (H, W, 3) uint8 tile.

    Returns:
        The features as one flat vector.
    """
    rgb = image.astype(np.float64) / 255.0
    grey = rgb.mean(axis=2)
    feats: list[float] = []
    for c in range(3):
        feats += np.percentile(rgb[..., c], _PERCENTILES).tolist()
        feats.append(float(rgb[..., c].std()))
    total = rgb.sum(axis=2) + 1e-6
    for c in range(3):
        feats.append(float(np.median(rgb[..., c] / total)))
    for q in _THRESHOLDS:
        feats += _moments(grey > np.percentile(grey, q))
    feats += _spectrum(grey)
    return np.asarray(feats, dtype=np.float64)


def isolation_scores(
    paths: list[Path], root: Path, *, seed: int = 0, n_trees: int = 500
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit an isolation forest to the tiles and score each one.

    Returns:
        The anomaly score per tile, higher for tiles the forest isolates sooner, and
        the seconds each tile took.
    """
    from sklearn.ensemble import IsolationForest  # noqa: PLC0415 slow to load, only iforest needs it

    keys: list[str] = []
    rows: list[npt.NDArray[np.float64]] = []
    seconds: dict[str, float] = {}
    for p in paths:
        start = time.perf_counter()
        with Image.open(p) as img:
            rows.append(tile_features(np.asarray(img.convert("RGB"))))
        key = relative_key(p, root)
        keys.append(key)
        seconds[key] = time.perf_counter() - start
    x = np.stack(rows)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    forest = IsolationForest(n_estimators=n_trees, random_state=seed).fit(x)
    scores = -forest.score_samples(x)
    return dict(zip(keys, scores.tolist(), strict=True)), seconds
