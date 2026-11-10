"""Checks for the injection benchmark's source model and its statistics."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from jwst_peculiar import benchmark
from jwst_peculiar.benchmark import Optics, Source

if TYPE_CHECKING:
    from pathlib import Path

SIZE = 64


def _source(kind: str, sed: str = "flat") -> Source:
    return Source("t.png", kind, sed, cx=30.0, cy=34.0, radius=9.0, theta=0.4, span=2.0)


def _band(micron: float | None = 2.0) -> dict[str, Any]:
    return {
        "filter": "F",
        "micron": micron,
        "jy_per_unit": 1e-6,
        "psf_fwhm_arcsec": 0.18,
    }


def test_shapes_stay_inside_the_tile() -> None:
    for kind in (*benchmark.KINDS, *benchmark.HELD_OUT):
        img = benchmark.intrinsic(_source(kind), SIZE)
        assert img.shape == (SIZE, SIZE)
        assert np.max(img) > 0.5
        assert np.max(img[0]) < 1e-3 or kind == "streak"
    ring = benchmark.intrinsic(_source("ring"), SIZE).sum()
    arc = benchmark.intrinsic(_source("arc"), SIZE).sum()
    # A 2 radian arc is about a third of the ring.
    assert abs(arc / ring - 2 / (2 * math.pi)) < 0.05


def _peaks(img: np.ndarray) -> list[tuple[int, int]]:
    """Find the pixels above half the maximum that beat their eight neighbours.

    Returns:
        Each peak's row and column.
    """
    pad = np.pad(img, 1)
    out = []
    for y, x in zip(*np.nonzero(img > 0.5 * img.max()), strict=True):
        around = pad[y : y + 3, x : x + 3]
        if img[y, x] == around.max():
            out.append((int(y), int(x)))
    return out


def test_held_out_shapes() -> None:
    # A span of 2 radians sits 0.68 of the way from 60 to 140 degrees, which gives
    # 3 + round(2.05) = 5 knots and a pair 2 + 2.05 pixels apart.
    assert len(_peaks(benchmark.intrinsic(_source("clumps"), SIZE))) == 5
    pair = _peaks(benchmark.intrinsic(_source("pair"), SIZE))
    assert len(pair) == 2
    (y1, x1), (y2, x2) = pair
    assert abs(math.hypot(y1 - y2, x1 - x2) - 4.05) < 1
    diffuse = benchmark.intrinsic(_source("diffuse"), SIZE)
    # A Gaussian of sigma 4.5 has 2 pi sigma^2 of volume for a peak of 1.
    assert abs(diffuse.sum() / (2 * math.pi * 4.5**2) - 1) < 0.01


def test_planted_flux_matches_the_magnitude() -> None:
    planes = np.ones((1, SIZE, SIZE), np.float32)
    out = benchmark.add_source(
        planes, _source("point"), 25.0, optics=Optics([_band()], pixel_scale=0.06)
    )
    added = out.astype(np.float64).sum() - planes.sum()
    # AB 25 is 0.3631 microjanskys, which is 0.3631 units at 1e-6 Jy per unit.
    assert abs(added - 0.3631) < 1e-4
    row, col = np.unravel_index(np.argmax(out[0]), out[0].shape)
    assert (int(row), int(col)) == (34, 30)


def test_a_red_source_is_brighter_at_long_wavelengths() -> None:
    planes = np.ones((2, SIZE, SIZE), np.float32)
    bands = [_band(2.0), _band(4.44)]
    out = benchmark.add_source(
        planes, _source("point", "red"), 25.0, optics=Optics(bands, pixel_scale=0.06)
    )
    short, long = (out[i].astype(np.float64).sum() - SIZE * SIZE for i in range(2))
    assert abs(long / short - (4.44 / 2.0) ** 2) < 1e-3
    # The magnitude is set at the geometric mean wavelength.
    assert abs(math.sqrt(short * long) - 0.3631) < 1e-4


def test_holes_stay_holes() -> None:
    planes = np.ones((1, SIZE, SIZE), np.float32)
    planes[0, 30:40, 25:35] = np.nan
    planes[0, 0:5] = 0
    out = benchmark.add_source(
        planes, _source("ring"), 20.0, optics=Optics([_band()], pixel_scale=0.06)
    )
    assert np.isnan(out[0, 30:40, 25:35]).all()
    assert (out[0, 0:5] == 0).all()


def test_ranks_put_nan_last() -> None:
    assert benchmark.ranks_of({"a": 1.0, "b": float("nan"), "c": 3.0}) == {
        "c": 1,
        "a": 2,
        "b": 3,
    }


def test_null_ranks_ignore_other_planted_tiles() -> None:
    scores = {"a": 5.0, "b": 4.0, "p1": 3.0, "c": 2.0, "p2": 6.0, "p3": float("nan")}
    # p2 outscores p1, but only the unplanted a and b count against p1.
    assert benchmark.null_ranks(scores, {"p1", "p2", "p3"}) == {
        "p1": 3,
        "p2": 1,
        "p3": 4,
    }


def test_wilson_interval() -> None:
    lo, hi = benchmark.wilson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = benchmark.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4)
    assert hi == 1.0
    lo, hi = benchmark.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    assert benchmark.wilson(0, 0) == (0.0, 1.0)


def test_summary_and_m50() -> None:
    # 100 tiles, so found means a rank of 5 or better.
    ranks = {24: [1, 2, 3, 4], 25: [1, 2, 9, 40], 26: [30, 50, 60, 70]}
    rows = [
        {"detector": "knn", "kind": "point", "mag": mag, "rank": r}
        for mag, rs in ranks.items()
        for r in rs
    ]
    summary = benchmark.summarise(rows, 100)
    point = {s["mag"]: s for s in summary if s["kind"] == "point"}
    assert [point[m]["recall"] for m in (24, 25, 26)] == [1.0, 0.5, 0.0]
    assert point[25]["median_rank"] == 5.5
    assert len([s for s in summary if s["kind"] == "all"]) == 3
    # Recall is exactly 0.5 at AB 25 and drops after it, so that's where it crosses.
    assert benchmark.m50(summary, "knn", "point") == 25.0
    assert benchmark.m50(summary, "pixels", "point") is None
    # The interval's low end is under 0.5 at AB 25, so it crosses between 24 and 25.
    lo = benchmark.m50(summary, "knn", "point", key="recall_lo")
    assert lo is not None
    assert 24 < lo < 25
    # Without null ranks nothing counts as beating every ordinary tile.
    assert {s["beats_all"] for s in summary} == {0}
    strict = benchmark.summarise(
        [{**r, "null_rank": 1 if r["rank"] == 1 else 2} for r in rows], 100
    )
    assert [s["beats_all"] for s in strict if s["kind"] == "point"] == [1, 1, 0]
    assert list(strict[0])[-1] == "beats_all"


def test_check_dataset_refuses_what_it_cannot_plant(tmp_path: Path) -> None:
    (tmp_path / "mosaics").mkdir()
    info: dict[str, Any] = {
        "render": {"kind": "asinh"},
        "pixel_scale_arcsec": 0.06,
        "planes": [_band(), _band(4.0)],
    }
    (tmp_path / "mosaics" / "m.json").write_text(json.dumps(info))
    manifest = {"t.png": {"mosaic": "m", "cutout": "cutouts/t.fits"}}

    assert (
        benchmark.check_dataset(tmp_path, manifest)["m"]["planes"][1]["micron"] == 4.0
    )
    with pytest.raises(SystemExit, match="--save_cutouts"):
        benchmark.check_dataset(tmp_path, {"t.png": {"mosaic": "m", "cutout": ""}})
    with pytest.raises(SystemExit, match="1 or 2 values"):
        benchmark.check_dataset(tmp_path, manifest, psf_fwhm=[0.1, 0.2, 0.3])
    swapped = benchmark.check_dataset(tmp_path, manifest, psf_fwhm=[0.3])
    assert [b["psf_fwhm_arcsec"] for b in swapped["m"]["planes"]] == [0.3, 0.3]

    info["planes"][0]["jy_per_unit"] = None
    (tmp_path / "mosaics" / "m.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit, match="Pass --jy_per_unit"):
        benchmark.check_dataset(tmp_path, manifest)
    del info["render"]
    (tmp_path / "mosaics" / "m.json").write_text(json.dumps(info))
    with pytest.raises(SystemExit, match="stretched"):
        benchmark.check_dataset(tmp_path, manifest)
