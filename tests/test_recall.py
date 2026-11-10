from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from jwst_peculiar.recall import group_interval, place, separation

if TYPE_CHECKING:
    from pathlib import Path


def test_separation_is_great_circle_arcseconds() -> None:
    assert np.isclose(separation(10.0, -72.0, (10.0, -73.0)), 3600.0)
    # A degree of RA at Dec 60 is half a degree on the sky.
    assert np.isclose(separation(11.0, 60.0, (10.0, 60.0)), 1800.0, rtol=1e-3)


def test_group_interval_draws_whole_groups() -> None:
    # Two groups whose members agree, so every draw's mean is 10, 20 or 30.
    lo, hi = group_interval([("a", 10.0), ("a", 10.0), ("b", 30.0)], n=2000)
    assert lo == 10.0
    assert hi == 30.0


def test_an_object_in_two_mosaics_is_placed_once(tmp_path: Path) -> None:
    run, data = tmp_path / "run", tmp_path / "data"
    (data / "mosaics").mkdir(parents=True)
    run.mkdir()
    (run / "args.yaml").write_text("seed: 0\n")
    (run / "measurements_final.csv").write_text(
        "Sample,AnomalyScore\na.png,2.0\nb.png,1.0\n"
    )
    # Two mosaics of the same sky, with one tile each.
    (data / "manifest.csv").write_text(
        "key,mosaic,x0,y0,size,ra,dec,cutout\n"
        "a.png,A,0,0,320,150,2,\nb.png,B,0,0,320,150,2,\n"
    )
    wcs = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": 100.0,
        "CRPIX2": 100.0,
        "CDELT1": -1e-5,
        "CDELT2": 1e-5,
    }
    for name in "AB":
        (data / "mosaics" / f"{name}.json").write_text(json.dumps({"wcs": wcs}))
    obj = {"id": "1.1", "group": "1", "ra": "150.0", "dec": "2.0"}
    placed, n_tiles, _ = place([obj], run, data)
    assert [p.tile for p in placed] == ["a.png"]
    assert n_tiles == 2
