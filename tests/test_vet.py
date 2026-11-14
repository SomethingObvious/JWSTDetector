# pyright: reportPrivateUsage=false
from __future__ import annotations

import argparse
import csv
from itertools import permutations
import json
from typing import TYPE_CHECKING

from astropy.coordinates import SkyCoord
from astropy.io import fits
import numpy as np
from PIL import Image
import pytest

from jwst_peculiar.vet import Source, _write, flag, hot_cell, significance

if TYPE_CHECKING:
    from pathlib import Path

NAN = float("nan")


def _planes(*bright_in: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    planes = rng.normal(size=(3, 280, 280))
    # Top left of the tile as scored, which is near row 260 in FITS order since the
    # scored tile is flipped top to bottom.
    for i in bright_in:
        planes[i, 258:263, 18:23] += 40.0
    return planes


def test_a_source_in_every_filter_is_not_flagged() -> None:
    planes = _planes(0, 1, 2)
    sigs = [significance(p, (1, 1), 28) for p in planes]
    assert min(sigs) > 20
    assert flag(sigs) == ""


def test_a_source_in_one_filter_is_flagged() -> None:
    sigs = [significance(p, (1, 1), 28) for p in _planes(2)]
    assert sigs[2] > 20
    assert flag(sigs) == "one filter"


def test_the_patch_is_found_through_the_flip() -> None:
    # The same patch counted from the bottom, without the flip, is empty sky.
    planes = _planes(0, 1, 2)
    assert flag([significance(p, (26, 1), 28) for p in planes]) == "no filter"


def test_bright_sources_are_not_flagged_for_their_colour() -> None:
    # A red galaxy far above the noise in two filters and well seen in the third.
    assert flag([480.0, 160.0, 40.0]) == ""
    assert flag([50.0, 48.0, 4.7]) == ""


@pytest.mark.parametrize(
    ("sigs", "expected"),
    [
        ([NAN, 2.0, 1.0], "no filter"),
        ([30.0, NAN, 1.0], "one filter"),
        ([30.0, NAN, NAN], "one filter"),
        ([30.0, 20.0, NAN], ""),
    ],
)
def test_a_missing_filter_never_vouches_wherever_it_sits(
    sigs: list[float], expected: str
) -> None:
    assert {flag(list(p)) for p in permutations(sigs)} == {expected}


def test_a_lone_filter_is_judged_like_the_strongest_of_many() -> None:
    assert flag([30.0]) == "one filter"
    assert flag([2.0]) == "no filter"


def test_the_hot_patch_is_one_the_tile_scores() -> None:
    scores = np.zeros((28, 28))
    scores[0, 3] = 9.0  # an object cut off by the tile's edge
    scores[10, 12] = 5.0
    owned = np.zeros((28, 28), dtype=bool)
    owned[7:21, 7:21] = True
    assert hot_cell(scores, owned) == (10, 12)


def _visit(data: Path, visit: str, filters: list[str]) -> None:
    for d in ("mosaics", "query", "cutouts"):
        (data / visit / d).mkdir(parents=True)
    (data / visit / "manifest.csv").write_text(
        "key,mosaic,x0,y0,size,ra,dec,cutout\n"
        f"{visit}.png,m,0,0,16,10.0,-5.0,cutouts/{visit}.fits\n",
        encoding="utf-8",
    )
    (data / visit / "mosaics" / "m.json").write_text(
        json.dumps({"filters": filters}), encoding="utf-8"
    )
    Image.new("RGB", (16, 16)).save(data / visit / "query" / f"{visit}.png")
    rng = np.random.default_rng(0)
    fits.PrimaryHDU(rng.normal(size=(len(filters), 16, 16))).writeto(
        data / visit / "cutouts" / f"{visit}.fits"
    )


def test_each_source_is_listed_under_its_own_filters(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _visit(data, "a", ["F115W", "F200W", "F444W"])
    _visit(data, "b", ["F090W", "F150W", "F277W", "F356W"])
    at = SkyCoord(10.0, -5.0, unit="deg")
    sources = [
        Source(1, "a", "a.png", 3.0, at, (1, 1), 4, [30.0, 20.0, 10.0]),
        Source(
            2, "b", "b.png", 2.0, at, (2, 3), 4, [30.0, 1.0, 1.0, 1.0], "one filter"
        ),
    ]
    args = argparse.Namespace(data=data, out=tmp_path / "out", no_catalogues=True)
    _write(args, sources, 1)
    with (args.out / "sources.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    mads = [{k: v for k, v in r.items() if k.endswith("_mad")} for r in rows]
    blank = dict.fromkeys(mads[0], "")
    assert list(blank) == [
        "F115W_mad",
        "F200W_mad",
        "F444W_mad",
        "F090W_mad",
        "F150W_mad",
        "F277W_mad",
        "F356W_mad",
    ]
    assert mads == [
        {**blank, "F115W_mad": "30.0", "F200W_mad": "20.0", "F444W_mad": "10.0"},
        {
            **blank,
            "F090W_mad": "30.0",
            "F150W_mad": "1.0",
            "F277W_mad": "1.0",
            "F356W_mad": "1.0",
        },
    ]
    assert [r["sheet"] for r in rows] == ["source001.png", "source002.png"]
    assert (args.out / "source002.png").is_file()


@pytest.mark.parametrize("filters", [["F200W"], ["F200W", "F444W"]])
def test_every_source_gets_a_sheet_however_few_its_filters(
    tmp_path: Path, filters: list[str]
) -> None:
    data = tmp_path / "data"
    _visit(data, "a", filters)
    at = SkyCoord(10.0, -5.0, unit="deg")
    sigs = [30.0] * len(filters)
    args = argparse.Namespace(data=data, out=tmp_path / "out", no_catalogues=True)
    _write(args, [Source(1, "a", "a.png", 3.0, at, (1, 1), 4, sigs)], 1)
    assert (args.out / "source001.png").is_file()
