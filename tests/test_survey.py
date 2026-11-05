from __future__ import annotations

import json
from typing import TYPE_CHECKING

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest

from jwstdetector.survey import (
    near_bright_star,
    on_grid,
    overlaps_other_visit,
    parse_key,
    read_stars,
    rehearse,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jwstdetector.utils import SkyCoords


def test_parse_key_reads_a_southern_tile() -> None:
    key = "jw01837-o014_t016_cx003200_cy000320_ra34.362028_decm5.170180_i00000012.png"
    assert parse_key(key) == (3200, 320, 34.362028, -5.17018)


def test_parse_key_refuses_a_key_without_a_position() -> None:
    with pytest.raises(ValueError, match="no _cx_cy_ra_dec_"):
        parse_key("tile_00012.png")


def test_grid_tiles_share_no_pixel() -> None:
    cx, cy = np.meshgrid(np.arange(320, 1600, 160), np.arange(320, 1600, 160))
    keep = on_grid(cx.ravel(), cy.ravel(), 320)
    assert keep.sum() == 16
    assert keep.size == 64


def _sky(*points: tuple[float, float]) -> SkyCoords:
    ra, dec = zip(*points, strict=True)
    return SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)


def test_overlap_counts_other_visits_only() -> None:
    tiles = _sky((150.0, 2.0), (150.1, 2.0))
    every = _sky((150.0, 2.0), (150.0, 2.0 + 10 / 3600), (150.1, 2.0))
    every_visit = np.array(["a", "a", "b"])
    hit = overlaps_other_visit(
        tiles,
        np.array(["a", "b"]),
        every=every,
        every_visit=every_visit,
        size_arcsec=20.0,
    )
    # The first tile's only neighbour within a diagonal is in its own visit.
    assert hit.tolist() == [False, False]
    shifted = _sky((150.0, 2.0 + 25 / 3600))
    moved = overlaps_other_visit(
        shifted, np.array(["b"]), every=every, every_visit=every_visit, size_arcsec=20.0
    )
    assert moved.tolist() == [True]


def test_star_mask_grows_with_brightness() -> None:
    tiles = _sky((150.0, 2.0 + 30 / 3600))
    g16 = near_bright_star(
        tiles, stars=_sky((150.0, 2.0)), star_g=np.array([15.9]), size_arcsec=20.0
    )
    g12 = near_bright_star(
        tiles, stars=_sky((150.0, 2.0)), star_g=np.array([12.0]), size_arcsec=20.0
    )
    faint = near_bright_star(
        tiles, stars=_sky((150.0, 2.0)), star_g=np.array([17.0]), size_arcsec=20.0
    )
    # At G = 12 the reach is 14.1 + 1.5 * 10**1.4 = 51.8 arcsec, at 15.9 it's 20.3.
    assert (g16.tolist(), g12.tolist(), faint.tolist()) == ([False], [True], [False])


def test_rehearsal_on_exchangeable_visits_stays_near_alpha() -> None:
    rng = np.random.default_rng(1)
    visits = {f"v{i:02d}": rng.normal(size=50) for i in range(40)}
    found = rehearse(visits, 4, 0.1, draws=300)
    assert (found > 0).mean() <= 0.15


def test_rehearsal_flags_visits_that_are_not_exchangeable() -> None:
    rng = np.random.default_rng(1)
    visits = {f"v{i:02d}": rng.normal(size=50) for i in range(40)}
    visits["v00"] = visits["v00"] + 5
    found = rehearse(visits, 4, 0.1, draws=300)
    # Every draw that picks v00 names tiles, which is one draw in ten.
    assert (found > 0).mean() >= 0.09


def test_star_list_must_reach_past_every_tile(tmp_path: Path) -> None:
    stars = tmp_path / "stars.json"
    doc = {"centre": [150.0, 2.0], "radius_deg": 1.0, "rows": [[150.0, 2.0, 12.0]]}
    stars.write_text(json.dumps(doc), encoding="utf-8")
    _, g = read_stars(stars, _sky((150.0, 2.5)))
    assert g.tolist() == [12.0]
    with pytest.raises(ValueError, match=r"only lists stars out to 1.0 degrees"):
        read_stars(stars, _sky((150.0, 2.8)))
