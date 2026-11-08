from __future__ import annotations

import json
from typing import TYPE_CHECKING

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest

from jwstdetector.survey import (
    Tiles,
    check_owned,
    cut_blocks,
    near_bright_star,
    overlaps_other_visit,
    parse_key,
    read_stars,
    rehearse,
    standardise,
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


def test_a_run_that_scored_every_patch_is_refused(tmp_path: Path) -> None:
    (tmp_path / "args.yaml").write_text("score_region: owned\nstride: 160\n")
    check_owned(tmp_path)
    (tmp_path / "args.yaml").write_text("score_region: all\nstride: 160\n")
    with pytest.raises(ValueError, match="wasn't scored with --score_region owned"):
        check_owned(tmp_path)


def _sky(*points: tuple[float, float]) -> SkyCoords:
    ra, dec = zip(*points, strict=True)
    return SkyCoord(np.array(ra) * u.deg, np.array(dec) * u.deg)


def _tiles(rows: list[tuple[float, float, str, str, int, int]]) -> Tiles:
    # Each row is RA, Dec, visit, mosaic, and the centre's x and y in the mosaic.
    return Tiles(
        coords=_sky(*[(r[0], r[1]) for r in rows]),
        visit=np.array([r[2] for r in rows]),
        mosaic=np.array([r[3] for r in rows]),
        centre=np.array([r[4:] for r in rows], dtype=np.int64).reshape(-1, 2),
    )


def test_overlap_counts_other_visits_only() -> None:
    every = _tiles([
        (150.0, 2.0, "a", "A", 0, 0),
        (150.0, 2.0 + 10 / 3600, "a", "A", 0, 160),
        (150.1, 2.0, "b", "B", 0, 0),
    ])
    tiles = _tiles([(150.0, 2.0, "a", "A", 0, 0), (150.1, 2.0, "b", "B", 0, 0)])
    # The first tile's only neighbour within a diagonal is in its own visit.
    hit = overlaps_other_visit(tiles, every, size_arcsec=20.0, size_px=320)
    assert hit.tolist() == [False, False]
    shifted = _tiles([(150.0, 2.0 + 25 / 3600, "b", "B", 0, 0)])
    moved = overlaps_other_visit(shifted, every, size_arcsec=20.0, size_px=320)
    assert moved.tolist() == [True]


def test_blocks_of_one_mosaic_overlap_only_where_their_boxes_do() -> None:
    # 1 px is 1/16 arcsecond, so 320 px is 20 arcseconds and inside the diagonal.
    every = _tiles([
        (150.0, 2.0, "b1", "A", 0, 0),
        (150.0, 2.0 + 20 / 3600, "b2", "A", 0, 320),
    ])
    tiles = _tiles([
        (150.0, 2.0, "b1", "A", 0, 0),
        (150.0, 2.0 + 10 / 3600, "b1", "A", 0, 160),
    ])
    hit = overlaps_other_visit(tiles, every, size_arcsec=20.0, size_px=320)
    assert hit.tolist() == [False, True]


def test_blocks_take_only_tiles_that_lie_wholly_inside(tmp_path: Path) -> None:
    data = tmp_path / "m"
    (data / "query").mkdir(parents=True)
    (data / "mosaics").mkdir()
    (data / "mosaics" / "M.json").write_text("{}", encoding="utf-8")
    rows = []
    for x0 in range(0, 1440, 160):
        key = f"M_cx{x0 + 160:06d}_cy000160_ra150.000000_dec2.000000_i{x0:08d}.png"
        (data / "query" / key).write_bytes(b"png")
        rows.append(f"{key},M,{x0},0,320,150,2,")
    (data / "manifest.csv").write_text(
        "key,mosaic,x0,y0,size,ra,dec,cutout\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    got = cut_blocks(data, tmp_path / "blocks", block_px=640, min_tiles=2)
    # Each 640 px block holds the tiles at 0, 160 and 320 of its own, and the tile at
    # 480 straddles the border. The third block only has room for one tile.
    assert got == {"M_b00_00": 3, "M_b00_01": 3}
    linked = tmp_path / "blocks" / "M_b00_01" / "query"
    assert sorted(int(k.name.split("_i")[1][:8]) for k in linked.iterdir()) == [
        640,
        800,
        960,
    ]


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
    # A fainter cut-off masks the G = 17 star too, whose reach is 14.1 + 3.8 arcsec.
    near = near_bright_star(
        _sky((150.0, 2.0 + 15 / 3600)),
        stars=_sky((150.0, 2.0)),
        star_g=np.array([17.0]),
        size_arcsec=20.0,
        g_max=19.0,
    )
    assert near.tolist() == [True]


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


def test_standardising_takes_out_a_visit_wide_shift() -> None:
    # The visit that was rendered darker no longer names tiles once each visit is on
    # its own median and MAD, and a scaled copy of a visit comes out the same.
    rng = np.random.default_rng(1)
    visits = {f"v{i:02d}": rng.normal(size=50) for i in range(40)}
    visits["v00"] = 3 * visits["v01"] + 5
    kept = {
        v: {f"{v}_{j}": float(s) for j, s in enumerate(x)} for v, x in visits.items()
    }
    z = standardise(kept)
    assert np.allclose(list(z["v00"].values()), list(z["v01"].values()))
    found = rehearse(
        {v: np.array(list(s.values())) for v, s in z.items()}, 4, 0.1, draws=300
    )
    assert (found > 0).mean() <= 0.15


def test_standardising_refuses_a_visit_with_no_spread() -> None:
    with pytest.raises(ValueError, match="3 kept scores have no spread"):
        standardise({"v1": {"a": 1.0, "b": 1.0, "c": 1.0}})


def test_star_list_must_reach_past_every_tile(tmp_path: Path) -> None:
    stars = tmp_path / "stars.json"
    doc = {"centre": [150.0, 2.0], "radius_deg": 1.0, "rows": [[150.0, 2.0, 12.0]]}
    stars.write_text(json.dumps(doc), encoding="utf-8")
    _, g = read_stars(stars, _sky((150.0, 2.5)))
    assert g.tolist() == [12.0]
    with pytest.raises(ValueError, match=r"0.3 degrees inside any cone"):
        read_stars(stars, _sky((150.0, 2.8)))


def test_star_list_can_cover_two_fields(tmp_path: Path) -> None:
    stars = tmp_path / "stars.json"
    doc = {
        "cones": [[150.0, 2.0, 1.0], [34.4, -5.2, 1.0]],
        "rows": [[150.0, 2.0, 12.0], [34.4, -5.2, 14.0]],
    }
    stars.write_text(json.dumps(doc), encoding="utf-8")
    _, g = read_stars(stars, _sky((150.0, 2.5), (34.4, -5.5)))
    assert g.tolist() == [12.0, 14.0]
    with pytest.raises(ValueError, match=r"at 34.4000 -6.0000 isn't"):
        read_stars(stars, _sky((150.0, 2.5), (34.4, -6.0)))
