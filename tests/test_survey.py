from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest

from jwst_peculiar.survey import (
    Tiles,
    can_name_one,
    check_owned,
    cut_blocks,
    kept_tiles,
    main,
    middle_scores,
    near_bright_star,
    overlaps_other_visit,
    parse_key,
    read_stars,
    rehearse,
    standardise,
)

if TYPE_CHECKING:
    from pathlib import Path

    from jwst_peculiar.utils import SkyCoords


def test_parse_key_reads_a_southern_tile() -> None:
    key = "jw01837-o014_t016_cx003200_cy000320_ra34.362028_decm5.170180_i00000012.png"
    assert parse_key(key) == (3200, 320, 34.362028, -5.17018)


def test_parse_key_refuses_a_key_without_a_position() -> None:
    with pytest.raises(ValueError, match="no _cx_cy_ra_dec_"):
        parse_key("tile_00012.png")


def test_a_run_that_scored_every_patch_is_refused(tmp_path: Path) -> None:
    for region in ("owned", "middle"):
        (tmp_path / "args.yaml").write_text(f"score_region: {region}\nstride: 160\n")
        check_owned(tmp_path)
    (tmp_path / "args.yaml").write_text("score_region: all\nstride: 160\n")
    with pytest.raises(ValueError, match="--score_region owned or middle, so"):
        check_owned(tmp_path)


def test_a_lone_tile_is_rescored_on_its_middle(tmp_path: Path) -> None:
    run, data = tmp_path / "run", tmp_path / "data"
    maps = run / "cross" / "anomaly_maps" / "seed=0"
    maps.mkdir(parents=True)
    data.mkdir()
    (run / "args.yaml").write_text("score_region: owned\nscore_mode: max\n")
    (run / "measurements_final.csv").write_text("Sample,AnomalyScore\na.png,9.0\n")
    (data / "manifest.csv").write_text(
        "key,mosaic,x0,y0,size,ra,dec,cutout\na.png,M,0,0,320,150,2,\n"
    )
    # With no neighbour the tile scored its edge, where the hottest patch is.
    dists = np.ones((28, 28), np.float32)
    dists[0, 3], dists[10, 12] = 9, 5
    np.save(maps / "a.npy", dists)
    assert middle_scores(run, data) == {"a.png": 5.0}


def test_middle_rescore_leaves_out_masked_patches(
    tmp_path: Path,
) -> None:
    run, data = tmp_path / "run", tmp_path / "data"
    key = "sub/M_cx000160_cy000160_ra150.000000_dec2.000000_i00000000.png"
    # Nested under its folder and named with every dot, as detection saves it.
    maps = run / "pass2" / "anomaly_maps" / "seed=3" / "sub"
    maps.mkdir(parents=True)
    data.mkdir()
    (run / "args.yaml").write_text(
        "score_region: owned\nscore_mode: peak_minus_median\nmasking: true\nseed: 3\n"
    )
    (run / "measurements_final.csv").write_text(f"Sample,AnomalyScore\n{key},9.0\n")
    (data / "manifest.csv").write_text(
        f"key,mosaic,x0,y0,size,ra,dec,cutout\n{key},M,0,0,320,150,2,\n"
    )
    # The middle is rows and columns 7 to 20. Over half of it was masked out and saved
    # as 0, which would pull the median from 1 to 0.
    dists = np.ones((28, 28), np.float32)
    dists[7:15, 7:21] = 0
    dists[18, 12] = 5
    np.save(maps / f"{key[4:-4]}.npy", dists)
    assert middle_scores(run, data) == {key: 4.0}


def _visit(
    data: Path,
    name: str,
    tiles: list[tuple[float, float]],
    *,
    size: int = 320,
    scale: float = 0.0625,
) -> list[str]:
    """Write a visit's manifest and mosaic record, one mosaic of `size` px tiles.

    Returns:
        The tile keys, in the order given.
    """
    root = data / name
    (root / "mosaics").mkdir(parents=True)
    (root / "mosaics" / f"M{name}.json").write_text(
        json.dumps({"pixel_scale_arcsec": scale}), encoding="utf-8"
    )
    keys = [
        f"M{name}_cx{400 * j:06d}_cy000160_ra{ra:.6f}_dec{dec:.6f}_i{j:08d}.png"
        for j, (ra, dec) in enumerate(tiles)
    ]
    rows = [
        f"{k},M{name},{400 * j},0,{size},{ra},{dec},"
        for j, (k, (ra, dec)) in enumerate(zip(keys, tiles, strict=True))
    ]
    (root / "manifest.csv").write_text(
        "key,mosaic,x0,y0,size,ra,dec,cutout\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return keys


def _row(i: int, dec: float) -> list[tuple[float, float]]:
    return [(150.0 + 0.05 * j, dec) for j in range(i)]


def _survey(
    tmp_path: Path, scores: dict[str, list[float]]
) -> tuple[list[str], dict[str, Any]]:
    """Lay out a search visit, two calibration visits and a bank, with their results.

    c1's last tile sits 30 arcsec from a G = 12 star and c2's last tile 10 arcsec
    from the bank's tile, and every other tile is 3 arcmin from the next.

    Returns:
        The search visit's keys and the split.
    """
    data, results = tmp_path / "data", tmp_path / "results"
    keys = {
        "s1": _visit(data, "s1", _row(3, 2.0)),
        "c1": _visit(data, "c1", _row(5, 2.05)),
        "c2": _visit(data, "c2", [*_row(4, 2.1), (150.0, 2.15 + 10 / 3600)]),
        "b": _visit(data, "b", [(150.0, 2.15)]),
    }
    for v, values in scores.items():
        (results / v).mkdir(parents=True)
        (results / v / "args.yaml").write_text("score_region: owned\n")
        (results / v / "measurements_final.csv").write_text(
            "Sample,AnomalyScore\n"
            + "".join(f"{k},{s}\n" for k, s in zip(keys[v], values, strict=True))
        )
    (tmp_path / "stars.json").write_text(
        json.dumps({
            "centre": [150.0, 2.0],
            "radius_deg": 1.0,
            "rows": [[150.2, 2.05 + 30 / 3600, 12.0]],
        }),
        encoding="utf-8",
    )
    split = {
        "alpha": 0.5,
        "tile_px": 320,
        "search": ["s1"],
        "bank": ["b"],
        "calibration": ["c1", "c2"],
    }
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    return keys["s1"], split


def _main(tmp_path: Path) -> int:
    return main([
        str(tmp_path / "split.json"),
        str(tmp_path / "stars.json"),
        str(tmp_path / "data"),
        str(tmp_path / "results"),
    ])


# The tiles the rules drop score 50, which would change every count had they stayed.
SCORES = {
    "s1": [100.0, 0.0, 4.5],
    "c1": [1.0, 3.0, 5.0, 7.0, 50.0],
    "c2": [2.0, 4.0, 6.0, 8.0, 50.0],
}


def test_search_names_only_the_tile_no_calibration_tile_reaches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    search, _ = _survey(tmp_path, SCORES)
    assert _main(tmp_path) == 0
    verdict = json.loads((tmp_path / "results" / "verdict.json").read_text())
    assert json.loads(capsys.readouterr().out) == verdict
    # Against 8 kept calibration tiles, p is 1/9, 5/9 and 1, and q 1/3, 5/6 and 1.
    # Each rehearsal draw tests 4 interleaved tiles against the other 4 and names
    # none.
    assert verdict == {
        "search_tiles": 3,
        "calibration_tiles": 8,
        "dropped": {"overlap": 1, "star": 1},
        "smallest_p": 1 / 9,
        "bh_needs_for_one": 0.5 / 3,
        "rehearsal_draws_with_any_detection": 0.0,
        "rehearsal_passed": True,
        "detections": [search[0]],
    }
    rows = (tmp_path / "results" / "search_pvalues.csv").read_text().splitlines()
    assert rows == [
        "key,score,p,q",
        f"{search[0]},100,0.111111,0.333333",
        f"{search[2]},4.5,0.555556,0.833333",
        f"{search[1]},0,1,1",
    ]


def test_nan_score_stops_the_search(tmp_path: Path) -> None:
    _survey(tmp_path, {**SCORES, "c2": [2.0, float("nan"), 6.0, 8.0, 50.0]})
    with pytest.raises(ValueError, match=r"^c2's tile Mc2_cx000400_\S+ scored NaN"):
        _main(tmp_path)


def test_rehearsal_needs_a_calibration_visit_left_over(
    tmp_path: Path,
) -> None:
    _survey(tmp_path, SCORES)
    split = {**json.loads((tmp_path / "split.json").read_text()), "search": ["s1", "x"]}
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="more than 2 calibration visits, not 2"):
        _main(tmp_path)


@pytest.mark.parametrize(
    ("argv", "code"),
    [
        ([], 2),
        (["split.json", "stars.json", "data"], 2),
        (["--check", "split.json", "stars.json", "data", "results"], 2),
        (["--middle", "split.json", "stars.json", "data"], 2),
        (["--fast", "split.json", "stars.json", "data", "results"], 2),
        (["--help"], 0),
    ],
)
def test_arguments_that_fit_no_form_print_the_usage(
    argv: list[str], code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == code
    out, err = capsys.readouterr()
    shown = out if code == 0 else err
    assert shown.startswith(
        "usage: python -m jwst_peculiar.survey [--middle] SPLIT_JSON STARS_JSON "
        "DATA_ROOT RESULTS_DIR\n"
        "       python -m jwst_peculiar.survey --check SPLIT_JSON STARS_JSON "
        "DATA_ROOT\n"
    ), shown
    if code:
        want = 3 if argv[:1] == ["--check"] else 4
        assert f"survey: error: expected {want} paths, got {argv}\n" in err


def test_tiles_of_another_size_are_refused(tmp_path: Path) -> None:
    _, split = _survey(tmp_path, SCORES)
    _visit(tmp_path / "train", "t", [(151.0, 2.0)], size=640)
    split["training_data"] = [str(tmp_path / "train" / "t")]
    with pytest.raises(ValueError, match=r"at 640 px across, .* tile_px of 320"):
        kept_tiles(
            split,
            stars_json=tmp_path / "stars.json",
            data=tmp_path / "data",
            results=None,
        )


def test_mosaics_of_another_pixel_scale_are_refused(tmp_path: Path) -> None:
    _, split = _survey(tmp_path, SCORES)
    _visit(tmp_path / "data", "c3", [(151.0, 2.0)], scale=0.03)
    split["calibration"] = [*split["calibration"], "c3"]
    with pytest.raises(ValueError, match=r"Mc3.json is 0.03 arcsec .* is 0.0625,"):
        kept_tiles(
            split,
            stars_json=tmp_path / "stars.json",
            data=tmp_path / "data",
            results=None,
        )


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


def test_keeping_the_first_copy_drops_only_the_later_visit() -> None:
    rows = [
        (150.0, 2.0, "a", "A", 0, 0),
        (150.0, 2.0 + 5 / 3600, "b", "B", 0, 0),
        (150.2, 2.0, "b", "B", 0, 640),
        (150.2, 2.0 + 5 / 3600, "bank", "C", 0, 0),
    ]
    every, tiles = _tiles(rows), _tiles(rows[:3])
    both = overlaps_other_visit(tiles, every, size_arcsec=20.0, size_px=320)
    assert both.tolist() == [True, True, True]
    # The bank isn't scored, so sky shared with it still drops b's second tile.
    first = overlaps_other_visit(
        tiles, every, size_arcsec=20.0, size_px=320, scored=frozenset({"a", "b"})
    )
    assert first.tolist() == [False, True, True]


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


def test_standardise_skips_a_visit_with_under_two_kept_tiles() -> None:
    assert (
        standardise({"v0": {}, "v1": {"a": 2.0}, "v2": {"a": 0.0, "b": 1.0, "c": 3.0}})[
            "v1"
        ]
        == {}
    )


def test_too_few_calibration_tiles_can_name_nothing() -> None:
    split = {"alpha": 0.1, "search": ["s"], "calibration": ["c"], "standardise": True}
    tiles = {"s": {f"s{i}": 0.0 for i in range(514)}}
    short = can_name_one(split, {**tiles, "c": {f"c{i}": 0.0 for i in range(3838)}})
    assert short["smallest_p"] == 1 / 3839
    assert not short["can_name_one"]
    # Two tiles above every calibration tile would both pass, though one can't.
    assert short["tiles_needed_together"] == 2
    enough = can_name_one(split, {**tiles, "c": {f"c{i}": 0.0 for i in range(5140)}})
    assert enough["can_name_one"]
    assert enough["tiles_needed_together"] == 1
