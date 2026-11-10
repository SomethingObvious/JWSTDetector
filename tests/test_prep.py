"""Checks for tiling mosaics, the stretches, the sensors and the photometry."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from PIL import Image
import pytest

from jwst_peculiar import photometry, prep
from jwst_peculiar.prep import (
    MiriSensor,
    Mosaic,
    MosaicStats,
    asinh_to_uint8,
    cut_tiles,
    find_image,
    has_signal,
    infer_tile_id_from_filename,
    make_output_name,
    measure_mosaic,
    tile_starts,
)
from jwst_peculiar.scoring import SCORE_MODES, aggregate_score
from jwst_peculiar.seeding import set_seed

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwst_peculiar.utils import FitsHDUList, FitsHeader

SKY, NOISE = 0.0, 1.0


def _tile_args(
    tile_size: int = 100, stride: int = 100, *, dry_run: bool = True
) -> argparse.Namespace:
    return argparse.Namespace(
        tile_size=tile_size,
        upscale=1,
        stride=stride,
        min_wht_frac=0.0,
        p_lo=1.0,
        p_hi=99.8,
        asinh_q=10.0,
        source_sigma=5.0,
        min_source_frac=2e-4,
        stretch_scope="mosaic",
        dry_run=dry_run,
        wcs_in_name=False,
        save_cutouts=False,
        bunit=None,
    )


def _cut(
    field: npt.NDArray[np.float32],
    args: argparse.Namespace | None = None,
    *,
    wht: npt.NDArray[np.float32] | None = None,
    query_dir: Path = Path("unused"),
) -> tuple[int, int, int, int]:
    """Cut one field as mosaic A5, with no WCS.

    Returns:
        The next index and the saved, empty and low-WHT counts.
    """
    return cut_tiles(
        args or _tile_args(), Mosaic("A5", field, wht=wht), query_dir, global_idx=0
    )


def _blank_sky(h: int, w: int, seed: int = 0) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed).normal(SKY, NOISE, (h, w)).astype(np.float32)


def _sky_with_sources(
    h: int, w: int, *, tile: int = 100, peak: float = 500.0, seed: int = 0
) -> npt.NDArray[np.float32]:
    """Make blank sky with a compact bright source in the middle of every tile.

    Returns:
        The field as float32.
    """
    field = _blank_sky(h, w, seed)
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            field[
                y + tile // 2 : y + tile // 2 + 5, x + tile // 2 : x + tile // 2 + 5
            ] = peak
    return field


def _wcs_header(ra: float = 150.0, dec: float = -2.0) -> FitsHeader:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [100.0, 100.0]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-3e-5, 3e-5]
    return w.to_header()


# --- the emptiness cut -------------------------------------------------------
def test_empty_cut_keeps_sources_drops_sky() -> None:
    # Measured on a PNG with a per-tile stretch, blank sky comes up to full brightness,
    # so a cut there keeps empty tiles and drops the ones with a bright source. Raw flux
    # against the mosaic noise gets it the right way round.
    field = _sky_with_sources(200, 200)
    stats = measure_mosaic(field, 1.0, 99.8)

    sky_tile = field[0:100, 0:100].copy()
    sky_tile[50:55, 50:55] = SKY
    assert has_signal(field[0:100, 0:100], stats) is True
    assert has_signal(sky_tile, stats) is False


def test_empty_cut_ignores_flux_units() -> None:
    # The same sky in MJy/sr or in counts must get the same verdict.
    field = _sky_with_sources(200, 200)
    for scale in (1e-6, 1.0, 1e6):
        stats = measure_mosaic(field * scale, 1.0, 99.8)
        assert has_signal(field[0:100, 0:100] * scale, stats) is True
        assert has_signal(_blank_sky(100, 100, seed=9) * scale, stats) is False


def test_empty_cut_degenerate_inputs() -> None:
    flat = MosaicStats(sky=0.0, sigma=0.0, lo=0.0, hi=1.0)
    assert has_signal(np.zeros((16, 16), np.float32), flat) is False
    assert has_signal(np.ones((16, 16), np.float32), flat) is True
    stats = measure_mosaic(_blank_sky(64, 64), 1.0, 99.8)
    assert has_signal(np.full((16, 16), np.nan, np.float32), stats) is False


def test_measure_mosaic_ignores_the_sources() -> None:
    stats = measure_mosaic(_sky_with_sources(400, 400), 1.0, 99.8)
    assert abs(stats.sky - SKY) < 0.1, stats.sky
    assert abs(stats.sigma - NOISE) < 0.15, stats.sigma
    assert stats.hi > stats.lo


def test_zero_filled_border_does_not_hide_the_noise() -> None:
    # A rotated mosaic can be mostly zero fill. Counted as data, that fill would make
    # sky and sigma exactly 0, and every tile of blank sky would pass the cut.
    field = np.random.default_rng(0).normal(5.0, 1.0, (1000, 1000)).astype(np.float32)
    field[:, :600] = 0.0
    stats = measure_mosaic(field, 1.0, 99.8)
    assert abs(stats.sky - 5.0) < 0.05, stats.sky
    assert abs(stats.sigma - 1.0) < 0.05, stats.sigma

    blank = field[0:100, 700:800]
    source = blank.copy()
    source[40:45, 40:45] = 50.0
    assert has_signal(blank, stats) is False
    assert has_signal(source, stats) is True


def test_sliver_of_coverage_is_dropped() -> None:
    # One noisy pixel above 5 sigma is 1% of the finite pixels in a tile that is 99%
    # NaN, far over the 2e-4 cut, so the fraction has to be of the whole tile.
    stats = measure_mosaic(_blank_sky(1000, 1000), 1.0, 99.8)
    sliver = np.full((100, 100), np.nan, np.float32)
    sliver[:, 0] = np.random.default_rng(1).normal(SKY, NOISE, 100)
    sliver[3, 0] = 6.0
    assert has_signal(sliver, stats) is False


# --- the stretch -------------------------------------------------------------
def test_asinh_stretch() -> None:
    img = np.linspace(0.0, 1000.0, 256, dtype=np.float32).reshape(16, 16)
    out = asinh_to_uint8(img, 0.0, 1000.0)
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255
    assert np.all(np.diff(out.reshape(-1).astype(np.int32)) >= 0)


def test_holes_take_the_sky_level() -> None:
    # A NaN hole should look like sky, not like a black shape for the detector to find.
    img = np.full((4, 4), 30.0, np.float32)
    img[0, 0], img[0, 1], img[0, 2] = np.nan, np.inf, -np.inf
    out = asinh_to_uint8(img, 0.0, 100.0, fill=30.0)
    assert np.all(out == out[3, 3]), out
    assert out[3, 3] == asinh_to_uint8(np.array([[30.0]], np.float32), 0.0, 100.0)[0, 0]


def test_mosaic_scope_keeps_tiles_comparable() -> None:
    # One shared stretch keeps a faint tile fainter than a bright one, where per-tile
    # percentiles would push both to the same range.
    field = _sky_with_sources(200, 200)
    stats = measure_mosaic(field, 1.0, 99.8)
    faint = asinh_to_uint8(np.full((64, 64), stats.lo, np.float32), stats.lo, stats.hi)
    bright = asinh_to_uint8(np.full((64, 64), stats.hi, np.float32), stats.lo, stats.hi)
    assert faint.mean() == 0
    assert bright.mean() == 255


# --- scoring -----------------------------------------------------------------
def test_aggregate_score_modes() -> None:
    flat = np.full(1000, 0.5, dtype=np.float32)
    needle = flat.copy()
    needle[123] = 5.0

    for mode in SCORE_MODES:
        assert aggregate_score(needle, mode) > aggregate_score(flat, mode), mode

    assert aggregate_score(needle, "max") == 5.0
    assert aggregate_score(needle, "peak_minus_median") == 4.5
    # A tile that is odd all over must not outrank a normal tile holding one
    # strange thing.
    assert aggregate_score(np.full(1000, 2.0, np.float32), "peak_minus_median") == 0.0


def test_aggregate_score_edge_cases() -> None:
    assert np.isnan(aggregate_score(np.array([]), "top1p"))
    assert aggregate_score(np.array([np.nan, np.inf, 2.0]), "max") == 2.0
    assert aggregate_score(np.array([1.0]), "top1p") == 1.0
    with pytest.raises(ValueError, match="Unknown score_mode"):
        aggregate_score(np.ones(4), "nonsense")


# --- tiling ------------------------------------------------------------------
def test_tile_starts() -> None:
    assert tile_starts(300, 100, 100) == [0, 100, 200]
    # The leftover 50 px get a crop flush with the edge rather than being skipped.
    assert tile_starts(250, 100, 100) == [0, 100, 150]
    assert tile_starts(1000, 672, 644) == [0, 328]
    assert tile_starts(100, 100, 50) == [0]
    assert tile_starts(99, 100, 50) == []


def test_tile_counts_cover_the_edges() -> None:
    # Every crop is a full 100 px, and a mosaic smaller than one crop gives none.
    for size, expected in ((250, 9), (300, 9), (99, 0)):
        _, saved, _, _ = _cut(_sky_with_sources(size, size))
        assert saved == expected, (size, saved)


def test_source_in_the_far_corner_is_tiled() -> None:
    # With 672 px crops every 644 px, the stride alone fits one crop into 1000 px and
    # never reaches the last 328 px of either axis.
    field = _blank_sky(1000, 1000)
    field[900:960, 900:960] = 500.0
    _, saved, skipped_empty, _ = _cut(field, _tile_args(tile_size=672, stride=644))
    assert (saved, skipped_empty) == (1, 3)


def test_blank_mosaic_yields_nothing() -> None:
    _, saved, skipped_empty, _ = _cut(_blank_sky(300, 300))
    assert (saved, skipped_empty) == (0, 9)


def test_dry_run_counts_match_a_real_run() -> None:
    field = _sky_with_sources(300, 300)
    with tempfile.TemporaryDirectory() as real_dir:
        real = _cut(field, _tile_args(dry_run=False), query_dir=Path(real_dir))
        dry = _cut(field, _tile_args(dry_run=True))
        assert real == dry == (9, 9, 0, 0), (real, dry)
        pngs = sorted(Path(real_dir).glob("*.png"))
        assert len(pngs) == 9
        with Image.open(pngs[0]) as img:
            assert (img.mode, img.size) == ("L", (100, 100))


def test_wht_gate() -> None:
    args = _tile_args()
    args.min_wht_frac = 0.5
    no_coverage = np.zeros((200, 200), np.float32)
    assert _cut(_sky_with_sources(200, 200), args, wht=no_coverage)[1:] == (0, 0, 4)
    half = no_coverage.copy()
    half[:, :100] = 1.0
    assert _cut(_sky_with_sources(200, 200), args, wht=half)[1:] == (2, 0, 2)

    # On the SMACS 0723 MIRI mosaic a tile missing half a percent of its pixels still
    # ranked second, so by default a notch of 1% has to cost the tile.
    args.min_wht_frac = prep.parse_args(["--indir", "x"]).min_wht_frac
    notched = np.ones((200, 200), np.float32)
    notched[:10, :10] = 0.0
    assert _cut(_sky_with_sources(200, 200), args, wht=notched)[1:] == (3, 0, 1)


# --- FITS handling -----------------------------------------------------------
def test_find_image() -> None:
    sci = np.ones((8, 8), np.float32)
    hdul: FitsHDUList = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(sci, name="SCI"),
    ])
    assert find_image(hdul, ("SCI",))[0].shape == (8, 8)
    # SCI must not come back as the weight map just because there is no WHT.
    with pytest.raises(ValueError, match="no 2D image called WHT"):
        find_image(hdul, ("WHT",), fallback=False)
    assert find_image(hdul, ("WHT",))[0] is hdul["SCI"].data

    stacked = fits.HDUList([fits.PrimaryHDU(sci[None])])
    assert find_image(stacked, ("SCI",))[0].shape == (8, 8)
    cube = fits.HDUList([fits.PrimaryHDU(np.ones((3, 8, 8), np.float32))])
    with pytest.raises(ValueError, match="no 2D image called SCI"):
        find_image(cube, ("SCI",))


def test_make_output_name() -> None:
    assert (
        make_output_name("A5", 100.4, 200.6, global_idx=7)
        == "A5_cx000100_cy000201_i00000007.png"
    )
    with_wcs = make_output_name("A5", 10, 20, global_idx=1, wcs_ra_dec=(150.1, -2.3))
    assert with_wcs == "A5_cx000010_cy000020_ra150.100000_decm2.300000_i00000001.png"


def test_sci_to_wht_path() -> None:
    assert MiriSensor.sci_to_wht_path("/data/a5_sci.fits") == "/data/a5_wht.fits"
    assert MiriSensor.sci_to_wht_path("/data/A5_SCI.FITS") == "/data/A5_wht.fits"
    assert MiriSensor.sci_to_wht_path("/data/x_i2d.fits") == "/data/x_i2d.fits"


def test_infer_tile_id() -> None:
    assert infer_tile_id_from_filename("A5_sci.fits") == "A5"
    assert infer_tile_id_from_filename("mosaic_B12_v1.fits") == "B12"
    assert infer_tile_id_from_filename("nothing_here.fits") is None


def test_miri_i2d_end_to_end() -> None:
    # A JWST i2d holds SCI and WHT in one file and has no tile id in its name.
    stem = "jw01234-o001_t001_miri_f770w_i2d"
    field = _sky_with_sources(200, 200)
    wht = np.ones_like(field)
    wht[:, 100:] = 0.0
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "raw").mkdir()
        fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(field, header=_wcs_header(), name="SCI"),
            fits.ImageHDU(wht, name="WHT"),
        ]).writeto(tmp / "raw" / f"{stem}.fits")

        prep.run(
            [
                "--sensor", "miri", "--indir", str(tmp / "raw"),
                "--out", str(tmp / "out"), "--pattern", "*i2d.fits",
                "--tile_size", "100", "--upscale", "1", "--stride", "100",
                "--min_wht_frac", "0.5", "--wcs_in_name",
            ]
        )  # fmt: skip

        names = sorted(p.name for p in (tmp / "out/query").glob("*.png"))
        assert len(names) == 2, names
        assert all(n.startswith(f"{stem}_cx000050_") for n in names), names
        m = re.search(r"cy(\d+)_ra([\d.]+)_dec(m?[\d.]+)_", names[1])
        assert m is not None, names[1]
        ra, dec = WCS(_wcs_header()).all_pix2world(49.5, 149.5, 0)
        assert m.group(1) == "000150"
        assert m.group(2) == f"{float(ra):.6f}"
        assert m.group(3) == f"m{-float(dec):.6f}"


def _tile_origin(name: str) -> tuple[int, int]:
    # The corner of a 100 px tile, from the centre pixel in its name.
    m = re.search(r"cx(\d+)_cy(\d+)", name)
    assert m is not None, name
    cx, cy = (int(v) for v in m.groups())
    return cx - 50, cy - 50


def test_nircam_colour_tiles_match_the_whole_mosaic() -> None:
    # Lupton's mapping works pixel by pixel, so a colour tile made on its own has to
    # match the same crop of a colour image made from the whole mosaic, top row first.
    rng = np.random.default_rng(3)
    planes = [
        np.abs(_sky_with_sources(200, 300, seed=s)) * rng.uniform(0.5, 2)
        for s in (1, 2, 3)
    ]
    planes[1][10, 10] = np.nan
    filters = ["f444w", "f277w", "f150w"]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "raw").mkdir()
        for filt, plane in zip(filters, planes, strict=True):
            name = f"mosaic_nircam_{filt}_COSMOS-Web_60mas_A5_v1.0_i2d.fits"
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(plane, name="SCI")]).writeto(
                tmp / "raw" / name
            )

        prep.run(
            [
                "--sensor", "nircam", "--indir", str(tmp / "raw"),
                "--out", str(tmp / "out"), "--tile_size", "100", "--upscale", "1",
                "--stride", "100", "--min_wht_frac", "0",
            ]
        )  # fmt: skip

        levels = [measure_mosaic(p, 1.0, 99.8) for p in planes]
        whole = prep.lupton(argparse.Namespace(stretch=0.05, Q=10.0), planes, levels)
        pngs = sorted((tmp / "out/query").glob("A5_*.png"))
        assert len(pngs) == 6
        for png in pngs:
            x, y = _tile_origin(png.name)
            with Image.open(png) as img:
                assert img.mode == "RGB"
                assert np.array_equal(
                    np.asarray(img), whole[y : y + 100, x : x + 100][::-1]
                )


_FILTERS = ("f444w", "f277w", "f150w")


def _three_visits(raw: Path) -> dict[str, list[npt.NDArray[np.float32]]]:
    # Three colour visits, the last with a bright star that takes 2% of its pixels.
    raw.mkdir()
    planes: dict[str, list[npt.NDArray[np.float32]]] = {}
    for i, tile in enumerate(("A1", "A2", "A3")):
        planes[tile] = [
            np.abs(_sky_with_sources(100, 200, peak=20.0 + 5 * i, seed=10 * i + s))
            for s in (1, 2, 3)
        ]
        for filt, plane in zip(_FILTERS, planes[tile], strict=True):
            if tile == "A3":
                plane[20:40, 20:40] += 400.0
            name = f"mosaic_nircam_{filt}_COSMOS-Web_60mas_{tile}_v1.0_i2d.fits"
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(plane, name="SCI"),
            ]).writeto(raw / name)
    return planes


def _nircam_args(raw: Path) -> list[str]:
    return [
        "--sensor", "nircam", "--indir", str(raw), "--tile_size", "100",
        "--upscale", "1", "--stride", "100", "--min_wht_frac", "0",
    ]  # fmt: skip


def test_levels_give_every_visit_the_survey_stretch() -> None:
    # The star pushes A3's percentile range up, and with --levels A3 gets the median
    # range of the three visits while keeping its own sky.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        planes = _three_visits(tmp / "raw")
        common = _nircam_args(tmp / "raw")

        prep.run([*common, "--out", str(tmp / "own")])
        prep.levels_main([str(tmp / "own"), "--out", str(tmp / "levels.json")])
        prep.run([
            *common,
            "--out",
            str(tmp / "survey"),
            "--levels",
            str(tmp / "levels.json"),
        ])

        own = {t: [measure_mosaic(p, 1.0, 99.8) for p in planes[t]] for t in planes}
        written = json.loads((tmp / "levels.json").read_text(encoding="utf-8"))
        assert written["mosaics"] == {"F444W": 3, "F277W": 3, "F150W": 3}
        for k, filt in enumerate(_FILTERS):
            ranges = sorted(own[t][k].hi - own[t][k].sky for t in own)
            assert own["A3"][k].hi - own["A3"][k].sky == ranges[2]
            assert written["range"][filt.upper()] == pytest.approx(ranges[1])

        record = json.loads(
            (tmp / "survey/mosaics/A3.json").read_text(encoding="utf-8")
        )
        shared = [
            replace(v, hi=v.sky + written["range"][f.upper()])
            for v, f in zip(own["A3"], _FILTERS, strict=True)
        ]
        assert record["render"]["levels"] == [
            {"sky": v.sky, "hi": v.hi} for v in shared
        ]
        whole = prep.lupton(
            argparse.Namespace(stretch=0.05, Q=10.0), planes["A3"], shared
        )
        pngs = sorted((tmp / "survey/query").glob("A3_*.png"))
        assert len(pngs) == 2
        for png in pngs:
            x, y = _tile_origin(png.name)
            with Image.open(png) as img:
                assert np.array_equal(
                    np.asarray(img), whole[y : y + 100, x : x + 100][::-1]
                )


def test_levels_refuse_a_per_tile_stretch_and_a_missing_filter() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _three_visits(tmp / "raw")
        levels = tmp / "levels.json"
        levels.write_text(json.dumps({"range": {"F444W": 1.0}}), encoding="utf-8")
        common = [*_nircam_args(tmp / "raw"), "--levels", str(levels)]
        with pytest.raises(SystemExit, match="needs --stretch_scope mosaic"):
            prep.run([*common, "--out", str(tmp / "a"), "--stretch_scope", "tile"])
        with pytest.raises(SystemExit, match="has no range for F277W"):
            prep.run([*common, "--out", str(tmp / "b")])


def test_cutout_pixels_and_wcs_match_the_mosaic() -> None:
    rng = np.random.default_rng(5)
    plane = rng.normal(size=(50, 60)).astype(np.float32)
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [150.0, 2.0]
    w.wcs.crpix = [20.0, 30.0]
    w.wcs.cd = [[-1e-5, 2e-6], [2e-6, 1e-5]]
    window = (slice(10, 30), slice(5, 25))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "c.fits"
        prep.write_cutout(path, Mosaic("c", plane, wcs=w, bunit="MJy/sr"), window)
        hdul: FitsHDUList
        with fits.open(path) as hdul:
            assert np.array_equal(hdul[0].data, plane[window])
            assert hdul[0].header["BUNIT"] == "MJy/sr"
            got = WCS(hdul[0].header).pixel_to_world_values(3, 4)
    assert np.allclose(got, w.pixel_to_world_values(5 + 3, 10 + 4))


def test_help_lists_the_sensor_flags() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        prep.parse_args(["--sensor", "nircam", "-h"])
    assert "--rgb_filters" in out.getvalue()


# --- reproducibility and the runners ------------------------------------------
def _global_draw() -> list[float]:
    draws: list[float] = np.random.rand(5).tolist()  # noqa: NPY002 set_seed pins this global stream
    return draws


def test_set_seed_reproducible() -> None:
    set_seed(7)
    first = _global_draw()
    set_seed(7)
    assert _global_draw() == first
    set_seed(8)
    assert _global_draw() != first


def test_cutouts_render_back_to_the_same_tiles() -> None:
    # The benchmark adds sources to the cutouts and renders them again, so a cutout
    # with nothing added has to come back as exactly the PNG prep saved.
    rng = np.random.default_rng(5)
    planes = [
        np.abs(_sky_with_sources(200, 300, seed=s)) * rng.uniform(0.5, 2)
        for s in (1, 2, 3)
    ]
    planes[2][40:45, 40:45] = np.nan
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "raw").mkdir()
        for filt, plane in zip(("f444w", "f277w", "f150w"), planes, strict=True):
            name = f"mosaic_nircam_{filt}_COSMOS-Web_60mas_A5_v1.0_i2d.fits"
            header = _wcs_header()
            header["BUNIT"] = "MJy/sr"
            header["TELESCOP"] = "JWST"
            # One filter has a strip with a third of the exposure, which the PNG
            # renders with its noise scaled down and the cutout has to say so.
            wht = np.full(plane.shape, 3.0)
            if filt == "f277w":
                wht[:, 220:] = 1.0
            fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(plane, header, name="SCI"),
                fits.ImageHDU(wht, name="WHT"),
            ]).writeto(tmp / "raw" / name)
        prep.run(
            [
                "--sensor", "nircam", "--indir", str(tmp / "raw"),
                "--out", str(tmp / "out"), "--tile_size", "100", "--upscale", "1",
                "--stride", "100", "--save_cutouts",
            ]
        )  # fmt: skip
        meta = json.loads((tmp / "out/mosaics/A5.json").read_text())
        assert meta["render"]["kind"] == "lupton"
        assert [b["filter"] for b in meta["planes"]] == ["F444W", "F277W", "F150W"]
        assert [round(b["micron"], 2) for b in meta["planes"]] == [4.44, 2.77, 1.5]
        pngs = sorted((tmp / "out/query").glob("*.png"))
        assert pngs
        shallow = 0
        hdul: FitsHDUList
        for png in pngs:
            with fits.open(
                tmp / "out/cutouts" / f"{png.stem}.fits", memmap=False
            ) as hdul:
                cutout, noise = hdul[0].data, hdul["NOISEEQ"].data
            shallow += int((noise[1] == round(255 / 3**0.5)).any())
            with Image.open(png) as img:
                rendered = prep.render_cutout(cutout, meta["render"], noise)
                assert np.array_equal(rendered, np.asarray(img))
        assert shallow == 2


def test_noise_factor_evens_out_the_sky() -> None:
    rng = np.random.default_rng(0)
    wht = np.full((200, 200), 4.0)
    wht[:, 100:] = 1.0
    sky = 10 + rng.normal(size=wht.shape) / np.sqrt(wht)
    f = prep.noise_factor(wht, typical=4.0)
    assert f[0, 0] == 255
    assert f[0, 199] == 128
    flat = prep.equalise(sky, 10.0, f)
    assert abs(flat[:, :100].std() / flat[:, 100:].std() - 1) < 0.05
    # A saturated core has no weight, so it keeps its flux.
    assert prep.noise_factor(np.zeros((1, 1)), 4.0)[0, 0] == 255


def test_fits_sensor_reads_any_single_image() -> None:
    # An HST-style drizzled image: the image in the primary HDU, counts per second
    # with PHOTFNU, and no weight map, which the generic sensor has to take as it is.
    field = _sky_with_sources(200, 200)
    header = _wcs_header()
    header.update({"TELESCOP": "HST", "FILTER": "F814W", "BUNIT": "ELECTRONS/S"})
    header.update({"PHOTFNU": 7.0e-8, "PHOTPLAM": 8045.0})
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        path = tmp / "hst_10092_acs_wfc_f814w_drz.fits"
        fits.PrimaryHDU(field, header).writeto(path)
        prep.run(
            [
                "--files", str(path), "--out", str(tmp / "out"), "--tile_size", "100",
                "--upscale", "1", "--stride", "100", "--wcs_in_name", "--save_cutouts",
            ]
        )  # fmt: skip
        assert len(list((tmp / "out/query").glob("*.png"))) == 4
        meta = json.loads(next((tmp / "out/mosaics").glob("*.json")).read_text())
        band = meta["planes"][0]
        assert band["filter"] == "F814W"
        assert band["telescope"] == "HST"
        assert band["jy_per_unit"] == 7.0e-8
        assert abs(band["micron"] - 0.8045) < 1e-9
        # HST's 2.4 m mirror at 0.8 microns.
        assert abs(band["psf_fwhm_arcsec"] - 1.03 * 0.8045e-6 / 2.4 * 206264.806) < 1e-9
        assert meta["render"]["kind"] == "asinh"


def test_fits_sensor_skips_weight_maps_and_clashing_names() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for name in (
            "M51_F814W_drz.fits",
            "M51_F814W_drz_wht.fits",
            "M51_F555W_drz.fits",
        ):
            (tmp / name).touch()
        args = prep.parse_args([
            "--indir",
            str(tmp),
            "--out",
            str(tmp / "out"),
            "--wht_pattern",
            "{stem}_wht.fits",
        ])
        specs = list(prep.SENSORS["fits"].iter_mosaic_specs(args))
        assert [s.tile_id for s in specs] == ["M51_F555W_drz", "M51_F814W_drz"]

        args.tile_regex = r"^(M51)_"
        with pytest.raises(SystemExit, match="would both be mosaic M51"):
            list(prep.SENSORS["fits"].iter_mosaic_specs(args))

        # A version number's dot isn't where the name ends.
        for name in ("x_v1.0_drz.fits", "x_v1.0_drz_wht.fits"):
            (tmp / "v" / name).parent.mkdir(exist_ok=True)
            (tmp / "v" / name).touch()
        args.indir, args.tile_regex = str(tmp / "v"), None
        specs = list(prep.SENSORS["fits"].iter_mosaic_specs(args))
        assert [s.tile_id for s in specs] == ["x_v1.0_drz"]


def test_a_mistyped_pattern_leaves_nothing_behind() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with pytest.raises(FileNotFoundError):
            prep.run([
                "--indir",
                str(tmp),
                "--pattern",
                "*.fts",
                "--out",
                str(tmp / "o"),
            ])
        assert not (tmp / "o").exists()
        with pytest.raises(SystemExit, match="--stride has to be at least 1"):
            prep.run(["--indir", str(tmp), "--out", str(tmp / "o"), "--stride", "0"])


def test_photometry_units() -> None:
    scale = 0.06
    pixel_sr = (scale / photometry.ARCSEC_PER_RAD) ** 2
    assert photometry.jy_per_unit([{"BUNIT": "MJy/sr"}], scale) == 1e6 * pixel_sr
    assert photometry.jy_per_unit([{"BUNIT": "nanomaggy"}], scale) == 3.631e-6
    assert photometry.jy_per_unit([{"BUNIT": "uJy"}], scale) == 1e-6
    assert photometry.jy_per_unit([{"BUNIT": "mJy/sr"}], scale) == 1e-3 * pixel_sr
    assert photometry.jy_per_unit([{"BUNIT": "mJy"}], scale) == 1e-3
    assert photometry.jy_per_unit([{"BUNIT": "MJY/SR"}], scale) == 1e6 * pixel_sr
    from_zero_point = photometry.jy_per_unit([{"BUNIT": "ADU", "MAGZERO": 30.0}], scale)
    assert from_zero_point is not None
    assert abs(from_zero_point - 3.631e-9) < 1e-20
    assert photometry.jy_per_unit([{"BUNIT": "ADU"}], scale) is None
    assert (
        photometry.filter_name([{"FILTER1": "CLEAR1L", "FILTER2": "F606W"}]) == "F606W"
    )
    # F162M sits in NIRCam's pupil wheel, behind the F150W2 that FILTER names.
    assert photometry.filter_name([{"FILTER": "F150W2", "PUPIL": "F162M"}]) == "F162M"
    assert photometry.filter_name([{"FILTER": "F444W", "PUPIL": "CLEAR"}]) == "F444W"
    assert photometry.wavelength_um([{"PHOTPLAM": 8045.0}], "F814W") == 0.8045
    assert photometry.wavelength_um([{"WAVELEN": 3.368}], "W1") == 3.368
    assert photometry.diffraction_fwhm_arcsec(3.4, "WISE") is None
    assert photometry.wavelength_um([{"TELESCOP": "JWST"}], "F1000W") == 10.0
    assert photometry.wavelength_um([{"TELESCOP": "HST"}], "F814W") is None
    assert photometry.diffraction_fwhm_arcsec(2.0, "SUBARU") is None
