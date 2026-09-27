"""Checks for the tiling, the emptiness cut and the scoring rules, none of which need a GPU.

Run it with `python test_helpers.py`, or with pytest. It needs numpy and astropy, and
the few checks that import torch or pandas skip themselves when those aren't there.
"""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from PIL import Image

import prep_data
from prep_data import (
    MiriSensor,
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
from src.scoring import SCORE_MODES, aggregate_score
from src.seeding import set_seed

SKY, NOISE = 0.0, 1.0


def _tile_args(tile_size=100, stride=100, dry_run=True):
    return SimpleNamespace(
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
    )


def _cut(field, args=None, wht=None, query_dir=Path("unused")):
    """(next index, saved, skipped as empty, skipped for low WHT) for one field."""
    return cut_tiles(args or _tile_args(), "A5", field, wht, None, query_dir, 0)


def _blank_sky(h, w, seed=0):
    return np.random.default_rng(seed).normal(SKY, NOISE, (h, w)).astype(np.float32)


def _sky_with_sources(h, w, tile=100, peak=500.0, seed=0):
    """Blank sky with a compact bright source in the middle of every tile."""
    field = _blank_sky(h, w, seed)
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            field[y + tile // 2 : y + tile // 2 + 5, x + tile // 2 : x + tile // 2 + 5] = peak
    return field


def _wcs_header(ra=150.0, dec=-2.0):
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
    # A tile that is odd all over must not outrank a normal tile holding one strange thing.
    assert aggregate_score(np.full(1000, 2.0, np.float32), "peak_minus_median") == 0.0


def test_aggregate_score_edge_cases() -> None:
    assert aggregate_score(np.array([]), "top1p") == 0.0
    assert aggregate_score(np.array([np.nan, np.inf, 2.0]), "max") == 2.0
    assert aggregate_score(np.array([1.0]), "top1p") == 1.0
    try:
        aggregate_score(np.ones(4), "nonsense")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown score_mode should raise")


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


# --- FITS handling -----------------------------------------------------------
def test_find_image() -> None:
    sci = np.ones((8, 8), np.float32)
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sci, name="SCI")])
    assert find_image(hdul, ("SCI",))[0].shape == (8, 8)
    # SCI must not come back as the weight map just because there is no WHT.
    try:
        find_image(hdul, ("WHT",), fallback=False)
    except ValueError:
        pass
    else:
        raise AssertionError("a missing WHT should raise")
    assert find_image(hdul, ("WHT",))[0] is hdul["SCI"].data

    stacked = fits.HDUList([fits.PrimaryHDU(sci[None])])
    assert find_image(stacked, ("SCI",))[0].shape == (8, 8)
    cube = fits.HDUList([fits.PrimaryHDU(np.ones((3, 8, 8), np.float32))])
    try:
        find_image(cube, ("SCI",))
    except ValueError:
        pass
    else:
        raise AssertionError("a real cube should raise")


def test_make_output_name() -> None:
    assert make_output_name("A5", 100.4, 200.6, 7) == "A5_cx000100_cy000201_i00000007.png"
    with_wcs = make_output_name("A5", 10, 20, 1, wcs_ra_dec=(150.1, -2.3))
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
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(field, header=_wcs_header(), name="SCI"),
                fits.ImageHDU(wht, name="WHT"),
            ]
        ).writeto(tmp / "raw" / f"{stem}.fits")

        prep_data.run(
            [
                "--indir", str(tmp / "raw"), "--out", str(tmp / "out"),
                "--pattern", "*i2d.fits", "--tile_size", "100", "--upscale", "1",
                "--stride", "100", "--min_wht_frac", "0.5", "--wcs_in_name",
            ]
        )  # fmt: skip

        names = sorted(p.name for p in (tmp / "out/query").glob("*.png"))
        assert len(names) == 2, names
        assert all(n.startswith(f"{stem}_cx000050_") for n in names), names
        m = re.search(r"cy(\d+)_ra([\d.]+)_dec(m?[\d.]+)_", names[1])
        ra, dec = WCS(_wcs_header()).all_pix2world(49.5, 149.5, 0)
        assert m.group(1) == "000150"
        assert m.group(2) == f"{float(ra):.6f}"
        assert m.group(3) == f"m{-float(dec):.6f}"


def test_nircam_colour_tiles_match_the_whole_mosaic() -> None:
    # Lupton's mapping works pixel by pixel, so a colour tile made on its own has to
    # match the same crop of a colour image made from the whole mosaic.
    rng = np.random.default_rng(3)
    planes = [np.abs(_sky_with_sources(200, 300, seed=s)) * rng.uniform(0.5, 2) for s in (1, 2, 3)]
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

        prep_data.run(
            [
                "--sensor", "nircam", "--indir", str(tmp / "raw"), "--out", str(tmp / "out"),
                "--tile_size", "100", "--upscale", "1", "--stride", "100",
            ]
        )  # fmt: skip

        whole = make_lupton_rgb(*(np.nan_to_num(p) for p in planes), stretch=0.5, Q=10.0)
        pngs = sorted((tmp / "out/query").glob("A5_*.png"))
        assert len(pngs) == 6
        for png in pngs:
            cx, cy = (int(v) for v in re.search(r"cx(\d+)_cy(\d+)", png.name).groups())
            x, y = cx - 50, cy - 50
            with Image.open(png) as img:
                assert img.mode == "RGB"
                assert np.array_equal(np.asarray(img), whole[y : y + 100, x : x + 100])


def test_help_lists_the_sensor_flags() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        prep_data.parse_args(["--sensor", "nircam", "-h"])
    assert "--rgb_filters" in out.getvalue()


# --- reproducibility and the runners ------------------------------------------
def test_set_seed_reproducible() -> None:
    set_seed(7)
    first = np.random.rand(5).tolist()
    set_seed(7)
    assert np.random.rand(5).tolist() == first
    set_seed(8)
    assert np.random.rand(5).tolist() != first


def test_subset_size() -> None:
    try:
        from run_query_bootstrap import subset_size
    except ImportError:
        print("skipping test_subset_size, torch isn't installed")
        return

    assert subset_size(1000, 0.1, 500, 5000) == 500
    assert subset_size(100_000, 0.1, 500, 5000) == 5000
    assert subset_size(10_000, 0.1, 500, 5000) == 1000
    assert subset_size(3, 0.1, 500, 5000) == 3


def test_summarize_normalize() -> None:
    try:
        import pandas as pd

        from summarize_results import normalize_measurements_df, parse_seed_from_anywhere
    except ImportError:
        print("skipping test_summarize_normalize, pandas or PyYAML isn't installed")
        return

    df = pd.DataFrame(
        {
            "Sample": ["a", "b"],
            "AnomalyScore": [1.0, 2.0],
            "MemoryBankTimeSec": [0.1, 0.1],
            "InferenceTimeSec": [0.2, 0.3],
        }
    )
    out = normalize_measurements_df(df, Path("x.csv"))
    assert list(out.columns) == [
        "Object",
        "Sample",
        "AnomalyScore",
        "MemoryBankTime",
        "InferenceTime",
    ]
    assert (out["Object"] == "query").all()
    assert out["InferenceTime"].tolist() == [0.2, 0.3]
    assert parse_seed_from_anywhere("measurements_seed=3.csv") == 3
    assert parse_seed_from_anywhere("nofile.csv") is None


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
