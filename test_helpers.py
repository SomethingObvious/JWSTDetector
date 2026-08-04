"""Framework-free checks for the CPU-side helpers.

Run with ``python test_helpers.py``. No pytest, no fixtures. These cover the parts
that decide what gets tiled and how a tile gets scored, none of which need a GPU,
torch, or a downloaded backbone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prep_data import (
    MiriSensor,
    MosaicSpec,
    MosaicStats,
    asinh_to_uint8,
    has_signal,
    infer_tile_id_from_filename,
    make_output_name,
    measure_mosaic,
)
from src.scoring import SCORE_MODES, aggregate_score
from src.seeding import set_seed

SKY, NOISE = 0.0, 1.0


def _tile_args(tile_size=100, stride=100, dry_run=False):
    return SimpleNamespace(
        tile_size=tile_size,
        upscale=1,
        stride=stride,
        wcs_in_name=False,
        min_wht_frac=0.0,
        p_lo=1.0,
        p_hi=99.8,
        asinh_q=10.0,
        source_sigma=5.0,
        min_source_frac=2e-4,
        stretch_scope="mosaic",
        dry_run=dry_run,
    )


def _blank_sky(h, w, seed=0):
    return np.random.default_rng(seed).normal(SKY, NOISE, (h, w)).astype(np.float32)


def _sky_with_sources(h, w, tile=100, peak=500.0, seed=0):
    """Blank sky with one compact bright source planted in every tile."""
    field = _blank_sky(h, w, seed)
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            field[y + tile // 2 : y + tile // 2 + 5, x + tile // 2 : x + tile // 2 + 5] = peak
    return field


# --- the emptiness cut -------------------------------------------------------
def test_empty_cut_keeps_sources_drops_sky() -> None:
    """The regression that motivated the rewrite.

    The old cut measured brightness on the stretched PNG. A per-tile stretch had
    already normalised blank sky up to full brightness, so the cut kept empty tiles
    and threw away the tiles holding a bright source, precisely backwards. Measuring
    raw flux against the mosaic's own noise gets the sign right.
    """
    field = _sky_with_sources(200, 200)
    stats = measure_mosaic(field, 1.0, 99.8)

    sky_tile = field[0:100, 0:100].copy()
    sky_tile[50:55, 50:55] = SKY  # remove the planted source
    source_tile = field[0:100, 0:100]

    assert has_signal(source_tile, stats) is True
    assert has_signal(sky_tile, stats) is False


def test_empty_cut_ignores_flux_units() -> None:
    # Same sky scaled by 1e6, as MJy/sr vs counts would be. The verdict must not move.
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


def test_measure_mosaic_is_robust_to_sources() -> None:
    # MAD must track the noise, not the bright pixels sitting on top of it.
    stats = measure_mosaic(_sky_with_sources(400, 400), 1.0, 99.8)
    assert abs(stats.sky - SKY) < 0.1, stats.sky
    assert abs(stats.sigma - NOISE) < 0.15, stats.sigma
    assert stats.hi > stats.lo


# --- the stretch -------------------------------------------------------------
def test_asinh_stretch() -> None:
    img = np.linspace(0.0, 1000.0, 256, dtype=np.float32).reshape(16, 16)
    out = asinh_to_uint8(img, 0.0, 1000.0)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    assert np.all(np.diff(out.reshape(-1).astype(np.int32)) >= 0)

    dirty = img.copy()
    dirty[0, 0], dirty[0, 1] = np.nan, np.inf
    assert asinh_to_uint8(dirty, 0.0, 1000.0).dtype == np.uint8


def test_mosaic_scope_keeps_tiles_comparable() -> None:
    # One shared stretch: a faint tile stays visibly fainter than a bright one.
    # Per-tile percentiles would drive both to the same range and lose the contrast.
    field = _sky_with_sources(200, 200)
    stats = measure_mosaic(field, 1.0, 99.8)
    faint = asinh_to_uint8(np.full((64, 64), stats.lo, np.float32), stats.lo, stats.hi)
    bright = asinh_to_uint8(np.full((64, 64), stats.hi, np.float32), stats.lo, stats.hi)
    assert faint.mean() < bright.mean()


# --- scoring -----------------------------------------------------------------
def test_aggregate_score_modes() -> None:
    flat = np.full(1000, 0.5, dtype=np.float32)
    needle = flat.copy()
    needle[123] = 5.0

    for mode in SCORE_MODES:
        assert aggregate_score(needle, mode) > aggregate_score(flat, mode), mode

    assert aggregate_score(needle, "max") == 5.0
    assert aggregate_score(needle, "peak_minus_median") == 4.5
    # A uniformly odd tile must not out-rank a normal tile holding one strange thing.
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
        raise AssertionError("unknown score_mode should raise")


# --- naming and tiling -------------------------------------------------------
def test_make_output_name() -> None:
    assert make_output_name("A5", 100.4, 200.6, 7) == "A5_cx000100_cy000201_i00000007.png"
    with_wcs = make_output_name("A5", 10, 20, 1, wcs_ra_dec=(150.1, -2.3))
    assert "ra150.100000" in with_wcs
    assert "decm2.300000" in with_wcs  # negatives use 'm' so the name stays filesystem-safe


def test_sci_to_wht_path() -> None:
    assert MiriSensor.sci_to_wht_path("/data/a5_sci.fits").endswith("a5_wht.fits")
    assert MiriSensor.sci_to_wht_path("/data/A5_SCI.FITS").endswith("wht.fits")


def test_infer_tile_id() -> None:
    assert infer_tile_id_from_filename("A5_sci.fits") == "A5"
    assert infer_tile_id_from_filename("mosaic_B12_v1.fits") == "B12"
    assert infer_tile_id_from_filename("nothing_here.fits") is None


def test_emit_tiles_tile_count() -> None:
    # No partial edge tiles: a tile starts only where the full crop fits.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    for size, expected in ((250, 4), (300, 9)):  # 100 px crop, 100 px stride
        _, saved, _, _ = sensor._emit_tiles(
            _tile_args(dry_run=True), spec, "unused", 0, _sky_with_sources(size, size), None, None
        )
        assert saved == expected, (size, saved)


def test_emit_tiles_drops_a_blank_mosaic() -> None:
    # A field with nothing in it should yield nothing, which the old cut got wrong.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    _, saved, skipped_empty, _ = sensor._emit_tiles(
        _tile_args(dry_run=True), spec, "unused", 0, _blank_sky(300, 300), None, None
    )
    assert saved == 0, saved
    assert skipped_empty == 9, skipped_empty


def test_emit_tiles_dry_run_parity() -> None:
    # --dry-run must not change counts or the global_idx sequence; it only skips
    # the PNG writes. Prove it by running both modes over the same field.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    sci = _sky_with_sources(300, 300)

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as dry_dir:
        real = sensor._emit_tiles(_tile_args(dry_run=False), spec, real_dir, 0, sci, None, None)
        dry = sensor._emit_tiles(_tile_args(dry_run=True), spec, dry_dir, 0, sci, None, None)
        assert real == dry, (real, dry)  # (global_idx, saved, skipped_empty, skipped_wht)
        assert len(list(Path(real_dir).glob("*.png"))) == real[1]
        assert len(list(Path(dry_dir).glob("*.png"))) == 0


def test_emit_tiles_wht_gate() -> None:
    # An all-zero weight map means no coverage, so every tile goes.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    args = _tile_args(dry_run=True)
    args.min_wht_frac = 0.5
    _, saved, _, skipped_wht = sensor._emit_tiles(
        args, spec, "unused", 0, _sky_with_sources(200, 200), None, np.zeros((200, 200), np.float32)
    )
    assert (saved, skipped_wht) == (0, 4), (saved, skipped_wht)


# --- reproducibility ---------------------------------------------------------
def test_set_seed_reproducible() -> None:
    set_seed(7)
    first = np.random.rand(5).tolist()
    set_seed(7)
    assert np.random.rand(5).tolist() == first
    set_seed(8)
    assert np.random.rand(5).tolist() != first


def test_reference_selection_reproducible() -> None:
    # The runners draw their reference subset from a local default_rng(seed).
    def pick(seed):
        return sorted(np.random.default_rng(seed).choice(100, size=10, replace=False).tolist())

    assert pick(0) == pick(0)
    assert pick(0) != pick(1)


def test_subset_size() -> None:
    try:
        from run_query_bootstrap import subset_size
    except ImportError:
        print("skip test_subset_size (torch not installed)")
        return

    assert subset_size(1000, 0.1, 500, 5000) == 500  # floor wins
    assert subset_size(100_000, 0.1, 500, 5000) == 5000  # ceiling wins
    assert subset_size(10_000, 0.1, 500, 5000) == 1000  # fraction wins
    assert subset_size(3, 0.1, 500, 5000) == 3  # never more tiles than exist


def test_summarize_normalize() -> None:
    try:
        import pandas as pd

        from summarize_results import normalize_measurements_df, parse_seed_from_anywhere
    except ImportError:
        print("skip test_summarize_normalize (pandas/yaml missing)")
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
