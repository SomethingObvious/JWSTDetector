"""Framework-free checks for the CPU-side pure helpers.

Run directly: ``python test_helpers.py``. No pytest, no fixtures. These cover the
tile-cutting maths and the reproducibility helper, i.e. the parts that do not
need a GPU, torch, faiss, or cv2.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from prep_data import (
    MiriSensor,
    MosaicSpec,
    infer_tile_id_from_filename,
    make_output_name,
    mostly_empty_rgb,
    robust_asinh_to_uint8,
)
from src.seeding import set_seed


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
        mean_thresh=2.0,
        var_thresh=1.0,
        dry_run=dry_run,
    )


def _busy_sci(h, w):
    # Textured field so tiles clear the empty/flat cut and actually get saved.
    return (np.random.default_rng(0).random((h, w)) * 1000.0).astype(np.float32)


def test_robust_asinh_to_uint8() -> None:
    img = np.linspace(0.0, 1000.0, 256, dtype=np.float32).reshape(16, 16)
    out = robust_asinh_to_uint8(img)
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    # Monotonic input stays monotonic non-decreasing after the stretch.
    flat = out.reshape(-1).astype(np.int32)
    assert np.all(np.diff(flat) >= 0)
    # NaNs/infs must not blow up the percentiles.
    dirty = img.copy()
    dirty[0, 0] = np.nan
    dirty[0, 1] = np.inf
    assert robust_asinh_to_uint8(dirty).dtype == np.uint8


def test_mostly_empty_rgb() -> None:
    blank = np.zeros((32, 32, 3), dtype=np.uint8)
    assert mostly_empty_rgb(blank) is True
    rng = np.random.default_rng(0)
    busy = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    assert mostly_empty_rgb(busy) is False


def test_make_output_name() -> None:
    name = make_output_name("A5", 100.4, 200.6, 7)
    assert name == "A5_cx000100_cy000201_i00000007.png"
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


def test_set_seed_reproducible() -> None:
    set_seed(7)
    first = np.random.rand(5).tolist()
    set_seed(7)
    assert np.random.rand(5).tolist() == first
    set_seed(8)
    assert np.random.rand(5).tolist() != first


def test_reference_selection_reproducible() -> None:
    # The runners draw their reference subset with a *local* default_rng(seed).
    # Same seed -> same tiles; different seed -> different tiles.
    def pick(seed):
        return sorted(np.random.default_rng(seed).choice(100, size=10, replace=False).tolist())

    assert pick(0) == pick(0)
    assert pick(0) != pick(1)


def test_emit_tiles_tile_count() -> None:
    # No partial edge tiles: tiles start only where the full crop fits.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    # 250x250 with crop=100 stride=100 -> starts {0,100} in each axis -> 2x2 = 4.
    _, saved, _, _ = sensor._emit_tiles(
        _tile_args(dry_run=True), spec, "unused", 0, _busy_sci(250, 250), None, None
    )
    assert saved == 4, saved
    # 300x300 -> starts {0,100,200} -> 3x3 = 9.
    _, saved9, _, _ = sensor._emit_tiles(
        _tile_args(dry_run=True), spec, "unused", 0, _busy_sci(300, 300), None, None
    )
    assert saved9 == 9, saved9


def test_emit_tiles_dry_run_parity() -> None:
    # --dry-run must not change counts or the global_idx sequence; it only skips
    # the PNG writes. Prove real-run parity by comparing both modes tile-for-tile.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    sci = _busy_sci(300, 300)

    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as dry_dir:
        real = sensor._emit_tiles(_tile_args(dry_run=False), spec, real_dir, 0, sci, None, None)
        dry = sensor._emit_tiles(_tile_args(dry_run=True), spec, dry_dir, 0, sci, None, None)
        assert real == dry, (real, dry)  # (global_idx, saved, skipped_empty, skipped_wht)

        n_real = len(list(Path(real_dir).glob("*.png")))
        n_dry = len(list(Path(dry_dir).glob("*.png")))
        assert n_real == real[1], (n_real, real[1])  # real run writes exactly `saved` files
        assert n_dry == 0, n_dry  # dry run writes nothing


def test_emit_tiles_wht_gate() -> None:
    # Low-WHT tiles are rejected; an all-zero weight map drops everything.
    sensor = MiriSensor()
    spec = MosaicSpec(tile_id="A5", sci_path="x")
    args = _tile_args(dry_run=True)
    args.min_wht_frac = 0.5
    sci = _busy_sci(200, 200)
    zeros = np.zeros((200, 200), dtype=np.float32)
    _, saved, _, skipped_wht = sensor._emit_tiles(args, spec, "unused", 0, sci, None, zeros)
    assert saved == 0 and skipped_wht == 4, (saved, skipped_wht)


def test_summarize_normalize() -> None:
    # summarize's pure column-normalization for both CSV layouts. Skipped if the
    # heavier deps (pandas/yaml) are not present in this environment.
    try:
        import pandas as pd

        from summarize_results import normalize_measurements_df, parse_seed_from_anywhere
    except Exception:
        print("skip test_summarize_normalize (pandas/yaml missing)")
        return

    q = pd.DataFrame(
        {
            "Sample": ["a", "b"],
            "AnomalyScore": [1.0, 2.0],
            "MemoryBankTimeSec": [0.1, 0.1],
            "InferenceTimeSec": [0.2, 0.3],
        }
    )
    out = normalize_measurements_df(q, Path("x.csv"))
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
