from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from jwst_peculiar.colour import (
    BANDS,
    NEIGHBOUR,
    Found,
    build,
    chi2_to_library,
    kernel_sigma_arcsec,
    lone,
    score,
    spikes,
)

if TYPE_CHECKING:
    from pathlib import Path


def _library(n: int = 40) -> np.ndarray:
    # Smooth spectra, a power law in wavelength with slopes between -1 and 2.
    wave = np.array([w for _, w, _ in BANDS])
    slopes = np.linspace(-1, 2, n)
    return np.array([100 * (wave / 2) ** a for a in slopes])


def test_every_band_is_smoothed_to_the_widest_psf() -> None:
    assert kernel_sigma_arcsec(0.155) == 0.0
    assert kernel_sigma_arcsec(0.048) == pytest.approx(
        np.sqrt(0.155**2 - 0.048**2) / 2.3548
    )


def test_a_scaled_library_spectrum_fits_exactly() -> None:
    lib = _library()
    chi = chi2_to_library(3 * lib[5:6], np.ones((1, len(BANDS))), lib)
    assert chi[0, 5] == pytest.approx(0.0, abs=1e-9)


def test_only_a_spectrum_unlike_the_library_scores_high() -> None:
    lib = _library()
    rng = np.random.default_rng(0)
    line = lib[10].copy()
    line[3] *= 3  # F210M three times its continuum
    faint = 0.05 * lib[10] + rng.normal(0, 2.0, len(BANDS))
    plain = lib[20] * 1.7
    flux = np.array([line, faint, plain])
    # Sky noise plus the 5% floor, as the module adds.
    err = np.sqrt(2.0**2 + (0.05 * flux) ** 2)
    none = (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    s = score(flux, err, lib, pairs=none)
    # With 7 degrees of freedom an ordinary source scores around 7.
    assert s[0] > 100
    assert s[1] < 15
    assert s[2] < 15


def test_self_matches_are_skipped() -> None:
    lib = _library(5)
    err = np.ones((1, len(BANDS)))
    alone = np.array([[100.0, 100, 100, 900, 100, 100, 100, 100]])
    library = np.vstack([lib, alone])
    with_itself = score(
        alone,
        err,
        library,
        pairs=(np.array([], dtype=np.int64), np.array([], dtype=np.int64)),
    )
    without = score(alone, err, library, pairs=(np.array([0]), np.array([5])))
    assert with_itself[0] < without[0]


def test_a_hot_pixel_is_a_spike_and_a_star_is_not() -> None:
    rng = np.random.default_rng(0)
    sci = rng.normal(0, 1.0, (64, 64))
    y, x = np.mgrid[:64, :64]
    # An F140M star on the 0.031 arcsec grid, FWHM 1.55 px.
    sci += 400 * np.exp(-((x - 20.3) ** 2 + (y - 20.6) ** 2) / (2 * 0.66**2))
    sci[44, 40] += 60
    ys, xs = spikes(sci, np.ones_like(sci, dtype=bool))
    assert list(zip(ys.tolist(), xs.tolist(), strict=True)) == [(44, 40)]


def test_a_band_without_a_measurement_drops_out_of_the_fit() -> None:
    lib = _library()
    flux = 3 * lib[5:6].copy()
    flux[0, 4] = 5000.0
    err = np.ones((1, len(BANDS)))
    err[0, 4] = np.inf
    assert chi2_to_library(flux, err, lib)[0, 5] == pytest.approx(0.0, abs=1e-9)
    assert np.all(chi2_to_library(flux, np.full_like(err, np.inf), lib) == 0)


def test_a_cosmic_ray_is_lone_but_a_line_or_a_cold_dwarf_is_not() -> None:
    sky = np.full((4, 8), 0.1)
    ray = np.zeros(8)
    ray[5] = 30.0
    line = np.ones(8)
    line[4] = 12.0
    cold = np.zeros(8)
    cold[7] = 30.0
    faint = np.zeros(8)
    faint[2] = 1.5
    hit = lone(np.array([ray, line, cold, faint]), sky)
    assert hit[0].tolist() == [False] * 5 + [True] + [False] * 2
    assert not hit[1:].any()


def _found(n_lib: int) -> Found:
    # Two faint sources first, so a library source's index in the catalogue isn't
    # its index in the library, then the library at 10 arcsec apart.
    wave = np.array([w for _, w, _ in BANDS])
    faint = [20 * (wave / 2) ** a for a in (0.0, 1.0)]
    bright = [1000 * (wave / 2) ** a for a in np.linspace(-1, 2, n_lib)]
    n = 2 + n_lib
    return Found(
        ra=np.full(n, 34.4),
        dec=-5.2 + np.arange(n) * 10 / 3600,
        flux=np.array(faint + bright),
        sky_err=np.ones((n, len(BANDS))),
        det_sn=np.full(n, 50.0),
        conc=np.full(n, 0.3),
        block=["b"] * n,
    )


def _no_bright_stars(tmp_path: Path) -> Path:
    path = tmp_path / "stars.json"
    path.write_text(
        json.dumps({
            "centre": [34.4, -5.2],
            "radius_deg": 1.0,
            "rows": [[34.5, -5.2, 20.0]],
        })
    )
    return path


def test_each_library_source_is_scored_without_itself(tmp_path: Path) -> None:
    found = _found(NEIGHBOUR + 1)
    cat = build([found], _no_bright_stars(tmp_path))
    assert cat.library.tolist() == list(range(2, NEIGHBOUR + 3))
    err = np.sqrt(1 + (0.05 * found.flux) ** 2)
    lib = np.arange(NEIGHBOUR + 1)
    expected = score(found.flux, err, found.flux[2:], pairs=(lib + 2, lib))
    assert np.isfinite(expected).all()
    assert cat.scores.tolist() == expected.tolist()


def test_a_library_too_small_to_leave_one_out_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=f"library of at least {NEIGHBOUR + 1}"):
        build([_found(NEIGHBOUR)], _no_bright_stars(tmp_path))
