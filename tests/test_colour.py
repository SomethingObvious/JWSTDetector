from __future__ import annotations

import numpy as np
import pytest

from jwst_peculiar.colour import (
    BANDS,
    chi2_to_library,
    kernel_sigma_arcsec,
    lone,
    score,
    spikes,
)


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
