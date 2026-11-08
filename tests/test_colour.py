from __future__ import annotations

import numpy as np
import pytest

from jwstdetector.colour import BANDS, chi2_to_library, kernel_sigma_arcsec, score


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
