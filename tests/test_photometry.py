from __future__ import annotations

import logging

import pytest

from jwst_peculiar.photometry import ARCSEC_PER_RAD, jy_per_unit, wavelength_um

SCALE = 0.063
PIXEL_SR = (SCALE / ARCSEC_PER_RAD) ** 2


@pytest.mark.parametrize(
    ("bunit", "jy"),
    [
        ("MJy/sr", 1e6 * PIXEL_SR),
        ("MJy / sr", 1e6 * PIXEL_SR),
        ("MJY/SR", 1e6 * PIXEL_SR),
        ("mJy/sr", 1e-3 * PIXEL_SR),
        ("Jy/pixel", 1.0),
        ("uJy", 1e-6),
        ("\u00b5Jy", 1e-6),
        ("nJy", 1e-9),
        ("nanomaggy", 3.631e-6),
    ],
)
def test_bunit_gives_janskys_per_pixel(bunit: str, jy: float) -> None:
    assert jy_per_unit([{"BUNIT": bunit}], SCALE) == jy


def test_counts_take_photfnu_or_a_zero_point() -> None:
    assert jy_per_unit([{"BUNIT": "ELECTRONS/S", "PHOTFNU": 1e-7}], SCALE) == 1e-7
    assert jy_per_unit([{"BUNIT": "ADU", "MAGZPT": 25.0}], SCALE) == pytest.approx(
        3631.0 * 10**-10, rel=1e-12
    )


def test_a_lower_case_milli_or_mega_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert jy_per_unit([{"BUNIT": "mjy/sr"}], SCALE) is None
    assert caplog.messages == ["BUNIT mjy/sr could be mJy or MJy, so it isn't used"]


def test_an_unambiguous_wavelength_comes_back_in_microns() -> None:
    assert wavelength_um([{"PHOTPLAM": 15000.0}], "F150W") == 1.5
    assert wavelength_um([{"PHOTPLAM": 1.5}], "F150W") == 1.5
    assert wavelength_um([{"WAVELEN": 3.368}], "W1") == 3.368


def test_a_wavelength_that_could_be_nanometres_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert wavelength_um([{"WAVELEN": 700}], "R") is None
    assert caplog.messages == [
        "WAVELEN is 700, which could be angstroms or nanometres, so it isn't used"
    ]
    # A JWST filter's name still says it.
    assert wavelength_um([{"WAVELEN": 4440, "TELESCOP": "JWST"}], "F444W") == 4.44
