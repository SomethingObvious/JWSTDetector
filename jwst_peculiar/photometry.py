"""What a FITS image's pixel values mean in janskys, and how sharp its PSF is.

Prep reads these off each file's headers and records them per filter in
mosaics/<id>.json, which is what lets the benchmark plant a source of a given AB
magnitude in an HST, Euclid, Spitzer or ground-based mosaic as well as a JWST one.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    Headers = Sequence[Mapping[str, Any]]

logger = logging.getLogger(__name__)

ARCSEC_PER_RAD = 206264.806
# No imaging filter sits below 100 angstroms or above 100 microns.
_MICRONS_BELOW = 100
AB_ZERO_JY = 3631.0

# Primary mirror diameters in metres, for space telescopes whose PSF is close to the
# diffraction limit. A ground-based image needs --psf_fwhm instead, and so do Spitzer
# and WISE, whose pixels are too coarse to reach it.
APERTURES = {
    "JWST": 6.5,
    "HST": 2.4,
    "EUCLID": 1.2,
    "ROMAN": 2.4,
}

FILTER_KEYS = ("FILTER", "FILTER1", "FILTER2", "FILTNAM1", "BAND", "FILTNAME", "FILT")
_JWST_FILTER = re.compile(r"F(\d{3,4})[WMNC2]{1,2}")
ZEROPOINT_KEYS = ("MAGZERO", "MAGZP", "ZEROPNT", "ZP", "MAGZPT", "PHOTZP", "ZP_AB")


def _get(headers: Headers, key: str) -> Any:  # noqa: ANN401 header values are bool, int, float or str
    for h in headers:
        value = h.get(key)
        if value not in (None, ""):
            return value
    return None


def telescope(headers: Headers) -> str:
    """Read the telescope's name off the headers.

    Returns:
        JWST, HST, EUCLID and so on, upper case, or "" when the headers don't say.
    """
    name = str(_get(headers, "TELESCOP") or "").upper()
    return next((t for t in APERTURES if t in name.replace(" ", "")), name)


def filter_name(headers: Headers, path: str | Path = "") -> str:
    """Read the filter from the usual keywords, skipping CLEAR wheels.

    Returns:
        The filter, upper case, or the file stem when no keyword names one.
    """
    # NIRCam's narrow filters and F162M, and NIRISS's short ones, sit in the pupil
    # wheel behind a wider filter, which is the one FILTER names.
    pupil = str(_get(headers, "PUPIL") or "").strip().upper()
    if _JWST_FILTER.fullmatch(pupil):
        return pupil
    for key in FILTER_KEYS:
        value = str(_get(headers, key) or "").strip().upper()
        if value and not value.startswith(("CLEAR", "N/A")):
            return value
    return Path(path).stem


def wavelength_um(headers: Headers, filt: str) -> float | None:
    """Find the filter's pivot wavelength from the headers or a JWST filter name.

    Returns:
        The wavelength in microns, or None when neither gives it.
    """
    for key in ("PHOTPLAM", "PIVOT", "WAVELEN", "WAVELENG"):
        value = _get(headers, key)
        if value is None:
            continue
        if float(value) <= _MICRONS_BELOW:
            return float(value)
        # PHOTPLAM is in angstroms by definition, but the others come in angstroms
        # or nanometres, and 7000 A and 700 nm are the same filter.
        if key == "PHOTPLAM":
            return float(value) * 1e-4
        logger.warning(
            "%s is %s, which could be angstroms or nanometres, so it isn't used",
            key,
            value,
        )
        break
    # JWST names give the wavelength in hundredths of a micron, so F444W is 4.44 and
    # F1000W is 10.00. HST names don't follow one rule, but HST files carry PHOTPLAM.
    m = _JWST_FILTER.fullmatch(filt.upper())
    if m and telescope(headers) == "JWST":
        return int(m.group(1)) / 100
    return None


# Case matters for the prefix only, since MJy and mJy are 1e9 apart. \u00b5 is the
# micro sign, which looks the same as the Greek mu but isn't.
_JY = re.compile(r"(micro|nano|[Mkmun\u00b5]?)(?i:jy)(?i:/(sr|steradian|pixel|pix))?")
_PREFIX = {"": 1.0, "M": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "\u00b5": 1e-6,
           "micro": 1e-6, "n": 1e-9, "nano": 1e-9}  # fmt: skip


def jy_per_unit(headers: Headers, pixel_scale_arcsec: float) -> float | None:
    """Work out how many janskys per pixel one unit of pixel value is.

    Surface brightness (MJy/sr) scales with the pixel's solid angle, and per-pixel
    units don't. An AB zero point covers counts in whatever units it was set for.

    Returns:
        Janskys per pixel per unit, or None when the headers don't say.
    """
    raw = re.sub(r"[\s_]", "", str(_get(headers, "BUNIT") or ""))
    from_unit = _jy_from_bunit(raw, pixel_scale_arcsec)
    if from_unit is not None:
        return from_unit
    return _jy_from_calibration(headers, raw.lower())


def _jy_from_bunit(raw: str, pixel_scale_arcsec: float) -> float | None:
    m = _JY.fullmatch(raw)
    # Written all in lower case, the unit has lost what told MJy from mJy.
    if m and m.group(1) == "m" and raw.islower():
        logger.warning("BUNIT %s could be mJy or MJy, so it isn't used", raw)
        return None
    if m:
        scale = _PREFIX[m.group(1)]
        per_sr = (m.group(2) or "").lower() in ("sr", "steradian")
        return scale * (pixel_scale_arcsec / ARCSEC_PER_RAD) ** 2 if per_sr else scale
    if raw.lower() in ("nanomaggy", "nanomaggies"):
        return 3.631e-6
    return None


def _jy_from_calibration(headers: Headers, bunit: str) -> float | None:
    photfnu = _get(headers, "PHOTFNU")
    if photfnu is not None and bunit in (
        "",
        "electrons/s",
        "electron/s",
        "e/s",
        "e-/s",
        "cps",
    ):
        return float(photfnu)
    for key in ZEROPOINT_KEYS:
        value = _get(headers, key)
        if value is not None:
            return AB_ZERO_JY * 10 ** (-0.4 * float(value))
    return None


def diffraction_fwhm_arcsec(micron: float, scope: str) -> float | None:
    """Estimate a space telescope's PSF FWHM as 1.03 lambda over D.

    For JWST longward of 2 microns that's within about 10% of the measured FWHM.

    Returns:
        The FWHM in arcseconds, or None for anything not in APERTURES.
    """
    if scope not in APERTURES:
        return None
    return 1.03 * micron * 1e-6 / APERTURES[scope] * ARCSEC_PER_RAD


def describe_plane(
    headers: Headers,
    path: str | Path,
    pixel_scale_arcsec: float | None,
    *,
    filt: str | None = None,
) -> dict[str, Any]:
    """Describe one filter of a mosaic the way prep records it for the benchmark.

    `filt` names the filter when the caller knows it better than the headers do.

    Returns:
        The filter, telescope, BUNIT, wavelength, flux scale and PSF FWHM.
    """
    filt = (filt or filter_name(headers, path)).upper()
    scope = telescope(headers)
    micron = wavelength_um(headers, filt)
    jy = jy_per_unit(headers, pixel_scale_arcsec) if pixel_scale_arcsec else None
    fwhm = diffraction_fwhm_arcsec(micron, scope) if micron is not None else None
    return {
        "filter": filt,
        "telescope": scope,
        "bunit": str(_get(headers, "BUNIT") or ""),
        "micron": micron,
        "jy_per_unit": jy,
        "psf_fwhm_arcsec": fwhm if fwhm is not None and math.isfinite(fwhm) else None,
    }
