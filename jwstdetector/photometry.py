"""What a FITS image's pixel values mean in janskys, and how sharp its PSF is.

Prep reads these off each file's headers and records them per filter in
mosaics/<id>.json, which is what lets the benchmark plant a source of a given AB
magnitude in an HST, Euclid, Spitzer or ground-based mosaic as well as a JWST one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

ARCSEC_PER_RAD = 206264.806
AB_ZERO_JY = 3631.0

# Primary mirror diameters in metres, for space telescopes whose PSF is close to the
# diffraction limit. A ground-based image needs --psf_fwhm instead.
APERTURES = {"JWST": 6.5, "HST": 2.4, "EUCLID": 1.2, "ROMAN": 2.4, "SPITZER": 0.85, "WISE": 0.4}

FILTER_KEYS = ("FILTER", "FILTER1", "FILTER2", "FILTNAM1", "BAND", "FILTNAME", "FILT")
ZEROPOINT_KEYS = ("MAGZERO", "MAGZP", "ZEROPNT", "ZP", "MAGZPT", "PHOTZP", "ZP_AB")


def _get(headers: Sequence[Mapping], key: str):
    for h in headers:
        value = h.get(key)
        if value not in (None, ""):
            return value
    return None


def telescope(headers: Sequence[Mapping]) -> str:
    """JWST, HST, EUCLID and so on, upper case, or "" when the headers don't say."""
    name = str(_get(headers, "TELESCOP") or "").upper()
    return next((t for t in APERTURES if t in name.replace(" ", "")), name)


def filter_name(headers: Sequence[Mapping], path: str | Path = "") -> str:
    """The filter from the usual keywords, skipping CLEAR wheels, or else the file stem."""
    for key in FILTER_KEYS:
        value = str(_get(headers, key) or "").strip().upper()
        if value and not value.startswith(("CLEAR", "N/A")):
            return value
    return Path(path).stem


def wavelength_um(headers: Sequence[Mapping], filt: str) -> float | None:
    """The filter's pivot wavelength in microns, from the headers or a JWST filter name."""
    for key, scale in (("PHOTPLAM", 1e-4), ("PIVOT", 1e-4), ("WAVELEN", 1e-4), ("WAVELENG", 1e-4)):
        value = _get(headers, key)
        if value is not None:
            return float(value) * scale
    # JWST names give the wavelength in hundredths of a micron, so F444W is 4.44 and
    # F1000W is 10.00. HST names don't follow one rule, but HST files carry PHOTPLAM.
    m = re.fullmatch(r"F(\d{3,4})[WMNC2]{1,2}", filt.upper())
    if m and telescope(headers) == "JWST":
        return int(m.group(1)) / 100
    return None


# Case matters for the prefix only, since MJy and mJy are 1e9 apart.
_JY = re.compile(r"(micro|nano|[Mkmunµ]?)(?i:jy)(?i:/(sr|steradian|pixel|pix))?")
_PREFIX = {"": 1.0, "M": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "µ": 1e-6, "micro": 1e-6,
           "n": 1e-9, "nano": 1e-9}  # fmt: skip


def jy_per_unit(headers: Sequence[Mapping], pixel_scale_arcsec: float) -> float | None:
    """Janskys per pixel for one unit of pixel value, or None when the headers don't say.

    Surface brightness (MJy/sr) scales with the pixel's solid angle, and per-pixel
    units don't. An AB zero point covers counts in whatever units it was set for.
    """
    raw = re.sub(r"[\s_]", "", str(_get(headers, "BUNIT") or ""))
    bunit = raw.lower()
    m = _JY.fullmatch(raw)
    if m:
        scale = _PREFIX[m.group(1)]
        per_sr = (m.group(2) or "").lower() in ("sr", "steradian")
        return scale * (pixel_scale_arcsec / ARCSEC_PER_RAD) ** 2 if per_sr else scale
    if bunit in ("nanomaggy", "nanomaggies"):
        return 3.631e-6
    photfnu = _get(headers, "PHOTFNU")
    if photfnu is not None and bunit in ("", "electrons/s", "electron/s", "e/s", "e-/s", "cps"):
        return float(photfnu)
    for key in ZEROPOINT_KEYS:
        value = _get(headers, key)
        if value is not None:
            return AB_ZERO_JY * 10 ** (-0.4 * float(value))
    return None


def diffraction_fwhm_arcsec(micron: float, scope: str) -> float | None:
    """1.03 lambda over D for a space telescope, which for JWST longward of 2 microns is
    within about 10% of the measured FWHM. None for anything not in APERTURES."""
    if scope not in APERTURES:
        return None
    return 1.03 * micron * 1e-6 / APERTURES[scope] * ARCSEC_PER_RAD


def describe_plane(
    headers: Sequence[Mapping],
    path: str | Path,
    pixel_scale_arcsec: float | None,
    filt: str | None = None,
) -> dict:
    """What prep records about one filter of a mosaic, for the benchmark.

    `filt` names the filter when the caller knows it better than the headers do.
    """
    filt = (filt or filter_name(headers, path)).upper()
    scope = telescope(headers)
    micron = wavelength_um(headers, filt)
    info = {
        "filter": filt,
        "telescope": scope,
        "bunit": str(_get(headers, "BUNIT") or ""),
        "micron": micron,
        "jy_per_unit": None,
        "psf_fwhm_arcsec": None,
    }
    if pixel_scale_arcsec:
        info["jy_per_unit"] = jy_per_unit(headers, pixel_scale_arcsec)
    if micron is not None:
        info["psf_fwhm_arcsec"] = diffraction_fwhm_arcsec(micron, scope)
    if info["psf_fwhm_arcsec"] is not None and not math.isfinite(info["psf_fwhm_arcsec"]):
        info["psf_fwhm_arcsec"] = None
    return info
