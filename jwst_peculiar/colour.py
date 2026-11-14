"""Ranks sources by how unlike every well-measured source their spectrum is.

    python -m jwst_peculiar.colour RAW STARS OUT MOSAIC [MOSAIC ...]

Each band of each mosaic (RAW/MOSAIC/MOSAIC_nircam_*-BAND_i2d.fits) is smoothed to
F460M's resolution and resampled onto the F460M grid, one 1,920 px block at a time.
Sources are found on the noise-weighted sum of the bands and measured in a 0.32
arcsec aperture in each, with errors from the same aperture on empty sky in the
block plus 5% of the flux. The sources at 30 sigma or more in every band are the
library, and each source scores the fifth-smallest chi-squared against a library
spectrum scaled to fit it, leaving out itself and anything within 1 arcsec. The 50
highest get a sheet each. Nothing here is a statistical claim.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import html
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D
from photutils.segmentation import SourceCatalog, deblend_sources, detect_sources
from reproject import reproject_interp
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    maximum_filter,
    uniform_filter,
)

from jwst_peculiar.survey import STAR_RADIUS_AT_19, read_stars
from jwst_peculiar.vet import catalogue_note

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwst_peculiar.utils import FitsHDUList, SkyCoords, SkyWcs

logger = logging.getLogger(__name__)

# Band, central wavelength in microns and PSF FWHM in arcsec, from JDox's NIRCam
# PSF table, bluest first. F460M is the widest, so every band is smoothed to it.
BANDS = (
    ("f140m", 1.405, 0.048),
    ("f162m", 1.627, 0.055),
    ("f182m", 1.845, 0.062),
    ("f210m", 2.095, 0.071),
    ("f250m", 2.503, 0.084),
    ("f300m", 2.989, 0.100),
    ("f360m", 3.624, 0.119),
    ("f460m", 4.630, 0.155),
)
BLOCK_PX = 1920
# A band covers a pixel if nearly all of what was resampled into it had data.
COVERED = 0.999
MIN_SKY_PIXELS = 500
MARGIN_PX = 32
# A star's halo reaches past the star rule and would merge whole regions into one
# source, so each band has a local sky taken off in boxes about 6 arcsec across.
SKY_BOX_PX = 96
APERTURE_ARCSEC = 0.16
# Point sources sit on a tight sequence of the flux ratio inside these two radii,
# which tells a brown dwarf from a compact galaxy of the same colours.
CORE_ARCSEC = 0.08
WIDE_ARCSEC = 0.24
# On o003, point sources sit at 0.35 to 0.40 and nearly everything over 0.45 is a hot
# pixel or cosmic ray in one band, so sources sharper than that are listed apart.
SHARPEST = 0.45
FLOOR = 0.05
# Smoothing spreads a hot pixel into something star shaped, so they are found on the
# native pixels first. 60 library stars put at most 0.28 of a 3 by 3 box in its
# brightest pixel, and the hot pixels at the top of the first full list put 0.34
# to 0.60.
SPIKE_SIGMA = 8.0
SPIKE_SHARE = 0.32
# A cosmic ray across a few pixels is too spread out for that, but it lights one band
# alone. [OIII] at a rest equivalent width of 4,000 A, about the most extreme known,
# lifts a medium band to at most 12 times the rest (F250M at z of 4). In Sonora Bobcat
# only F460M goes past that, to 2,500 times at 200 K, so it's never treated this way.
LONE = 20.0
DETECT_SIGMA = 3.0
DETECT_PIXELS = 5
KEEP_SN = 10.0
LIBRARY_SN = 30.0
NEIGHBOUR = 5
APART_ARCSEC = 1.0
STAR_G_MAX = 19.0
TOP = 50


def kernel_sigma_arcsec(fwhm: float, target: float = BANDS[-1][2]) -> float:
    """The Gaussian width that takes a PSF of `fwhm` to one of `target`.

    Returns:
        Sigma in arcsec, 0 for the target band itself.
    """
    return float(np.sqrt(max(target**2 - fwhm**2, 0.0)) / 2.3548)


def chi2_to_library(
    flux: npt.NDArray[np.float64],
    err: npt.NDArray[np.float64],
    library: npt.NDArray[np.float64],
    *,
    skip: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.float64]:
    """Chi-squared of each source against each library spectrum scaled to fit it.

    `skip` (sources by library) marks pairs that don't count, a source against
    itself or a neighbour, which come back as infinity.

    Returns:
        One row per source, one column per library spectrum.
    """
    w = 1.0 / err**2
    flux = np.where(np.isfinite(err), flux, 0.0)
    a = (flux * w) @ library.T
    b = w @ (library**2).T
    c = np.sum(flux**2 * w, axis=1)[:, None]
    fit = np.divide(a**2, b, out=np.zeros_like(a), where=b > 0)
    chi2 = np.maximum(c - fit, 0.0)
    if skip is not None:
        chi2[skip] = np.inf
    return chi2


def score(
    flux: npt.NDArray[np.float64],
    err: npt.NDArray[np.float64],
    library: npt.NDArray[np.float64],
    *,
    pairs: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
) -> npt.NDArray[np.float64]:
    """The NEIGHBOUR-th smallest chi-squared of each source over the library.

    `pairs` lists the (source, library) pairs that don't count.

    Returns:
        One score per source.
    """
    out = np.empty(len(flux))
    src, lib = pairs
    step = 2000
    for lo in range(0, len(flux), step):
        hi = min(lo + step, len(flux))
        skip = np.zeros((hi - lo, len(library)), dtype=bool)
        mine = (src >= lo) & (src < hi)
        skip[src[mine] - lo, lib[mine]] = True
        part = chi2_to_library(flux[lo:hi], err[lo:hi], library, skip=skip)
        out[lo:hi] = np.partition(part, NEIGHBOUR - 1, axis=1)[:, NEIGHBOUR - 1]
    return out


def lone(
    flux: npt.NDArray[np.float64], sky_err: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Bands other than F460M holding LONE times every other band of their source.

    A band without a detection counts at its 1 sigma.

    Returns:
        One row per source, one column per band.
    """
    floor = np.maximum(flux, sky_err)
    out = np.zeros(flux.shape, dtype=bool)
    for b in range(flux.shape[1] - 1):
        others = np.delete(floor, b, axis=1).max(axis=1)
        out[:, b] = flux[:, b] > LONE * others
    return out


def _band_file(raw: Path, mosaic: str, band: str) -> Path:
    found = sorted((raw / mosaic).glob(f"{mosaic}_nircam_*{band}_i2d.fits"))
    if len(found) != 1:
        msg = f"Expected one {band} file in {raw / mosaic}, found {len(found)}."
        raise FileNotFoundError(msg)
    return found[0]


def spikes(
    sci: npt.NDArray[np.float64], good: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pixels far sharper than the PSF allows, on a band's native grid.

    Returns:
        Their rows and columns.
    """
    noise = _mad(sci[good]) if good.any() else 1.0
    box = uniform_filter(np.clip(sci, 0.0, None), 3) * 9
    share = np.divide(sci, box, out=np.zeros_like(sci), where=box > 0)
    hit = (
        good
        & (sci > SPIKE_SIGMA * noise)
        & (sci == maximum_filter(sci, 3))
        & (share > SPIKE_SHARE)
    )
    ys, xs = np.nonzero(hit)
    return ys.astype(np.int64), xs.astype(np.int64)


def _resampled(  # noqa: PLR0914 one band, read, smoothed and resampled
    path: Path, target: SkyWcs, *, shape: tuple[int, int], fwhm: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    """One band smoothed to F460M and resampled onto the target block.

    Returns:
        The image, where it has data and its spikes as x, y on the target.
    """
    hdul: FitsHDUList = fits.open(path, memmap=True)
    with hdul as h:
        wcs = WCS(h["SCI"].header)
        ny, nx = h["SCI"].shape
        x, y = wcs.world_to_pixel(
            target.pixel_to_world(
                [0, shape[1], 0, shape[1]], [0, 0, shape[0], shape[0]]
            )
        )
        pad = 16
        xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad, nx)
        y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad, ny)
        if x1 <= x0 or y1 <= y0:
            return np.zeros(shape), np.zeros(shape, dtype=bool), np.empty((0, 2))
        sci = np.array(h["SCI"].section[y0:y1, x0:x1], dtype=np.float64)
        wht = np.array(h["WHT"].section[y0:y1, x0:x1], dtype=np.float64)
    cut = wcs[y0:y1, x0:x1]
    good = np.isfinite(sci) & (wht > 0)
    sci[~good] = 0.0
    sy, sx = spikes(sci, good)
    at = target.world_to_pixel(cut.pixel_to_world(sx, sy))
    spiked = np.column_stack([np.atleast_1d(at[0]), np.atleast_1d(at[1])])
    sigma_px = kernel_sigma_arcsec(fwhm) / float(
        proj_plane_pixel_scales(cut).mean() * 3600
    )
    if sigma_px > 0:
        sci = gaussian_filter(sci, sigma_px)
    img, _ = reproject_interp((sci, cut), target, shape_out=shape)
    cover, _ = reproject_interp((good.astype(np.float64), cut), target, shape_out=shape)
    return np.nan_to_num(img), np.nan_to_num(cover) > COVERED, spiked


@dataclass
class Found:
    """Sources of one block, measured in every band."""

    ra: npt.NDArray[np.float64]
    dec: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    sky_err: npt.NDArray[np.float64]
    det_sn: npt.NDArray[np.float64]
    conc: npt.NDArray[np.float64]
    block: list[str]


def _sky_scatter(
    img: npt.NDArray[np.float64], empty: npt.NDArray[np.bool_], radius: float
) -> float:
    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(empty)
    if len(ys) < MIN_SKY_PIXELS:
        return float("nan")
    pick = rng.choice(len(ys), size=min(2000, len(ys)), replace=False)
    ap = CircularAperture(np.column_stack([xs[pick], ys[pick]]), r=radius)
    sums = np.asarray(aperture_photometry(img, ap)["aperture_sum"], dtype=np.float64)
    return float(1.4826 * np.median(np.abs(sums - np.median(sums))))


def _bands_on(
    raw: Path, mosaic: str, *, target: SkyWcs, shape: tuple[int, int]
) -> tuple[
    list[npt.NDArray[np.float64]],
    npt.NDArray[np.bool_],
    list[npt.NDArray[np.float64]],
]:
    imgs, cover, spiked = [], np.ones(shape, dtype=bool), []
    for band, _, fwhm in BANDS:
        img, ok, at = _resampled(
            _band_file(raw, mosaic, band), target, shape=shape, fwhm=fwhm
        )
        imgs.append(img)
        spiked.append(at)
        cover &= ok
    if cover.sum() < 0.1 * cover.size:
        return imgs, cover, spiked
    for i, img in enumerate(imgs):
        sky = Background2D(img, SKY_BOX_PX, filter_size=3, coverage_mask=~cover)
        imgs[i] = np.where(cover, img - sky.background, 0.0)
    return imgs, cover, spiked


def _mad(x: npt.NDArray[np.float64]) -> float:
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) or 1.0


def _detect(
    det: npt.NDArray[np.float64], cover: npt.NDArray[np.bool_]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]] | None:
    segm = detect_sources(
        det, DETECT_SIGMA * _mad(det[cover]), DETECT_PIXELS, mask=~cover
    )
    if segm is None:
        return None
    # Without this, close pairs share one centroid and a mix of their colours.
    segm = deblend_sources(det, segm, DETECT_PIXELS, progress_bar=False)
    cat = SourceCatalog(det, segm)
    # The peak, not the segment's centroid, as a segment can hold a source and a
    # stretch of halo or detector stripe beside it.
    xy = np.asarray(cat.centroid_quad, dtype=np.float64)
    bad = ~np.all(np.isfinite(xy), axis=1)
    xy[bad] = np.column_stack([cat.xcentroid, cat.ycentroid])[bad]
    return xy, np.asarray(segm.data > 0)


def measure_block(  # noqa: PLR0914, PLR0915 one block, in the order the module docstring gives
    raw: Path,
    mosaic: str,
    *,
    ref: SkyWcs,
    corner: tuple[int, int],
    size: tuple[int, int],
) -> Found | None:
    """Find and measure the sources whose centre lies in one block.

    Returns:
        The block's sources, or None if it has too little data or no sources.
    """
    by, bx = corner
    shape = (size[0] + 2 * MARGIN_PX, size[1] + 2 * MARGIN_PX)
    target = ref.deepcopy()
    target.wcs.crpix = target.wcs.crpix - [bx - MARGIN_PX, by - MARGIN_PX]
    imgs, cover, spiked = _bands_on(raw, mosaic, target=target, shape=shape)
    if cover.sum() < 0.1 * cover.size:
        return None
    det = np.where(cover, sum(img / _mad(img[cover]) for img in imgs), 0.0)
    found = _detect(det, cover)
    if found is None:
        return None
    xy, in_source = found
    inner = (
        (xy[:, 0] >= MARGIN_PX)
        & (xy[:, 0] < MARGIN_PX + size[1])
        & (xy[:, 1] >= MARGIN_PX)
        & (xy[:, 1] < MARGIN_PX + size[0])
    )
    xy = xy[inner & np.all(np.isfinite(xy), axis=1)]
    if not len(xy):
        return None
    pixel = float(proj_plane_pixel_scales(ref).mean() * 3600)
    radius = APERTURE_ARCSEC / pixel
    ap = CircularAperture(xy, r=radius)
    # Uncovered pixels are zero, so a source on a gap would be measured short.
    gaps = aperture_photometry(
        (~cover).astype(np.float64), CircularAperture(xy, r=WIDE_ARCSEC / pixel)
    )["aperture_sum"]
    whole = np.asarray(gaps, dtype=np.float64) == 0
    empty = cover & ~binary_dilation(in_source, iterations=6)
    flux = np.column_stack([
        np.asarray(aperture_photometry(img, ap)["aperture_sum"]) for img in imgs
    ])
    sky_err = np.tile([_sky_scatter(img, empty, radius) for img in imgs], (len(xy), 1))
    # A band with a spike in the aperture counts as unmeasured, not as the source.
    for band, at in enumerate(spiked):
        if len(at):
            near = np.hypot(*(xy[:, None, :] - at[None, :, :]).transpose(2, 0, 1))
            sky_err[(near <= radius + 1).any(axis=1), band] = np.inf
    det_flux = np.asarray(
        aperture_photometry(det, ap)["aperture_sum"], dtype=np.float64
    )
    det_sn = det_flux / _sky_scatter(det, empty, radius)
    core, wide = (
        np.asarray(
            aperture_photometry(det, CircularAperture(xy, r=r / pixel))["aperture_sum"],
            dtype=np.float64,
        )
        for r in (CORE_ARCSEC, WIDE_ARCSEC)
    )
    sky = target.pixel_to_world(xy[:, 0], xy[:, 1])
    keep = whole & np.isfinite(det_sn)
    name = f"{mosaic}_b{by // BLOCK_PX:02d}_{bx // BLOCK_PX:02d}"
    return Found(
        ra=np.asarray(sky.ra.deg)[keep],
        dec=np.asarray(sky.dec.deg)[keep],
        flux=flux[keep],
        sky_err=sky_err[keep],
        det_sn=det_sn[keep],
        conc=(core / wide)[keep],
        block=[name] * int(keep.sum()),
    )


def measure_mosaic(raw: Path, mosaic: str) -> list[Found]:
    """Every block of one mosaic, on the F460M grid as prep cuts it.

    Returns:
        The blocks that have sources.
    """
    hdul: FitsHDUList = fits.open(_band_file(raw, mosaic, "f460m"), memmap=True)
    with hdul as h:
        ref = WCS(h["SCI"].header)
        ny, nx = h["SCI"].shape
    out = []
    for by in range(0, ny, BLOCK_PX):
        for bx in range(0, nx, BLOCK_PX):
            size = (min(BLOCK_PX, ny - by), min(BLOCK_PX, nx - bx))
            got = measure_block(raw, mosaic, ref=ref, corner=(by, bx), size=size)
            logger.info(
                "%s block %d %d: %d sources",
                mosaic,
                by,
                bx,
                0 if got is None else len(got.ra),
            )
            if got is not None:
                out.append(got)
    return out


def _near_star(coords: SkyCoords, stars_json: Path) -> npt.NDArray[np.bool_]:
    stars, g = read_stars(stars_json, coords)
    bright = g < STAR_G_MAX
    out = np.zeros(len(coords), dtype=bool)
    if not bright.any():
        return out
    # The survey adds a tile's half-diagonal to the halo, but a source is a point.
    # With the tile's reach the rule throws away 16% of o003.
    reach = STAR_RADIUS_AT_19 * 10 ** (0.2 * (19 - g[bright]))
    idx_star, idx_src, sep, _ = coords.search_around_sky(
        stars[bright], float(np.max(reach)) * u.arcsec
    )
    out[idx_src[sep.arcsec < reach[idx_star]]] = True
    return out


def _unique(
    coords: SkyCoords, order: npt.NDArray[np.int64], arcsec: float
) -> npt.NDArray[np.bool_]:
    """Keep the first of any sources closer than `arcsec`, in `order`.

    Returns:
        One flag per source.
    """
    keep = np.ones(len(coords), dtype=bool)
    i, j, _, _ = coords.search_around_sky(coords, arcsec * u.arcsec)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    pairs = sorted(zip(i[i != j], j[i != j], strict=True), key=lambda p: rank[p[0]])
    for a, b in pairs:
        if keep[a] and rank[a] < rank[b]:
            keep[b] = False
    return keep


@dataclass
class Catalogue:
    """Every kept source, with its errors, score and the library."""

    coords: SkyCoords
    block: list[str]
    det_sn: npt.NDArray[np.float64]
    conc: npt.NDArray[np.float64]
    flux: npt.NDArray[np.float64]
    err: npt.NDArray[np.float64]
    library: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float64]


def build(found: list[Found], stars_json: Path) -> Catalogue:  # noqa: PLR0914 the cuts in the order docs/minerva.md gives
    """Apply the cuts, form the library and score every source.

    Returns:
        The catalogue.

    Raises:
        SystemExit: If too few sources are bright enough to form the library.
    """
    flux = np.concatenate([f.flux for f in found])
    sky_err = np.concatenate([f.sky_err for f in found])
    det_sn = np.concatenate([f.det_sn for f in found])
    conc = np.concatenate([f.conc for f in found])
    coords: SkyCoords = SkyCoord(
        np.concatenate([f.ra for f in found]),
        np.concatenate([f.dec for f in found]),
        unit="deg",
    )
    keep = (det_sn >= KEEP_SN) & np.all(sky_err > 0, axis=1)
    keep &= ~_near_star(coords, stars_json)
    # Two mosaics can share sky, so a source measured twice keeps its better copy.
    keep &= _unique(coords, np.argsort(-det_sn), 0.3)
    idx = np.nonzero(keep)[0]
    flux, sky_err, coords = flux[idx], sky_err[idx], coords[idx]
    sky_err = np.where(lone(flux, sky_err), np.inf, sky_err)
    err = np.sqrt(sky_err**2 + (FLOOR * flux) ** 2)
    lib = np.nonzero(np.all(flux / sky_err >= LIBRARY_SN, axis=1))[0]
    # A library source leaves itself out, so it needs NEIGHBOUR others.
    if len(lib) < NEIGHBOUR + 1:
        msg = (
            f"Only {len(lib)} sources reach {LIBRARY_SN} sigma in every band, and "
            f"the score needs a library of at least {NEIGHBOUR + 1}."
        )
        raise SystemExit(msg)
    # astropy gives the library's index first, then the source's.
    by_lib, by_src, _, _ = coords.search_around_sky(
        coords[lib], APART_ARCSEC * u.arcsec
    )
    pairs = (by_src.astype(np.int64), by_lib.astype(np.int64))
    blocks = [b for f in found for b in f.block]
    return Catalogue(
        coords=coords,
        block=[blocks[i] for i in idx],
        det_sn=det_sn[idx],
        conc=conc[idx],
        flux=flux,
        err=err,
        library=lib,
        scores=score(flux, err, flux[lib], pairs=pairs),
    )


def _stamp(
    path: Path, at: SkyCoords, half_arcsec: float = 3.0
) -> npt.NDArray[np.float64]:
    hdul: FitsHDUList = fits.open(path, memmap=True)
    with hdul as h:
        wcs: SkyWcs = WCS(h["SCI"].header)
        x, y = (float(v) for v in wcs.world_to_pixel(at))
        half = int(half_arcsec / float(proj_plane_pixel_scales(wcs).mean() * 3600))
        ny, nx = h["SCI"].shape
        ys = slice(max(int(y) - half, 0), min(int(y) + half, ny))
        xs = slice(max(int(x) - half, 0), min(int(x) + half, nx))
        cut = np.array(h["SCI"].section[ys, xs], dtype=np.float64)
    return np.nan_to_num(cut)


def _best_fits(cat: Catalogue, k: int) -> list[npt.NDArray[np.float64]]:
    lib = cat.flux[cat.library]
    chi = chi2_to_library(cat.flux[k : k + 1], cat.err[k : k + 1], lib)[0]
    chi[cat.coords[k].separation(cat.coords[cat.library]).arcsec < APART_ARCSEC] = (
        np.inf
    )
    w = 1.0 / cat.err[k] ** 2
    return [
        lib[b] * float(np.sum(cat.flux[k] * lib[b] * w) / np.sum(lib[b] ** 2 * w))
        for b in np.argsort(chi)[:NEIGHBOUR]
    ]


def _sheet(cat: Catalogue, k: int, *, raw: Path, rank: int, out: Path) -> None:
    import matplotlib as mpl  # noqa: PLC0415 matplotlib is slow to load

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 matplotlib is slow to load

    mosaic = cat.block[k].rsplit("_b", 1)[0]
    fig = plt.figure(figsize=(24, 6), dpi=80)
    for i, (band, _, _) in enumerate(BANDS):
        a = fig.add_subplot(2, 8, i + 1)
        s = _stamp(_band_file(raw, mosaic, band), cat.coords[k])
        z = np.arcsinh((s - np.median(s)) / _mad(s) / 3)
        a.imshow(z, origin="lower", cmap="gray", vmin=-1, vmax=4)
        a.set_title(band.upper())
        a.set_xticks([])
        a.set_yticks([])
    a = fig.add_subplot(2, 1, 2)
    wave = [w for _, w, _ in BANDS]
    a.errorbar(wave, cat.flux[k], yerr=cat.err[k], fmt="o", color="k", label="source")
    for n, fit in enumerate(_best_fits(cat, k)):
        a.plot(wave, fit, "-", alpha=0.6, label="library fits" if n == 0 else None)
    a.set_xlabel("wavelength (micron)")
    a.set_ylabel("flux in the aperture")
    a.legend()
    at = cat.coords[k]
    fig.suptitle(
        f"#{rank}  {cat.block[k]}  score {cat.scores[k]:.0f}  "
        f"{at.ra.deg:.5f} {at.dec.deg:+.5f}"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _write_csv(cat: Catalogue, out: Path) -> None:
    names = [b for b, _, _ in BANDS]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ra",
            "dec",
            "block",
            "det_sn",
            "conc",
            "score",
            "in_library",
            *[f"{b}_flux" for b in names],
            *[f"{b}_err" for b in names],
        ])
        in_lib = np.zeros(len(cat.flux), dtype=bool)
        in_lib[cat.library] = True
        for k in range(len(cat.flux)):
            w.writerow([
                f"{cat.coords[k].ra.deg:.6f}",
                f"{cat.coords[k].dec.deg:.6f}",
                cat.block[k],
                f"{cat.det_sn[k]:.1f}",
                f"{cat.conc[k]:.3f}",
                f"{cat.scores[k]:.2f}",
                int(in_lib[k]),
                *[f"{x:.5g}" for x in cat.flux[k]],
                *[f"{x:.3g}" for x in cat.err[k]],
            ])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jwst_peculiar.colour",
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0],
    )
    p.add_argument("raw", type=Path)
    p.add_argument("stars", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("mosaics", nargs="+")
    p.add_argument("--no_catalogues", action="store_true", help="skip SIMBAD and NED")
    return p


def main(argv: list[str] | None = None) -> int:
    """Write OUT/sources.csv, OUT/index.html and a sheet for each of the top sources.

    Returns:
        0.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    cat = build(
        [f for m in args.mosaics for f in measure_mosaic(args.raw, m)], args.stars
    )
    logger.info("%d sources kept, %d in the library", len(cat.flux), len(cat.library))
    _write_csv(cat, args.out / "sources.csv")
    parts = []
    order = np.argsort(-cat.scores)
    sharp = cat.conc > SHARPEST
    for rank, k in enumerate(order[~sharp[order]][:TOP], start=1):
        sheet = f"colour{rank:03d}.png"
        _sheet(cat, int(k), raw=args.raw, rank=rank, out=args.out / sheet)
        near = {} if args.no_catalogues else catalogue_note(cat.coords[k])
        notes = "<br>".join(html.escape(f"{c}: {n}") for c, n in near.items())
        parts.append(
            f"<h3>#{rank} {html.escape(cat.block[k])}, score {cat.scores[k]:.0f}</h3>"
            f"<p>{notes}</p><img src='{sheet}' width='100%'>"
        )
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Colour List</title>"
        "<body style='font-family:sans-serif;background:#111;color:#ddd;margin:16px'>"
        f"<p>{len(cat.flux)} sources, {len(cat.library)} in the library, "
        f"{int(sharp.sum())} sharper than a star and not ranked.</p>"
        f"{''.join(parts)}</body>",
        encoding="utf-8",
    )
    summary = {
        "sources": len(cat.flux),
        "library": len(cat.library),
        "sharper_than_a_star": int(sharp.sum()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
