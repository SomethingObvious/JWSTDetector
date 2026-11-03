"""Cuts FITS mosaics into PNG tiles for the detector, one mosaic at a time.

Any image with a celestial WCS works through --sensor fits, from any telescope, as one
grey band per file or three files combined into colour with --rgb. The miri and
nircam sensors are presets for JWST archive and COSMOS-Web file layouts.

Each mosaic stays memmapped while it is cut, so memory stays flat however big the
mosaic is. The sky level, its noise and the stretch limits are measured once per
mosaic, and every threshold is expressed against those, which means the same flag
values work whatever units the mosaic is in and however deep the exposure went.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import logging
import re
import shutil
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS, FITSFixedWarning
from PIL import Image
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm

from jwstdetector.photometry import describe_plane

logger = logging.getLogger(__name__)


def find_image(
    hdul: fits.HDUList, names: Sequence[str], *, fallback: bool = True
) -> tuple[np.ndarray, fits.Header]:
    """Data and header of the first 2D image HDU with one of `names`.

    With `fallback` it settles for the first 2D image of any name, which suits a file
    holding a single image. Turn it off when looking for WHT beside SCI in one file, or
    a missing WHT quietly comes back as SCI.
    """
    wanted = [hdu for name in names for hdu in hdul if hdu.name == name.upper()]
    for hdu in wanted + (list(hdul) if fallback else []):
        if not hdu.is_image or hdu.data is None:
            continue
        data = hdu.data
        # Some mosaics are stored as (1, H, W), which is still just one image.
        while data.ndim > 2 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 2:
            return data, hdu.header
    raise ValueError(f"There is no 2D image called {' or '.join(names)} in {hdul.filename()}")


@dataclass
class MosaicStats:
    """Sky level, noise and stretch limits, measured once for a whole mosaic."""

    sky: float
    sigma: float
    lo: float
    hi: float


def subsample(data: np.ndarray, max_values: int = 1 << 24) -> np.ndarray:
    """Up to `max_values` evenly spaced pixels that hold real data.

    Mosaics can run to several GB, so the statistics use a strided view of the memmap.
    Pixels outside the footprint are left out too. The JWST pipeline fills them with
    NaN but COSMOS-Web uses 0, and when most of a rotated mosaic is zero fill the
    median and MAD both come out as 0, after which every tile of blank sky passes
    the emptiness cut.
    """
    h, w = data.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_values))))
    sample = np.asarray(data[::step, ::step], dtype=np.float32).reshape(-1)
    return sample[np.isfinite(sample) & (sample != 0)]


def measure_mosaic(data: np.ndarray, p_lo: float, p_hi: float) -> MosaicStats:
    """Median sky, MAD noise and percentile stretch limits for one mosaic."""
    sample = subsample(data)
    if sample.size == 0:
        return MosaicStats(sky=0.0, sigma=0.0, lo=0.0, hi=1.0)

    sky = float(np.median(sample))
    # The MAD scaled to a Gaussian sigma, which unlike std doesn't see the sources.
    sigma = float(np.median(np.abs(sample - sky)) * 1.4826)
    lo, hi = (float(v) for v in np.percentile(sample, [p_lo, p_hi]))
    if hi <= lo:
        lo, hi = float(sample.min()), float(sample.max()) + 1e-6
    return MosaicStats(sky=sky, sigma=sigma, lo=lo, hi=hi)


def asinh_to_uint8(
    img: np.ndarray, lo: float, hi: float, asinh_q: float = 10.0, fill: float = 0.0
) -> np.ndarray:
    """Flux to 8 bits through an asinh stretch between fixed limits, with NaN set to `fill`.

    The limits are arguments on purpose. Stretching each tile between its own
    percentiles brings faint sky and a bright source up to the same range, which
    hides the very difference the detector is looking for. Holes get the sky level
    so they don't turn into black shapes that the detector then flags, and so does
    zero fill, which is how COSMOS-Web marks the same thing.
    """
    img = np.asarray(img, dtype=np.float32)
    img = np.where(np.isfinite(img) & (img != 0), img, np.float32(fill))
    x = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    # Rounding rather than truncating, which would put the upper limit itself at 254.
    return np.rint(255.0 * np.arcsinh(asinh_q * x) / np.arcsinh(asinh_q)).astype(np.uint8)


def has_signal(
    cut: np.ndarray, stats: MosaicStats, source_sigma: float = 5.0, min_frac: float = 2e-4
) -> bool:
    """True when enough of the tile rises above the sky to be worth embedding.

    It reads raw flux against the mosaic's noise, so the default works for MIRI and
    NIRCam alike, as pure sky puts next to nothing above 5 sigma. The fraction is of
    the whole tile rather than its finite pixels, because a sliver of coverage at a
    mosaic edge would otherwise pass on a single noisy pixel.
    """
    finite = np.isfinite(cut)
    if not np.any(finite & (cut != 0)):
        return False
    if stats.sigma <= 0.0:
        return bool(np.any(finite & (cut != stats.sky)))
    above = finite & (cut > stats.sky + source_sigma * stats.sigma)
    return float(above.sum()) / cut.size >= min_frac


def covered_fraction(wht: np.ndarray) -> float:
    """Share of a window with a positive weight, counting enclosed holes as covered.

    A saturated star core has zero weight inside the footprint. Counting it as missing
    would throw away every tile over the star, and edge tiles are what the check is
    for. A gap that reaches the window's border isn't enclosed, so it still counts.
    """
    covered = np.isfinite(wht) & (wht > 0)
    return float(binary_fill_holes(covered).mean())


def noise_factor(wht: np.ndarray, typical: float) -> np.ndarray:
    """sqrt(weight / typical) capped at 1, in 255ths, for every pixel of a window.

    Multiplying the flux above the sky by it gives every part of a mosaic the noise of
    its typical depth, which is how a detection image is weighted. Without it, sky
    covered by one exposure in place of three renders 1.7 times noisier, and on SMACS
    0723 tiles reaching into those strips took the top four places. Pixels with no
    weight, like a saturated star's core, keep their flux.
    """
    w = np.asarray(wht, dtype=np.float32)
    f = np.sqrt(np.clip(w / typical, 0.0, 1.0))
    f = np.where(np.isfinite(w) & (w > 0), f, 1.0)
    return np.round(f * 255).astype(np.uint8)


def equalise(x: np.ndarray, sky: float, factor: np.ndarray | None) -> np.ndarray:
    """Flux above `sky` scaled by a noise_factor, leaving NaN and zero fill as they are."""
    x = np.asarray(x, dtype=np.float32)
    if factor is None:
        return x
    f = factor.astype(np.float32) / 255
    return np.where(np.isfinite(x) & (x != 0), (x - sky) * f + sky, x).astype(np.float32)


def typical_weights(whts: Sequence[np.ndarray], starts_y, starts_x, crop: int) -> list[float]:
    """The median weight of each map over up to 64 tile windows spread across the mosaic.

    Windows, since a reprojected plane can only be read a window at a time."""
    windows = [(y, x) for y in starts_y for x in starts_x]
    windows = windows[:: max(1, len(windows) // 64)]
    typical = []
    for w in whts:
        medians = []
        for y, x in windows:
            part = np.asarray(w[y : y + crop, x : x + crop])[::4, ::4]
            part = part[np.isfinite(part) & (part > 0)]
            if part.size:
                medians.append(np.median(part))
        typical.append(float(np.median(medians)) if medians else 1.0)
    return typical


def tile_starts(length: int, crop: int, stride: int) -> list[int]:
    """Where crops start along one axis.

    When the stride doesn't land on the far edge, one more crop goes flush against it,
    as otherwise a strip up to a stride wide at the right and bottom of every mosaic
    never gets looked at.
    """
    if length < crop:
        return []
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


_TILE_FALLBACK_RE = re.compile(r"(?:^|[_\-])([A-Z]\d{1,2})(?:[_\-]|$)")


def infer_tile_id_from_filename(path: str, tile_regex: str | None = None) -> str | None:
    base = Path(path).name
    if tile_regex:
        m = re.search(tile_regex, base)
        if m:
            return m.group(1) if m.groups() else m.group(0)
    m = _TILE_FALLBACK_RE.search(base)
    return m.group(1) if m else None


def _safe_float_tag(x: float, ndp: int = 6) -> str:
    s = f"{x:.{ndp}f}"
    return "m" + s[1:] if s.startswith("-") else s


def celestial_wcs(header: fits.Header, source: str, required: bool = True) -> WCS | None:
    """The header's sky WCS, or None for a header without one unless it's `required`."""
    with warnings.catch_warnings():
        # JWST headers get a few harmless date and unit fixes that astropy warns about.
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header).celestial
    if not wcs.has_celestial:
        if required:
            raise ValueError(f"{source} has no sky coordinates in its header. Drop --wcs_in_name.")
        return None
    return wcs


def make_output_name(
    tile_id: str,
    cx: float,
    cy: float,
    global_idx: int,
    wcs_ra_dec: tuple[float, float] | None = None,
) -> str:
    parts = [tile_id, f"cx{round(cx):06d}", f"cy{round(cy):06d}"]
    if wcs_ra_dec is not None:
        ra, dec = wcs_ra_dec
        parts += [f"ra{_safe_float_tag(ra)}", f"dec{_safe_float_tag(dec)}"]
    parts.append(f"i{global_idx:08d}")
    return "_".join(parts) + ".png"


def cut_tiles(
    args: argparse.Namespace,
    tile_id: str,
    signal: np.ndarray,
    wht: np.ndarray | Sequence[np.ndarray] | None,
    wcs: WCS | None,
    query_dir: Path,
    global_idx: int,
    colour: Callable[[tuple[slice, slice]], np.ndarray] | None = None,
    planes: Sequence[np.ndarray] | None = None,
    records: list[dict] | None = None,
) -> tuple[int, int, int, int]:
    """Cut, filter and save the tiles of one mosaic.

    `signal` is the raw flux that the emptiness cut and the grey stretch read, and
    `colour` renders the saved tile for a window in place of the grey stretch. With
    several weight maps a window has to be covered in all of them. `planes` are the
    float planes a --save_cutouts FITS file holds, `signal` alone by default, and
    each saved tile appends its manifest row to `records`.
    Returns the next global index and the saved, empty and low-WHT counts.
    """
    whts = [] if wht is None else list(wht) if isinstance(wht, (list, tuple)) else [wht]
    for w in whts:
        if w.shape != signal.shape:
            raise ValueError(
                f"[{tile_id}] the weight map is {w.shape[1]}x{w.shape[0]} but the image is "
                f"{signal.shape[1]}x{signal.shape[0]}, so they can't be from the same mosaic"
            )
    crop = int(args.tile_size / args.upscale)
    stride = args.stride if args.stride is not None else max(1, crop // 2)
    stats = measure_mosaic(signal, args.p_lo, args.p_hi)
    logger.info(
        "[%s] sky=%.4g sigma=%.4g stretch=[%.4g, %.4g]",
        tile_id,
        stats.sky,
        stats.sigma,
        stats.lo,
        stats.hi,
    )

    saved = skipped_empty = skipped_wht = 0
    h, w = signal.shape
    typical = None
    if getattr(args, "noise_eq", False) and whts and len(whts) == len(planes or [signal]):
        typical = typical_weights(
            whts, tile_starts(h, crop, stride), tile_starts(w, crop, stride), crop
        )
    for y in tile_starts(h, crop, stride):
        for x in tile_starts(w, crop, stride):
            window = np.s_[y : y + crop, x : x + crop]
            if args.min_wht_frac > 0 and any(
                covered_fraction(np.asarray(w[window])) < args.min_wht_frac for w in whts
            ):
                skipped_wht += 1
                continue

            # Judge emptiness on raw flux, before a stretch flattens the difference
            # between blank sky and a tile with something in it.
            raw = np.asarray(signal[window], dtype=np.float32)
            if not has_signal(raw, stats, args.source_sigma, args.min_source_frac):
                skipped_empty += 1
                continue

            if not args.dry_run:
                noise = None
                if typical is not None:
                    noise = [noise_factor(m[window], t) for m, t in zip(whts, typical, strict=True)]
                if colour is not None:
                    tile = colour(window, noise)
                else:
                    s = stats
                    if args.stretch_scope == "tile":
                        s = measure_mosaic(raw, args.p_lo, args.p_hi)
                    flux = equalise(raw, s.sky, noise[0] if noise else None)
                    tile = asinh_to_uint8(flux, s.lo, s.hi, args.asinh_q, fill=s.sky)
                cx, cy = x + (crop - 1) / 2, y + (crop - 1) / 2
                radec = None
                if wcs is not None:
                    ra, dec = wcs.all_pix2world(cx, cy, 0)
                    radec = (float(ra), float(dec))
                # FITS row 1 is the bottom of the sky, so it goes at the bottom of the PNG.
                img = Image.fromarray(np.ascontiguousarray(tile[::-1]))
                if img.size != (args.tile_size, args.tile_size):
                    img = img.resize((args.tile_size, args.tile_size), Image.Resampling.BICUBIC)
                name = make_output_name(
                    tile_id, cx, cy, global_idx, radec if args.wcs_in_name else None
                )
                img.save(query_dir / name)
                cutout = None
                if getattr(args, "save_cutouts", False):
                    cutout = Path("cutouts") / f"{Path(name).stem}.fits"
                    write_cutout(
                        query_dir.parent / cutout,
                        planes or [signal],
                        window,
                        wcs,
                        args.bunit,
                        noise,
                    )
                if records is not None:
                    records.append(
                        {
                            "key": name,
                            "mosaic": tile_id,
                            "x0": x,
                            "y0": y,
                            "size": crop,
                            "ra": f"{radec[0]:.7f}" if radec else "",
                            "dec": f"{radec[1]:.7f}" if radec else "",
                            "cutout": cutout.as_posix() if cutout else "",
                        }
                    )
            global_idx += 1
            saved += 1

    logger.info(
        "[%s] %s %d tiles, skipped %d as empty and %d for low WHT",
        tile_id,
        "would save" if args.dry_run else "saved",
        saved,
        skipped_empty,
        skipped_wht,
    )
    return global_idx, saved, skipped_empty, skipped_wht


def write_cutout(
    path: Path,
    planes: Sequence[np.ndarray],
    window: tuple[slice, slice],
    wcs: WCS | None,
    bunit: str | None,
    noise: Sequence[np.ndarray] | None = None,
) -> None:
    """The raw float pixels under one tile, as a FITS file with its own WCS.

    Zero fill becomes NaN, which is what FITS readers take as no data. The noise
    factors the PNG was rendered with go in a NOISEEQ extension, so render_cutout can
    make the same PNG again.
    """
    data = np.stack([np.asarray(p[window], dtype=np.float32) for p in planes])
    data[data == 0] = np.nan
    header = fits.Header()
    if wcs is not None:
        ys, xs = window
        sub_wcs = wcs.deepcopy()
        sub_wcs.wcs.crpix = [wcs.wcs.crpix[0] - xs.start, wcs.wcs.crpix[1] - ys.start]
        header.update(sub_wcs.to_header())
    if bunit:
        header["BUNIT"] = bunit
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus = [fits.PrimaryHDU(data[0] if len(planes) == 1 else data, header=header)]
    if noise is not None:
        factors = np.stack(noise)
        hdus.append(fits.ImageHDU(factors[0] if len(noise) == 1 else factors, name="NOISEEQ"))
        hdus[-1].header["COMMENT"] = "sqrt(weight / typical weight) capped at 1, times 255"
    fits.HDUList(hdus).writeto(path, overwrite=True)


def describe_mosaic(
    out: Path,
    tile_id: str,
    signal: np.ndarray,
    wcs: WCS | None,
    headers: Sequence[fits.Header],
    filters: Sequence[str],
    preview: Callable[[np.ndarray, np.ndarray], np.ndarray],
    render: dict | None = None,
    bunit: str | None = None,
    planes: Sequence[dict] | None = None,
) -> None:
    """mosaics/<id>.json and a preview PNG, which the web viewer reads.

    `render` records how the tiles were stretched, which is what lets render_cutout
    turn a cutout back into the same PNG after the benchmark adds a source to it.
    """
    h, w = signal.shape
    step = max(1, int(np.ceil(max(h, w) / 2048)))
    ys, xs = np.arange(0, h, step), np.arange(0, w, step)
    # Top row first, so the preview is the right way up like the tiles.
    Image.fromarray(preview(ys[::-1], xs)).save(out / "mosaics" / f"{tile_id}.png")

    meta: dict = {
        "shape": [h, w],
        "preview_scale": len(xs) / w,
        "filters": [f.upper() for f in filters],
    }
    if wcs is not None:
        cd = np.diag(wcs.wcs.cdelt) @ wcs.wcs.get_pc()
        meta["wcs"] = {
            "CTYPE1": wcs.wcs.ctype[0],
            "CTYPE2": wcs.wcs.ctype[1],
            "CRPIX1": float(wcs.wcs.crpix[0]),
            "CRPIX2": float(wcs.wcs.crpix[1]),
            "CRVAL1": float(wcs.wcs.crval[0]),
            "CRVAL2": float(wcs.wcs.crval[1]),
            "CD1_1": float(cd[0, 0]),
            "CD1_2": float(cd[0, 1]),
            "CD2_1": float(cd[1, 0]),
            "CD2_2": float(cd[1, 1]),
        }
        scale = np.sqrt(abs(np.linalg.det(cd))) * 3600
        meta["pixel_scale_arcsec"] = round(float(scale), 5)
    cards = {}
    for header in headers:
        for card in header.cards:
            if card.keyword and card.keyword not in ("HISTORY", "COMMENT", "") and len(cards) < 400:
                cards.setdefault(card.keyword, str(card.value))
    meta["header"] = cards
    if bunit:
        meta["bunit"] = bunit
    if render is not None:
        meta["render"] = render
    if planes:
        meta["planes"] = list(planes)
    (out / "mosaics" / f"{tile_id}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def pixel_scale(wcs: WCS | None) -> float | None:
    """Arcseconds per pixel, from the area of one pixel."""
    if wcs is None:
        return None
    cd = np.diag(wcs.wcs.cdelt) @ wcs.wcs.get_pc()
    return float(np.sqrt(abs(np.linalg.det(cd))) * 3600)


def plane_info(headers, path: str, own: WCS | None, target: WCS | None, filt: str = "") -> dict:
    """describe_plane for a file that may have been reprojected onto `target`'s grid.

    Interpolating keeps a pixel's value, so one unit of it now covers the target's
    pixel area. The area ratio fixes surface brightness and per-pixel units alike.
    """
    own_scale, target_scale = pixel_scale(own), pixel_scale(target or own)
    info = describe_plane(headers, path, own_scale, filt or None)
    if info["jy_per_unit"] is not None and own_scale and target_scale:
        info["jy_per_unit"] *= (target_scale / own_scale) ** 2
    return info


def asinh_render(stats: MosaicStats, args: argparse.Namespace) -> dict | None:
    """The render record for a grey mosaic, or None when each tile has its own stretch."""
    if args.stretch_scope != "mosaic":
        return None
    return {
        "kind": "asinh",
        "lo": stats.lo,
        "hi": stats.hi,
        "q": args.asinh_q,
        "sky": stats.sky,
        "tile_size": args.tile_size,
    }


def render_cutout(data: np.ndarray, render: dict, noise: np.ndarray | None = None) -> np.ndarray:
    """The 8-bit tile `prep` saves for these float pixels, top row first, at its saved size.

    `data` is a cutout as write_cutout leaves it, (H, W) or (3, H, W) with FITS row 1
    first, and `render` is the record mosaics/<id>.json keeps of how its mosaic was
    stretched. Both stretches send NaN and zero to the sky level, so the NaN that
    write_cutout puts in holes renders exactly as the zero fill did.
    """
    if render["kind"] == "lupton":
        levels = [
            MosaicStats(sky=v["sky"], sigma=0.0, lo=0.0, hi=v["hi"]) for v in render["levels"]
        ]
        opts = SimpleNamespace(stretch=render["stretch"], Q=render["Q"])
        noise = [None] * len(levels) if noise is None else list(noise)
        planes = [equalise(d, v.sky, n) for d, v, n in zip(data, levels, noise, strict=True)]
        tile = lupton(opts, planes, levels)
    elif render["kind"] == "asinh":
        flux = equalise(data, render["sky"], noise)
        tile = asinh_to_uint8(flux, render["lo"], render["hi"], render["q"], fill=render["sky"])
    else:
        raise ValueError(f"Unknown render kind {render['kind']!r}")
    img = Image.fromarray(np.ascontiguousarray(tile[::-1]))
    size = int(render["tile_size"])
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(img)


class Reprojected:
    """One plane of a mosaic seen on another plane's pixel grid, a window at a time.

    JWST level-3 mosaics of one field come out on slightly different grids per filter
    (a pixel wider here, twice the pixel scale for the short-wave channel there), and
    stacking the arrays as they are would put different sky in each colour.
    """

    def __init__(self, data: np.ndarray, wcs: WCS, target: WCS, shape: tuple[int, int]):
        self.data, self.wcs, self.target, self.shape = data, wcs, target, shape

    def __getitem__(self, window: tuple[slice, slice]) -> np.ndarray:
        from reproject import reproject_interp

        ys, xs = window
        out_wcs = self.target[ys, xs]
        out_shape = (ys.stop - ys.start, xs.stop - xs.start)
        # Only the part of this plane under the window gets read off the memmap.
        corners_y = [ys.start, ys.start, ys.stop, ys.stop]
        corners_x = [xs.start, xs.stop, xs.start, xs.stop]
        ra, dec = self.target.all_pix2world(corners_x, corners_y, 0)
        px, py = self.wcs.all_world2pix(ra, dec, 0)
        h, w = self.data.shape
        x0, x1 = max(0, int(np.floor(px.min())) - 4), min(w, int(np.ceil(px.max())) + 5)
        y0, y1 = max(0, int(np.floor(py.min())) - 4), min(h, int(np.ceil(py.max())) + 5)
        if x1 <= x0 or y1 <= y0:
            return np.full(out_shape, np.nan, np.float32)
        src = np.asarray(self.data[y0:y1, x0:x1], dtype=np.float32)
        out, _ = reproject_interp((src, self.wcs[y0:y1, x0:x1]), out_wcs, out_shape, order=1)
        return out.astype(np.float32)


def on_grid(data: np.ndarray, wcs: WCS, target: WCS, shape: tuple[int, int]):
    """`data` itself when it already shares the target grid, otherwise a Reprojected view."""
    same = (
        data.shape == shape
        and np.allclose(wcs.wcs.crpix, target.wcs.crpix)
        and np.allclose(wcs.pixel_scale_matrix, target.pixel_scale_matrix, rtol=1e-6, atol=0)
        and np.allclose(wcs.wcs.crval, target.wcs.crval, rtol=0, atol=1e-9)
    )
    return data if same else Reprojected(data, wcs, target, shape)


@dataclass
class MosaicSpec:
    tile_id: str
    sci_path: str
    wht_path: str | None = None


class Sensor:
    name: str = "base"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        raise NotImplementedError

    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        query_dir: Path,
        global_idx: int,
        records: list[dict],
    ) -> int:
        raise NotImplementedError


class MiriSensor(Sensor):
    """One mosaic per file, with WHT either inside it (i2d) or in a *wht.fits beside it."""

    name = "miri"
    _SCI2WHT_RE = re.compile(r"(?i)sci\.fits$")

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--pattern", default="*sci.fits", help="Glob for the science mosaics.")
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")

    @classmethod
    def sci_to_wht_path(cls, sci_path: str) -> str:
        """The *wht.fits beside a *sci.fits, or the file itself when it isn't named that way."""
        if cls._SCI2WHT_RE.search(sci_path):
            return cls._SCI2WHT_RE.sub("wht.fits", sci_path)
        return sci_path

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        sci_paths = sorted(str(p) for p in Path(args.indir).glob(args.pattern) if p.is_file())
        if not sci_paths:
            raise FileNotFoundError(f"Nothing in {args.indir} matches {args.pattern}")

        for sci_path in sci_paths:
            # JWST archive names carry no tile id, and the file stem at least says
            # which mosaic a tile came from.
            tile_id = (
                args.tile_id
                or infer_tile_id_from_filename(sci_path, args.tile_regex)
                or Path(sci_path).stem
            )
            wht_path = None
            if args.min_wht_frac > 0:
                wht_path = self.sci_to_wht_path(sci_path)
                if not Path(wht_path).is_file():
                    raise FileNotFoundError(
                        f"--min_wht_frac needs a weight map for {sci_path}, and {wht_path} "
                        "isn't there. Pass --min_wht_frac 0 to tile it without one."
                    )
            yield MosaicSpec(tile_id=tile_id, sci_path=sci_path, wht_path=wht_path)

    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        query_dir: Path,
        global_idx: int,
        records: list[dict],
    ) -> int:
        # Everything is cut while the files are open, since the arrays are memmaps.
        with contextlib.ExitStack() as stack:
            sci_hdul = stack.enter_context(fits.open(spec.sci_path, memmap=True))
            sci, header = find_image(sci_hdul, (args.sci_ext, "SCI"))
            wht = None
            if spec.wht_path == spec.sci_path:
                try:
                    wht, _ = find_image(sci_hdul, (args.wht_ext, "WHT"), fallback=False)
                except ValueError:
                    raise ValueError(
                        f"{spec.sci_path} has no {args.wht_ext} extension for --min_wht_frac. "
                        "Pass --min_wht_frac 0 to tile it without one."
                    ) from None
            elif spec.wht_path is not None:
                wht_hdul = stack.enter_context(fits.open(spec.wht_path, memmap=True))
                wht, _ = find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))
            wcs = celestial_wcs(header, spec.sci_path, required=args.wcs_in_name)
            args.bunit = header.get("BUNIT")
            if not args.dry_run:
                stats = measure_mosaic(sci, args.p_lo, args.p_hi)
                describe_mosaic(
                    query_dir.parent,
                    spec.tile_id,
                    sci,
                    wcs,
                    [sci_hdul[0].header, header],
                    [str(sci_hdul[0].header.get("FILTER", header.get("FILTER", "")))],
                    lambda ys, xs: asinh_to_uint8(
                        sci[ys[:, None], xs[None, :]], stats.lo, stats.hi, args.asinh_q, stats.sky
                    ),
                    render=asinh_render(stats, args),
                    bunit=args.bunit,
                    planes=[plane_info([sci_hdul[0].header, header], spec.sci_path, wcs, wcs)],
                )
            return cut_tiles(
                args, spec.tile_id, sci, wht, wcs, query_dir, global_idx, records=records
            )[0]


class NircamSensor(Sensor):
    """COSMOS-Web style mosaics, one file per filter, combined into colour tiles.

    The first filter is the detection plane for the emptiness cut. Colour is made one
    tile at a time, which gives the same pixels as the whole mosaic at once (Lupton's
    mapping works pixel by pixel) without holding three mosaics and their float64
    copies in RAM.
    """

    name = "nircam"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--pixel_scale", type=str, default="60mas")
        p.add_argument("--version", type=str, default="v1.0")
        p.add_argument("--ext", type=str, default="i2d")
        p.add_argument(
            "--pattern",
            type=str,
            default="mosaic_nircam_{filter}_COSMOS-Web_{pixel_scale}_{tile}_{version}_{ext}.fits",
        )
        p.add_argument("--rgb_filters", nargs="+", default=["f444w", "f277w", "f150w"])
        p.add_argument("--tiles", nargs="*", default=None)
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")
        p.add_argument(
            "--stretch",
            type=float,
            default=0.05,
            help="Lupton stretch, in units of each filter's sky to --p_hi range.",
        )
        p.add_argument("--Q", type=float, default=10.0, help="Lupton softening.")

    def _infer_tiles_from_dir(self, indir: str) -> list[str]:
        names = (infer_tile_id_from_filename(str(p)) for p in Path(indir).glob("*.fits"))
        tiles = sorted({t for t in names if t})
        if not tiles:
            raise ValueError(f"Couldn't find a tile id in any file name in {indir}. Pass --tiles.")
        return tiles

    def _build_path(self, args: argparse.Namespace, tile_id: str, filt: str) -> str:
        fname = args.pattern.format(
            filter=filt,
            pixel_scale=args.pixel_scale,
            tile=tile_id,
            version=args.version,
            ext=args.ext,
        )
        return str(Path(args.indir) / fname)

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        for tile_id in args.tiles or self._infer_tiles_from_dir(args.indir):
            yield MosaicSpec(tile_id=tile_id, sci_path="")

    def _inputs(self, args: argparse.Namespace, spec: MosaicSpec) -> list[tuple[str, str]]:
        if len(args.rgb_filters) not in (1, 3):
            raise ValueError(f"--rgb_filters takes 1 or 3 filters, not {len(args.rgb_filters)}")
        out = []
        for filt in args.rgb_filters:
            path = self._build_path(args, spec.tile_id, filt)
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"There is no {filt} mosaic for tile {spec.tile_id} at {path}"
                )
            out.append((filt, path))
        return out

    def _weights(self, args, stack, hdul, path: str, filt: str) -> np.ndarray:
        try:
            return find_image(hdul, (args.wht_ext, "WHT"), fallback=False)[0]
        except ValueError:
            pass
        # COSMOS-Web ships the weights as a separate file named like the science one.
        wht_path = Path(path.replace(f"_{args.ext}.fits", "_wht.fits"))
        if wht_path == Path(path) or not wht_path.is_file():
            raise FileNotFoundError(
                f"--min_wht_frac needs a weight map for {filt}, and {path} has no "
                f"{args.wht_ext} extension and there is no {wht_path.name} beside it. "
                "Pass --min_wht_frac 0 to tile it without one."
            )
        wht_hdul = stack.enter_context(fits.open(wht_path, memmap=True))
        return find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))[0]

    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        query_dir: Path,
        global_idx: int,
        records: list[dict],
    ) -> int:
        inputs = self._inputs(args, spec)
        with contextlib.ExitStack() as stack:
            planes, whts, headers, target, levels, infos = [], [], [], None, [], []
            for filt, path in inputs:
                hdul = stack.enter_context(fits.open(path, memmap=True))
                data, hdr = find_image(hdul, (args.sci_ext, "SCI"))
                levels.append(measure_mosaic(data, args.p_lo, args.p_hi))
                wht = (
                    self._weights(args, stack, hdul, path, filt) if args.min_wht_frac > 0 else None
                )
                if not infos:
                    # The first filter sets the pixel grid and is the detection plane.
                    target = celestial_wcs(hdr, path, required=args.wcs_in_name)
                    shape = data.shape
                    headers = [hdul[0].header, hdr]
                    args.bunit = hdr.get("BUNIT")
                    own = target
                elif target is None:
                    # Without a WCS the files have to share a grid already.
                    own = None
                    if data.shape != shape:
                        raise ValueError(
                            f"{path} is {data.shape[1]}x{data.shape[0]} and the first file is "
                            f"{shape[1]}x{shape[0]}, and without a WCS they can't be aligned"
                        )
                else:
                    own = celestial_wcs(hdr, path)
                    data = on_grid(data, own, target, shape)
                    if wht is not None:
                        wht = on_grid(wht, own, target, shape)
                infos.append(plane_info([hdul[0].header, hdr], path, own, target, filt))
                planes.append(data)
                if wht is not None:
                    whts.append(wht)

            colour = None
            if len(planes) == 3:

                def colour(window: tuple[slice, slice], noise=None) -> np.ndarray:
                    noise = noise or [None] * len(planes)
                    flux = [
                        equalise(p[window], v.sky, n)
                        for p, v, n in zip(planes, levels, noise, strict=True)
                    ]
                    return lupton(args, flux, levels)

            signal = planes[0]
            if not args.dry_run:
                describe_mosaic(
                    query_dir.parent,
                    spec.tile_id,
                    signal,
                    target,
                    headers,
                    [i["filter"] for i in infos],
                    lambda ys, xs: self._preview(args, planes, levels, ys, xs),
                    render=self._render(args, levels),
                    bunit=args.bunit,
                    planes=infos,
                )
            return cut_tiles(
                args,
                spec.tile_id,
                signal,
                whts,
                target,
                query_dir,
                global_idx,
                colour,
                planes=planes,
                records=records,
            )[0]

    @staticmethod
    def _render(args, levels) -> dict | None:
        if len(levels) == 1:
            return asinh_render(levels[0], args)
        return {
            "kind": "lupton",
            "stretch": args.stretch,
            "Q": args.Q,
            "levels": [{"sky": v.sky, "hi": v.hi} for v in levels],
            "tile_size": args.tile_size,
        }

    @staticmethod
    def _preview(args, planes, levels, ys, xs) -> np.ndarray:
        def sample(p):
            if isinstance(p, Reprojected):
                full = p[np.s_[0 : p.shape[0], 0 : p.shape[1]]]
                return full[ys[:, None], xs[None, :]]
            return np.nan_to_num(np.asarray(p[ys[:, None], xs[None, :]], np.float32))

        if len(planes) == 3:
            return lupton(args, [sample(p) for p in planes], levels)
        stats = levels[0]
        return asinh_to_uint8(sample(planes[0]), stats.lo, stats.hi, args.asinh_q, stats.sky)


def lupton(
    args: argparse.Namespace, planes: Sequence[np.ndarray], levels: Sequence[MosaicStats]
) -> np.ndarray:
    """Lupton colour from three planes, each first put on its own mosaic's sky and range.

    Pipeline mosaics keep their sky, and it differs a lot between filters (F444W sits
    far above F277W), so feeding raw flux to Lupton's mapping turns the whole field one
    colour. The levels are per mosaic, not per tile, so tiles stay comparable.
    """
    scaled = []
    for p, level in zip(planes, levels, strict=True):
        x = np.asarray(p, dtype=np.float32)
        x = np.where(np.isfinite(x) & (x != 0), x, level.sky)
        scaled.append((x - level.sky) / max(level.hi - level.sky, 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        return make_lupton_rgb(*scaled, stretch=args.stretch, Q=args.Q)


def _weight_path(path: str, pattern: str) -> Path:
    return Path(path).with_name(pattern.format(stem=Path(path).name.split(".")[0]))


class FitsSensor(NircamSensor):
    """Any FITS image with a celestial WCS, from any telescope.

    Each file matching --pattern in --indir, or each of --files, is cut as its own grey
    mosaic. --rgb takes three files of one field instead, red first, and cuts them as one
    colour mosaic on the first file's grid. The image is the --sci_ext extension when
    there is one and the first 2D image otherwise. A weight map is used when the file
    has a --wht_ext extension or a sibling named by --wht_pattern, and the coverage
    check is skipped with a warning when there's neither.
    """

    name = "fits"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--pattern", default="*.fits", help="Glob for the mosaics in --indir.")
        p.add_argument("--files", nargs="+", default=None, help="FITS files, one mosaic each.")
        p.add_argument(
            "--rgb",
            nargs=3,
            default=None,
            metavar="FILE",
            help="Red, green and blue files of one field.",
        )
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")
        p.add_argument(
            "--wht_pattern",
            default=None,
            help="The weight map's name from the image's, like {stem}_wht.fits. Weights in "
            "a --wht_ext extension are found without it.",
        )
        p.add_argument("--stretch", type=float, default=0.05, help="Lupton stretch for --rgb.")
        p.add_argument("--Q", type=float, default=10.0, help="Lupton softening for --rgb.")

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        if args.rgb:
            yield MosaicSpec(
                tile_id=args.tile_id or Path(args.rgb[0]).stem, sci_path="|".join(args.rgb)
            )
            return
        if args.files:
            paths = [str(p) for p in args.files]
        elif args.indir:
            paths = sorted(str(p) for p in Path(args.indir).glob(args.pattern) if p.is_file())
        else:
            raise SystemExit("--sensor fits needs --indir, --files or --rgb")
        if args.wht_pattern:
            weights = {str(_weight_path(p, args.wht_pattern)) for p in paths}
            paths = [p for p in paths if p not in weights]
        if not paths:
            raise FileNotFoundError(f"Nothing in {args.indir} matches {args.pattern}")
        seen: dict[str, str] = {}
        for path in paths:
            tile_id = args.tile_id if len(paths) == 1 and args.tile_id else None
            if not tile_id and args.tile_regex:
                tile_id = infer_tile_id_from_filename(path, args.tile_regex)
            tile_id = tile_id or Path(path).stem
            if tile_id in seen:
                raise SystemExit(
                    f"{seen[tile_id]} and {path} would both be mosaic {tile_id}. "
                    "Give a --tile_regex that tells them apart."
                )
            seen[tile_id] = path
            yield MosaicSpec(tile_id=tile_id, sci_path=path)

    def _inputs(self, args: argparse.Namespace, spec: MosaicSpec) -> list[tuple[str, str]]:  # noqa: ARG002
        paths = spec.sci_path.split("|")
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(f"There is no FITS file at {path}")
        return [("", path) for path in paths]

    def _weights(self, args, stack, hdul, path: str, filt: str) -> np.ndarray | None:  # noqa: ARG002
        try:
            return find_image(hdul, (args.wht_ext, "WHT"), fallback=False)[0]
        except ValueError:
            pass
        if args.wht_pattern:
            wht_path = _weight_path(path, args.wht_pattern)
            if wht_path.is_file():
                wht_hdul = stack.enter_context(fits.open(wht_path, memmap=True))
                return find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))[0]
        logger.warning(
            "%s has no weight map, so tiles on the edge of its coverage aren't dropped. "
            "Pass --wht_pattern if it has one beside it.",
            Path(path).name,
        )
        return None


SENSORS: dict[str, Sensor] = {"fits": FitsSensor(), "miri": MiriSensor(), "nircam": NircamSensor()}


def clear_previous_output(out: Path, overwrite: bool) -> None:
    """Stop when --out holds an earlier run's tiles, or clear them with --overwrite.

    Tile names carry a counter that starts at 0 every run, so a second run into the same
    folder leaves the first run's tiles beside its own, including exact copies under new
    names that the detector then treats as separate tiles vouching for each other.
    """
    old = [out / name for name in ("query", "cutouts", "mosaics", "manifest.csv")]
    old = [p for p in old if p.exists()]
    if not old:
        return
    if not overwrite:
        raise SystemExit(
            f"{out} already holds tiles from an earlier run. Pass --overwrite to clear them, "
            "or choose another --out."
        )
    for p in old:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sensor",
        choices=sorted(SENSORS),
        default="fits",
        help="fits for any image with a WCS, or the miri and nircam presets for JWST layouts.",
    )
    p.add_argument("--indir", default=None, help="Folder holding the FITS mosaics.")
    p.add_argument(
        "--out", default="datasets/jwst/object1", help="Output root. Tiles go in out/query/."
    )
    p.add_argument("--tile_id", default=None, help="Use this tile id instead of inferring one.")
    p.add_argument("--tile_regex", default=None, help="Regex that pulls the tile id from a name.")
    p.add_argument(
        "--wcs_in_name",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Put the RA and Dec of each tile centre in its file name.",
    )
    p.add_argument("--tile_size", type=int, default=518, help="Saved tile size in pixels.")
    p.add_argument(
        "--upscale", type=int, default=2, help="Crop tile_size/upscale pixels and resize up."
    )
    p.add_argument(
        "--stride", type=int, default=None, help="Step between crops. Half a crop by default."
    )

    p.add_argument(
        "--min_wht_frac",
        "--minwhtfrac",
        dest="min_wht_frac",
        type=float,
        default=1.0,
        help="Drop tiles where less than this fraction of pixels has a positive weight. "
        "Below 1, tiles on the edge of the coverage get in and rank near the top.",
    )
    p.add_argument(
        "--source_sigma",
        type=float,
        default=5.0,
        help="A pixel counts as signal this many noise sigmas above the sky.",
    )
    p.add_argument(
        "--min_source_frac",
        type=float,
        default=2e-4,
        help="Drop tiles with less than this fraction of signal pixels.",
    )
    # These measured the stretched PNG and are gone. They fail loudly in run() rather
    # than being ignored.
    p.add_argument("--mean_thresh", "--meanthresh", dest="mean_thresh", help=argparse.SUPPRESS)
    p.add_argument("--var_thresh", "--varthresh", dest="var_thresh", help=argparse.SUPPRESS)

    p.add_argument(
        "--noise_eq",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Scale the flux above the sky by sqrt(weight / typical weight), capped at 1, "
        "so shallow parts of a mosaic aren't noisier than the rest. Needs weight maps.",
    )
    p.add_argument(
        "--stretch_scope",
        choices=("mosaic", "tile"),
        default="mosaic",
        help="Measure the stretch once per mosaic, which keeps tiles comparable, or per tile.",
    )
    p.add_argument(
        "--p_lo", "--plo", dest="p_lo", type=float, default=1.0, help="Low stretch percentile."
    )
    p.add_argument(
        "--p_hi", "--phi", dest="p_hi", type=float, default=99.8, help="High stretch percentile."
    )
    p.add_argument(
        "--asinh_q", "--asinhq", dest="asinh_q", type=float, default=10.0, help="Asinh softening."
    )

    p.add_argument(
        "--gc_each_mosaic",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the garbage collector after each mosaic.",
    )
    p.add_argument(
        "--save_cutouts",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Also save each tile's raw float pixels as FITS with its WCS, for the web viewer.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear tiles left in --out by an earlier run instead of stopping.",
    )
    p.add_argument("--limit", type=int, default=None, help="Stop after this many mosaics.")
    p.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        default=False,
        action="store_true",
        help="Cut and filter but write nothing, for tuning the thresholds.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Each sensor has its own flags, so find the sensor first. This parser has no
    # --help, which lets -h reach the full parser and list the sensor's flags too.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sensor", choices=sorted(SENSORS), default="fits")
    sensor = pre.parse_known_args(argv)[0].sensor
    p = build_parser()
    SENSORS[sensor].add_args(p)
    return p.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.mean_thresh is not None or args.var_thresh is not None:
        raise SystemExit(
            "--mean_thresh and --var_thresh are gone. They measured the stretched PNG, where a\n"
            "per-tile stretch had already brought blank sky up to full brightness, so they kept\n"
            "empty tiles and threw out tiles holding a bright source. Use --source_sigma and\n"
            "--min_source_frac, which read raw flux."
        )
    if args.limit is not None and args.limit < 0:
        raise SystemExit(f"--limit can't be negative, got {args.limit}")
    if args.asinh_q <= 0:
        raise SystemExit(f"--asinh_q has to be above 0, got {args.asinh_q}")
    if args.sensor != "fits" and not args.indir:
        raise SystemExit(f"--sensor {args.sensor} needs --indir")

    out = Path(args.out)
    query_dir = out / "query"
    if not args.dry_run:
        clear_previous_output(out, args.overwrite)
        query_dir.mkdir(parents=True)
        (out / "mosaics").mkdir()
    sensor = SENSORS[args.sensor]
    specs = list(sensor.iter_mosaic_specs(args))[: args.limit]

    global_idx = 0
    records: list[dict] = []
    for spec in tqdm(specs, desc="Mosaics", unit="mosaic"):
        global_idx = sensor.process_one_mosaic(args, spec, query_dir, global_idx, records)
        if args.gc_each_mosaic:
            gc.collect()
    if not args.dry_run:
        with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["key"])
            writer.writeheader()
            writer.writerows(records)

    logger.info(
        "%s %d tiles from %d mosaics in %s",
        "Would write" if args.dry_run else "Wrote",
        global_idx,
        len(specs),
        query_dir,
    )


if __name__ == "__main__":
    run()
