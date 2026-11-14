"""Cuts FITS mosaics into PNG tiles for the detector, one mosaic at a time.

Any image with a celestial WCS works through --sensor fits, from any telescope, as one
grey band per file or three files combined into colour with --rgb. The miri and
nircam sensors are presets for JWST archive and COSMOS-Web file layouts.

Each mosaic stays memmapped while it is cut, so memory stays flat however big the
mosaic is. The sky level, its noise and the stretch limits are measured once per
mosaic, and every threshold is set against those, so the same flags work whatever
the mosaic's units and however deep the exposure went.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass, field, replace
import gc
import json
import logging
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Any, TypeAlias
import warnings

from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS, FITSFixedWarning
import numpy as np
from PIL import Image
from reproject import reproject_interp
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
from typing_extensions import override

from jwst_peculiar.photometry import describe_plane

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import numpy.typing as npt

    from jwst_peculiar.utils import FitsHDUList, FitsHeader, SkyWcs

    Array: TypeAlias = npt.NDArray[Any]
    Window: TypeAlias = tuple[slice, slice]
    # A mosaic's pixels, either as they are or seen on another plane's grid.
    Plane: TypeAlias = "npt.NDArray[Any] | Reprojected"
    Colour: TypeAlias = Callable[[Window, "list[Array] | None"], Array]

logger = logging.getLogger(__name__)

_IMAGE_NDIM = 2
_RGB_PLANES = 3
_MAX_HEADER_CARDS = 400


def find_image(
    hdul: FitsHDUList, names: Sequence[str], *, fallback: bool = True
) -> tuple[Array, FitsHeader]:
    """Find the first 2D image HDU with one of `names`.

    With `fallback` it settles for the first 2D image of any name, which suits a file
    holding a single image. Turn it off when looking for WHT beside SCI in one file, or
    a missing WHT quietly comes back as SCI.

    Returns:
        The image's data and header.

    Raises:
        ValueError: If there's no such image.
    """
    wanted = [hdu for name in names for hdu in hdul if hdu.name == name.upper()]
    for hdu in wanted + (list(hdul) if fallback else []):
        if not hdu.is_image or hdu.data is None:
            continue
        data = hdu.data
        # Some mosaics are stored as (1, H, W), which is still just one image.
        while data.ndim > _IMAGE_NDIM and data.shape[0] == 1:
            data = data[0]
        if data.ndim == _IMAGE_NDIM:
            return data, hdu.header
    msg = f"There is no 2D image called {' or '.join(names)} in {hdul.filename()}"
    raise ValueError(msg)


@dataclass
class MosaicStats:
    """Sky level, noise and stretch limits, measured once for a whole mosaic."""

    sky: float
    sigma: float
    lo: float
    hi: float


def subsample(data: Array, max_values: int = 1 << 24) -> Array:
    """Pick up to `max_values` evenly spaced pixels that hold real data.

    Mosaics can run to several GB, so the statistics use a strided view of the memmap.
    Pixels outside the footprint are left out too. The JWST pipeline fills them with
    NaN but COSMOS-Web uses 0, and when most of a rotated mosaic is zero fill the
    median and MAD both come out as 0, after which every tile of blank sky passes
    the emptiness cut.

    Returns:
        The pixels as a flat float32 array.
    """
    h, w = data.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_values))))
    sample = np.asarray(data[::step, ::step], dtype=np.float32).reshape(-1)
    kept: Array = sample[np.isfinite(sample) & (sample != 0)]
    return kept


def measure_mosaic(data: Array, p_lo: float, p_hi: float) -> MosaicStats:
    """Measure the median sky, MAD noise and percentile stretch limits of a mosaic.

    Returns:
        The mosaic's stats, or a flat 0 to 1 range when it holds no data.
    """
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


def survey_levels(datasets: Sequence[Path]) -> dict[str, Any]:
    """Take the median stretch range of each filter over the mosaics prep cut.

    A per-mosaic percentile follows whatever is brightest in the mosaic, so one
    bright star can push a filter's range several times above the other visits' and
    render the whole visit darker. Measured over many visits the median doesn't move.

    Returns:
        The levels file's contents, a range per filter with how many mosaics it saw.

    Raises:
        SystemExit: If no mosaic in `datasets` has a Lupton render record.
    """
    ranges: dict[str, list[float]] = {}
    for root in datasets:
        for path in sorted(Path(root).glob("mosaics/*.json")):
            info = json.loads(path.read_text(encoding="utf-8"))
            render = info.get("render") or {}
            if render.get("kind") != "lupton":
                continue
            for filt, v in zip(info["filters"], render["levels"], strict=True):
                ranges.setdefault(str(filt or "").upper(), []).append(
                    v["hi"] - v["sky"]
                )
    if not ranges:
        msg = f"No mosaic under {', '.join(map(str, datasets))} was cut in colour."
        raise SystemExit(msg)
    return {
        "range": {f: float(np.median(r)) for f, r in ranges.items()},
        "mosaics": {f: len(r) for f, r in ranges.items()},
    }


def read_levels(path: Path, filt: str) -> float:
    """Read one filter's range from a file `survey_levels` wrote.

    Returns:
        The range above the sky, in the mosaic's flux units.

    Raises:
        SystemExit: If the file has no range for `filt`.
    """
    ranges: dict[str, float] = json.loads(path.read_text(encoding="utf-8"))["range"]
    key = filt.upper()
    if key not in ranges:
        msg = f"{path} has no range for {filt or 'this file'}, only {sorted(ranges)}."
        raise SystemExit(msg)
    return ranges[key]


def levels_main(argv: list[str] | None = None) -> None:
    """Write one stretch range per filter, measured over datasets prep already cut."""
    p = argparse.ArgumentParser(prog="jwst-peculiar levels")
    p.add_argument("datasets", nargs="+", type=Path, help="Folders prep wrote.")
    p.add_argument("--out", type=Path, required=True, help="The levels file.")
    args = p.parse_args(argv)
    levels = survey_levels(args.datasets)
    args.out.write_text(json.dumps(levels, indent=1) + "\n", encoding="utf-8")


def asinh_to_uint8(
    img: Array, lo: float, hi: float, *, asinh_q: float = 10.0, fill: float = 0.0
) -> Array:
    """Stretch flux to 8 bits through an asinh between fixed limits.

    The limits are arguments on purpose. Stretching each tile between its own
    percentiles brings faint sky and a bright source up to the same range, which
    hides the very difference the detector is looking for. NaN, and the zero fill
    COSMOS-Web uses for the same thing, become `fill`, the sky level, so holes don't
    turn into black shapes the detector flags.

    Returns:
        The stretched image as uint8.
    """
    img = np.asarray(img, dtype=np.float32)
    img = np.where(np.isfinite(img) & (img != 0), img, np.float32(fill))
    x = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    # Rounded, as truncating would put the upper limit itself at 254.
    stretched: Array = np.rint(255.0 * np.arcsinh(asinh_q * x) / np.arcsinh(asinh_q))
    return stretched.astype(np.uint8)


def has_signal(
    cut: Array,
    stats: MosaicStats,
    *,
    source_sigma: float = 5.0,
    min_frac: float = 2e-4,
) -> bool:
    """Check whether enough of the tile rises above the sky to be worth embedding.

    It reads raw flux against the mosaic's noise. Pure sky puts next to nothing above
    5 sigma, so the default works for MIRI and NIRCam alike. The fraction is of the
    whole tile rather than its finite pixels, or a sliver of coverage at a mosaic edge
    would pass on a single noisy pixel.

    Returns:
        True when the tile has signal.
    """
    finite = np.isfinite(cut)
    if not np.any(finite & (cut != 0)):
        return False
    if stats.sigma <= 0.0:
        return bool(np.any(finite & (cut != stats.sky)))
    above = finite & (cut > stats.sky + source_sigma * stats.sigma)
    return float(above.sum()) / cut.size >= min_frac


def covered_fraction(wht: Array) -> float:
    """Measure the share of a window with a positive weight.

    Enclosed holes count as covered. A saturated star core has zero weight inside the
    footprint, and counting it as missing would throw away every tile over the star,
    when edge tiles are what the check is for. A gap that reaches the window's border
    isn't enclosed, so it still counts.

    Returns:
        The covered fraction, from 0 to 1.
    """
    covered = np.isfinite(wht) & (wht > 0)
    return float(binary_fill_holes(covered).mean())


def noise_factor(wht: Array, typical: float) -> Array:
    """Work out sqrt(weight / typical), capped at 1, for every pixel of a window.

    Multiplying the flux above the sky by it gives every part of a mosaic the noise of
    its typical depth, which is how a detection image is weighted. Without it, sky
    covered by one exposure in place of three renders 1.7 times noisier, and on SMACS
    0723 tiles reaching into those strips took the top four places. Pixels with no
    weight, like a saturated star's core, keep their flux.

    Returns:
        The factors in 255ths, as uint8.
    """
    w = np.asarray(wht, dtype=np.float32)
    f = np.sqrt(np.clip(w / typical, 0.0, 1.0))
    f = np.where(np.isfinite(w) & (w > 0), f, 1.0)
    return np.round(f * 255).astype(np.uint8)


def equalise(x: Array, sky: float, factor: Array | None) -> Array:
    """Scale the flux above `sky` by a noise_factor.

    Returns:
        The scaled flux as float32, with NaN and zero fill left as they are.
    """
    x = np.asarray(x, dtype=np.float32)
    if factor is None:
        return x
    f = factor.astype(np.float32) / 255
    return np.where(np.isfinite(x) & (x != 0), (x - sky) * f + sky, x).astype(
        np.float32
    )


def typical_weights(
    whts: Sequence[Plane], starts_y: list[int], starts_x: list[int], *, crop: int
) -> list[float]:
    """Take the median weight of each map over at most 127 tile windows.

    The windows are spread across the mosaic, as a reprojected plane can only be read
    a window at a time.

    Returns:
        One typical weight per map, 1.0 for a map with no positive weight.
    """
    windows = [(y, x) for y in starts_y for x in starts_x]
    windows = windows[:: max(1, len(windows) // 64)]
    typical: list[float] = []
    for w in whts:
        medians: list[np.floating[Any]] = []
        for y, x in windows:
            part = np.asarray(w[y : y + crop, x : x + crop])[::4, ::4]
            part = part[np.isfinite(part) & (part > 0)]
            if part.size:
                medians.append(np.median(part))
        typical.append(float(np.median(medians)) if medians else 1.0)
    return typical


def tile_starts(length: int, crop: int, stride: int) -> list[int]:
    """List where crops start along one axis.

    When the stride doesn't land on the far edge, one more crop goes flush against it,
    as otherwise a strip up to a stride wide at the right and bottom of every mosaic
    never gets looked at.

    Returns:
        The start of every crop, or nothing when the axis is shorter than a crop.
    """
    if length < crop:
        return []
    starts = list(range(0, length - crop + 1, stride))
    if starts[-1] != length - crop:
        starts.append(length - crop)
    return starts


_TILE_FALLBACK_RE = re.compile(r"(?:^|[_\-])([A-Z]\d{1,2})(?:[_\-]|$)")


def infer_tile_id_from_filename(path: str, tile_regex: str | None = None) -> str | None:
    """Pull a tile id like A5 out of a file name.

    Returns:
        The first group of `tile_regex` when it matches, or else a letter and one or
        two digits standing alone in the name, or None.
    """
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


def celestial_wcs(
    header: FitsHeader, source: str, *, required: bool = True
) -> SkyWcs | None:
    """Read the header's sky WCS.

    Returns:
        The WCS, or None for a header without one when it isn't `required`.

    Raises:
        ValueError: If it's `required` and the header has none.
    """
    with warnings.catch_warnings():
        # JWST headers get a few harmless date and unit fixes that astropy warns about.
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header).celestial
    if not wcs.has_celestial:
        if required:
            msg = f"{source} has no sky coordinates in its header. Drop --wcs_in_name."
            raise ValueError(msg)
        return None
    return wcs


def make_output_name(
    tile_id: str,
    cx: float,
    cy: float,
    *,
    global_idx: int,
    wcs_ra_dec: tuple[float, float] | None = None,
) -> str:
    """Name a tile after its mosaic, centre pixel and index, and its sky position.

    Returns:
        The PNG file name.
    """
    parts = [tile_id, f"cx{round(cx):06d}", f"cy{round(cy):06d}"]
    if wcs_ra_dec is not None:
        ra, dec = wcs_ra_dec
        # A WCS can give an RA just below 0, and survey keys only sign the Dec.
        parts += [f"ra{_safe_float_tag(ra % 360)}", f"dec{_safe_float_tag(dec)}"]
    parts.append(f"i{global_idx:08d}")
    return "_".join(parts) + ".png"


@dataclass
class Mosaic:
    """One mosaic as cut_tiles reads it.

    `signal` is the raw flux that the emptiness cut and the grey stretch read, and
    `colour` renders the saved tile for a window in place of the grey stretch. With
    several weight maps a window has to be covered in all of them. `planes` are the
    float planes a --save_cutouts FITS file holds, `signal` alone by default.
    """

    tile_id: str
    signal: Array
    wcs: SkyWcs | None = None
    wht: Plane | list[Plane] | tuple[Plane, ...] | None = None
    colour: Colour | None = None
    planes: Sequence[Plane] | None = None
    bunit: str | None = None


def _save_tile_png(tile: Array, size: int) -> Image.Image:
    # FITS row 1 is the bottom of the sky, so it goes at the bottom of the PNG.
    img = Image.fromarray(np.ascontiguousarray(tile[::-1]))
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return img


@dataclass
class _Cutter:
    args: argparse.Namespace
    mosaic: Mosaic
    query_dir: Path
    records: list[dict[str, Any]] | None
    whts: list[Plane]
    crop: int
    starts: tuple[list[int], list[int]]
    stats: MosaicStats
    typical: list[float] | None = None

    @classmethod
    def start(
        cls,
        args: argparse.Namespace,
        mosaic: Mosaic,
        query_dir: Path,
        *,
        records: list[dict[str, Any]] | None,
    ) -> _Cutter:
        whts = _weight_maps(mosaic)
        crop = int(args.tile_size / args.upscale)
        stride = args.stride if args.stride is not None else max(1, crop // 2)
        stats = measure_mosaic(mosaic.signal, args.p_lo, args.p_hi)
        logger.info(
            "[%s] sky=%.4g sigma=%.4g stretch=[%.4g, %.4g]",
            mosaic.tile_id,
            stats.sky,
            stats.sigma,
            stats.lo,
            stats.hi,
        )
        h, w = mosaic.signal.shape
        starts = (tile_starts(h, crop, stride), tile_starts(w, crop, stride))
        cutter = cls(args, mosaic, query_dir, records, whts, crop, starts, stats)
        if (
            getattr(args, "noise_eq", False)
            and whts
            and len(whts) == len(mosaic.planes or [mosaic.signal])
        ):
            cutter.typical = typical_weights(whts, *starts, crop=crop)
        elif getattr(args, "noise_eq", False) and args.min_wht_frac <= 0:
            logger.info(
                "[%s] --min_wht_frac 0 reads no weight map, so the noise isn't "
                "evened out",
                mosaic.tile_id,
            )
        return cutter

    def windows(self) -> Iterator[Window]:
        for y in self.starts[0]:
            for x in self.starts[1]:
                yield np.s_[y : y + self.crop, x : x + self.crop]

    def low_weight(self, window: Window) -> bool:
        return self.args.min_wht_frac > 0 and any(
            covered_fraction(np.asarray(w[window])) < self.args.min_wht_frac
            for w in self.whts
        )

    def save(self, window: Window, raw: Array, global_idx: int) -> None:
        noise = self._noise(window)
        tile = self._render(window, raw, noise)
        ys, xs = window
        cx, cy = xs.start + (self.crop - 1) / 2, ys.start + (self.crop - 1) / 2
        radec = None
        if self.mosaic.wcs is not None:
            ra, dec = self.mosaic.wcs.all_pix2world(cx, cy, 0)
            radec = (float(ra), float(dec))
        img = _save_tile_png(tile, self.args.tile_size)
        name = make_output_name(
            self.mosaic.tile_id,
            cx,
            cy,
            global_idx=global_idx,
            wcs_ra_dec=radec if self.args.wcs_in_name else None,
        )
        img.save(self.query_dir / name)
        cutout = self._write_cutout(name, window, noise)
        if self.records is not None:
            self.records.append({
                "key": name,
                "mosaic": self.mosaic.tile_id,
                "x0": xs.start,
                "y0": ys.start,
                "size": self.crop,
                "ra": f"{radec[0]:.7f}" if radec else "",
                "dec": f"{radec[1]:.7f}" if radec else "",
                "cutout": cutout.as_posix() if cutout else "",
            })

    def _noise(self, window: Window) -> list[Array] | None:
        if self.typical is None:
            return None
        return [
            noise_factor(m[window], t)
            for m, t in zip(self.whts, self.typical, strict=True)
        ]

    def _render(self, window: Window, raw: Array, noise: list[Array] | None) -> Array:
        if self.mosaic.colour is not None:
            return self.mosaic.colour(window, noise)
        s = self.stats
        if self.args.stretch_scope == "tile":
            s = measure_mosaic(raw, self.args.p_lo, self.args.p_hi)
        flux = equalise(raw, s.sky, noise[0] if noise else None)
        return asinh_to_uint8(flux, s.lo, s.hi, asinh_q=self.args.asinh_q, fill=s.sky)

    def _write_cutout(
        self, name: str, window: Window, noise: list[Array] | None
    ) -> Path | None:
        if not getattr(self.args, "save_cutouts", False):
            return None
        cutout = Path("cutouts") / f"{Path(name).stem}.fits"
        write_cutout(self.query_dir.parent / cutout, self.mosaic, window, noise=noise)
        return cutout


def _weight_maps(mosaic: Mosaic) -> list[Plane]:
    wht = mosaic.wht
    whts = [] if wht is None else list(wht) if isinstance(wht, (list, tuple)) else [wht]
    signal = mosaic.signal
    for w in whts:
        if w.shape != signal.shape:
            msg = (
                f"[{mosaic.tile_id}] the weight map is {w.shape[1]}x{w.shape[0]} but "
                f"the image is {signal.shape[1]}x{signal.shape[0]}, so they can't be "
                "from the same mosaic"
            )
            raise ValueError(msg)
    return whts


def cut_tiles(
    args: argparse.Namespace,
    mosaic: Mosaic,
    query_dir: Path,
    *,
    global_idx: int,
    records: list[dict[str, Any]] | None = None,
) -> tuple[int, int, int, int]:
    """Cut, filter and save the tiles of one mosaic.

    Each saved tile appends its manifest row to `records`.

    Returns:
        The next global index and the saved, empty and low-WHT counts.

    Raises:
        ValueError: If a weight map isn't the same size as the image.
    """
    cutter = _Cutter.start(args, mosaic, query_dir, records=records)
    saved = skipped_empty = skipped_wht = 0
    for window in cutter.windows():
        if cutter.low_weight(window):
            skipped_wht += 1
            continue

        # Judge emptiness on raw flux, before a stretch flattens the difference
        # between blank sky and a tile with something in it.
        raw = np.asarray(mosaic.signal[window], dtype=np.float32)
        if not has_signal(
            raw,
            cutter.stats,
            source_sigma=args.source_sigma,
            min_frac=args.min_source_frac,
        ):
            skipped_empty += 1
            continue

        if not args.dry_run:
            cutter.save(window, raw, global_idx)
        global_idx += 1
        saved += 1

    logger.info(
        "[%s] %s %d tiles, skipped %d as empty and %d for low WHT",
        mosaic.tile_id,
        "would save" if args.dry_run else "saved",
        saved,
        skipped_empty,
        skipped_wht,
    )
    return global_idx, saved, skipped_empty, skipped_wht


def write_cutout(
    path: Path,
    mosaic: Mosaic,
    window: Window,
    *,
    noise: Sequence[Array] | None = None,
) -> None:
    """Write the raw float pixels under one tile as a FITS file with its own WCS.

    Zero fill becomes NaN, which is what FITS readers take as no data. The noise
    factors the PNG was rendered with go in a NOISEEQ extension, so render_cutout can
    make the same PNG again.
    """
    planes = mosaic.planes or [mosaic.signal]
    wcs = mosaic.wcs
    data = np.stack([np.asarray(p[window], dtype=np.float32) for p in planes])
    data[data == 0] = np.nan
    header = fits.Header()
    if wcs is not None:
        ys, xs = window
        sub_wcs = wcs.deepcopy()
        sub_wcs.wcs.crpix = [wcs.wcs.crpix[0] - xs.start, wcs.wcs.crpix[1] - ys.start]
        header.update(sub_wcs.to_header())
    if mosaic.bunit:
        header["BUNIT"] = mosaic.bunit
    path.parent.mkdir(parents=True, exist_ok=True)
    hdus: list[Any] = [
        fits.PrimaryHDU(data[0] if len(planes) == 1 else data, header=header)
    ]
    if noise is not None:
        factors = np.stack(noise)
        hdus.append(
            fits.ImageHDU(factors[0] if len(noise) == 1 else factors, name="NOISEEQ")
        )
        hdus[-1].header["COMMENT"] = (
            "sqrt(weight / typical weight) capped at 1, times 255"
        )
    fits.HDUList(hdus).writeto(path, overwrite=True)


@dataclass
class MosaicInfo:
    """What describe_mosaic records about a mosaic beyond its pixels.

    `render` records how the tiles were stretched, which is what lets render_cutout
    turn a cutout back into the same PNG after the benchmark adds a source to it.
    `plane_records` describe each filter, as plane_info returns them.
    """

    headers: Sequence[FitsHeader]
    filters: Sequence[str]
    render: dict[str, Any] | None = None
    plane_records: Sequence[dict[str, Any]] | None = None


def _wcs_record(wcs: SkyWcs) -> tuple[dict[str, Any], float]:
    cd = np.diag(wcs.wcs.cdelt) @ wcs.wcs.get_pc()
    record = {
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
    return record, round(float(scale), 5)


def _header_cards(headers: Sequence[FitsHeader]) -> dict[str, str]:
    cards: dict[str, str] = {}
    for header in headers:
        for card in header.cards:
            if (
                card.keyword
                and card.keyword not in ("HISTORY", "COMMENT", "")
                and len(cards) < _MAX_HEADER_CARDS
            ):
                cards.setdefault(card.keyword, str(card.value))
    return cards


def describe_mosaic(
    out: Path,
    mosaic: Mosaic,
    info: MosaicInfo,
    *,
    preview: Callable[[Array, Array], Array],
) -> None:
    """Write mosaics/<id>.json and a preview PNG, which the web viewer reads."""
    h, w = mosaic.signal.shape
    step = max(1, int(np.ceil(max(h, w) / 2048)))
    ys, xs = np.arange(0, h, step), np.arange(0, w, step)
    # Top row first, so the preview is the right way up like the tiles.
    Image.fromarray(preview(ys[::-1], xs)).save(
        out / "mosaics" / f"{mosaic.tile_id}.png"
    )

    meta: dict[str, Any] = {
        "shape": [h, w],
        "preview_scale": len(xs) / w,
        "filters": [f.upper() for f in info.filters],
    }
    if mosaic.wcs is not None:
        meta["wcs"], meta["pixel_scale_arcsec"] = _wcs_record(mosaic.wcs)
    meta["header"] = _header_cards(info.headers)
    if mosaic.bunit:
        meta["bunit"] = mosaic.bunit
    if info.render is not None:
        meta["render"] = info.render
    if info.plane_records:
        meta["planes"] = list(info.plane_records)
    (out / "mosaics" / f"{mosaic.tile_id}.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8"
    )


def pixel_scale(wcs: SkyWcs | None) -> float | None:
    """Work out the pixel scale from the area of one pixel.

    Returns:
        Arcseconds per pixel, or None without a WCS.
    """
    if wcs is None:
        return None
    cd = np.diag(wcs.wcs.cdelt) @ wcs.wcs.get_pc()
    return float(np.sqrt(abs(np.linalg.det(cd))) * 3600)


def plane_info(
    headers: Sequence[FitsHeader],
    path: str,
    *,
    own: SkyWcs | None,
    target: SkyWcs | None,
    filt: str = "",
) -> dict[str, Any]:
    """Run describe_plane for a file that may have been reprojected onto `target`.

    Interpolating keeps a pixel's value, so one unit of it now covers the target's
    pixel area. The area ratio fixes surface brightness and per-pixel units alike.

    Returns:
        describe_plane's record, with the flux scale for the target's pixels.
    """
    own_scale, target_scale = pixel_scale(own), pixel_scale(target or own)
    info = describe_plane(headers, path, own_scale, filt=filt or None)
    if info["jy_per_unit"] is not None and own_scale and target_scale:
        info["jy_per_unit"] *= (target_scale / own_scale) ** 2
    return info


def asinh_render(stats: MosaicStats, args: argparse.Namespace) -> dict[str, Any] | None:
    """Make the render record for a grey mosaic.

    Returns:
        The record, or None when each tile has its own stretch.
    """
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


def render_cutout(
    data: Array, render: dict[str, Any], noise: Array | None = None
) -> Array:
    """Make the 8-bit tile `prep` saves for these float pixels.

    `data` is a cutout as write_cutout leaves it, (H, W) or (3, H, W) with FITS row 1
    first, and `render` is the record mosaics/<id>.json keeps of how its mosaic was
    stretched. Both stretches send NaN and zero to the sky level, so the NaN that
    write_cutout puts in holes renders exactly as the zero fill did.

    Returns:
        The tile, top row first, at its saved size.

    Raises:
        ValueError: If `render` is of a kind this doesn't know.
    """
    if render["kind"] == "lupton":
        levels = [
            MosaicStats(sky=v["sky"], sigma=0.0, lo=0.0, hi=v["hi"])
            for v in render["levels"]
        ]
        opts = argparse.Namespace(stretch=render["stretch"], Q=render["Q"])
        factors: list[Array | None] = (
            [None] * len(levels) if noise is None else list(noise)
        )
        planes = [
            equalise(d, v.sky, n) for d, v, n in zip(data, levels, factors, strict=True)
        ]
        tile = lupton(opts, planes, levels)
    elif render["kind"] == "asinh":
        flux = equalise(data, render["sky"], noise)
        tile = asinh_to_uint8(
            flux, render["lo"], render["hi"], asinh_q=render["q"], fill=render["sky"]
        )
    else:
        msg = f"Unknown render kind {render['kind']!r}"
        raise ValueError(msg)
    return np.asarray(_save_tile_png(tile, int(render["tile_size"])))


@dataclass(eq=False)
class Reprojected:
    """One plane of a mosaic seen on another plane's pixel grid, a window at a time.

    JWST level-3 mosaics of one field come out on slightly different grids per filter
    (a pixel wider here, twice the pixel scale for the short-wave channel there), and
    stacking the arrays as they are would put different sky in each colour.
    """

    data: Array
    wcs: SkyWcs
    target: SkyWcs
    shape: tuple[int, ...]

    def __getitem__(self, window: Window) -> Array:
        """Reproject one window of the target grid.

        Returns:
            The window's pixels as float32, NaN where this plane has no data.
        """
        ys, xs = window
        out_shape = (ys.stop - ys.start, xs.stop - xs.start)
        x0, x1, y0, y1 = self._source_box(ys, xs)
        if x1 <= x0 or y1 <= y0:
            return np.full(out_shape, np.nan, np.float32)
        src = np.asarray(self.data[y0:y1, x0:x1], dtype=np.float32)
        out, _ = reproject_interp(
            (src, self.wcs[y0:y1, x0:x1]),
            self.target[ys, xs],
            out_shape,
            order="bilinear",
        )
        reprojected: Array = out.astype(np.float32)
        return reprojected

    def _source_box(self, ys: slice, xs: slice) -> tuple[int, int, int, int]:
        # Only the part of this plane under the window gets read off the memmap.
        corners_y = [ys.start, ys.start, ys.stop, ys.stop]
        corners_x = [xs.start, xs.stop, xs.start, xs.stop]
        ra, dec = self.target.all_pix2world(corners_x, corners_y, 0)
        px, py = self.wcs.all_world2pix(ra, dec, 0)
        h, w = self.data.shape
        x0, x1 = max(0, int(np.floor(px.min())) - 4), min(w, int(np.ceil(px.max())) + 5)
        y0, y1 = max(0, int(np.floor(py.min())) - 4), min(h, int(np.ceil(py.max())) + 5)
        return x0, x1, y0, y1


def on_grid(
    data: Array, wcs: SkyWcs, *, target: SkyWcs, shape: tuple[int, ...]
) -> Plane:
    """Put `data` on the target grid.

    Returns:
        `data` itself when it already shares the grid, otherwise a Reprojected view.
    """
    same = (
        data.shape == shape
        and np.allclose(wcs.wcs.crpix, target.wcs.crpix)
        and np.allclose(
            wcs.pixel_scale_matrix, target.pixel_scale_matrix, rtol=1e-6, atol=0
        )
        and np.allclose(wcs.wcs.crval, target.wcs.crval, rtol=0, atol=1e-9)
    )
    return data if same else Reprojected(data, wcs, target, shape)


@dataclass
class MosaicSpec:
    """Where one mosaic's files are, before any of them is opened."""

    tile_id: str
    sci_path: str
    wht_path: str | None = None


class Sensor:
    """A file layout prep knows how to find mosaics in and cut."""

    name: str = "base"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        """Add this sensor's own flags to the parser."""
        raise NotImplementedError

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        """Find the mosaics to cut."""
        raise NotImplementedError

    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        *,
        query_dir: Path,
        global_idx: int,
        records: list[dict[str, Any]],
    ) -> int:
        """Cut one mosaic, appending its tiles' manifest rows to `records`."""
        raise NotImplementedError


class MiriSensor(Sensor):
    """One mosaic per file, with WHT inside it (i2d) or in a *wht.fits beside it."""

    name = "miri"
    _SCI2WHT_RE = re.compile(r"(?i)sci\.fits$")

    @override
    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pattern", default="*sci.fits", help="Glob for the science mosaics."
        )
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")

    @classmethod
    def sci_to_wht_path(cls, sci_path: str) -> str:
        """Name the weight map that goes with a science mosaic.

        Returns:
            The *wht.fits beside a *sci.fits, or the file itself when it isn't named
            that way.
        """
        if cls._SCI2WHT_RE.search(sci_path):
            return cls._SCI2WHT_RE.sub("wht.fits", sci_path)
        return sci_path

    @override
    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        sci_paths = sorted(
            str(p) for p in Path(args.indir).glob(args.pattern) if p.is_file()
        )
        if not sci_paths:
            msg = f"Nothing in {args.indir} matches {args.pattern}"
            raise FileNotFoundError(msg)

        seen: dict[str, str] = {}
        for sci_path in sci_paths:
            # JWST archive names carry no tile id, and the file stem at least says
            # which mosaic a tile came from.
            tile_id = (
                args.tile_id
                or infer_tile_id_from_filename(sci_path, args.tile_regex)
                or Path(sci_path).stem
            )
            if tile_id in seen:
                msg = (
                    f"{seen[tile_id]} and {sci_path} would both be mosaic {tile_id}. "
                    "Give a --tile_regex that tells them apart."
                )
                raise SystemExit(msg)
            seen[tile_id] = sci_path
            wht_path = None
            if args.min_wht_frac > 0:
                wht_path = self.sci_to_wht_path(sci_path)
                if not Path(wht_path).is_file():
                    msg = (
                        f"--min_wht_frac needs a weight map for {sci_path}, and "
                        f"{wht_path} isn't there. Pass --min_wht_frac 0 to tile it "
                        "without one."
                    )
                    raise FileNotFoundError(msg)
            yield MosaicSpec(tile_id=tile_id, sci_path=sci_path, wht_path=wht_path)

    @staticmethod
    def _weight_map(
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        sci_hdul: FitsHDUList,
        *,
        spec: MosaicSpec,
    ) -> Array | None:
        if spec.wht_path == spec.sci_path:
            try:
                return find_image(sci_hdul, (args.wht_ext, "WHT"), fallback=False)[0]
            except ValueError:
                msg = (
                    f"{spec.sci_path} has no {args.wht_ext} extension for "
                    "--min_wht_frac. Pass --min_wht_frac 0 to tile it without one."
                )
                raise ValueError(msg) from None
        if spec.wht_path is not None:
            wht_hdul = stack.enter_context(fits.open(spec.wht_path, memmap=True))
            return find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))[0]
        return None

    @override
    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        *,
        query_dir: Path,
        global_idx: int,
        records: list[dict[str, Any]],
    ) -> int:
        # Everything is cut while the files are open, since the arrays are memmaps.
        with contextlib.ExitStack() as stack:
            sci_hdul: FitsHDUList = stack.enter_context(
                fits.open(spec.sci_path, memmap=True)
            )
            sci, header = find_image(sci_hdul, (args.sci_ext, "SCI"))
            wht = self._weight_map(args, stack, sci_hdul, spec=spec)
            wcs = celestial_wcs(header, spec.sci_path, required=args.wcs_in_name)
            args.bunit = header.get("BUNIT")
            mosaic = Mosaic(spec.tile_id, sci, wcs=wcs, wht=wht, bunit=args.bunit)
            if not args.dry_run:
                headers = [sci_hdul[0].header, header]
                filt = sci_hdul[0].header.get("FILTER", header.get("FILTER", ""))
                stats = measure_mosaic(sci, args.p_lo, args.p_hi)
                info = MosaicInfo(
                    headers=headers,
                    filters=[str(filt)],
                    render=asinh_render(stats, args),
                    plane_records=[
                        plane_info(headers, spec.sci_path, own=wcs, target=wcs)
                    ],
                )
                describe_mosaic(
                    query_dir.parent,
                    mosaic,
                    info,
                    preview=lambda ys, xs: asinh_to_uint8(
                        sci[ys[:, None], xs[None, :]],
                        stats.lo,
                        stats.hi,
                        asinh_q=args.asinh_q,
                        fill=stats.sky,
                    ),
                )
            return cut_tiles(
                args, mosaic, query_dir, global_idx=global_idx, records=records
            )[0]


@dataclass
class _File:
    """One opened file of a multi-file mosaic."""

    data: Plane
    headers: list[FitsHeader]
    level: MosaicStats
    wht: Plane | None


@dataclass
class _Layers:
    """The planes of one multi-file mosaic, all on the first file's pixel grid."""

    signal: Array
    target: SkyWcs | None
    headers: list[FitsHeader]
    planes: list[Plane] = field(default_factory=list)
    whts: list[Plane] = field(default_factory=list)
    levels: list[MosaicStats] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)

    def add(self, f: _File, info: dict[str, Any], levels: Path | None) -> None:
        if levels is not None:
            # The sky stays per mosaic and only the range is shared.
            hi = f.level.sky + read_levels(levels, str(info["filter"] or ""))
            f.level = replace(f.level, hi=hi)
        self.levels.append(f.level)
        self.infos.append(info)
        self.planes.append(f.data)
        if f.wht is not None:
            self.whts.append(f.wht)


class NircamSensor(Sensor):
    """COSMOS-Web style mosaics, one file per filter, combined into colour tiles.

    The first filter is the detection plane for the emptiness cut. Colour is made one
    tile at a time, which gives the same pixels as the whole mosaic at once (Lupton's
    mapping works pixel by pixel) without holding three mosaics and their float64
    copies in RAM.
    """

    name = "nircam"

    @override
    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--pixel_scale", type=str, default="60mas")
        p.add_argument("--version", type=str, default="v1.0")
        p.add_argument("--ext", type=str, default="i2d")
        p.add_argument(
            "--pattern",
            type=str,
            default="mosaic_nircam_{filter}_COSMOS-Web_{pixel_scale}_{tile}_{version}"
            "_{ext}.fits",
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
        names = (
            infer_tile_id_from_filename(str(p)) for p in Path(indir).glob("*.fits")
        )
        tiles = sorted({t for t in names if t})
        if not tiles:
            msg = f"Couldn't find a tile id in any file name in {indir}. Pass --tiles."
            raise ValueError(msg)
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

    @override
    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        for tile_id in args.tiles or self._infer_tiles_from_dir(args.indir):
            yield MosaicSpec(tile_id=tile_id, sci_path="")

    def _inputs(
        self, args: argparse.Namespace, spec: MosaicSpec
    ) -> list[tuple[str, str]]:
        if len(args.rgb_filters) not in {1, _RGB_PLANES}:
            msg = f"--rgb_filters takes 1 or 3 filters, not {len(args.rgb_filters)}"
            raise ValueError(msg)
        out: list[tuple[str, str]] = []
        for filt in args.rgb_filters:
            path = self._build_path(args, spec.tile_id, filt)
            if not Path(path).is_file():
                msg = f"There is no {filt} mosaic for tile {spec.tile_id} at {path}"
                raise FileNotFoundError(msg)
            out.append((filt, path))
        return out

    def _weights(
        self,
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        hdul: FitsHDUList,
        *,
        path: str,
        filt: str,
    ) -> Array | None:
        try:
            return find_image(hdul, (args.wht_ext, "WHT"), fallback=False)[0]
        except ValueError:
            pass
        # COSMOS-Web ships the weights as a separate file named like the science one.
        wht_path = Path(path.replace(f"_{args.ext}.fits", "_wht.fits"))
        if wht_path == Path(path) or not wht_path.is_file():
            msg = (
                f"--min_wht_frac needs a weight map for {filt}, and {path} has no "
                f"{args.wht_ext} extension and there is no {wht_path.name} beside it. "
                "Pass --min_wht_frac 0 to tile it without one."
            )
            raise FileNotFoundError(msg)
        wht_hdul = stack.enter_context(fits.open(wht_path, memmap=True))
        return find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))[0]

    def _read(
        self,
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        *,
        path: str,
        filt: str,
    ) -> _File:
        hdul: FitsHDUList = stack.enter_context(fits.open(path, memmap=True))
        data, hdr = find_image(hdul, (args.sci_ext, "SCI"))
        level = measure_mosaic(data, args.p_lo, args.p_hi)
        wht = (
            self._weights(args, stack, hdul, path=path, filt=filt)
            if args.min_wht_frac > 0
            else None
        )
        return _File(data, [hdul[0].header, hdr], level, wht)

    def _first_plane(
        self,
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        *,
        path: str,
        filt: str,
    ) -> _Layers:
        # The first filter sets the pixel grid and is the detection plane.
        f = self._read(args, stack, path=path, filt=filt)
        target = celestial_wcs(f.headers[1], path, required=args.wcs_in_name)
        args.bunit = f.headers[1].get("BUNIT")
        layers = _Layers(signal=np.asarray(f.data), target=target, headers=f.headers)
        info = plane_info(f.headers, path, own=target, target=target, filt=filt)
        layers.add(f, info, args.levels)
        return layers

    def _next_plane(
        self,
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        layers: _Layers,
        *,
        path: str,
        filt: str,
    ) -> None:
        f = self._read(args, stack, path=path, filt=filt)
        shape = layers.signal.shape
        if layers.target is None:
            # Without a WCS the files have to share a grid already.
            own = None
            if f.data.shape != shape:
                msg = (
                    f"{path} is {f.data.shape[1]}x{f.data.shape[0]} and the first "
                    f"file is {shape[1]}x{shape[0]}, and without a WCS they can't be "
                    "aligned"
                )
                raise ValueError(msg)
        else:
            own = celestial_wcs(f.headers[1], path)
            f.data = on_grid(np.asarray(f.data), own, target=layers.target, shape=shape)
            if f.wht is not None:
                f.wht = on_grid(
                    np.asarray(f.wht), own, target=layers.target, shape=shape
                )
        info = plane_info(f.headers, path, own=own, target=layers.target, filt=filt)
        layers.add(f, info, args.levels)

    @override
    def process_one_mosaic(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
        *,
        query_dir: Path,
        global_idx: int,
        records: list[dict[str, Any]],
    ) -> int:
        (first_filt, first_path), *rest = self._inputs(args, spec)
        with contextlib.ExitStack() as stack:
            layers = self._first_plane(args, stack, path=first_path, filt=first_filt)
            for filt, path in rest:
                self._next_plane(args, stack, layers, path=path, filt=filt)
            if args.levels is not None and len(layers.planes) != _RGB_PLANES:
                # A grey mosaic's tiles take their stretch from _Cutter, not these.
                msg = f"--levels is for colour mosaics, and {spec.tile_id} is grey"
                raise SystemExit(msg)
            if args.stretch_scope == "tile" and len(layers.planes) == _RGB_PLANES:
                msg = "--stretch_scope tile is for grey mosaics, and "
                msg += f"{spec.tile_id} is colour"
                raise SystemExit(msg)
            mosaic = Mosaic(
                spec.tile_id,
                layers.signal,
                wcs=layers.target,
                wht=layers.whts,
                colour=self._colour(args, layers),
                planes=layers.planes,
                bunit=args.bunit,
            )
            if not args.dry_run:
                info = MosaicInfo(
                    headers=layers.headers,
                    filters=[i["filter"] for i in layers.infos],
                    render=self._render(args, layers.levels),
                    plane_records=layers.infos,
                )
                describe_mosaic(
                    query_dir.parent,
                    mosaic,
                    info,
                    preview=lambda ys, xs: self._preview(args, layers, ys=ys, xs=xs),
                )
            return cut_tiles(
                args, mosaic, query_dir, global_idx=global_idx, records=records
            )[0]

    @staticmethod
    def _colour(args: argparse.Namespace, layers: _Layers) -> Colour | None:
        if len(layers.planes) != _RGB_PLANES:
            return None

        def colour(window: Window, noise: list[Array] | None = None) -> Array:
            factors: list[Array] | list[None] = noise or [None] * len(layers.planes)
            flux = [
                equalise(p[window], v.sky, n)
                for p, v, n in zip(layers.planes, layers.levels, factors, strict=True)
            ]
            return lupton(args, flux, layers.levels)

        return colour

    @staticmethod
    def _render(
        args: argparse.Namespace, levels: list[MosaicStats]
    ) -> dict[str, Any] | None:
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
    def _preview(
        args: argparse.Namespace, layers: _Layers, *, ys: Array, xs: Array
    ) -> Array:
        def sample(p: Plane) -> Array:
            if isinstance(p, Reprojected):
                full = p[np.s_[0 : p.shape[0], 0 : p.shape[1]]]
                return full[ys[:, None], xs[None, :]]
            return np.nan_to_num(np.asarray(p[ys[:, None], xs[None, :]], np.float32))

        if len(layers.planes) == _RGB_PLANES:
            return lupton(args, [sample(p) for p in layers.planes], layers.levels)
        stats = layers.levels[0]
        return asinh_to_uint8(
            sample(layers.planes[0]),
            stats.lo,
            stats.hi,
            asinh_q=args.asinh_q,
            fill=stats.sky,
        )


def lupton(
    args: argparse.Namespace,
    planes: Sequence[Array],
    levels: Sequence[MosaicStats],
) -> Array:
    """Make Lupton colour from three planes, each put on its mosaic's sky and range.

    Pipeline mosaics keep their sky, and it differs a lot between filters (F444W sits
    far above F277W), so feeding raw flux to Lupton's mapping turns the whole field one
    colour. The levels are per mosaic, not per tile, so tiles stay comparable.

    Returns:
        The colour image as (H, W, 3) uint8.
    """
    scaled: list[Array] = []
    for p, level in zip(planes, levels, strict=True):
        x = np.asarray(p, dtype=np.float32)
        x = np.where(np.isfinite(x) & (x != 0), x, level.sky)
        scaled.append((x - level.sky) / max(level.hi - level.sky, 1e-12))
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb: Array = make_lupton_rgb(*scaled, stretch=args.stretch, Q=args.Q)
    return rgb


def _weight_path(path: str, pattern: str) -> Path:
    return Path(path).with_name(pattern.format(stem=Path(path).stem))


def _fits_paths(args: argparse.Namespace) -> list[str]:
    if args.files:
        paths = [str(p) for p in args.files]
    elif args.indir:
        paths = sorted(
            str(p) for p in Path(args.indir).glob(args.pattern) if p.is_file()
        )
    else:
        msg = "--sensor fits needs --indir, --files or --rgb"
        raise SystemExit(msg)
    if args.wht_pattern:
        weights = {str(_weight_path(p, args.wht_pattern)) for p in paths}
        paths = [p for p in paths if p not in weights]
    if not paths:
        msg = f"Nothing in {args.indir} matches {args.pattern}"
        raise FileNotFoundError(msg)
    return paths


def _fits_tile_id(args: argparse.Namespace, path: str, *, only: bool) -> str:
    tile_id = args.tile_id if only and args.tile_id else None
    if not tile_id and args.tile_regex:
        tile_id = infer_tile_id_from_filename(path, args.tile_regex)
    return tile_id or Path(path).stem


class FitsSensor(NircamSensor):
    """Any FITS image with a celestial WCS, from any telescope.

    Each file matching --pattern in --indir, or each of --files, is cut as its own
    grey mosaic. --rgb takes three files of one field instead, red first, and cuts
    them as one colour mosaic on the first file's grid. The image is the --sci_ext
    extension when there is one and the first 2D image otherwise. A weight map is
    used when the file has a --wht_ext extension or a sibling named by --wht_pattern,
    and the coverage check is skipped with a warning when there's neither.
    """

    name = "fits"

    @override
    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pattern", default="*.fits", help="Glob for the mosaics in --indir."
        )
        p.add_argument(
            "--files", nargs="+", default=None, help="FITS files, one mosaic each."
        )
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
            help="The weight map's name from the image's, like {stem}_wht.fits. "
            "Weights in a --wht_ext extension are found without it.",
        )
        p.add_argument(
            "--stretch", type=float, default=0.05, help="Lupton stretch for --rgb."
        )
        p.add_argument(
            "--Q", type=float, default=10.0, help="Lupton softening for --rgb."
        )

    @override
    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        if args.rgb:
            yield MosaicSpec(
                tile_id=args.tile_id or Path(args.rgb[0]).stem,
                sci_path="|".join(args.rgb),
            )
            return
        paths = _fits_paths(args)
        seen: dict[str, str] = {}
        for path in paths:
            tile_id = _fits_tile_id(args, path, only=len(paths) == 1)
            if tile_id in seen:
                msg = (
                    f"{seen[tile_id]} and {path} would both be mosaic {tile_id}. "
                    "Give a --tile_regex that tells them apart."
                )
                raise SystemExit(msg)
            seen[tile_id] = path
            yield MosaicSpec(tile_id=tile_id, sci_path=path)

    @override
    def _inputs(
        self,
        args: argparse.Namespace,
        spec: MosaicSpec,
    ) -> list[tuple[str, str]]:
        paths = spec.sci_path.split("|")
        for path in paths:
            if not Path(path).is_file():
                msg = f"There is no FITS file at {path}"
                raise FileNotFoundError(msg)
        return [("", path) for path in paths]

    @override
    def _weights(
        self,
        args: argparse.Namespace,
        stack: contextlib.ExitStack[bool | None],
        hdul: FitsHDUList,
        *,
        path: str,
        filt: str,
    ) -> Array | None:
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
            "%s has no weight map, so tiles on the edge of its coverage aren't "
            "dropped. Pass --wht_pattern if it has one beside it.",
            Path(path).name,
        )
        return None


SENSORS: dict[str, Sensor] = {
    "fits": FitsSensor(),
    "miri": MiriSensor(),
    "nircam": NircamSensor(),
}


def clear_previous_output(out: Path, *, overwrite: bool) -> None:
    """Stop when --out holds an earlier run's tiles, or clear them with --overwrite.

    Tile names carry a counter that starts at 0 every run, so a second run into the same
    folder leaves the first run's tiles beside its own, including exact copies under new
    names that the detector then treats as separate tiles vouching for each other.

    Raises:
        SystemExit: If there are earlier tiles and `overwrite` is off.
    """
    old = [out / name for name in ("query", "cutouts", "mosaics", "manifest.csv")]
    old = [p for p in old if p.exists()]
    if not old:
        return
    if not overwrite:
        msg = (
            f"{out} already holds tiles from an earlier run. Pass --overwrite to clear "
            "them, or choose another --out."
        )
        raise SystemExit(msg)
    for p in old:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for the flags every sensor shares.

    Returns:
        The parser, without any sensor's own flags.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sensor",
        choices=sorted(SENSORS),
        default="fits",
        help="fits for any image with a WCS, or the miri and nircam presets for JWST "
        "layouts.",
    )
    p.add_argument("--indir", default=None, help="Folder holding the FITS mosaics.")
    p.add_argument(
        "--out",
        default="datasets/jwst/object1",
        help="Output root. Tiles go in out/query/.",
    )
    p.add_argument(
        "--tile_id", default=None, help="Use this tile id instead of inferring one."
    )
    p.add_argument(
        "--tile_regex", default=None, help="Regex that pulls the tile id from a name."
    )
    p.add_argument(
        "--wcs_in_name",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Put the RA and Dec of each tile centre in its file name.",
    )
    p.add_argument(
        "--tile_size", type=int, default=518, help="Saved tile size in pixels."
    )
    p.add_argument(
        "--upscale",
        type=int,
        default=2,
        help="Crop tile_size/upscale pixels and resize up.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step between crops. Half a crop by default.",
    )

    p.add_argument(
        "--min_wht_frac",
        "--minwhtfrac",
        dest="min_wht_frac",
        type=float,
        default=1.0,
        help="Drop tiles where less than this fraction of pixels has a positive "
        "weight. Below 1, tiles on the edge of the coverage get in and rank near "
        "the top.",
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
    p.add_argument(
        "--noise_eq",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Scale the flux above the sky by sqrt(weight / typical weight), capped "
        "at 1, so shallow parts of a mosaic aren't noisier than the rest. Needs "
        "weight maps.",
    )
    p.add_argument(
        "--stretch_scope",
        choices=("mosaic", "tile"),
        default="mosaic",
        help="Measure the stretch once per mosaic, which keeps tiles comparable, or "
        "per tile.",
    )
    p.add_argument(
        "--levels",
        type=Path,
        default=None,
        help="Use one stretch range per filter for every mosaic, from a file "
        "`jwst-peculiar levels` wrote, so all the visits of a survey look alike. "
        "Colour mosaics only.",
    )
    p.add_argument(
        "--p_lo",
        "--plo",
        dest="p_lo",
        type=float,
        default=1.0,
        help="Low stretch percentile.",
    )
    p.add_argument(
        "--p_hi",
        "--phi",
        dest="p_hi",
        type=float,
        default=99.8,
        help="High stretch percentile.",
    )
    p.add_argument(
        "--asinh_q",
        "--asinhq",
        dest="asinh_q",
        type=float,
        default=10.0,
        help="Asinh softening.",
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
        help="Also save each tile's raw float pixels as FITS with its WCS, for the "
        "web viewer.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear tiles left in --out by an earlier run instead of stopping.",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="Stop after this many mosaics."
    )
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
    """Parse the shared flags and the chosen sensor's own.

    Returns:
        The parsed arguments.
    """
    # Each sensor has its own flags, so find the sensor first. This parser has no
    # --help, which lets -h reach the full parser and list the sensor's flags too.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sensor", choices=sorted(SENSORS), default="fits")
    sensor = pre.parse_known_args(argv)[0].sensor
    p = build_parser()
    SENSORS[sensor].add_args(p)
    return p.parse_args(argv)


def _check_levels(args: argparse.Namespace) -> None:
    if args.stretch_scope != "mosaic":
        msg = "--levels replaces the stretch of each mosaic, so it needs "
        msg += "--stretch_scope mosaic"
        raise SystemExit(msg)
    if not args.levels.is_file():
        msg = f"There is no levels file at {args.levels}"
        raise SystemExit(msg)
    if args.sensor == "miri":
        msg = "--levels is for colour mosaics, and --sensor miri cuts grey ones"
        raise SystemExit(msg)


def _check_args(args: argparse.Namespace) -> None:
    for flag in ("tile_size", "upscale", "stride"):
        value = getattr(args, flag)
        if value is not None and value < 1:
            msg = f"--{flag} has to be at least 1, got {value}"
            raise SystemExit(msg)
    if args.limit is not None and args.limit < 0:
        msg = f"--limit can't be negative, got {args.limit}"
        raise SystemExit(msg)
    if args.asinh_q <= 0:
        msg = f"--asinh_q has to be above 0, got {args.asinh_q}"
        raise SystemExit(msg)
    if args.levels is not None:
        _check_levels(args)
    if args.sensor != "fits" and not args.indir:
        msg = f"--sensor {args.sensor} needs --indir"
        raise SystemExit(msg)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["key"])
        writer.writeheader()
        writer.writerows(records)


def run(argv: list[str] | None = None) -> None:
    """Cut every mosaic the sensor finds into tiles, and write the manifest."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    _check_args(args)

    out = Path(args.out)
    query_dir = out / "query"
    sensor = SENSORS[args.sensor]
    # Before any folder is made, so a mistyped --pattern doesn't leave an output the
    # next run refuses to write over.
    specs = list(sensor.iter_mosaic_specs(args))[: args.limit]
    if not args.dry_run:
        clear_previous_output(out, overwrite=args.overwrite)
        query_dir.mkdir(parents=True)
        (out / "mosaics").mkdir()

    global_idx = 0
    records: list[dict[str, Any]] = []
    for spec in tqdm(specs, desc="Mosaics", unit="mosaic"):
        global_idx = sensor.process_one_mosaic(
            args, spec, query_dir=query_dir, global_idx=global_idx, records=records
        )
        if args.gc_each_mosaic:
            gc.collect()
    if not args.dry_run:
        _write_manifest(out / "manifest.csv", records)

    logger.info(
        "%s %d tiles from %d mosaics in %s",
        "Would write" if args.dry_run else "Wrote",
        global_idx,
        len(specs),
        query_dir,
    )


if __name__ == "__main__":
    run()
