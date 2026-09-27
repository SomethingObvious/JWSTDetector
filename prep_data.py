#!/usr/bin/env python3
"""Cuts JWST FITS mosaics into PNG tiles for the detector, one mosaic at a time.

Each mosaic stays memmapped while it is cut, so memory stays flat however big the
mosaic is. The sky level, its noise and the stretch limits are measured once per
mosaic, and every threshold is expressed against those, which means the same flag
values work whatever units the mosaic is in and however deep the exposure went.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import re
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS, FITSFixedWarning
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("prep_data")


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
    so they don't turn into black shapes that the detector then flags.
    """
    img = np.nan_to_num(img, nan=fill, posinf=fill, neginf=fill).astype(np.float32, copy=False)
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
    if stats.sigma <= 0.0:
        return bool(np.any(finite & (cut != stats.sky)))
    above = finite & (cut > stats.sky + source_sigma * stats.sigma)
    return float(above.sum()) / cut.size >= min_frac


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


def celestial_wcs(header: fits.Header, source: str) -> WCS:
    with warnings.catch_warnings():
        # JWST headers get a few harmless date and unit fixes that astropy warns about.
        warnings.simplefilter("ignore", FITSFixedWarning)
        wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise ValueError(f"{source} has no sky coordinates in its header. Drop --wcs_in_name.")
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
    wht: np.ndarray | None,
    wcs: WCS | None,
    query_dir: Path,
    global_idx: int,
    colour: Callable[[tuple[slice, slice]], np.ndarray] | None = None,
) -> tuple[int, int, int, int]:
    """Cut, filter and save the tiles of one mosaic.

    `signal` is the raw flux that the emptiness cut and the grey stretch read, and
    `colour` renders the saved tile for a window in place of the grey stretch.
    Returns the next global index and the saved, empty and low-WHT counts.
    """
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
    for y in tile_starts(h, crop, stride):
        for x in tile_starts(w, crop, stride):
            window = np.s_[y : y + crop, x : x + crop]
            if wht is not None and args.min_wht_frac > 0:
                wcut = wht[window]
                if np.mean(np.isfinite(wcut) & (wcut > 0)) < args.min_wht_frac:
                    skipped_wht += 1
                    continue

            # Judge emptiness on raw flux, before a stretch flattens the difference
            # between blank sky and a tile with something in it.
            raw = np.asarray(signal[window], dtype=np.float32)
            if not has_signal(raw, stats, args.source_sigma, args.min_source_frac):
                skipped_empty += 1
                continue

            if not args.dry_run:
                if colour is not None:
                    tile = colour(window)
                else:
                    s = stats
                    if args.stretch_scope == "tile":
                        s = measure_mosaic(raw, args.p_lo, args.p_hi)
                    tile = asinh_to_uint8(raw, s.lo, s.hi, args.asinh_q, fill=s.sky)
                cx, cy = x + (crop - 1) / 2, y + (crop - 1) / 2
                radec = None
                if wcs is not None:
                    ra, dec = wcs.all_pix2world(cx, cy, 0)
                    radec = (float(ra), float(dec))
                img = Image.fromarray(tile)
                if img.size != (args.tile_size, args.tile_size):
                    img = img.resize((args.tile_size, args.tile_size), Image.Resampling.BICUBIC)
                img.save(query_dir / make_output_name(tile_id, cx, cy, global_idx, radec))
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
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: Path, global_idx: int
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
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: Path, global_idx: int
    ) -> int:
        # Everything is cut while the files are open, since the arrays are memmaps.
        with contextlib.ExitStack() as stack:
            sci_hdul = stack.enter_context(fits.open(spec.sci_path, memmap=True))
            sci, header = find_image(sci_hdul, (args.sci_ext, "SCI"))
            wht = None
            if spec.wht_path == spec.sci_path:
                wht, _ = find_image(sci_hdul, (args.wht_ext, "WHT"), fallback=False)
            elif spec.wht_path is not None:
                wht_hdul = stack.enter_context(fits.open(spec.wht_path, memmap=True))
                wht, _ = find_image(wht_hdul, (args.wht_ext, "WHT", "SCI"))
            wcs = celestial_wcs(header, spec.sci_path) if args.wcs_in_name else None
            return cut_tiles(args, spec.tile_id, sci, wht, wcs, query_dir, global_idx)[0]


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
        p.add_argument("--nircam_load_wht", default=False, action=argparse.BooleanOptionalAction)
        p.add_argument("--wht_filter", type=str, default=None)
        p.add_argument("--stretch", type=float, default=0.5, help="Lupton stretch.")
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

    def process_one_mosaic(
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: Path, global_idx: int
    ) -> int:
        if len(args.rgb_filters) not in (1, 3):
            raise ValueError(f"--rgb_filters takes 1 or 3 filters, not {len(args.rgb_filters)}")
        load_wht = args.nircam_load_wht and args.min_wht_frac > 0
        wht_filter = args.wht_filter or args.rgb_filters[0]

        with contextlib.ExitStack() as stack:
            planes, header, wht = [], None, None
            for filt in args.rgb_filters:
                path = self._build_path(args, spec.tile_id, filt)
                if not Path(path).is_file():
                    raise FileNotFoundError(
                        f"There is no {filt} mosaic for tile {spec.tile_id} at {path}"
                    )
                hdul = stack.enter_context(fits.open(path, memmap=True))
                data, hdr = find_image(hdul, (args.sci_ext, "SCI"))
                planes.append(data)
                header = hdr if header is None else header
                if load_wht and filt == wht_filter:
                    wht, _ = find_image(hdul, (args.wht_ext, "WHT"), fallback=False)
            if load_wht and wht is None:
                raise ValueError(f"--wht_filter {wht_filter} isn't one of --rgb_filters")

            colour = None
            if len(planes) == 3:

                def colour(window: tuple[slice, slice]) -> np.ndarray:
                    r, g, b = (np.nan_to_num(np.asarray(p[window], np.float32)) for p in planes)
                    return make_lupton_rgb(r, g, b, stretch=args.stretch, Q=args.Q)

            wcs = celestial_wcs(header, spec.tile_id) if args.wcs_in_name else None
            signal = planes[0]
            return cut_tiles(args, spec.tile_id, signal, wht, wcs, query_dir, global_idx, colour)[0]


SENSORS: dict[str, Sensor] = {"miri": MiriSensor(), "nircam": NircamSensor()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cut JWST FITS mosaics into PNG tiles for anomaly detection."
    )
    p.add_argument(
        "--sensor", choices=sorted(SENSORS), default="miri", help="Instrument layout to read."
    )
    p.add_argument("--indir", required=True, help="Folder holding the FITS mosaics.")
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
    pre.add_argument("--sensor", choices=sorted(SENSORS), default="miri")
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

    query_dir = Path(args.out) / "query"
    if not args.dry_run:
        query_dir.mkdir(parents=True, exist_ok=True)
    sensor = SENSORS[args.sensor]
    specs = list(sensor.iter_mosaic_specs(args))[: args.limit]

    global_idx = 0
    for spec in tqdm(specs, desc="Mosaics", unit="mosaic"):
        global_idx = sensor.process_one_mosaic(args, spec, query_dir, global_idx)
        if args.gc_each_mosaic:
            gc.collect()

    logger.info(
        "%s %d tiles from %d mosaics in %s",
        "Would write" if args.dry_run else "Wrote",
        global_idx,
        len(specs),
        query_dir,
    )


if __name__ == "__main__":
    run()
