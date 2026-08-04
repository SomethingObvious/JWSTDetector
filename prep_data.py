#!/usr/bin/env python3
"""Cuts JWST FITS mosaics into PNG tiles for the detector, one mosaic at a time.

Each mosaic is cut, closed and collected before the next one, so memory stays flat
however many there are, and the tiles go in out/query/. The sky level and its noise
are measured once per mosaic from a strided sample, and every threshold is expressed
against those, which means the same flag values work whatever units the mosaic is in
and however deep the exposure went.
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger("prep_data")

try:
    from astropy.visualization import make_lupton_rgb
except Exception:
    make_lupton_rgb = None

try:
    from astropy.wcs import WCS
except Exception:
    WCS = None


def _find_hdu_with_data(hdul, prefer_extnames: Sequence[str]) -> tuple[np.ndarray, fits.Header]:
    """Data and header of the first 2D image named in `prefer_extnames`, or of any name.

    It hands back hdu.data itself, a memmap view, so the mosaic never has to fit in RAM.
    """
    for extname in prefer_extnames:
        for hdu in hdul:
            if getattr(hdu, "name", None) == extname and getattr(hdu, "data", None) is not None:
                data = hdu.data
                if data is not None and getattr(data, "ndim", 0) >= 2:
                    return data, hdu.header

    for hdu in hdul:
        if getattr(hdu, "data", None) is None:
            continue
        data = hdu.data
        if data is not None and getattr(data, "ndim", 0) >= 2:
            return data, hdu.header

    raise RuntimeError("There is no 2D image in this FITS file.")


@dataclass
class MosaicStats:
    """Sky level, noise and stretch limits, measured once for a whole mosaic."""

    sky: float
    sigma: float
    lo: float
    hi: float


def subsample(data: np.ndarray, max_values: int = 1 << 24) -> np.ndarray:
    """Up to `max_values` evenly spaced finite pixels of a mosaic.

    Mosaics can run to several GB, so the statistics use a strided view of the memmap.
    """
    h, w = data.shape[:2]
    step = max(1, int(np.ceil(np.sqrt(h * w / max_values))))
    sample = np.asarray(data[::step, ::step], dtype=np.float32).reshape(-1)
    return sample[np.isfinite(sample)]


def measure_mosaic(data: np.ndarray, p_lo: float, p_hi: float) -> MosaicStats:
    """Median sky, MAD noise and percentile stretch limits for one mosaic."""
    sample = subsample(data)
    if sample.size == 0:
        return MosaicStats(sky=0.0, sigma=0.0, lo=0.0, hi=1.0)

    sky = float(np.median(sample))
    # The MAD scaled to a Gaussian sigma, which unlike std doesn't see the sources.
    sigma = float(np.median(np.abs(sample - sky)) * 1.4826)
    lo, hi = (float(v) for v in np.percentile(sample, [p_lo, p_hi]))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(sample.min()), float(sample.max()) + 1e-6
    return MosaicStats(sky=sky, sigma=sigma, lo=lo, hi=hi)


def asinh_to_uint8(img: np.ndarray, lo: float, hi: float, asinh_q: float = 10.0) -> np.ndarray:
    """Flux to 8 bits through an asinh stretch between fixed limits.

    The limits are arguments on purpose. Stretching each tile between its own
    percentiles brings faint sky and a bright source up to the same range, which
    hides the very difference the detector is looking for.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    x = np.clip((img - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    return (255.0 * np.arcsinh(asinh_q * x) / np.arcsinh(asinh_q)).astype(np.uint8)


def has_signal(
    cut: np.ndarray, stats: MosaicStats, source_sigma: float = 5.0, min_frac: float = 2e-4
) -> bool:
    """True when enough of the tile rises above the sky to be worth embedding.

    It reads raw flux against the mosaic's noise, so the default works for MIRI and
    NIRCam alike, as pure sky puts next to nothing above 5 sigma.
    """
    if stats.sigma <= 0.0:
        return bool(np.any(np.isfinite(cut) & (cut != stats.sky)))
    finite = np.isfinite(cut)
    if not finite.any():
        return False
    above = finite & (cut > stats.sky + source_sigma * stats.sigma)
    return float(above.sum()) / float(finite.sum()) >= min_frac


def ensure_dirs_query_only(out_root: str) -> str:
    query_dir = Path(out_root) / "query"
    query_dir.mkdir(parents=True, exist_ok=True)
    return str(query_dir)


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
    if s.startswith("-"):
        s = "m" + s[1:]
    return s


def maybe_center_radec(wcs_obj, cx: float, cy: float) -> tuple[float, float] | None:
    if wcs_obj is None:
        return None
    try:
        ra, dec = wcs_obj.all_pix2world(cx, cy, 0)
        if np.isfinite(ra) and np.isfinite(dec):
            return float(ra), float(dec)
    except Exception:
        return None
    return None


def make_output_name(
    tile_id: str,
    cx: float,
    cy: float,
    global_idx: int,
    wcs_ra_dec: tuple[float, float] | None = None,
    extra_tag: str | None = None,
) -> str:
    cx_i = round(cx)
    cy_i = round(cy)
    parts = [tile_id, f"cx{cx_i:06d}", f"cy{cy_i:06d}"]
    if wcs_ra_dec is not None:
        ra, dec = wcs_ra_dec
        parts.append(f"ra{_safe_float_tag(ra, 6)}")
        parts.append(f"dec{_safe_float_tag(dec, 6)}")
    if extra_tag:
        parts.append(extra_tag)
    parts.append(f"i{global_idx:08d}")
    return "_".join(parts) + ".png"


@dataclass
class MosaicSpec:
    tile_id: str
    sci_path: str
    wht_path: str | None = None
    src_tag: str | None = None


class Sensor:
    name: str = "base"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        raise NotImplementedError

    def process_one_mosaic(
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int
    ) -> int:
        raise NotImplementedError


class MiriSensor(Sensor):
    """One SCI mosaic per file, with an optional WHT mosaic beside it.

    The WHT file has the same name with the trailing sci.fits swapped for wht.fits, so
    a5_sci.fits pairs with a5_wht.fits.
    """

    name = "miri"
    _SCI2WHT_RE = re.compile(r"(?i)sci\.fits$")

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--pattern", default="*sci.fits", help="Glob for the science mosaics."
        )
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")

    @classmethod
    def sci_to_wht_path(cls, sci_path: str) -> str:
        if cls._SCI2WHT_RE.search(sci_path):
            return cls._SCI2WHT_RE.sub("wht.fits", sci_path)
        p = Path(sci_path)
        if p.suffix.lower() == ".fits" and p.stem.lower().endswith("sci"):
            return str(p.with_name(f"{p.stem[:-3]}wht{p.suffix}"))
        return sci_path

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        sci_paths = sorted(str(p) for p in Path(args.indir).glob(args.pattern) if p.is_file())
        if not sci_paths:
            raise RuntimeError(f"Nothing in {args.indir} matches {args.pattern}")

        for sci_path in sci_paths:
            tile_id = (
                args.tile_id or infer_tile_id_from_filename(sci_path, args.tile_regex) or "UNKNOWN"
            )
            wht_path = None
            if args.min_wht_frac > 0:
                wht_path = self.sci_to_wht_path(sci_path)
                if not Path(wht_path).is_file():
                    raise RuntimeError(
                        f"--min_wht_frac needs a weight map for {sci_path}, and {wht_path} "
                        "isn't there. Pass --min_wht_frac 0 to tile it without one."
                    )
            yield MosaicSpec(
                tile_id=tile_id,
                sci_path=sci_path,
                wht_path=wht_path,
                src_tag=Path(sci_path).stem,
            )

    def _emit_tiles(self, args, spec, query_dir, global_idx, sci_data, sci_hdr, wht_data):
        """Cut, filter and save one SCI array's tiles, with the WHT cut when there is wht_data."""
        crop = int(args.tile_size / args.upscale)
        stride = args.stride if args.stride is not None else max(1, crop // 2)

        wcs_obj = None
        if args.wcs_in_name and WCS is not None:
            try:
                wcs_obj = WCS(sci_hdr)
            except Exception:
                wcs_obj = None

        stats = measure_mosaic(sci_data, args.p_lo, args.p_hi)
        logger.info(
            "[%s] sky=%.4g sigma=%.4g stretch=[%.4g, %.4g]",
            spec.tile_id,
            stats.sky,
            stats.sigma,
            stats.lo,
            stats.hi,
        )

        saved = skipped_empty = skipped_wht = 0
        h, w = sci_data.shape[:2]
        for y in range(0, h - crop + 1, stride):
            for x in range(0, w - crop + 1, stride):
                cx = x + (crop - 1) / 2.0
                cy = y + (crop - 1) / 2.0

                if wht_data is not None:
                    wcut = wht_data[y : y + crop, x : x + crop]
                    good_frac = float(np.mean(np.isfinite(wcut) & (wcut > 0)))
                    if good_frac < args.min_wht_frac:
                        skipped_wht += 1
                        continue

                cut = np.asarray(sci_data[y : y + crop, x : x + crop], dtype=np.float32)

                # Judge emptiness on raw flux, before a stretch flattens the difference
                # between blank sky and a tile with something in it.
                if not has_signal(cut, stats, args.source_sigma, args.min_source_frac):
                    skipped_empty += 1
                    continue

                tile_stats = (
                    stats
                    if args.stretch_scope == "mosaic"
                    else measure_mosaic(cut, args.p_lo, args.p_hi)
                )
                tile_u8 = asinh_to_uint8(cut, tile_stats.lo, tile_stats.hi, args.asinh_q)
                rgb_u8 = np.stack([tile_u8] * 3, axis=-1)

                radec = maybe_center_radec(wcs_obj, cx, cy) if args.wcs_in_name else None
                sample_name = make_output_name(spec.tile_id, cx, cy, global_idx, wcs_ra_dec=radec)
                if not args.dry_run:
                    out_img = Image.fromarray(rgb_u8).resize(
                        (args.tile_size, args.tile_size), resample=Image.BICUBIC
                    )
                    out_img.save(Path(query_dir) / sample_name)

                global_idx += 1
                saved += 1

        return global_idx, saved, skipped_empty, skipped_wht

    def process_one_mosaic(
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int
    ) -> int:
        # Everything is cut while the files are open, since the arrays are memmaps.
        with fits.open(spec.sci_path, memmap=True) as hdul_sci:
            sci_data, sci_hdr = _find_hdu_with_data(hdul_sci, prefer_extnames=(args.sci_ext, "SCI"))

            if spec.wht_path is not None:
                with fits.open(spec.wht_path, memmap=True) as hdul_wht:
                    wht_data, _ = _find_hdu_with_data(
                        hdul_wht, prefer_extnames=(args.wht_ext, "WHT", "SCI")
                    )
                    global_idx, saved, skipped_empty, skipped_wht = self._emit_tiles(
                        args, spec, query_dir, global_idx, sci_data, sci_hdr, wht_data
                    )
            else:
                global_idx, saved, skipped_empty, skipped_wht = self._emit_tiles(
                    args, spec, query_dir, global_idx, sci_data, sci_hdr, None
                )

        verb = "Would save" if args.dry_run else "Saved"
        logger.info("[%s] %s %d PNG tiles", spec.tile_id, verb, saved)
        logger.info(
            "[%s] Skipped %d as empty and %d for low WHT", spec.tile_id, skipped_empty, skipped_wht
        )

        if args.gc_each_mosaic:
            gc.collect()

        return global_idx


class NircamSensor(Sensor):
    """COSMOS-Web style mosaics, one file per filter, combined into colour tiles.

    It still goes one tile id at a time, but every filter for that tile is loaded into
    RAM in full to build the colour image.
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
        p.add_argument("--stretch", type=float, default=0.5)
        p.add_argument("--Q", type=float, default=10.0)

    def _infer_tiles_from_dir(self, indir: str) -> list[str]:
        all_fits = [str(p) for p in Path(indir).glob("*.fits")]
        tiles: list[str] = []
        for fp in all_fits:
            tid = infer_tile_id_from_filename(fp, None)
            if tid:
                tiles.append(tid)
        tiles = sorted(set(tiles))
        if not tiles:
            raise RuntimeError(
                "Couldn't find a tile id in any file name in --indir. Pass --tiles."
            )
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
        tiles = (
            args.tiles
            if args.tiles and len(args.tiles) > 0
            else self._infer_tiles_from_dir(args.indir)
        )
        for tile_id in tiles:
            # A mosaic here is one tile id with all of its filters.
            yield MosaicSpec(tile_id=tile_id, sci_path="", wht_path=None, src_tag=tile_id)

    def process_one_mosaic(
        self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int
    ) -> int:
        if len(args.rgb_filters) not in (1, 3):
            raise ValueError("--rgb_filters takes 1 or 3 filters")
        if make_lupton_rgb is None and len(args.rgb_filters) == 3:
            raise RuntimeError("Three NIRCam filters need make_lupton_rgb, and this astropy doesn't have it")

        crop = int(args.tile_size / args.upscale)
        stride = args.stride if args.stride is not None else max(1, crop // 2)

        saved = 0
        skipped_empty = 0
        skipped_wht = 0

        planes: list[np.ndarray] = []
        hdr_for_wcs = None
        wht_filt = args.wht_filter or args.rgb_filters[0]
        wht = None

        for filt in args.rgb_filters:
            fp = self._build_path(args, spec.tile_id, filt)
            if not Path(fp).is_file():
                raise RuntimeError(f"There is no mosaic for tile {spec.tile_id} in {filt} at {fp}")

            with fits.open(fp, memmap=True) as hdul:
                data, hdr = _find_hdu_with_data(hdul, prefer_extnames=(args.sci_ext, "SCI"))
                planes.append(
                    np.nan_to_num(np.asarray(data), nan=0.0, posinf=0.0, neginf=0.0).astype(
                        np.float32
                    )
                )
                if hdr_for_wcs is None and filt == wht_filt:
                    hdr_for_wcs = hdr

                if args.nircam_load_wht and args.min_wht_frac > 0 and filt == wht_filt:
                    try:
                        wht_arr, _ = _find_hdu_with_data(
                            hdul, prefer_extnames=(args.wht_ext, "WHT")
                        )
                        # Copied, as the tile loop reads `wht` after this file is closed,
                        # and a memmap view would be reading a closed file by then.
                        wht = np.array(wht_arr)
                    except Exception:
                        wht = None

        wcs_obj = None
        if args.wcs_in_name and WCS is not None and hdr_for_wcs is not None:
            try:
                wcs_obj = WCS(hdr_for_wcs)
            except Exception:
                wcs_obj = None

        # The emptiness cut reads raw flux from the first filter, even when the saved
        # tile is a three-filter colour composite.
        signal_plane = planes[0]
        stats = measure_mosaic(signal_plane, args.p_lo, args.p_hi)
        logger.info(
            "[%s] sky=%.4g sigma=%.4g stretch=[%.4g, %.4g]",
            spec.tile_id,
            stats.sky,
            stats.sigma,
            stats.lo,
            stats.hi,
        )

        if len(planes) == 1:
            img = planes[0]
        else:
            r, g, b = planes
            img = make_lupton_rgb(r, g, b, stretch=args.stretch, Q=args.Q)

        h, w = img.shape[:2]

        for y in range(0, h - crop + 1, stride):
            for x in range(0, w - crop + 1, stride):
                cx = x + (crop - 1) / 2.0
                cy = y + (crop - 1) / 2.0

                if wht is not None and args.min_wht_frac > 0:
                    wcut = wht[y : y + crop, x : x + crop]
                    good_frac = float(np.mean(np.isfinite(wcut) & (wcut > 0)))
                    if good_frac < args.min_wht_frac:
                        skipped_wht += 1
                        continue

                raw = signal_plane[y : y + crop, x : x + crop]
                if not has_signal(raw, stats, args.source_sigma, args.min_source_frac):
                    skipped_empty += 1
                    continue

                if img.ndim == 2:
                    tile_stats = (
                        stats
                        if args.stretch_scope == "mosaic"
                        else measure_mosaic(raw, args.p_lo, args.p_hi)
                    )
                    tile_u8 = asinh_to_uint8(raw, tile_stats.lo, tile_stats.hi, args.asinh_q)
                    rgb_u8 = np.stack([tile_u8] * 3, axis=-1)
                else:
                    # make_lupton_rgb normally hands back stretched uint8 already.
                    cut = img[y : y + crop, x : x + crop, :]
                    if cut.dtype == np.uint8:
                        rgb_u8 = cut
                    else:
                        cutf = np.nan_to_num(cut.astype(np.float32), nan=0.0)
                        if cutf.max() <= 1.0:
                            cutf = 255.0 * cutf
                        rgb_u8 = np.clip(cutf, 0.0, 255.0).astype(np.uint8)

                radec = maybe_center_radec(wcs_obj, cx, cy) if args.wcs_in_name else None
                sample_name = make_output_name(spec.tile_id, cx, cy, global_idx, wcs_ra_dec=radec)
                if not args.dry_run:
                    out_img = Image.fromarray(rgb_u8).resize(
                        (args.tile_size, args.tile_size), resample=Image.BICUBIC
                    )
                    out_img.save(Path(query_dir) / sample_name)

                global_idx += 1
                saved += 1

        verb = "Would save" if args.dry_run else "Saved"
        logger.info("[%s] %s %d PNG tiles", spec.tile_id, verb, saved)
        logger.info(
            "[%s] Skipped %d as empty and %d for low WHT", spec.tile_id, skipped_empty, skipped_wht
        )

        if args.gc_each_mosaic:
            del planes
            del img
            del wht
            gc.collect()

        return global_idx


SENSORS: dict[str, Sensor] = {"miri": MiriSensor(), "nircam": NircamSensor()}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Cut JWST FITS mosaics into PNG tiles for anomaly detection, "
            "one mosaic at a time so memory stays flat."
        )
    )
    p.add_argument(
        "--sensor",
        choices=sorted(SENSORS.keys()),
        default="miri",
        help="Instrument layout to read.",
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
        help="Put the RA and Dec of each tile centre in its file name. It needs a valid WCS.",
    )
    p.add_argument(
        "--tile_size", type=int, default=518, help="Saved tile size in pixels."
    )
    p.add_argument(
        "--upscale", type=int, default=2, help="Crop tile_size/upscale pixels and resize up."
    )
    p.add_argument(
        "--stride", type=int, default=None, help="Step between crops. Half a crop by default."
    )

    # The old spellings without underscores still work.
    p.add_argument(
        "--min_wht_frac",
        "--minwhtfrac",
        dest="min_wht_frac",
        type=float,
        default=0.0,
        help="Drop tiles where less than this fraction of pixels has a positive weight. 0 turns it off.",
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
        "--p_hi",
        "--phi",
        dest="p_hi",
        type=float,
        default=99.8,
        help="High stretch percentile.",
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
        "--limit",
        type=int,
        default=None,
        help="Stop after this many mosaics, for a quick trial run.",
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


def parse_args_two_stage() -> argparse.Namespace:
    p = build_parser()
    known, _ = p.parse_known_args()
    SENSORS[known.sensor].add_args(p)
    return p.parse_args()


def run() -> None:
    args = parse_args_two_stage()
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
    if args.dry_run:
        logger.info("Dry run, so tiles get cut and counted but no PNGs are written.")

    query_dir = ensure_dirs_query_only(args.out)
    sensor = SENSORS[args.sensor]

    specs = list(sensor.iter_mosaic_specs(args))
    if args.limit is not None:
        if args.limit < 0:
            # A negative slice would quietly drop mosaics off the end instead.
            raise SystemExit(f"--limit can't be negative, got {args.limit}")
        specs = specs[: args.limit]

    global_idx = 0
    for spec in tqdm(specs, desc="Mosaics", unit="mosaic"):
        global_idx = sensor.process_one_mosaic(args, spec, query_dir, global_idx)

        # Collected again out here, as if each mosaic had been its own run.
        if args.gc_each_mosaic:
            gc.collect()

    logger.info("Done")
    logger.info("Sensor was %s", args.sensor)
    logger.info("Read %d mosaics", len(specs))
    logger.info("%s %d PNG tiles in all", "Would have saved" if args.dry_run else "Saved", global_idx)


if __name__ == "__main__":
    run()
