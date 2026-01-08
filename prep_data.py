#!/usr/bin/env python3
"""
prep_data.py (streaming per mosaic)

Key behavior:
- Processes exactly ONE mosaic at a time (SCI + optional WHT), then closes FITS, deletes arrays,
  runs gc.collect(), and moves to the next mosaic.
- For MIRI: expects paired files with same name except 'sci.fits' vs 'wht.fits'
  (e.g., a5_sci.fits -> a5_wht.fits).
- Prints per-mosaic:
    Saved PNG tiles: {saved}
    Skipped: {skipped_empty} empty-ish, {skipped_wht} low-WHT

Output structure:
out/
  query/
"""

from __future__ import annotations

import os
import re
import glob
import gc
import argparse
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from PIL import Image

try:
    from astropy.visualization import make_lupton_rgb
except Exception:
    make_lupton_rgb = None

try:
    from astropy.wcs import WCS
except Exception:
    WCS = None


# -----------------------------
# Core utilities
# -----------------------------
def _find_hdu_with_data(hdul, prefer_extnames: Sequence[str]) -> Tuple[np.ndarray, fits.Header]:
    """
    Return (data, header) for the first 2D+ HDU matching preferred extnames, else first 2D+ HDU.

    IMPORTANT: returns hdu.data directly (memmap view) to avoid loading the whole mosaic into RAM.
    """
    # preferred by extname
    for extname in prefer_extnames:
        for hdu in hdul:
            if getattr(hdu, "name", None) == extname and getattr(hdu, "data", None) is not None:
                data = hdu.data
                if data is not None and getattr(data, "ndim", 0) >= 2:
                    return data, hdu.header

    # fallback: first image-like HDU
    for hdu in hdul:
        if getattr(hdu, "data", None) is None:
            continue
        data = hdu.data
        if data is not None and getattr(data, "ndim", 0) >= 2:
            return data, hdu.header

    raise RuntimeError("No 2D image data found in FITS file.")


def robust_asinh_to_uint8(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.8, asinh_q: float = 10.0) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(img))
        hi = float(np.max(img) + 1e-6)
    x = (img - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0.0, 1.0)
    x = np.arcsinh(asinh_q * x) / np.arcsinh(asinh_q)
    return (255.0 * x).astype(np.uint8)


def mostly_empty_rgb(tile_u8_rgb: np.ndarray, mean_thresh: float = 2.0, var_thresh: float = 1.0) -> bool:
    g = tile_u8_rgb.mean(axis=-1)
    return (float(g.mean()) < mean_thresh) or (float(g.var()) < var_thresh)


def ensure_dirs_query_only(out_root: str) -> str:
    query_dir = os.path.join(out_root, "query")
    os.makedirs(query_dir, exist_ok=True)
    return query_dir


# -----------------------------
# Naming helpers
# -----------------------------
_TILE_FALLBACK_RE = re.compile(r"(?:^|[_\-])([A-Z]\d{1,2})(?:[_\-]|$)")


def infer_tile_id_from_filename(path: str, tile_regex: Optional[str] = None) -> Optional[str]:
    base = os.path.basename(path)
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


def maybe_center_radec(wcs_obj, cx: float, cy: float) -> Optional[Tuple[float, float]]:
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
    wcs_ra_dec: Optional[Tuple[float, float]] = None,
    extra_tag: Optional[str] = None,
) -> str:
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    parts = [tile_id, f"cx{cx_i:06d}", f"cy{cy_i:06d}"]
    if wcs_ra_dec is not None:
        ra, dec = wcs_ra_dec
        parts.append(f"ra{_safe_float_tag(ra, 6)}")
        parts.append(f"dec{_safe_float_tag(dec, 6)}")
    if extra_tag:
        parts.append(extra_tag)
    parts.append(f"i{global_idx:08d}")
    return "_".join(parts) + ".png"


# -----------------------------
# Mosaic specs + sensors
# -----------------------------
@dataclass
class MosaicSpec:
    tile_id: str
    sci_path: str
    wht_path: Optional[str] = None
    src_tag: Optional[str] = None


class Sensor:
    name: str = "base"

    def add_args(self, p: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        raise NotImplementedError

    def process_one_mosaic(self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int) -> int:
        raise NotImplementedError


class MiriSensor(Sensor):
    """
    MIRI streaming: one SCI FITS at a time + optional WHT FITS.

    Pairing rule:
      <same name> except trailing 'sci.fits' -> 'wht.fits'
      Example: ".../a5_sci.fits" -> ".../a5_wht.fits"
    """
    name = "miri"
    _SCI2WHT_RE = re.compile(r"(?i)sci\.fits$")  # case-insensitive trailing

    def add_args(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--pattern", default="*sci.fits", help="Glob for MIRI SCI files (e.g., '*sci.fits').")
        p.add_argument("--sci_ext", default="SCI")
        p.add_argument("--wht_ext", default="WHT")

    @classmethod
    def sci_to_wht_path(cls, sci_path: str) -> str:
        if cls._SCI2WHT_RE.search(sci_path):
            return cls._SCI2WHT_RE.sub("wht.fits", sci_path)
        base, ext = os.path.splitext(sci_path)
        if ext.lower() == ".fits" and base.lower().endswith("sci"):
            return base[:-3] + "wht" + ext
        return sci_path

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        sci_paths = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
        sci_paths = [fp for fp in sci_paths if os.path.isfile(fp)]
        if not sci_paths:
            raise RuntimeError(f"No SCI FITS found in {args.indir} with pattern {args.pattern}")

        for sci_path in sci_paths:
            tile_id = args.tile_id or infer_tile_id_from_filename(sci_path, args.tile_regex) or "UNKNOWN"
            wht_path = None
            if args.min_wht_frac > 0:
                wht_path = self.sci_to_wht_path(sci_path)
                if not os.path.isfile(wht_path):
                    raise RuntimeError(
                        "min_wht_frac>0 but matching WHT file not found.\n"
                        f"SCI: {sci_path}\nWHT: {wht_path}\n"
                        "Expected same name except sci.fits -> wht.fits"
                    )
            yield MosaicSpec(
                tile_id=tile_id,
                sci_path=sci_path,
                wht_path=wht_path,
                src_tag=os.path.splitext(os.path.basename(sci_path))[0],
            )

    def process_one_mosaic(self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int) -> int:
        crop = int(args.tile_size / args.upscale)
        stride = args.stride if args.stride is not None else max(1, crop // 2)

        saved = 0
        skipped_empty = 0
        skipped_wht = 0

        # Open FITS with memmap so sci_data is not fully loaded into RAM. [file:13]
        with fits.open(spec.sci_path, memmap=True) as hdul_sci:
            sci_data, sci_hdr = _find_hdu_with_data(hdul_sci, prefer_extnames=(args.sci_ext, "SCI"))

            wht_data = None
            if spec.wht_path is not None:
                with fits.open(spec.wht_path, memmap=True) as hdul_wht:
                    wht_data, _ = _find_hdu_with_data(hdul_wht, prefer_extnames=(args.wht_ext, "WHT", "SCI"))
                    # Note: wht_data is only valid inside this context if it is memmapped.
                    # To use it outside, keep processing inside this nested context.
                    # So: do all tile cutting while WHT file is open.
                    wcs_obj = None
                    if args.wcs_in_name and WCS is not None:
                        try:
                            wcs_obj = WCS(sci_hdr)
                        except Exception:
                            wcs_obj = None

                    h, w = sci_data.shape[:2]
                    for y in range(0, h - crop + 1, stride):
                        for x in range(0, w - crop + 1, stride):
                            cx = x + (crop - 1) / 2.0
                            cy = y + (crop - 1) / 2.0

                            wcut = wht_data[y : y + crop, x : x + crop]
                            good_frac = float(np.mean(np.isfinite(wcut) & (wcut > 0)))
                            if good_frac < args.min_wht_frac:
                                skipped_wht += 1
                                continue

                            cut = sci_data[y : y + crop, x : x + crop]
                            tile_u8 = robust_asinh_to_uint8(cut, p_lo=args.p_lo, p_hi=args.p_hi, asinh_q=args.asinh_q)
                            rgb_u8 = np.stack([tile_u8, tile_u8, tile_u8], axis=-1)

                            if mostly_empty_rgb(rgb_u8, mean_thresh=args.mean_thresh, var_thresh=args.var_thresh):
                                skipped_empty += 1
                                continue

                            out_img = Image.fromarray(rgb_u8).resize((args.tile_size, args.tile_size), resample=Image.BICUBIC)
                            radec = maybe_center_radec(wcs_obj, cx, cy) if args.wcs_in_name else None
                            sample_name = make_output_name(spec.tile_id, cx, cy, global_idx, wcs_ra_dec=radec)
                            out_img.save(os.path.join(query_dir, sample_name))

                            global_idx += 1
                            saved += 1
            else:
                wcs_obj = None
                if args.wcs_in_name and WCS is not None:
                    try:
                        wcs_obj = WCS(sci_hdr)
                    except Exception:
                        wcs_obj = None

                h, w = sci_data.shape[:2]
                for y in range(0, h - crop + 1, stride):
                    for x in range(0, w - crop + 1, stride):
                        cx = x + (crop - 1) / 2.0
                        cy = y + (crop - 1) / 2.0

                        cut = sci_data[y : y + crop, x : x + crop]
                        tile_u8 = robust_asinh_to_uint8(cut, p_lo=args.p_lo, p_hi=args.p_hi, asinh_q=args.asinh_q)
                        rgb_u8 = np.stack([tile_u8, tile_u8, tile_u8], axis=-1)

                        if mostly_empty_rgb(rgb_u8, mean_thresh=args.mean_thresh, var_thresh=args.var_thresh):
                            skipped_empty += 1
                            continue

                        out_img = Image.fromarray(rgb_u8).resize((args.tile_size, args.tile_size), resample=Image.BICUBIC)
                        radec = maybe_center_radec(wcs_obj, cx, cy) if args.wcs_in_name else None
                        sample_name = make_output_name(spec.tile_id, cx, cy, global_idx, wcs_ra_dec=radec)
                        out_img.save(os.path.join(query_dir, sample_name))

                        global_idx += 1
                        saved += 1

        # per-mosaic reporting (requested)
        print(f"[{spec.tile_id}] Saved PNG tiles: {saved}")
        print(f"[{spec.tile_id}] Skipped: {skipped_empty} empty-ish, {skipped_wht} low-WHT")

        # hard free (requested semantics)
        if args.gc_each_mosaic:
            gc.collect()

        return global_idx


class NircamSensor(Sensor):
    """
    NIRCam: still streams per tile_id (one tile at a time), but note that it must load
    1 or 3 planes for that tile in memory to create the RGB image.
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

    def _infer_tiles_from_dir(self, indir: str) -> List[str]:
        all_fits = glob.glob(os.path.join(indir, "*.fits"))
        tiles: List[str] = []
        for fp in all_fits:
            tid = infer_tile_id_from_filename(fp, None)
            if tid:
                tiles.append(tid)
        tiles = sorted(set(tiles))
        if not tiles:
            raise RuntimeError("Could not infer any tiles from filenames in --indir. Provide --tiles.")
        return tiles

    def _build_path(self, args: argparse.Namespace, tile_id: str, filt: str) -> str:
        fname = args.pattern.format(
            filter=filt,
            pixel_scale=args.pixel_scale,
            tile=tile_id,
            version=args.version,
            ext=args.ext,
        )
        return os.path.join(args.indir, fname)

    def iter_mosaic_specs(self, args: argparse.Namespace) -> Iterator[MosaicSpec]:
        tiles = args.tiles if args.tiles and len(args.tiles) > 0 else self._infer_tiles_from_dir(args.indir)
        for tile_id in tiles:
            # We treat a "mosaic" as one tile_id group.
            yield MosaicSpec(tile_id=tile_id, sci_path="", wht_path=None, src_tag=tile_id)

    def process_one_mosaic(self, args: argparse.Namespace, spec: MosaicSpec, query_dir: str, global_idx: int) -> int:
        if len(args.rgb_filters) not in (1, 3):
            raise ValueError("--rgb_filters must have length 1 or 3")
        if make_lupton_rgb is None and len(args.rgb_filters) == 3:
            raise RuntimeError("3-filter NIRCam RGB requires astropy.visualization.make_lupton_rgb")

        crop = int(args.tile_size / args.upscale)
        stride = args.stride if args.stride is not None else max(1, crop // 2)

        saved = 0
        skipped_empty = 0
        skipped_wht = 0

        planes: List[np.ndarray] = []
        hdr_for_wcs = None
        wht_filt = args.wht_filter or args.rgb_filters[0]
        wht = None

        for filt in args.rgb_filters:
            fp = self._build_path(args, spec.tile_id, filt)
            if not os.path.isfile(fp):
                raise RuntimeError(f"Missing FITS for tile={spec.tile_id}, filter={filt}: {fp}")

            with fits.open(fp, memmap=True) as hdul:
                data, hdr = _find_hdu_with_data(hdul, prefer_extnames=(args.sci_ext, "SCI"))
                planes.append(np.nan_to_num(np.asarray(data), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
                if hdr_for_wcs is None and filt == wht_filt:
                    hdr_for_wcs = hdr

                if args.nircam_load_wht and args.min_wht_frac > 0 and filt == wht_filt:
                    try:
                        wht_arr, _ = _find_hdu_with_data(hdul, prefer_extnames=(args.wht_ext, "WHT"))
                        wht = np.asarray(wht_arr)
                    except Exception:
                        wht = None

        wcs_obj = None
        if args.wcs_in_name and WCS is not None and hdr_for_wcs is not None:
            try:
                wcs_obj = WCS(hdr_for_wcs)
            except Exception:
                wcs_obj = None

        if len(planes) == 1:
            img = planes[0]
        else:
            r, g, b = planes
            img = make_lupton_rgb(r, g, b, stretch=args.stretch, Q=args.Q)

        if img.ndim == 2:
            h, w = img.shape
        else:
            h, w, _ = img.shape

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

                if img.ndim == 2:
                    cut = img[y : y + crop, x : x + crop]
                    tile_u8 = robust_asinh_to_uint8(cut, p_lo=args.p_lo, p_hi=args.p_hi, asinh_q=args.asinh_q)
                    rgb_u8 = np.stack([tile_u8, tile_u8, tile_u8], axis=-1)
                else:
                    cut = img[y : y + crop, x : x + crop, :]
                    if cut.dtype != np.uint8:
                        cutf = np.nan_to_num(cut.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        if cutf.max() <= 1.0:
                            cutf = 255.0 * cutf
                        rgb_u8 = np.clip(cutf, 0.0, 255.0).astype(np.uint8)
                    else:
                        rgb_u8 = cut

                if mostly_empty_rgb(rgb_u8, mean_thresh=args.mean_thresh, var_thresh=args.var_thresh):
                    skipped_empty += 1
                    continue

                out_img = Image.fromarray(rgb_u8).resize((args.tile_size, args.tile_size), resample=Image.BICUBIC)
                radec = maybe_center_radec(wcs_obj, cx, cy) if args.wcs_in_name else None
                sample_name = make_output_name(spec.tile_id, cx, cy, global_idx, wcs_ra_dec=radec)
                out_img.save(os.path.join(query_dir, sample_name))

                global_idx += 1
                saved += 1

        print(f"[{spec.tile_id}] Saved PNG tiles: {saved}")
        print(f"[{spec.tile_id}] Skipped: {skipped_empty} empty-ish, {skipped_wht} low-WHT")

        if args.gc_each_mosaic:
            del planes
            del img
            del wht
            gc.collect()

        return global_idx


SENSORS: Dict[str, Sensor] = {"miri": MiriSensor(), "nircam": NircamSensor()}


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--sensor", choices=sorted(SENSORS.keys()), default="miri")
    p.add_argument("--indir", required=True)
    p.add_argument("--out", default="datasets/jwst/object1")
    p.add_argument("--tile_id", default=None)
    p.add_argument("--tile_regex", default=None)
    p.add_argument("--wcs_in_name", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--tile_size", type=int, default=518)
    p.add_argument("--upscale", type=int, default=2)
    p.add_argument("--stride", type=int, default=None)

    # Filtering: accept both old and new spellings (dest uses underscores)
    p.add_argument("--min_wht_frac", "--minwhtfrac", dest="min_wht_frac", type=float, default=0.0)
    p.add_argument("--mean_thresh", "--meanthresh", dest="mean_thresh", type=float, default=2.0)
    p.add_argument("--var_thresh", "--varthresh", dest="var_thresh", type=float, default=1.0)

    # Robust scaling (grayscale): accept both spellings
    p.add_argument("--p_lo", "--plo", dest="p_lo", type=float, default=1.0)
    p.add_argument("--p_hi", "--phi", dest="p_hi", type=float, default=99.8)
    p.add_argument("--asinh_q", "--asinhq", dest="asinh_q", type=float, default=10.0)

    # Memory control
    p.add_argument("--gc_each_mosaic", default=True, action=argparse.BooleanOptionalAction)
    return p


def parse_args_two_stage() -> argparse.Namespace:
    p = build_parser()
    known, _ = p.parse_known_args()
    SENSORS[known.sensor].add_args(p)
    return p.parse_args()


def run() -> None:
    args = parse_args_two_stage()
    query_dir = ensure_dirs_query_only(args.out)

    global_idx = 0
    mosaics_processed = 0

    for spec in SENSORS[args.sensor].iter_mosaic_specs(args):
        mosaics_processed += 1
        global_idx = SENSORS[args.sensor].process_one_mosaic(args, spec, query_dir, global_idx)

        # mimic "program end" between mosaics
        if args.gc_each_mosaic:
            gc.collect()

    print("Done.")
    print(f"Sensor: {args.sensor}")
    print(f"Processed mosaics: {mosaics_processed}")
    print(f"Total saved PNG tiles: {global_idx}")


if __name__ == "__main__":
    run()
