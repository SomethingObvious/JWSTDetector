r"""Plants synthetic sources in the FITS cutouts and measures how many the detectors find.

    jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench_smacs
    jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench_smacs \
        --mags 24 25 26 --seeds 3 --detectors knn pixels -- --k_neighbors 3

A source is added to a tile's float cutout at a given AB magnitude, in whatever units
the mosaic is in, blurred with each filter's PSF, and the tile is rendered back to its
PNG with the exact stretch its mosaic was cut with. For JWST the PSF is STScI's model
from STPSF, with its Airy rings and six spikes, turned to the mosaic by the telescope's
roll. Anything else, or JWST without STPSF installed, gets a Gaussian as wide as the
diffraction limit. Prep records the units and PSF widths from each file's headers, and
--jy_per_unit and --psf_fwhm cover data whose headers don't say, such as most
ground-based images. Every magnitude of a seed gets
the same tiles, shapes, positions and colours, so the curves differ by brightness
alone. Real anomalies in the field compete with the planted ones, which is why found
means ranked in the top 5% rather than first.

It needs a dataset cut with --save_cutouts. Anything after `--` goes to
every `jwstdetector run`. Writes injections.csv with each planted source's rank,
summary.csv with recall and its 95% Wilson interval per detector, kind and
magnitude, and completeness.png.
"""  # noqa: E501 --help prints this, so the summary line stays whole

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import functools
import importlib.util
import itertools
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, Any

from astropy.io import fits
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, rotate
from scipy.signal import fftconvolve

from jwstdetector import bootstrap
from jwstdetector.bundle import mosaic_wcs
from jwstdetector.data import read_manifest, read_measurements
from jwstdetector.prep import render_cutout

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    import numpy.typing as npt

    from jwstdetector.utils import FitsHDUList

logger = logging.getLogger(__name__)

KINDS = ("point", "arc", "ring", "streak")
# f_nu goes as wavelength to this power, so red sources are brighter in F444W.
SEDS = {"flat": 0.0, "red": 2.0, "blue": -2.0}
# What each detector adds to every `jwstdetector run`. Pixels is the kNN search on raw
# 16 px patches instead of features, which is what shows whether the backbone helps.
DETECTORS = {
    "knn": ["--detector", "knn"],
    "subspace": ["--detector", "subspace", "--layers", "mid"],
    "pixels": ["--detector", "knn", "--model_name", "pixels", "--knn_metric", "L2"],
    "iforest": ["--detector", "iforest"],
}
TOP_SHARE = 0.05
# The recall that the m50 magnitude marks.
M50_RECALL = 0.5
PSFS = ("auto", "webb", "gaussian")
# STPSF's instrument names, from the INSTRUME header card.
_WEBB_INSTRUMENTS = {
    "NIRCAM": "NIRCam",
    "MIRI": "MIRI",
    "NIRISS": "NIRISS",
    "NIRSPEC": "NIRSpec",
}
# About 6 arcseconds at NIRCam's long-wavelength scale, past which the wings are under
# a thousandth of the peak.
_PSF_PIXELS = 97


@dataclass(frozen=True)
class Source:
    """One planted source: its tile, shape, colour and placement in tile pixels."""

    tile: str
    kind: str
    sed: str
    cx: float
    cy: float
    radius: float
    theta: float
    span: float


@dataclass(frozen=True)
class Optics:
    """What blurring and scaling a source needs from a mosaic's bands.

    Each band needs jy_per_unit and psf_fwhm_arcsec, and a micron for the SED to
    colour it. `kernels` holds each band's PSF image, and without them it's a
    Gaussian.
    """

    bands: list[dict[str, Any]]
    pixel_scale: float
    kernels: list[npt.NDArray[np.float64]] | None = None


def plan(
    rng: np.random.Generator, keys: list[str], n: int, *, kinds: list[str], size: int
) -> list[Source]:
    """Pick which tiles get a source and its shape, drawn once per seed.

    Returns:
        `n` sources on `n` different tiles.
    """
    picked = rng.choice(len(keys), size=n, replace=False)
    out: list[Source] = []
    for i, idx in enumerate(picked):
        cx, cy = rng.uniform(0.3, 0.7, 2) * size
        out.append(
            Source(
                tile=keys[idx],
                kind=kinds[i % len(kinds)],
                sed=list(SEDS)[rng.integers(len(SEDS))],
                cx=float(cx),
                cy=float(cy),
                radius=float(rng.uniform(0.06, 0.15) * size),
                theta=float(rng.uniform(-np.pi, np.pi)),
                span=float(np.deg2rad(rng.uniform(60, 140))),
            )
        )
    return out


def intrinsic(source: Source, size: int) -> npt.NDArray[np.float64]:
    """The source before the PSF, in FITS pixel order.

    Arcs, rings and streaks are half a pixel wide, so the PSF sets how wide they look,
    which is how a thin lensed arc or a satellite trail looks too.

    Returns:
        A (size, size) map.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = x - source.cx, y - source.cy
    shape: npt.NDArray[np.float64]
    if source.kind == "point":
        shape = np.exp(-(dx**2 + dy**2) / (2 * 0.25**2))
        return shape
    if source.kind == "streak":
        d = dx * np.sin(source.theta) - dy * np.cos(source.theta)
        shape = np.exp(-(d**2) / (2 * 0.5**2))
        return shape
    ring: npt.NDArray[np.float64] = np.exp(
        -((np.hypot(dx, dy) - source.radius) ** 2) / (2 * 0.5**2)
    )
    if source.kind == "ring":
        return ring
    angle = (np.arctan2(dy, dx) - source.theta) % (2 * np.pi)
    shape = ring * (angle < source.span)
    return shape


@functools.cache
def webb_psf(
    instrument: str, filt: str, pixel_scale: float, *, fov: int = _PSF_PIXELS
) -> npt.NDArray[np.float64]:
    """STPSF's model of a filter's PSF, sampled at `pixel_scale` in the detector frame.

    Returns:
        The PSF image, normalised to sum to 1.
    """
    import stpsf  # noqa: PLC0415 stpsf is optional, and auto falls back to a Gaussian

    inst = stpsf.instrument(instrument)
    inst.filter = filt.upper()
    # NIRCam picks its short or long-wavelength scale from the filter, so this goes
    # after.
    inst.pixelscale = pixel_scale
    hdul: FitsHDUList = inst.calc_psf(fov_pixels=fov, oversample=4)
    psf = np.asarray(hdul["DET_DIST"].data, np.float64)
    return psf / psf.sum()


def roll_angle(info: dict[str, Any]) -> float:
    """Find how far to turn a detector-frame PSF to put it on the mosaic.

    Returns:
        The angle in degrees that lays the PSF's V3 axis along PA_V3 on the mosaic.
    """
    t = np.radians(float(info["header"]["PA_V3"]))
    # Intermediate world coordinates run east and north, so the V3 axis is
    # (sin t, cos t).
    dx, dy = np.linalg.solve(
        mosaic_wcs(info).pixel_scale_matrix, [np.sin(t), np.cos(t)]
    )
    return float(np.degrees(np.arctan2(dx, dy)))


def psf_kernels(
    info: dict[str, Any], mode: str
) -> list[npt.NDArray[np.float64]] | None:
    """Make the Webb PSF for each band of a mosaic.

    Returns:
        One PSF image per band, or None where the Gaussian applies.

    Raises:
        SystemExit: `mode` is webb but the header doesn't give INSTRUME and PA_V3.
    """
    instrument = _WEBB_INSTRUMENTS.get(str(info["header"].get("INSTRUME", "")).upper())
    if mode == "gaussian" or (
        mode == "auto" and importlib.util.find_spec("stpsf") is None
    ):
        return None
    if instrument is None or "PA_V3" not in info["header"]:
        if mode == "webb":
            msg = (
                "--psf webb needs a JWST mosaic whose header gives INSTRUME and PA_V3."
            )
            raise SystemExit(msg)
        return None
    angle = roll_angle(info)
    kernels: list[npt.NDArray[np.float64]] = []
    for band in info["planes"]:
        psf = webb_psf(instrument, band["filter"], round(info["pixel_scale_arcsec"], 5))
        turned = rotate(psf, angle, reshape=False, order=1, mode="constant")
        kernels.append(turned / turned.sum())
    return kernels


def _band_mags(mag: float, bands: list[dict[str, Any]], sed: str) -> list[float]:
    """Spread total AB magnitude `mag` over the bands by the SED.

    The magnitude is the flux inside the tile, at the bands' geometric mean
    wavelength. Without wavelengths every band gets the same magnitude.

    Returns:
        The AB magnitude in each band.
    """
    microns = [float(b["micron"]) for b in bands if b.get("micron") is not None]
    if len(microns) < len(bands):
        return [mag] * len(bands)
    ref = float(np.exp(np.mean(np.log(microns))))
    beta = SEDS[sed]
    return [mag - 2.5 * beta * math.log10(micron / ref) for micron in microns]


def add_source(
    planes: npt.NDArray[np.float32], source: Source, mag: float, *, optics: Optics
) -> npt.NDArray[np.float32]:
    """Add the source to `planes` (n_filters, H, W) at total AB magnitude `mag`.

    Without PSF kernels it's a Gaussian, which has no Airy rings or spikes, so planted
    point sources are a little easier to tell from stars than they should be. Holes
    stay holes.

    Returns:
        A copy of `planes` with the source in it.
    """
    base = intrinsic(source, planes.shape[-1])
    mags = _band_mags(mag, optics.bands, source.sed)
    out = planes.astype(np.float64).copy()
    for i, (band, m) in enumerate(zip(optics.bands, mags, strict=True)):
        units = 3631.0 * 10 ** (-0.4 * m) / band["jy_per_unit"]
        if optics.kernels is None:
            sigma = band["psf_fwhm_arcsec"] / 2.3548 / optics.pixel_scale
            shape = gaussian_filter(base, sigma, mode="constant")
        else:
            shape = np.clip(fftconvolve(base, optics.kernels[i], mode="same"), 0, None)
        shape /= shape.sum()
        live = np.isfinite(out[i]) & (out[i] != 0)
        out[i] = np.where(live, out[i] + shape * units, out[i])
    return out.astype(np.float32)


def ranks_of(scores: dict[str, float]) -> dict[str, int]:
    """Rank tiles by score, highest first, with NaN last.

    Returns:
        Each tile's rank, starting at 1.
    """
    order = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
    return {k: i + 1 for i, k in enumerate(order)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n, which behaves at 0 and n.

    Returns:
        The interval's low and high ends.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(rows: list[dict[str, Any]], n_tiles: int) -> list[dict[str, Any]]:
    """Count how many planted sources each detector found, by kind and magnitude.

    Returns:
        One row of summary.csv per detector, kind and magnitude, with "all" for every
        kind together.
    """
    top = max(1, round(TOP_SHARE * n_tiles))
    groups: dict[tuple[str, str, float], list[int]] = {}
    for r in rows:
        for kind in (r["kind"], "all"):
            groups.setdefault((r["detector"], kind, r["mag"]), []).append(r["rank"])
    out: list[dict[str, Any]] = []
    for (det, kind, mag), ranks in sorted(groups.items()):
        found = sum(r <= top for r in ranks)
        lo, hi = wilson(found, len(ranks))
        out.append({
            "detector": det,
            "kind": kind,
            "mag": mag,
            "planted": len(ranks),
            "found": found,
            "recall": found / len(ranks),
            "recall_lo": lo,
            "recall_hi": hi,
            "median_rank": float(np.median(ranks)),
        })
    return out


def m50(summary: list[dict[str, Any]], detector: str, kind: str) -> float | None:
    """Find the magnitude where recall falls through 50%, interpolated.

    Returns:
        That magnitude, or None if recall never falls through 50%.
    """
    pts = sorted(
        (s["mag"], s["recall"])
        for s in summary
        if (s["detector"], s["kind"]) == (detector, kind)
    )
    for (m1, r1), (m2, r2) in itertools.pairwise(pts):
        if r1 >= M50_RECALL > r2:
            return float(m1 + (r1 - M50_RECALL) / (r1 - r2) * (m2 - m1))
    return None


# Categorical slots 1 to 4 of the validated reference palette. The first three clear
# the colour-blind floors for every pair, but the fourth is close to orange, so each
# line also gets its own marker shape.
COLOURS = {
    "knn": "#2a78d6",
    "subspace": "#eb6834",
    "pixels": "#1baf7a",
    "iforest": "#eda100",
}
MARKERS = {"knn": "o", "subspace": "s", "pixels": "^", "iforest": "D"}
LABELS = {
    "knn": "kNN",
    "subspace": "Subspace",
    "pixels": "kNN on pixels",
    "iforest": "Isolation forest",
}


def _plot_kind(
    ax: Axes,
    summary: list[dict[str, Any]],
    *,
    kind: str,
    detectors: list[str],
) -> None:
    ax.set_facecolor("#fcfcfb")
    ax.axhline(TOP_SHARE, color="#8a8986", lw=1, ls=(0, (3, 3)), label="Chance")
    for det in detectors:
        pts = sorted(
            (s["mag"], s["recall"], s["recall_lo"], s["recall_hi"])
            for s in summary
            if s["detector"] == det and s["kind"] == kind
        )
        mags, rec, lo, hi = (np.array(v) for v in zip(*pts, strict=True))
        ax.fill_between(mags, lo, hi, color=COLOURS[det], alpha=0.12, lw=0)
        ax.plot(
            mags,
            rec,
            color=COLOURS[det],
            lw=2,
            marker=MARKERS[det],
            ms=4.5,
            label=LABELS[det],
        )
    ax.set_title(kind.capitalize(), fontsize=10, color="#0b0b0b", loc="left")
    ax.set_xlabel("AB magnitude", fontsize=9, color="#52514e")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")


def plot(summary: list[dict[str, Any]], path: Path, n_tiles: int) -> None:
    """Draw recall against magnitude, one panel per kind and one line per detector."""
    import matplotlib as mpl  # noqa: PLC0415 matplotlib is slow to load

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 matplotlib is slow to load

    detectors = [d for d in DETECTORS if any(s["detector"] == d for s in summary)]
    kinds = [k for k in KINDS if any(s["kind"] == k for s in summary)]
    fig, axes = plt.subplots(
        1, len(kinds), figsize=(3.1 * len(kinds), 3.3), sharey=True, dpi=160
    )
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, kind in zip(axes, kinds, strict=True):
        _plot_kind(ax, summary, kind=kind, detectors=detectors)
    axes[0].set_ylabel(
        f"Found in the top 5% ({max(1, round(TOP_SHARE * n_tiles))} tiles)",
        fontsize=9,
        color="#52514e",
    )
    # The chance line goes in the legend, as any label on it would sit on the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncol=len(labels),
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def link_tree(src: Path, dst: Path) -> None:
    """Hard links where the filesystem allows, as the copies are only read."""

    def link(s: str, d: str) -> None:
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    shutil.copytree(src, dst, copy_function=link)


def _read_cutout(
    path: Path,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[Any] | None]:
    hdul: FitsHDUList
    with fits.open(path, memmap=False) as hdul:
        planes = np.asarray(hdul[0].data, dtype=np.float32)
        noise = hdul["NOISEEQ"].data if "NOISEEQ" in hdul else None
    # A cutout with one filter is a single plane.
    return planes.reshape(-1, *planes.shape[-2:]), noise


def plant(
    data: Path,
    source_root: Path,
    sources: list[Source],
    *,
    mag: float,
    mosaics: dict[str, dict[str, Any]],
) -> None:
    """Add the sources to their tiles' cutouts and render the PNGs under `data`."""
    manifest = read_manifest(source_root)
    for s in sources:
        meta = manifest[s.tile]
        info = mosaics[meta["mosaic"]]
        planes, noise = _read_cutout(source_root / meta["cutout"])
        optics = Optics(info["planes"], info["pixel_scale_arcsec"], info.get("kernels"))
        planes = add_source(planes, s, mag, optics=optics)
        tile = render_cutout(
            planes if len(planes) > 1 else planes[0], info["render"], noise
        )
        dest = data / "query" / s.tile
        # Unlinked first, since it's a hard link to the original dataset's tile.
        dest.unlink()
        Image.fromarray(tile).save(dest)


def _read_mosaic_record(root: Path, mosaic: str) -> dict[str, Any]:
    """Read mosaics/<id>.json and check it has what planting needs.

    Returns:
        The mosaic's record.

    Raises:
        SystemExit: The record has no stretch, WCS or photometry.
    """
    info: dict[str, Any] = json.loads(
        (root / "mosaics" / f"{mosaic}.json").read_text("utf-8")
    )
    if "render" not in info:
        msg = (
            f"{root}/mosaics/{mosaic}.json doesn't say how its tiles were "
            "stretched. Cut the dataset again with this version, with "
            "--stretch_scope mosaic."
        )
        raise SystemExit(msg)
    if "planes" not in info or "pixel_scale_arcsec" not in info:
        msg = (
            f"{mosaic} has no WCS or no photometry record. Cut it again with "
            "this version of `jwstdetector prep`."
        )
        raise SystemExit(msg)
    return info


def _fill_bands(
    bands: list[dict[str, Any]],
    mosaic: str,
    *,
    field: str,
    override: list[float] | None,
    flag: str,
) -> None:
    """Apply `override` to every band, then check that none still lacks `field`.

    Raises:
        SystemExit: `override` has the wrong count, or a band still lacks `field`.
    """
    if override:
        values = override * len(bands) if len(override) == 1 else override
        if len(values) != len(bands):
            msg = f"{flag} takes 1 or {len(bands)} values for {mosaic}"
            raise SystemExit(msg)
        for b, v in zip(bands, values, strict=True):
            b[field] = v
    missing = [b["filter"] for b in bands if not b.get(field)]
    if missing:
        msg = (
            f"The headers of {mosaic} don't give {field} for "
            f"{', '.join(missing)}. Pass {flag}."
        )
        raise SystemExit(msg)


def check_dataset(
    root: Path,
    manifest: dict[str, dict[str, Any]],
    *,
    psf_fwhm: list[float] | None = None,
    jy_per_unit: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Check every mosaic can be planted in, applying the overrides.

    Returns:
        Each mosaic's record, with the overrides applied.

    Raises:
        SystemExit: A tile has no FITS cutout.
    """
    mosaics: dict[str, dict[str, Any]] = {}
    for meta in manifest.values():
        if not meta.get("cutout"):
            msg = f"{root} has no FITS cutouts. Cut it again with --save_cutouts."
            raise SystemExit(msg)
        mosaic = meta["mosaic"]
        if mosaic not in mosaics:
            info = _read_mosaic_record(root, mosaic)
            for field, override, flag in (
                ("psf_fwhm_arcsec", psf_fwhm, "--psf_fwhm"),
                ("jy_per_unit", jy_per_unit, "--jy_per_unit"),
            ):
                _fill_bands(
                    info["planes"], mosaic, field=field, override=override, flag=flag
                )
            mosaics[mosaic] = info
    return mosaics


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data_root", type=Path, required=True, help="A `jwstdetector prep` folder."
    )
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mags", type=float, nargs="+", default=[23, 24, 25, 26, 27, 28])
    ap.add_argument(
        "--seeds", type=int, default=3, help="Independent placements per magnitude."
    )
    ap.add_argument("--n_inject", type=int, default=12, help="Sources planted per run.")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument(
        "--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS)
    )
    ap.add_argument(
        "--feature_cache",
        type=Path,
        default=None,
        help="Shared by every run, DATA_ROOT/.features by default, so each run only "
        "embeds the tiles it planted a source in.",
    )
    ap.add_argument(
        "--psf_fwhm",
        type=float,
        nargs="+",
        default=None,
        help="PSF FWHM in arcseconds, one for every band or one per band, in place of "
        "the headers.",
    )
    ap.add_argument(
        "--jy_per_unit",
        type=float,
        nargs="+",
        default=None,
        help="Janskys per pixel for one unit of pixel value, one or one per band, in "
        "place of the headers.",
    )
    ap.add_argument(
        "--psf",
        default="auto",
        choices=PSFS,
        help="webb is STPSF's model for JWST mosaics, gaussian is a Gaussian of each "
        "filter's width, and auto takes webb when it applies and STPSF is installed.",
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="Start again in an existing OUT_DIR."
    )
    return ap


@dataclass
class _Bench:
    """Everything each planted run needs, fixed for the whole benchmark."""

    args: argparse.Namespace
    root: Path
    keys: list[str]
    size: int
    mosaics: dict[str, dict[str, Any]]
    runner_args: list[str]


def _load_bench(args: argparse.Namespace, runner_args: list[str]) -> _Bench:
    """Check the dataset and pick each mosaic's PSF.

    Returns:
        The benchmark's fixed inputs.

    Raises:
        SystemExit: OUT_DIR exists without --overwrite, the dataset has no manifest,
            or there are too few tiles for the sources asked for.
    """
    if args.out_dir.exists():
        if not args.overwrite:
            msg = f"{args.out_dir} already exists. Pass --overwrite to replace it."
            raise SystemExit(msg)
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    root = args.data_root.resolve()
    manifest = read_manifest(root)
    if not manifest:
        msg = f"{root} has no manifest.csv. Cut it again with `jwstdetector prep`."
        raise SystemExit(msg)
    mosaics = check_dataset(
        root, manifest, psf_fwhm=args.psf_fwhm, jy_per_unit=args.jy_per_unit
    )
    for name, info in mosaics.items():
        info["kernels"] = psf_kernels(info, args.psf)
        logger.info(
            "%s gets %s",
            name,
            "the Webb PSF from STPSF"
            if info["kernels"] is not None
            else "a Gaussian PSF",
        )
    keys = sorted(manifest)
    if args.n_inject >= len(keys) // 4:
        msg = (
            f"Planting {args.n_inject} sources in {len(keys)} tiles would make them "
            "too common to count as anomalies. Use fewer."
        )
        raise SystemExit(msg)
    size = next(iter(manifest.values()))["size"]
    return _Bench(args, root, keys, size, mosaics, runner_args)


def _run_planted(
    bench: _Bench, sources: list[Source], *, seed: int, mag: float
) -> list[dict[str, Any]]:
    """Plant one seed's sources at one magnitude and run every detector on them.

    Returns:
        One injections.csv row per source and detector.
    """
    args = bench.args
    run_dir = args.out_dir / f"seed{seed}_mag{mag:g}"
    data = run_dir / "data"
    link_tree(bench.root / "query", data / "query")
    shutil.copyfile(bench.root / "manifest.csv", data / "manifest.csv")
    plant(data, bench.root, sources, mag=mag, mosaics=bench.mosaics)
    cache = str(args.feature_cache or bench.root / ".features")
    rows: list[dict[str, Any]] = []
    for det in args.detectors:
        bootstrap.main(
            [
                "--data_root", str(data),
                "--out_dir", str(run_dir / det),
                "--feature_cache", cache,
                "--seed", str(seed),
                *DETECTORS[det],
                *bench.runner_args,
            ]
        )  # fmt: skip
        rank = ranks_of(read_measurements(run_dir / det / "measurements_final.csv"))
        rows.extend(
            {
                "seed": seed,
                "mag": mag,
                "detector": det,
                "tile": s.tile,
                "kind": s.kind,
                "sed": s.sed,
                "rank": rank[s.tile],
            }
            for s in sources
        )
    shutil.rmtree(data)
    logger.info("Seed %d at AB %g done", seed, mag)
    return rows


def _write_results(
    out_dir: Path,
    rows: list[dict[str, Any]],
    plans: dict[int, list[dict[str, Any]]],
    *,
    n_tiles: int,
) -> list[dict[str, Any]]:
    """Write injections.csv, plan.json, summary.csv and completeness.png.

    Returns:
        The summary rows.
    """
    with (out_dir / "injections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "plan.json").write_text(json.dumps(plans, indent=1), encoding="utf-8")
    summary = summarise(rows, n_tiles)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(summary, out_dir / "completeness.png", n_tiles)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Plant sources at every magnitude and seed, and measure how many are found.

    Returns:
        0 once the results are written.
    """
    argv = sys.argv[1:] if argv is None else argv
    runner_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, runner_args = argv[:cut], argv[cut + 1 :]

    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    bench = _load_bench(args, runner_args)

    rows: list[dict[str, Any]] = []
    plans: dict[int, list[dict[str, Any]]] = {}
    for seed in range(args.seeds):
        sources = plan(
            np.random.default_rng(seed),
            bench.keys,
            args.n_inject,
            kinds=args.kinds,
            size=bench.size,
        )
        plans[seed] = [asdict(s) for s in sources]
        for mag in args.mags:
            rows.extend(_run_planted(bench, sources, seed=seed, mag=mag))

    summary = _write_results(args.out_dir, rows, plans, n_tiles=len(bench.keys))
    for det in args.detectors:
        parts = []
        for kind in [*args.kinds, "all"]:
            m = m50(summary, det, kind)
            parts.append(f"{kind} {m:.1f}" if m is not None else f"{kind} none")
        logger.info("%s, 50%% found at AB: %s", LABELS[det], ", ".join(parts))
    logger.info("Wrote %s", args.out_dir / "summary.csv")
    return 0
