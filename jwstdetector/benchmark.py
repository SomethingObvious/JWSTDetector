"""Plants synthetic sources in the FITS cutouts and measures how many the detectors find.

    jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench_smacs
    jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench_smacs \\
        --mags 24 25 26 --seeds 3 --detectors knn pixels -- --k_neighbors 3

A source is added to a tile's float cutout at a given AB magnitude, in whatever units
the mosaic is in, blurred with a Gaussian PSF of each filter's width, and the tile is
rendered back to its PNG with the exact stretch its mosaic was cut with. Prep records
the units and the PSF from each file's headers (the diffraction limit for JWST, HST,
Euclid and other space telescopes), and --jy_per_unit and --psf_fwhm cover data whose
headers don't say, such as most ground-based images. Every magnitude of a seed gets
the same tiles, shapes, positions and colours, so the curves differ by brightness
alone. Real anomalies in the field compete with the planted ones, which is why found
means ranked in the top 5% rather than first.

It needs a dataset cut with --save_cutouts. Anything after `--` goes to
every `jwstdetector run`. Writes injections.csv with each planted source's rank,
summary.csv with recall and its 95% Wilson interval per detector, kind and
magnitude, and completeness.png.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter

from jwstdetector import bootstrap
from jwstdetector.data import read_manifest, read_measurements
from jwstdetector.prep import render_cutout

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
}
TOP_SHARE = 0.05


@dataclass(frozen=True)
class Source:
    tile: str
    kind: str
    sed: str
    cx: float
    cy: float
    radius: float
    theta: float
    span: float


def plan(rng: np.random.Generator, keys: list[str], n: int, kinds: list[str], size: int):
    """Which tiles get a source and its shape, drawn once per seed."""
    picked = rng.choice(len(keys), size=n, replace=False)
    out = []
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


def intrinsic(source: Source, size: int) -> np.ndarray:
    """The source before the PSF, as a (size, size) map in FITS pixel order.

    Arcs, rings and streaks are half a pixel wide, so the PSF sets how wide they look,
    which is how a thin lensed arc or a satellite trail looks too.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    dx, dy = x - source.cx, y - source.cy
    if source.kind == "point":
        return np.exp(-(dx**2 + dy**2) / (2 * 0.25**2))
    if source.kind == "streak":
        d = dx * np.sin(source.theta) - dy * np.cos(source.theta)
        return np.exp(-(d**2) / (2 * 0.5**2))
    ring = np.exp(-((np.hypot(dx, dy) - source.radius) ** 2) / (2 * 0.5**2))
    if source.kind == "ring":
        return ring
    angle = (np.arctan2(dy, dx) - source.theta) % (2 * np.pi)
    return ring * (angle < source.span)


def add_source(
    planes: np.ndarray, source: Source, mag: float, bands: list[dict], pixel_scale: float
) -> np.ndarray:
    """`planes` (n_filters, H, W) with the source added at total AB magnitude `mag`.

    Each band needs jy_per_unit and psf_fwhm_arcsec, and a micron for the SED to
    colour it. The magnitude is the flux inside the tile, at the bands'
    geometric mean wavelength. Without wavelengths every band gets the same magnitude.
    A Gaussian has no Airy rings or spikes, so planted point sources are a little
    easier to tell from stars than they should be. Holes stay holes.
    """
    base = intrinsic(source, planes.shape[-1])
    microns = [b.get("micron") for b in bands]
    known = all(m is not None for m in microns)
    ref = float(np.exp(np.mean(np.log(microns)))) if known else 1.0
    out = planes.astype(np.float64).copy()
    for i, band in enumerate(bands):
        beta = SEDS[source.sed] if known else 0.0
        m = mag - 2.5 * beta * math.log10(microns[i] / ref) if known else mag
        units = 3631.0 * 10 ** (-0.4 * m) / band["jy_per_unit"]
        sigma = band["psf_fwhm_arcsec"] / 2.3548 / pixel_scale
        shape = gaussian_filter(base, sigma, mode="constant")
        shape /= shape.sum()
        live = np.isfinite(out[i]) & (out[i] != 0)
        out[i] = np.where(live, out[i] + shape * units, out[i])
    return out.astype(np.float32)


def ranks_of(scores: dict[str, float]) -> dict[str, int]:
    order = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
    return {k: i + 1 for i, k in enumerate(order)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n, which behaves at 0 and n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(rows: list[dict], n_tiles: int) -> list[dict]:
    top = max(1, round(TOP_SHARE * n_tiles))
    groups: dict[tuple, list[int]] = {}
    for r in rows:
        for kind in (r["kind"], "all"):
            groups.setdefault((r["detector"], kind, r["mag"]), []).append(r["rank"])
    out = []
    for (det, kind, mag), ranks in sorted(groups.items()):
        found = sum(r <= top for r in ranks)
        lo, hi = wilson(found, len(ranks))
        out.append(
            {
                "detector": det,
                "kind": kind,
                "mag": mag,
                "planted": len(ranks),
                "found": found,
                "recall": found / len(ranks),
                "recall_lo": lo,
                "recall_hi": hi,
                "median_rank": float(np.median(ranks)),
            }
        )
    return out


def m50(summary: list[dict], detector: str, kind: str) -> float | None:
    """The magnitude where recall falls through 50%, interpolated, or None if it never does."""
    pts = sorted(
        (s["mag"], s["recall"]) for s in summary if (s["detector"], s["kind"]) == (detector, kind)
    )
    for (m1, r1), (m2, r2) in itertools.pairwise(pts):
        if r1 >= 0.5 > r2:
            return m1 + (r1 - 0.5) / (r1 - r2) * (m2 - m1)
    return None


# Categorical slots 1 to 3 of the validated reference palette, which clear the
# colour-blind separation floors for every pair of three on a light surface.
COLOURS = {"knn": "#2a78d6", "subspace": "#eb6834", "pixels": "#1baf7a"}
LABELS = {"knn": "kNN", "subspace": "Subspace", "pixels": "kNN on pixels"}


def plot(summary: list[dict], path: Path, n_tiles: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detectors = [d for d in DETECTORS if any(s["detector"] == d for s in summary)]
    kinds = [k for k in KINDS if any(s["kind"] == k for s in summary)]
    fig, axes = plt.subplots(1, len(kinds), figsize=(3.1 * len(kinds), 3.3), sharey=True, dpi=160)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, kind in zip(axes, kinds, strict=True):
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
            ax.plot(mags, rec, color=COLOURS[det], lw=2, marker="o", ms=4.5, label=LABELS[det])
        ax.set_title(kind.capitalize(), fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel("AB magnitude", fontsize=9, color="#52514e")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.tick_params(colors="#52514e", labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c3")
    axes[0].set_ylabel(
        f"Found in the top 5% ({max(1, round(TOP_SHARE * n_tiles))} tiles)",
        fontsize=9,
        color="#52514e",
    )
    # The chance line goes in the legend, as any label on it would sit on the curves.
    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = handles[1:] + handles[:1], labels[1:] + labels[:1]
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def link_tree(src: Path, dst: Path) -> None:
    """Hard links where the filesystem allows, as the copies are only read."""

    def link(s, d):
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)

    shutil.copytree(src, dst, copy_function=link)


def plant(data: Path, source_root: Path, sources: list[Source], mag: float, mosaics: dict) -> None:
    manifest = read_manifest(source_root)
    for s in sources:
        meta = manifest[s.tile]
        info = mosaics[meta["mosaic"]]
        with fits.open(source_root / meta["cutout"], memmap=False) as hdul:
            planes = np.asarray(hdul[0].data, dtype=np.float32)
            noise = hdul["NOISEEQ"].data if "NOISEEQ" in hdul else None
        planes = planes[None] if planes.ndim == 2 else planes
        planes = add_source(planes, s, mag, info["planes"], info["pixel_scale_arcsec"])
        tile = render_cutout(planes if len(planes) > 1 else planes[0], info["render"], noise)
        dest = data / "query" / s.tile
        # Unlinked first, since it's a hard link to the original dataset's tile.
        dest.unlink()
        Image.fromarray(tile).save(dest)


def check_dataset(
    root: Path,
    manifest: dict,
    psf_fwhm: list[float] | None = None,
    jy_per_unit: list[float] | None = None,
) -> dict:
    """Each mosaic's record, with the overrides applied, once it's known to be plantable."""

    def per_band(values, n):
        return values * n if len(values) == 1 else values

    mosaics = {}
    for meta in manifest.values():
        if not meta.get("cutout"):
            raise SystemExit(f"{root} has no FITS cutouts. Cut it again with --save_cutouts.")
        if meta["mosaic"] not in mosaics:
            info = json.loads((root / "mosaics" / f"{meta['mosaic']}.json").read_text("utf-8"))
            if "render" not in info:
                raise SystemExit(
                    f"{root}/mosaics/{meta['mosaic']}.json doesn't say how its tiles were "
                    "stretched. Cut the dataset again with this version, with "
                    "--stretch_scope mosaic."
                )
            if "planes" not in info or "pixel_scale_arcsec" not in info:
                raise SystemExit(
                    f"{meta['mosaic']} has no WCS or no photometry record. Cut it again with "
                    "this version of `jwstdetector prep`."
                )
            bands = info["planes"]
            for field, override, flag in (
                ("psf_fwhm_arcsec", psf_fwhm, "--psf_fwhm"),
                ("jy_per_unit", jy_per_unit, "--jy_per_unit"),
            ):
                if override:
                    values = per_band(override, len(bands))
                    if len(values) != len(bands):
                        raise SystemExit(
                            f"{flag} takes 1 or {len(bands)} values for {meta['mosaic']}"
                        )
                    for b, v in zip(bands, values, strict=True):
                        b[field] = v
                missing = [b["filter"] for b in bands if not b.get(field)]
                if missing:
                    raise SystemExit(
                        f"The headers of {meta['mosaic']} don't give {field} for "
                        f"{', '.join(missing)}. Pass {flag}."
                    )
            mosaics[meta["mosaic"]] = info
    return mosaics


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runner_args: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, runner_args = argv[:cut], argv[cut + 1 :]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_root", type=Path, required=True, help="A `jwstdetector prep` folder.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mags", type=float, nargs="+", default=[23, 24, 25, 26, 27, 28])
    ap.add_argument("--seeds", type=int, default=3, help="Independent placements per magnitude.")
    ap.add_argument("--n_inject", type=int, default=12, help="Sources planted per run.")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS), choices=KINDS)
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS), choices=list(DETECTORS))
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
        help="PSF FWHM in arcseconds, one for every band or one per band, in place of the headers.",
    )
    ap.add_argument(
        "--jy_per_unit",
        type=float,
        nargs="+",
        default=None,
        help="Janskys per pixel for one unit of pixel value, one or one per band, in place of "
        "the headers.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Start again in an existing OUT_DIR.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)
    root = args.data_root.resolve()
    manifest = read_manifest(root)
    if not manifest:
        raise SystemExit(f"{root} has no manifest.csv. Cut it again with `jwstdetector prep`.")
    mosaics = check_dataset(root, manifest, args.psf_fwhm, args.jy_per_unit)
    keys = sorted(manifest)
    if args.n_inject >= len(keys) // 4:
        raise SystemExit(
            f"Planting {args.n_inject} sources in {len(keys)} tiles would make them too "
            "common to count as anomalies. Use fewer."
        )
    size = next(iter(manifest.values()))["size"]
    cache = str(args.feature_cache or root / ".features")

    rows = []
    plans = {}
    for seed in range(args.seeds):
        sources = plan(np.random.default_rng(seed), keys, args.n_inject, args.kinds, size)
        plans[seed] = [asdict(s) for s in sources]
        for mag in args.mags:
            run_dir = args.out_dir / f"seed{seed}_mag{mag:g}"
            data = run_dir / "data"
            link_tree(root / "query", data / "query")
            shutil.copyfile(root / "manifest.csv", data / "manifest.csv")
            plant(data, root, sources, mag, mosaics)
            for det in args.detectors:
                bootstrap.main(
                    [
                        "--data_root", str(data),
                        "--out_dir", str(run_dir / det),
                        "--feature_cache", cache,
                        "--seed", str(seed),
                        *DETECTORS[det],
                        *runner_args,
                    ]
                )  # fmt: skip
                scores = read_measurements(run_dir / det / "measurements_final.csv")
                rank = ranks_of(scores)
                for s in sources:
                    rows.append(
                        {
                            "seed": seed,
                            "mag": mag,
                            "detector": det,
                            "tile": s.tile,
                            "kind": s.kind,
                            "sed": s.sed,
                            "rank": rank[s.tile],
                        }
                    )
            shutil.rmtree(data)
            logger.info("Seed %d at AB %g done", seed, mag)

    with (args.out_dir / "injections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out_dir / "plan.json").write_text(json.dumps(plans, indent=1), encoding="utf-8")
    summary = summarise(rows, len(keys))
    with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    plot(summary, args.out_dir / "completeness.png", len(keys))

    for det in args.detectors:
        parts = []
        for kind in [*args.kinds, "all"]:
            m = m50(summary, det, kind)
            parts.append(f"{kind} {m:.1f}" if m is not None else f"{kind} none")
        logger.info("%s, 50%% found at AB: %s", LABELS[det], ", ".join(parts))
    logger.info("Wrote %s", args.out_dir / "summary.csv")
    return 0
