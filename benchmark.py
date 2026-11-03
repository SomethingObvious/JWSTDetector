#!/usr/bin/env python3
"""Plants fake sources in real tiles and checks how high the detector ranks them.

    python benchmark.py --data_root datasets/smacs_nircam --out_dir bench_smacs
    python benchmark.py --data_root datasets/smacs_nircam --out_dir bench_smacs \\
        --strength 0.15 -- --detector subspace --layers mid

It copies the query tiles to OUT_DIR/data, adds a point source, arc, ring or streak
to a random few of them, runs run_query_bootstrap.py on the copy and writes where
each planted tile landed to injections.csv. Anything after `--` goes to the runner.
Real anomalies in the field compete with the planted ones, so a planted tile ranked
just below them still counts as found by recall at 5%.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import run_query_bootstrap
from jwstdetector.data import read_measurements

logger = logging.getLogger("benchmark")

KINDS = ("point", "arc", "ring", "streak")
COLOURS = {
    "red": (1.0, 0.35, 0.1),
    "blue": (0.3, 0.55, 1.0),
    "white": (1.0, 1.0, 1.0),
}


def profile(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """A unit-peak brightness map of one fake source, somewhere away from the edges."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = rng.uniform(0.3, 0.7, 2) * size
    if kind == "point":
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        return np.exp(-r2 / (2 * 1.8**2))
    if kind == "streak":
        theta = rng.uniform(0, np.pi)
        d = (x - cx) * np.sin(theta) - (y - cy) * np.cos(theta)
        return np.exp(-(d**2) / (2 * 1.0**2))
    radius = rng.uniform(0.06, 0.15) * size
    ring = np.exp(-((np.hypot(x - cx, y - cy) - radius) ** 2) / (2 * 1.5**2))
    if kind == "ring":
        return ring
    # An arc is the part of a ring within a random 60 to 140 degrees of one direction.
    start = rng.uniform(-np.pi, np.pi)
    span = np.deg2rad(rng.uniform(60, 140))
    angle = (np.arctan2(y - cy, x - cx) - start) % (2 * np.pi)
    return ring * (angle < span)


def inject(path: Path, kind: str, colour: str, strength: float, rng: np.random.Generator) -> None:
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    if img.shape[0] != img.shape[1]:
        raise SystemExit(f"{path.name} isn't square, and the benchmark expects square tiles")
    source = profile(kind, img.shape[0], rng)[..., None] * np.array(COLOURS[colour])
    img = np.clip(img + 255 * strength * source, 0, 255)
    Image.fromarray(img.round().astype(np.uint8)).save(path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runner_args = []
    if "--" in argv:
        cut = argv.index("--")
        argv, runner_args = argv[:cut], argv[cut + 1 :]

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data_root", type=Path, required=True, help="A prep_data.py --out folder.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_inject", type=int, default=12, help="How many tiles get a fake source.")
    ap.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Peak brightness a source adds, as a share of the full 0 to 255 range.",
    )
    ap.add_argument("--kinds", default=",".join(KINDS), help="Any of " + ", ".join(KINDS) + ".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="Start again in an existing OUT_DIR.")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    kinds = args.kinds.split(",")
    if not set(kinds) <= set(KINDS):
        raise SystemExit(f"--kinds takes {', '.join(KINDS)}, not {args.kinds}")
    if args.out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.out_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.out_dir)

    data = args.out_dir / "data"
    shutil.copytree(args.data_root / "query", data / "query")
    if (args.data_root / "manifest.csv").is_file():
        shutil.copyfile(args.data_root / "manifest.csv", data / "manifest.csv")

    tiles = sorted((data / "query").glob("*.png"))
    if args.n_inject >= len(tiles) // 4:
        raise SystemExit(
            f"Planting {args.n_inject} sources in {len(tiles)} tiles would make them "
            "too common to count as anomalies. Use fewer."
        )
    rng = np.random.default_rng(args.seed)
    picked = rng.choice(len(tiles), size=args.n_inject, replace=False)
    planted = {}
    for i, idx in enumerate(picked):
        kind = kinds[i % len(kinds)]
        colour = list(COLOURS)[rng.integers(len(COLOURS))]
        inject(tiles[idx], kind, colour, args.strength, rng)
        planted[tiles[idx].name] = (kind, colour)

    results = args.out_dir / "results"
    run_query_bootstrap.main(["--data_root", str(data), "--out_dir", str(results), *runner_args])

    scores = read_measurements(results / "measurements_final.csv")
    order = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
    rank = {k: i + 1 for i, k in enumerate(order)}
    rows = sorted(
        ((rank[name], name, kind, colour) for name, (kind, colour) in planted.items()),
    )
    with (args.out_dir / "injections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tile", "kind", "colour"])
        w.writerows(rows)

    n = len(order)
    top = max(1, round(0.05 * n))
    found = sum(r <= top for r, *_ in rows)
    ranks = [r for r, *_ in rows]
    logger.info(
        "Planted %d sources at strength %.2f in %d tiles. %d are in the top 5%% (%d tiles), "
        "median rank %d, worst %d",
        len(rows),
        args.strength,
        n,
        found,
        top,
        int(np.median(ranks)),
        max(ranks),
    )
    for kind in kinds:
        mine = [r for r, _, k, _ in rows if k == kind]
        if mine:
            logger.info("  %-6s ranks %s", kind, ", ".join(map(str, mine)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
