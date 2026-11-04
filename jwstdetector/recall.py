"""How a run ranks objects already known to be there, such as a cluster's lensed images.

    jwstdetector recall results_smacs datasets/smacs_nircam --objects lensed.csv

The objects come from a CSV with ra and dec columns in degrees, and an optional id
column like 1.2, where the part before the dot names a group such as a lens system.
Each object is placed on the one tile that scores its patch, as the run itself
assigns them, and gets that tile's rank and the patch distance at its position.
Positions in a lensing catalogue are a pixel or two off at best and a lensed image
spans a few patches, so the distance is the highest within one patch of it. Every
patch it's compared with gets the same treatment, as the highest of nine random
patches already sits near the 93rd percentile of single ones.

The patch percentile and the AUC compare those distances with every patch the run
scored, which needs no labels for the rest of the field. Known objects that are
ordinary for their field, like most faint lensed galaxies, rightly score low, so
this measures what a detector of the unusual finds among them, not completeness.

Lensed images crowd around the cluster core, which is unusual all over, so a ranking
that only knew where the core was would already do well on the tile figures. With
--centre, each object is also compared only with patches at the same distance from
that point, within --ring arcseconds, which takes the location out.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import maximum_filter
import yaml

from jwstdetector.bundle import map_path, mosaic_wcs
from jwstdetector.data import (
    final_pass_dir,
    owned_patches,
    read_manifest,
    read_measurements,
)
from jwstdetector.detection import Footprint

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwstdetector.utils import SkyWcs

logger = logging.getLogger(__name__)

SHARES = (0.05, 0.10)
_TILE_ONLY_GRID = 28


@dataclass(frozen=True)
class Placed:
    """A known object on the tile that scores it."""

    id: str
    group: str
    tile: str
    rank: int
    distance: float
    percentile: float
    in_tile: float
    at_radius: float


@dataclass
class _Field:
    """Every patch a run scored, and what placing an object on one needs."""

    n_tiles: int
    manifest: dict[str, dict[str, Any]]
    rank: dict[str, int]
    maps: dict[str, npt.NDArray[np.float64]]
    owned: dict[str, npt.NDArray[np.bool_]]
    has_maps: bool
    unsorted: npt.NDArray[np.float64]
    field: npt.NDArray[np.float64]
    radii: npt.NDArray[np.float64] | None
    centre: tuple[float, float] | None
    ring: float

    @property
    def grid(self) -> tuple[int, ...]:
        """The patch grid every map shares, as (rows, cols)."""
        return next(iter(self.maps.values())).shape


def read_objects(path: Path) -> list[dict[str, str]]:
    """Read the known objects, giving each an id and a group when it has none.

    Returns:
        One row per object, with its CSV columns as strings.

    Raises:
        SystemExit: The CSV has no rows, or no ra and dec columns.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"ra", "dec"} <= set(rows[0]):
        msg = f"{path} needs ra and dec columns in degrees."
        raise SystemExit(msg)
    for i, r in enumerate(rows):
        r.setdefault("id", str(i + 1))
        r["group"] = r.get("group") or r["id"].split(".")[0]
    return rows


def _patch_radii(
    owned: npt.NDArray[np.bool_],
    meta: dict[str, Any],
    *,
    wcs: SkyWcs,
    centre: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Find how far each owned patch of a tile is from `centre`.

    Returns:
        The distance in arcseconds to the middle of each owned patch.
    """
    rows, cols = np.nonzero(owned)
    grid = owned.shape
    x = meta["x0"] + (cols + 0.5) * meta["size"] / grid[1] - 0.5
    y = meta["y0"] + meta["size"] - (rows + 0.5) * meta["size"] / grid[0] - 0.5
    ra, dec = wcs.pixel_to_world_values(x, y)
    return separation(ra, dec, centre)


def separation(
    ra: npt.NDArray[np.float64] | float | str,
    dec: npt.NDArray[np.float64] | float | str,
    centre: tuple[float, float],
) -> npt.NDArray[np.float64]:
    """Great circle distance from `centre` in arcseconds, all positions in degrees.

    Returns:
        The distance of each position, the same shape as `ra` and `dec`.
    """
    ra0, dec0 = np.radians(centre)
    ra = np.radians(np.asarray(ra, dtype=np.float64))
    dec = np.radians(np.asarray(dec, dtype=np.float64))
    cos = np.sin(dec) * np.sin(dec0) + np.cos(dec) * np.cos(dec0) * np.cos(ra - ra0)
    arcsec: npt.NDArray[np.float64] = np.degrees(np.arccos(np.clip(cos, -1, 1))) * 3600
    return arcsec


def _read_maps(
    maps_dir: Path, manifest: dict[str, dict[str, Any]], scores: dict[str, float]
) -> dict[str, npt.NDArray[np.float64]]:
    if maps_dir.is_dir():
        return {
            k: maximum_filter(np.load(map_path(maps_dir, k)), size=3, mode="nearest")
            for k in manifest
        }
    # A detector with no patch maps, like the isolation forest, gives every patch its
    # tile's score, so the percentiles rank tiles and the in-tile figure means
    # nothing.
    return {k: np.full((_TILE_ONLY_GRID, _TILE_ONLY_GRID), scores[k]) for k in manifest}


def _owned(
    manifest: dict[str, dict[str, Any]], grid: tuple[int, ...], *, region: str
) -> dict[str, npt.NDArray[np.bool_]]:
    if region != "owned":
        return {k: np.ones(grid, dtype=bool) for k in manifest}
    footprints = {
        k: Footprint(m["mosaic"], m["x0"], m["y0"], m["size"])
        for k, m in manifest.items()
    }
    return owned_patches(footprints, (grid[0], grid[1]))


def _field_radii(
    data_root: Path,
    manifest: dict[str, dict[str, Any]],
    owned: dict[str, npt.NDArray[np.bool_]],
    *,
    centre: tuple[float, float],
) -> npt.NDArray[np.float64]:
    wcs_of: dict[str, SkyWcs] = {}
    for m in manifest.values():
        if m["mosaic"] not in wcs_of:
            meta = json.loads(
                (data_root / "mosaics" / f"{m['mosaic']}.json").read_text("utf-8")
            )
            wcs_of[m["mosaic"]] = mosaic_wcs(meta)
    return np.concatenate([
        _patch_radii(owned[k], m, wcs=wcs_of[m["mosaic"]], centre=centre)
        for k, m in manifest.items()
    ])


def _load_field(
    run_dir: Path,
    data_root: Path,
    *,
    centre: tuple[float, float] | None,
    ring: float,
) -> _Field:
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) or {}
    maps_dir = final_pass_dir(run_dir) / f"anomaly_maps/seed={args.get('seed', 0)}"
    scores = read_measurements(run_dir / "measurements_final.csv")
    manifest = {k: m for k, m in read_manifest(data_root).items() if k in scores}
    ranked = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
    maps = _read_maps(maps_dir, manifest, scores)
    grid = next(iter(maps.values())).shape
    owned = _owned(manifest, grid, region=args.get("score_region", "all"))
    unsorted = np.concatenate([maps[k][owned[k]] for k in manifest])
    radii = None
    if centre is not None:
        radii = _field_radii(data_root, manifest, owned, centre=centre)
    return _Field(
        n_tiles=len(scores),
        manifest=manifest,
        rank={k: i for i, k in enumerate(ranked, start=1)},
        maps=maps,
        owned=owned,
        has_maps=maps_dir.is_dir(),
        unsorted=unsorted,
        field=np.sort(unsorted),
        radii=radii,
        centre=centre,
        ring=ring,
    )


def _placed(
    run: _Field, obj: dict[str, str], key: str, *, patch: tuple[int, int]
) -> Placed:
    d = float(run.maps[key][patch])
    at_radius = np.nan
    if run.radii is not None and run.centre is not None:
        near = np.abs(run.radii - separation(obj["ra"], obj["dec"], run.centre))
        at_radius = 100 * float(np.mean(run.unsorted[near <= run.ring] <= d))
    percentile = 100 * np.searchsorted(run.field, d, side="right") / len(run.field)
    return Placed(
        id=obj["id"],
        group=obj["group"],
        tile=key,
        rank=run.rank[key],
        distance=d,
        percentile=percentile,
        in_tile=100 * float(np.mean(run.maps[key][run.owned[key]] <= d))
        if run.has_maps
        else np.nan,
        at_radius=at_radius,
    )


def _place_on(
    run: _Field, obj: dict[str, str], keys: list[str], *, xy: tuple[float, float]
) -> Placed | None:
    """Place one object on the first of `keys` that owns the patch at mosaic `xy`.

    Returns:
        The placed object, or None when none of those tiles scores that spot.
    """
    x, y = xy
    rows, cols = run.grid[0], run.grid[1]
    for k in keys:
        m = run.manifest[k]
        c = int((x + 0.5 - m["x0"]) * cols // m["size"])
        # PNG rows run down from the tile's highest FITS row.
        r = int((m["y0"] + m["size"] - y - 0.5) * rows // m["size"])
        if 0 <= r < rows and 0 <= c < cols and run.owned[k][r, c]:
            return _placed(run, obj, k, patch=(r, c))
    return None


def place(
    objects: list[dict[str, str]],
    run_dir: Path,
    data_root: Path,
    *,
    centre: tuple[float, float] | None = None,
    ring: float = 3.0,
) -> tuple[list[Placed], int, npt.NDArray[np.float64]]:
    """Puts each object on the tile that owns its patch.

    Returns:
        The placed objects, the number of tiles and the sorted distance of every
        scored patch in the run.
    """
    run = _load_field(run_dir, data_root, centre=centre, ring=ring)
    by_mosaic: dict[str, list[str]] = {}
    for k, m in run.manifest.items():
        by_mosaic.setdefault(m["mosaic"], []).append(k)
    placed: list[Placed] = []
    for mosaic, keys in by_mosaic.items():
        meta = json.loads((data_root / "mosaics" / f"{mosaic}.json").read_text("utf-8"))
        xs, ys = mosaic_wcs(meta).world_to_pixel_values(
            [float(o["ra"]) for o in objects], [float(o["dec"]) for o in objects]
        )
        for o, x, y in zip(objects, xs, ys, strict=True):
            p = _place_on(run, o, keys, xy=(x, y))
            if p is not None:
                placed.append(p)
    return placed, run.n_tiles, run.field


def auc(positives: npt.NDArray[np.float64], field: npt.NDArray[np.float64]) -> float:
    """Chance that a known object's distance beats a random scored patch.

    Ties count as half.

    Returns:
        The area under the ROC curve, from 0 to 1.
    """
    lo = np.searchsorted(field, positives, side="left")
    hi = np.searchsorted(field, positives, side="right")
    return float(np.mean((lo + hi) / 2) / len(field))


def group_interval(values: list[tuple[str, float]], n: int = 5000) -> list[float]:
    """95% bootstrap interval of the mean, drawing whole groups.

    The images of one lens system are not independent of each other.

    Returns:
        The interval's low and high ends.
    """
    by: dict[str, list[float]] = {}
    for g, v in values:
        by.setdefault(g, []).append(v)
    groups = list(by.values())
    rng = np.random.default_rng(0)
    means = [
        np.mean([
            v for i in rng.integers(len(groups), size=len(groups)) for v in groups[i]
        ])
        for _ in range(n)
    ]
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def _at_radius_summary(placed: list[Placed]) -> dict[str, Any]:
    at_radius = [p.at_radius for p in placed if not np.isnan(p.at_radius)]
    if not at_radius:
        return {}
    # A percentile among patches at the same radius, so 50 is what a ranking that
    # only knew where the cluster was would get.
    return {
        "median_at_radius": float(np.median(at_radius)),
        "mean_at_radius": float(np.mean(at_radius)),
        "mean_at_radius_95": group_interval([
            (p.group, p.at_radius) for p in placed if not np.isnan(p.at_radius)
        ]),
    }


def summarise(
    placed: list[Placed], n_tiles: int, field: npt.NDArray[np.float64]
) -> dict[str, Any]:
    """Sum up how highly the run ranks the placed objects.

    Returns:
        The figures recall.json holds and the summary lines print.
    """
    out: dict[str, Any] = {"objects": len(placed), "tiles": n_tiles}
    if not placed:
        return out
    d = np.array([p.distance for p in placed])
    out["median_percentile"] = float(np.median([p.percentile for p in placed]))
    # The same rank among the other patches of its own tile, which a tile that is odd
    # all over, like the core of a cluster, can't raise.
    out["median_in_tile"] = float(np.median([p.in_tile for p in placed]))
    if np.isnan(out["median_in_tile"]):
        del out["median_in_tile"]
    out.update(_at_radius_summary(placed))
    out["auc"] = auc(d, field)
    groups = sorted({p.group for p in placed})
    out["groups"] = len(groups)
    for share in SHARES:
        top = max(1, round(share * n_tiles))
        tag = f"top{round(100 * share)}"
        out[f"{tag}_objects"] = sum(p.rank <= top for p in placed)
        out[f"{tag}_groups"] = sum(
            any(p.rank <= top for p in placed if p.group == g) for g in groups
        )
        # The share of objects a ranking that knew nothing would put there.
        out[f"{tag}_chance"] = top / n_tiles
    return out


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="A `jwstdetector run` output folder.")
    ap.add_argument("data_root", type=Path, help="The prep folder the run scored.")
    ap.add_argument(
        "--objects", type=Path, required=True, help="CSV with ra, dec and id."
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="Per-object CSV, RUN_DIR/recall.csv."
    )
    ap.add_argument(
        "--centre",
        type=float,
        nargs=2,
        metavar=("RA", "DEC"),
        default=None,
        help="Cluster centre in degrees, to compare each object only with patches as "
        "far from it.",
    )
    ap.add_argument(
        "--ring",
        type=float,
        default=3.0,
        help="Half-width in arcseconds of that comparison.",
    )
    return ap


def _write_placed(out: Path, placed: list[Placed]) -> None:
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id",
            "group",
            "tile",
            "rank",
            "distance",
            "percentile",
            "in_tile",
            "at_radius",
        ])
        for p in placed:
            w.writerow([
                p.id,
                p.group,
                p.tile,
                p.rank,
                f"{p.distance:.6f}",
                f"{p.percentile:.2f}",
                f"{p.in_tile:.2f}",
                f"{p.at_radius:.2f}",
            ])


def _print_summary(summary: dict[str, Any]) -> None:
    in_tile = (
        f", median percentile within its own tile {summary['median_in_tile']:.1f}."
        if "median_in_tile" in summary
        else ", from tile scores alone as the run saved no patch maps."
    )
    lines = [
        f"{summary['objects']} objects in {summary['groups']} groups. Median patch "
        f"percentile {summary['median_percentile']:.1f}, AUC {summary['auc']:.3f}"
        + in_tile
    ]
    if "median_at_radius" in summary:
        lo, hi = summary["mean_at_radius_95"]
        lines.append(
            f"Against patches as far from the centre: median percentile "
            f"{summary['median_at_radius']:.1f}, "
            f"mean {summary['mean_at_radius']:.1f} "
            f"(95% {lo:.1f} to {hi:.1f} drawing whole groups), where 50 is chance."
        )
    for share in SHARES:
        tag = f"top{round(100 * share)}"
        objects = summary[f"{tag}_objects"]
        lines.append(
            f"Tile in the top {round(100 * share)}%: {objects} objects "
            f"({100 * objects / summary['objects']:.0f}%, chance "
            f"{100 * summary[f'{tag}_chance']:.0f}%), {summary[f'{tag}_groups']} of "
            f"{summary['groups']} groups."
        )
    sys.stdout.write("".join(line + "\n" for line in lines))


def main(argv: list[str] | None = None) -> int:
    """Place the known objects on a run's tiles and write how highly they rank.

    Returns:
        0 once the CSV and its summary are written.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    objects = read_objects(args.objects)
    centre = (args.centre[0], args.centre[1]) if args.centre else None
    placed, n_tiles, field = place(
        objects, args.run_dir, args.data_root, centre=centre, ring=args.ring
    )
    if len(placed) < len(objects):
        logger.info(
            "%d of %d objects fall on no scored tile, as they're off the mosaic or "
            "within a tile of its edge",
            len(objects) - len(placed),
            len(objects),
        )
    out = args.out or args.run_dir / "recall.csv"
    _write_placed(out, placed)
    summary = summarise(placed, n_tiles, field)
    (out.with_suffix(".json")).write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )
    if placed:
        _print_summary(summary)
    return 0
