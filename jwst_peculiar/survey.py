"""The pre-registered COSMOS-Web search, from tile scores to the verdict.

    python -m jwst_peculiar.survey SPLIT_JSON STARS_JSON DATA_ROOT RESULTS_DIR
    python -m jwst_peculiar.survey --check SPLIT_JSON STARS_JSON DATA_ROOT

The second form needs no scores. It applies the rules to every tile and says whether
the search could name one tile alone, and how many would have to beat every
calibration tile together otherwise, which is worth knowing before scoring anything.

docs/search.md has the design and the reason for each rule.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import yaml

from jwst_peculiar.data import owned_patches, read_manifest
from jwst_peculiar.detection import Footprint
from jwst_peculiar.scoring import aggregate_score
from jwst_peculiar.significance import bh_q, conformal_p

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwst_peculiar.utils import SkyCoords

_KEY = re.compile(r"_cx(\d+)_cy(\d+)_ra([\d.]+)_dec(m?)([\d.]+)_")

# Stars brighter than G = 16 get a mask of radius 1.5 arcsec at G = 19, growing as
# sqrt(flux). Measured on PRIMER, which this search never touches.
STAR_G_MAX = 16.0
STAR_RADIUS_AT_19 = 1.5
# The star list has to reach this far past every tile, which covers a G = 5 star.
STAR_MARGIN_DEG = 0.3

REHEARSAL_DRAWS = 1000
# The rehearsal fails, and the search names nothing, if more than this share of
# calibration-only draws find anything. BH at alpha keeps that share at alpha when
# nothing is there.
REHEARSAL_LIMIT = 0.15


def parse_key(key: str) -> tuple[int, int, float, float]:
    """Centre x and y in mosaic pixels and RA and Dec, read from a tile key.

    Returns:
        (cx, cy, ra, dec).

    Raises:
        ValueError: If the key has no position in it.
    """
    m = _KEY.search(key)
    if m is None:
        msg = f"{key} has no _cx_cy_ra_dec_ position in it."
        raise ValueError(msg)
    dec = float(m[5]) * (-1 if m[4] else 1)
    return int(m[1]), int(m[2]), float(m[3]), dec


def mosaic_of(key: str) -> str:
    """The mosaic a tile key was cut from, which is everything before its position.

    Returns:
        The mosaic's tile id.

    Raises:
        ValueError: If the key has no position in it.
    """
    m = _KEY.search(key)
    if m is None:
        msg = f"{key} has no _cx_cy_ra_dec_ position in it."
        raise ValueError(msg)
    return key[: m.start()]


def check_owned(run_dir: Path) -> None:
    """Refuse a run that scored every patch, as its tiles would count shared sky twice.

    Raises:
        ValueError: If the run's args.yaml doesn't say score_region: owned or middle.
    """
    text = (run_dir / "args.yaml").read_text(encoding="utf-8")
    if not re.search(r"^score_region: (owned|middle)$", text, re.MULTILINE):
        msg = (
            f"{run_dir} wasn't scored with --score_region owned, so its overlapping "
            "tiles share sky and can't all count."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class Tiles:
    """Where a set of tiles sit, on the sky and in their mosaics."""

    coords: SkyCoords
    visit: npt.NDArray[np.str_]
    mosaic: npt.NDArray[np.str_]
    centre: npt.NDArray[np.int64]


def overlaps_other_visit(
    tiles: Tiles,
    every: Tiles,
    *,
    size_arcsec: float,
    size_px: int,
    scored: frozenset[str] | None = None,
) -> npt.NDArray[np.bool_]:
    """True for tiles that may share sky with a tile of another visit.

    Visits sit at different position angles, so two tiles from different mosaics can
    only be apart when their centres are a full diagonal apart. Two from the same
    mosaic share its grid, so they overlap only if their boxes do, which lets blocks
    of one mosaic sit side by side.

    `scored` names the search and calibration visits of a split that keeps the first
    copy. Where two of those share sky, only the tile of the visit whose name sorts
    later goes, as one copy can't be its own twin. That depends on the names alone,
    so it treats search and calibration visits alike. Sky shared with anything else,
    the bank or the training tiles, still drops the tile.

    Returns:
        One flag per tile in `tiles`.
    """
    reach = size_arcsec * np.sqrt(2) * u.arcsec
    near_idx, self_idx, _, _ = tiles.coords.search_around_sky(every.coords, reach)
    other = every.visit[near_idx] != tiles.visit[self_idx]
    if scored is not None:
        later = every.visit[near_idx] > tiles.visit[self_idx]
        other &= ~(later & np.isin(every.visit[near_idx], sorted(scored)))
    same_mosaic = every.mosaic[near_idx] == tiles.mosaic[self_idx]
    apart = (
        np.abs(every.centre[near_idx] - tiles.centre[self_idx]).max(axis=1) >= size_px
    )
    out = np.zeros(len(tiles.coords), dtype=bool)
    out[self_idx[other & ~(same_mosaic & apart)]] = True
    return out


def near_bright_star(
    coords: SkyCoords,
    *,
    stars: SkyCoords,
    star_g: npt.NDArray[np.float64],
    size_arcsec: float,
    g_max: float = STAR_G_MAX,
) -> npt.NDArray[np.bool_]:
    """True for tiles within reach of a bright star's spikes.

    Returns:
        One flag per tile in `coords`.
    """
    bright = star_g < g_max
    out = np.zeros(len(coords), dtype=bool)
    if not bright.any():
        return out
    g = star_g[bright]
    reach = size_arcsec / np.sqrt(2) + STAR_RADIUS_AT_19 * 10 ** (0.2 * (19 - g))
    star_idx, tile_idx, sep, _ = coords.search_around_sky(
        stars[bright], float(np.max(reach)) * u.arcsec
    )
    out[tile_idx[sep.arcsec < reach[star_idx]]] = True
    return out


def rehearse(
    scores: dict[str, npt.NDArray[np.float64]],
    n_search: int,
    alpha: float,
    *,
    draws: int,
    seed: int = 0,
) -> npt.NDArray[np.int64]:
    """Run the search on calibration visits alone, where every detection is false.

    Each draw takes `n_search` visits as the search and the rest as calibration.

    Returns:
        The number of tiles BH names in each draw.
    """
    rng = np.random.default_rng(seed)
    names = sorted(scores)
    found = np.zeros(draws, dtype=np.int64)
    for i in range(draws):
        pick = set(rng.choice(len(names), size=n_search, replace=False).tolist())
        test = np.concatenate([scores[names[j]] for j in sorted(pick)])
        cal = np.concatenate([scores[n] for j, n in enumerate(names) if j not in pick])
        found[i] = int((bh_q(conformal_p(test, cal)) <= alpha).sum())
    return found


def _read_scores(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as f:
        return {r["Sample"]: float(r["AnomalyScore"]) for r in csv.DictReader(f)}


def middle_scores(run_dir: Path, data_dir: Path) -> dict[str, float]:
    """Score a finished run again on the middle of each tile, from its saved maps.

    The maps hold every patch's distance, so this is what `--score_region middle`
    would have given the run, without embedding anything again.

    Returns:
        The score of every tile the run scored.
    """
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    maps = run_dir / "cross" / "anomaly_maps" / f"seed={args.get('seed', 0)}"
    scored = _read_scores(run_dir / "measurements_final.csv")
    footprints = {
        k: Footprint(m["mosaic"], m["x0"], m["y0"], m["size"])
        for k, m in read_manifest(data_dir).items()
    }
    out: dict[str, float] = {}
    mine: dict[str, npt.NDArray[np.bool_]] = {}
    for key in scored:
        dists = np.load(maps / f"{Path(key).stem}.npy")
        if not mine:
            mine = owned_patches(
                footprints, (dists.shape[0], dists.shape[1]), middle=True
            )
        out[key] = aggregate_score(
            dists[mine[key]],
            mode=args.get("score_mode", "top1p"),
            top_frac=args.get("score_top_frac", 0.01),
            quantile=args.get("score_quantile", 0.9995),
        )
    return out


def read_stars(
    path: Path, tiles: SkyCoords
) -> tuple[SkyCoords, npt.NDArray[np.float64]]:
    """The bright-star list, after checking its cones cover every tile with margin.

    The list has one cone (`centre` and `radius_deg`) or several (`cones`, each RA,
    Dec and radius in degrees), for a survey of more than one field.

    Returns:
        The stars' positions and G magnitudes.

    Raises:
        ValueError: If a tile is within STAR_MARGIN_DEG of every cone's edge.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    cones = doc.get("cones") or [[*doc["centre"], doc["radius_deg"]]]
    inside = np.zeros(len(tiles), dtype=bool)
    for ra, dec, radius in cones:
        sep: Any = SkyCoord(ra, dec, unit="deg").separation(tiles)
        inside |= np.asarray(sep.deg, dtype=np.float64) <= radius - STAR_MARGIN_DEG
    if not inside.all():
        worst = tiles[~inside][0]
        msg = (
            f"The tile at {worst.ra.deg:.4f} {worst.dec.deg:+.4f} isn't at least "
            f"{STAR_MARGIN_DEG} degrees inside any cone of {path}."
        )
        raise ValueError(msg)
    arr = np.array(doc["rows"], dtype=np.float64).reshape(-1, 3)
    return SkyCoord(arr[:, 0], arr[:, 1], unit="deg"), arr[:, 2]


def tiles_of(keys: list[str], visits: list[str]) -> Tiles:
    """Read where each tile sits from its key.

    Returns:
        The tiles, in the order given.
    """
    p = [parse_key(k) for k in keys]
    return Tiles(
        coords=SkyCoord([x[2] for x in p], [x[3] for x in p], unit="deg"),
        visit=np.array(visits),
        mosaic=np.array([mosaic_of(k) for k in keys]),
        centre=np.array([x[:2] for x in p], dtype=np.int64).reshape(-1, 2),
    )


def _every_tile(data: Path, visits: list[str], training: list[Path]) -> Tiles:
    # A training tile counts as another visit to every tile, so a tile on sky the
    # model was trained on is dropped like one seen twice.
    keys: list[str] = []
    names: list[str] = []
    roots = [(data / v, v) for v in visits] + [(t, "training") for t in training]
    for root, name in roots:
        with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                keys.append(r["key"])
                names.append(name)
    return tiles_of(keys, names)


@dataclass(frozen=True)
class _Rules:
    size_arcsec: float
    size_px: int
    every: Tiles
    stars: SkyCoords
    star_g: npt.NDArray[np.float64]
    star_g_max: float
    scored: frozenset[str] | None


def _filter_visit(
    visit: str, scores: dict[str, float], rules: _Rules
) -> tuple[dict[str, float], dict[str, int]]:
    keys = sorted(scores)
    tiles = tiles_of(keys, [visit] * len(keys))
    overlap = overlaps_other_visit(
        tiles,
        rules.every,
        size_arcsec=rules.size_arcsec,
        size_px=rules.size_px,
        scored=rules.scored,
    )
    c = tiles.coords
    star = near_bright_star(
        c,
        stars=rules.stars,
        star_g=rules.star_g,
        size_arcsec=rules.size_arcsec,
        g_max=rules.star_g_max,
    )
    keep = ~overlap & ~star
    counts = {"overlap": int(overlap.sum()), "star": int((~overlap & star).sum())}
    return {k: scores[k] for k, ok in zip(keys, keep, strict=True) if ok}, counts


def _manifest_keys(root: Path) -> dict[str, float]:
    with (root / "manifest.csv").open(encoding="utf-8", newline="") as f:
        return {r["key"]: 0.0 for r in csv.DictReader(f)}


def kept_tiles(
    split: dict[str, Any],
    *,
    stars_json: Path,
    data: Path,
    results: Path | None,
    middle: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Apply the overlap and star rules to each search and calibration visit.

    With no results every tile scores 0, which is enough to count what the rules keep.
    With `middle`, each run's tiles are scored again on their middle first.

    Returns:
        The kept scores per visit, and how many tiles each rule dropped.
    """
    every_visit = [*split["search"], *split["bank"], *split["calibration"]]
    mosaic = next((data / every_visit[0] / "mosaics").glob("*.json"))
    scale = float(json.loads(mosaic.read_text(encoding="utf-8"))["pixel_scale_arcsec"])
    training = [Path(t) for t in split.get("training_data", [])]
    every = _every_tile(data, every_visit, training)
    stars, star_g = read_stars(stars_json, every.coords)
    rules = _Rules(
        size_arcsec=int(split["tile_px"]) * scale,
        size_px=int(split["tile_px"]),
        every=every,
        stars=stars,
        star_g=star_g,
        star_g_max=float(split.get("star_g_max", STAR_G_MAX)),
        scored=frozenset([*split["search"], *split["calibration"]])
        if split.get("keep_first_copy")
        else None,
    )
    kept: dict[str, dict[str, float]] = {}
    dropped = {"overlap": 0, "star": 0}
    for v in [*split["search"], *split["calibration"]]:
        if results is None:
            scores = _manifest_keys(data / v)
        else:
            check_owned(results / v)
            scores = (
                middle_scores(results / v, data / v)
                if middle
                else _read_scores(results / v / "measurements_final.csv")
            )
        kept[v], counts = _filter_visit(v, scores, rules)
        for rule, n in counts.items():
            dropped[rule] += n
    return kept, dropped


def cut_blocks(
    dataset: Path, out: Path, *, block_px: int, min_tiles: int
) -> dict[str, int]:
    """Split a cut mosaic into square blocks that can stand in for visits.

    A tile goes to a block only if all of it lies inside, so tiles of two blocks never
    share a pixel and the tiles that straddle a border belong to none. The tiles,
    cutouts and mosaic records are hard links, not copies.

    Returns:
        How many tiles each written block holds.
    """
    with (dataset / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    blocks: dict[tuple[str, int, int], list[dict[str, str]]] = {}
    for r in rows:
        x0, y0, size = int(r["x0"]), int(r["y0"]), int(r["size"])
        bx, by = x0 // block_px, y0 // block_px
        if (x0 + size - 1) // block_px == bx and (y0 + size - 1) // block_px == by:
            blocks.setdefault((r["mosaic"], by, bx), []).append(r)
    written: dict[str, int] = {}
    for (mosaic, by, bx), members in sorted(blocks.items()):
        if len(members) < min_tiles:
            continue
        name = f"{mosaic}_b{by:02d}_{bx:02d}"
        root = out / name
        (root / "query").mkdir(parents=True)
        shutil.copytree(dataset / "mosaics", root / "mosaics")
        for r in members:
            os.link(dataset / "query" / r["key"], root / "query" / r["key"])
            if r.get("cutout"):
                (root / r["cutout"]).parent.mkdir(exist_ok=True)
                os.link(dataset / r["cutout"], root / r["cutout"])
        with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(members)
        written[name] = len(members)
    return written


def standardise(kept: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Put each visit's scores on its own median and MAD.

    A visit rendered darker or brighter than the rest shifts every one of its scores,
    and in COSMOS-Web one such visit made most of the calibration tail. Done the same
    way for every visit, it keeps search and calibration visits exchangeable.

    Returns:
        The kept scores per visit, each as `(score - median) / (1.4826 * MAD)`.

    Raises:
        ValueError: If a visit's scores have no spread to divide by.
    """
    out: dict[str, dict[str, float]] = {}
    for v, scores in kept.items():
        if len(scores) < 2:  # noqa: PLR2004 the rules left no spread to place it on
            out[v] = {}
            continue
        x = np.array(list(scores.values()))
        med = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - med)))
        if not mad > 0:
            msg = f"{v}'s {x.size} kept scores have no spread to divide by"
            raise ValueError(msg)
        out[v] = {k: (s - med) / mad for k, s in scores.items()}
    return out


def can_name_one(
    split: dict[str, Any], kept: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Whether the best possible conformal p can pass BH for a single tile.

    BH passes k tiles at the smallest p once that p is at most k alpha over the
    search tiles, so a search that can't name one tile can still name several that
    each beat every calibration tile.

    Returns:
        The tile counts, the smallest p, what BH needs, the answer, and how many
        tiles at the smallest p it takes for any to pass.
    """
    if split.get("standardise"):
        kept = {v: s if len(s) >= 2 else {} for v, s in kept.items()}  # noqa: PLR2004 as standardise leaves them
    n_search = sum(len(kept[v]) for v in split["search"])
    n_cal = sum(len(kept[v]) for v in split["calibration"])
    floor = 1 / (n_cal + 1)
    needs = float(split["alpha"]) / n_search if n_search else 0.0
    return {
        "search_tiles": n_search,
        "calibration_tiles": n_cal,
        "smallest_p": floor,
        "bh_needs_for_one": needs,
        "can_name_one": bool(n_search) and floor <= needs,
        "tiles_needed_together": math.ceil(floor / needs) if n_search else 0,
    }


def _write_pvalues(
    path: Path,
    *,
    search: dict[str, float],
    p: npt.NDArray[np.float64],
    q: npt.NDArray[np.float64],
) -> None:
    keys = sorted(search)
    scores = np.array([search[k] for k in keys])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "score", "p", "q"])
        for i in np.lexsort((-scores, p)):
            w.writerow([keys[i], f"{scores[i]:.6g}", f"{p[i]:.6g}", f"{q[i]:.6g}"])


def _check(argv: list[str]) -> int:
    split = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    kept, _ = kept_tiles(
        split, stars_json=Path(argv[1]), data=Path(argv[2]), results=None
    )
    power = can_name_one(split, kept)
    sys.stdout.write(json.dumps(power, indent=1) + "\n")
    return 0 if power["can_name_one"] else 1


def main(argv: list[str]) -> int:  # noqa: PLR0914 one pass, in the order docs/search.md gives
    """Write search_pvalues.csv and verdict.json, and print the verdict.

    With --middle first, the tiles are scored again on their middle and the two files
    get _middle in their names, so a search's own verdict stays beside the rescore.

    Returns:
        0, also when nothing is found. With --check, 1 if no tile could be named
        alone.

    Raises:
        ValueError: If no search tile is left once the rules have run.
    """
    if argv[0] == "--check":
        return _check(argv[1:])
    middle = argv[0] == "--middle"
    argv = argv[middle:]
    split = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    results = Path(argv[3])
    alpha = float(split["alpha"])
    kept, dropped = kept_tiles(
        split,
        stars_json=Path(argv[1]),
        data=Path(argv[2]),
        results=results,
        middle=middle,
    )
    if split.get("standardise"):
        kept = standardise(kept)

    search = {k: x for v in split["search"] for k, x in kept[v].items()}
    cal = np.array([x for v in split["calibration"] for x in kept[v].values()])
    keys = sorted(search)
    if not keys:
        msg = "Every search tile was dropped by the overlap or star rules."
        raise ValueError(msg)
    p = conformal_p(np.array([search[k] for k in keys]), cal)
    q = bh_q(p)

    found = rehearse(
        {v: np.array(list(kept[v].values())) for v in split["calibration"]},
        len(split["search"]),
        alpha,
        draws=REHEARSAL_DRAWS,
    )
    rate = float((found > 0).mean())
    passed = rate <= REHEARSAL_LIMIT
    verdict = {
        "search_tiles": len(keys),
        "calibration_tiles": int(cal.size),
        "dropped": dropped,
        "smallest_p": 1 / (cal.size + 1),
        "bh_needs_for_one": alpha / len(keys),
        "rehearsal_draws_with_any_detection": rate,
        "rehearsal_passed": passed,
        "detections": [
            k for k, qk in zip(keys, q, strict=True) if passed and qk <= alpha
        ],
    }
    text = json.dumps(verdict, indent=1) + "\n"
    # Both files go out together, once the rehearsal has run.
    tag = "_middle" if middle else ""
    _write_pvalues(results / f"search_pvalues{tag}.csv", search=search, p=p, q=q)
    (results / f"verdict{tag}.json").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
