"""The pre-registered COSMOS-Web search, from tile scores to the verdict.

    python -m jwstdetector.survey SPLIT_JSON STARS_JSON DATA_ROOT RESULTS_DIR

docs/search.md has the design and the reason for each rule.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np

from jwstdetector.significance import bh_q, conformal_p

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwstdetector.utils import SkyCoords

_KEY = re.compile(r"_cx(\d+)_cy(\d+)_ra([\d.]+)_dec(m?)([\d.]+)_")

# Stars brighter than this get a mask, of this many arcseconds at G = 19 and growing
# by sqrt(flux), measured on PRIMER, which this search never touches.
STAR_G_MAX = 16.0
STAR_RADIUS_AT_19 = 1.5
# The star list has to reach this far past every tile, which covers a G = 5 star.
STAR_MARGIN_DEG = 0.3

REHEARSAL_DRAWS = 1000
# The rehearsal fails, and the search names nothing, if more than this share of
# draws of calibration visits alone gets any detection at all. BH at alpha keeps
# that share at alpha when nothing is there.
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
        ValueError: If the run's args.yaml doesn't say score_region: owned.
    """
    text = (run_dir / "args.yaml").read_text(encoding="utf-8")
    if not re.search(r"^score_region: owned$", text, re.MULTILINE):
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
    tiles: Tiles, every: Tiles, *, size_arcsec: float, size_px: int
) -> npt.NDArray[np.bool_]:
    """True for tiles that may share sky with a tile of another visit.

    Visits sit at different position angles, so two tiles from different mosaics can
    only be apart when their centres are a full diagonal apart. Two from the same
    mosaic share its grid, so they overlap only if their boxes do, which lets blocks
    of one mosaic sit side by side.

    Returns:
        One flag per tile in `tiles`.
    """
    reach = size_arcsec * np.sqrt(2) * u.arcsec
    near_idx, self_idx, _, _ = tiles.coords.search_around_sky(every.coords, reach)
    other = every.visit[near_idx] != tiles.visit[self_idx]
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
) -> npt.NDArray[np.bool_]:
    """True for tiles within reach of a bright star's spikes.

    Returns:
        One flag per tile in `coords`.
    """
    bright = star_g < STAR_G_MAX
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


def read_stars(
    path: Path, tiles: SkyCoords
) -> tuple[SkyCoords, npt.NDArray[np.float64]]:
    """The bright-star list, after checking its cone covers every tile with margin.

    Returns:
        The stars' positions and G magnitudes.

    Raises:
        ValueError: If a tile is within STAR_MARGIN_DEG of the cone's edge.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    centre = SkyCoord(*doc["centre"], unit="deg")
    sep: Any = centre.separation(tiles)
    furthest = float(np.max(np.asarray(sep.deg, dtype=np.float64)))
    if furthest > float(doc["radius_deg"]) - STAR_MARGIN_DEG:
        msg = (
            f"A tile is {furthest:.3f} degrees from the centre of {path}, which only "
            f"lists stars out to {doc['radius_deg']} degrees."
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


def _filter_visit(
    visit: str, scores: dict[str, float], rules: _Rules
) -> tuple[dict[str, float], dict[str, int]]:
    keys = sorted(scores)
    tiles = tiles_of(keys, [visit] * len(keys))
    overlap = overlaps_other_visit(
        tiles, rules.every, size_arcsec=rules.size_arcsec, size_px=rules.size_px
    )
    c = tiles.coords
    star = near_bright_star(
        c, stars=rules.stars, star_g=rules.star_g, size_arcsec=rules.size_arcsec
    )
    keep = ~overlap & ~star
    counts = {"overlap": int(overlap.sum()), "star": int((~overlap & star).sum())}
    return {k: scores[k] for k, ok in zip(keys, keep, strict=True) if ok}, counts


def _kept_tiles(
    split: dict[str, Any], *, stars_json: Path, data: Path, results: Path
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Apply the overlap and star rules to each search and calibration visit.

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
    )
    kept: dict[str, dict[str, float]] = {}
    dropped = {"overlap": 0, "star": 0}
    for v in [*split["search"], *split["calibration"]]:
        check_owned(results / v)
        scores = _read_scores(results / v / "measurements_final.csv")
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
        x = np.array(list(scores.values()))
        med = float(np.median(x))
        mad = 1.4826 * float(np.median(np.abs(x - med)))
        if not mad > 0:
            msg = f"{v}'s {x.size} kept scores have no spread to divide by"
            raise ValueError(msg)
        out[v] = {k: (s - med) / mad for k, s in scores.items()}
    return out


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


def main(argv: list[str]) -> int:  # noqa: PLR0914 one pass, in the order docs/search.md gives
    """Write search_pvalues.csv and verdict.json, and print the verdict.

    Returns:
        0, also when nothing is found.

    Raises:
        ValueError: If no search tile is left once the rules have run.
    """
    split = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    results = Path(argv[3])
    alpha = float(split["alpha"])
    kept, dropped = _kept_tiles(
        split, stars_json=Path(argv[1]), data=Path(argv[2]), results=results
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
    _write_pvalues(results / "search_pvalues.csv", search=search, p=p, q=q)

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
    (results / "verdict.json").write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
