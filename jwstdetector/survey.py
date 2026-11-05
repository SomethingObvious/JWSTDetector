"""The pre-registered COSMOS-Web search, from tile scores to the verdict.

    python -m jwstdetector.survey SPLIT_JSON STARS_JSON DATA_ROOT RESULTS_DIR

docs/search.md has the design and the reason for each rule.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
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


def on_grid(
    cx: npt.NDArray[np.int64], cy: npt.NDArray[np.int64], grid: int
) -> npt.NDArray[np.bool_]:
    """True for tiles centred on a multiple of `grid` pixels, which share no pixel.

    Returns:
        One flag per tile.
    """
    keep: npt.NDArray[np.bool_] = (cx % grid == 0) & (cy % grid == 0)
    return keep


def overlaps_other_visit(
    coords: SkyCoords,
    visits: npt.NDArray[np.str_],
    *,
    every: SkyCoords,
    every_visit: npt.NDArray[np.str_],
    size_arcsec: float,
) -> npt.NDArray[np.bool_]:
    """True for tiles that may share sky with a tile of another visit.

    Visits sit at different position angles, so two tiles can only be apart when
    their centres are a full diagonal apart.

    Returns:
        One flag per tile in `coords`.
    """
    reach = size_arcsec * np.sqrt(2) * u.arcsec
    near_idx, self_idx, _, _ = coords.search_around_sky(every, reach)
    out = np.zeros(len(coords), dtype=bool)
    out[self_idx[every_visit[near_idx] != visits[self_idx]]] = True
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


def _sky_of_every_tile(
    data: Path, visits: list[str]
) -> tuple[SkyCoords, npt.NDArray[np.str_]]:
    names, pos = [], []
    for v in visits:
        with (data / v / "manifest.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                names.append(v)
                pos.append(parse_key(r["key"])[2:])
    p = np.array(pos)
    return SkyCoord(p[:, 0], p[:, 1], unit="deg"), np.array(names)


@dataclass(frozen=True)
class _Rules:
    grid: int
    size_arcsec: float
    every: SkyCoords
    every_visit: npt.NDArray[np.str_]
    stars: SkyCoords
    star_g: npt.NDArray[np.float64]


def _filter_visit(
    visit: str, scores: dict[str, float], rules: _Rules
) -> tuple[dict[str, float], dict[str, int]]:
    keys = sorted(scores)
    p = np.array([parse_key(k) for k in keys])
    c = SkyCoord(p[:, 2], p[:, 3], unit="deg")
    grid = on_grid(p[:, 0].astype(np.int64), p[:, 1].astype(np.int64), rules.grid)
    overlap = overlaps_other_visit(
        c,
        np.full(len(keys), visit),
        every=rules.every,
        every_visit=rules.every_visit,
        size_arcsec=rules.size_arcsec,
    )
    star = near_bright_star(
        c, stars=rules.stars, star_g=rules.star_g, size_arcsec=rules.size_arcsec
    )
    keep = grid & ~overlap & ~star
    counts = {
        "off_grid": int((~grid).sum()),
        "overlap": int((grid & overlap).sum()),
        "star": int((grid & ~overlap & star).sum()),
    }
    return {k: scores[k] for k, ok in zip(keys, keep, strict=True) if ok}, counts


def _kept_tiles(
    split: dict[str, Any], *, stars_json: Path, data: Path, results: Path
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Apply the grid, overlap and star rules to each search and calibration visit.

    Returns:
        The kept scores per visit, and how many tiles each rule dropped.
    """
    every_visit = [*split["search"], *split["bank"], *split["calibration"]]
    mosaic = next((data / every_visit[0] / "mosaics").glob("*.json"))
    scale = float(json.loads(mosaic.read_text(encoding="utf-8"))["pixel_scale_arcsec"])
    every, every_name = _sky_of_every_tile(data, every_visit)
    stars, star_g = read_stars(stars_json, every)
    rules = _Rules(
        grid=int(split["tile_grid_px"]),
        size_arcsec=int(split["tile_grid_px"]) * scale,
        every=every,
        every_visit=every_name,
        stars=stars,
        star_g=star_g,
    )
    kept: dict[str, dict[str, float]] = {}
    dropped = {"off_grid": 0, "overlap": 0, "star": 0}
    for v in [*split["search"], *split["calibration"]]:
        scores = _read_scores(results / v / "measurements_final.csv")
        kept[v], counts = _filter_visit(v, scores, rules)
        for rule, n in counts.items():
            dropped[rule] += n
    return kept, dropped


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

    search = {k: x for v in split["search"] for k, x in kept[v].items()}
    cal = np.array([x for v in split["calibration"] for x in kept[v].values()])
    keys = sorted(search)
    if not keys:
        msg = "Every search tile was dropped by the grid, overlap or star rules."
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
