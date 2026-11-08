"""Writes a vetting list of a survey's oddest sources, one sheet each.

    python -m jwstdetector.vet SPLIT STARS DATA RESULTS OUT [--top 50]

Every scored tile counts, calibration as well as search, since nothing here is a
statistical claim. Tiles go through the survey's overlap and star rules and onto
their visit's median and MAD, and each source is listed once, at the tile where it
scores highest. The hottest patch of each tile is checked in every filter of its
cutout, and a source that shows in one filter only, or in none, is flagged and
listed after the rest, as those are mostly detector artefacts and trails. Each
source gets the nearest SIMBAD and NED entries and a sheet with the tile and every
filter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from jwstdetector.catalogues import fetch, nearest
from jwstdetector.survey import kept_tiles, parse_key, standardise

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwstdetector.utils import SkyCoords

# The hot patch shows in no filter below SHOWS MADs. It shows in one filter only if
# that filter is RATIO times the next and the next is under QUIET, since a peak over
# a patch reaches 3 to 5 MADs on noise alone, and bright sources differ by more than
# RATIO between filters just from depth. Picked on COSMOS-Web's trial list of 50.
SHOWS = 5.0
QUIET = 8.0
RATIO = 1.6
# Overlapping tiles see the same source, so hot spots this close are one source.
SAME_SOURCE_ARCSEC = 1.5
MATCH_ARCSEC = 2.0


def _mad(x: npt.NDArray[np.float64]) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) or 1.0


def _edges(n: int, grid: int) -> npt.NDArray[np.int64]:
    return np.linspace(0, n, grid + 1).astype(np.int64)


def significance(
    plane: npt.NDArray[np.float64], cell: tuple[int, int], grid: int
) -> float:
    """How far one filter stands off its tile at a patch, in MADs, bright or dark.

    `plane` is in FITS order (row 0 at the bottom) and `cell` is the patch's row and
    column in the tile as scored, which is flipped top to bottom. Two measures count,
    the patch's mean against the tile's other patch means (for anything spread out)
    and the 3x3-smoothed peak against the smoothed tile (for anything compact).

    Returns:
        The larger of the two.
    """
    plane = np.nan_to_num(plane, nan=float(np.nanmedian(plane)))
    ys, xs = _edges(plane.shape[0], grid), _edges(plane.shape[1], grid)
    means = np.array([
        [plane[ys[i] : ys[i + 1], xs[j] : xs[j + 1]].mean() for j in range(grid)]
        for i in range(grid)
    ])
    row, col = grid - 1 - cell[0], cell[1]
    mean_sig = abs(means[row, col] - np.median(means)) / _mad(means)
    smooth = uniform_filter(plane, size=3)
    # One patch either side, as the hottest patch can sit just off the source.
    near = smooth[
        ys[max(row - 1, 0)] : ys[min(row + 2, grid)],
        xs[max(col - 1, 0)] : xs[min(col + 2, grid)],
    ]
    peak_sig = float(np.max(np.abs(near - np.median(smooth)))) / _mad(smooth)
    return max(float(mean_sig), peak_sig)


def flag(sigs: list[float]) -> str:
    """Name what's odd about where the hot patch shows, if anything.

    Returns:
        "one filter", "no filter" or "".
    """
    top, second = sorted(sigs, reverse=True)[:2]
    if top < SHOWS:
        return "no filter"
    if second < QUIET and top >= RATIO * second:
        return "one filter"
    return ""


@dataclass
class Source:
    """One source on the list, at the tile where it scores highest."""

    rank: int
    visit: str
    key: str
    z: float
    hot: SkyCoords
    cell: tuple[int, int]
    sigs: list[float] = field(default_factory=list)
    flag: str = ""
    near: dict[str, str] = field(default_factory=dict)


def _mosaic_record(data: Path, visit: str) -> dict[str, Any]:
    record = next((data / visit / "mosaics").glob("*.json"))
    meta: dict[str, Any] = json.loads(record.read_text(encoding="utf-8"))
    return meta


def _hot_spot(
    results: Path, data: Path, *, visit: str, key: str
) -> tuple[tuple[int, int], SkyCoords, int]:
    path = (
        results / visit / "cross" / "anomaly_maps" / "seed=0" / f"{Path(key).stem}.npy"
    )
    scores = np.load(path)
    gi, gj = (int(i) for i in np.unravel_index(np.argmax(scores), scores.shape))
    cx, cy, _, _ = parse_key(key)
    meta = _mosaic_record(data, visit)
    size = int(meta.get("tile_px", 320))
    step = size / scores.shape[0]
    # The tile as scored is flipped top to bottom against the mosaic.
    hot = WCS(meta["wcs"]).pixel_to_world(
        cx - size / 2 + (gj + 0.5) * step - 0.5,
        cy + size / 2 - (gi + 0.5) * step - 0.5,
    )
    return (gi, gj), hot, scores.shape[0]


def _cutouts(data: Path, visits: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for v in visits:
        with (data / v / "manifest.csv").open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("cutout"):
                    out[r["key"]] = data / v / r["cutout"]
    return out


def _planes(path: Path) -> npt.NDArray[np.float64]:
    planes: npt.NDArray[np.float64] = np.asarray(fits.getdata(path), dtype=np.float64)
    return planes


def catalogue_note(hot: SkyCoords) -> dict[str, str]:
    """Look up what SIMBAD and NED have near a source.

    Returns:
        The nearest entry within MATCH_ARCSEC in each catalogue that has one.
    """
    note: dict[str, str] = {}
    at = SkyCoord([hot.ra.deg], [hot.dec.deg], unit="deg")
    for cat in ("SIMBAD", "NED"):
        got = fetch(cat, [(hot.ra.deg, hot.dec.deg)], MATCH_ARCSEC / 60)
        if not got["ok"]:
            note[cat] = f"didn't answer: {got['error'][:80]}"
            continue
        idx, sep = nearest(at, got["rows"])
        if idx[0] < 0 or sep[0] > MATCH_ARCSEC:
            note[cat] = f'nothing within {MATCH_ARCSEC:g}"'
            continue
        name, _, _, kind, extra = got["rows"][int(idx[0])]
        detail = ", ".join(x for x in (kind, extra) if x)
        note[cat] = f'{name} ({detail or "?"}) at {sep[0]:.1f}"'
    return note


def _stretch(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
    med, spread = float(np.median(a)), _mad(a)
    top = max(float(np.percentile(a, 99.7)) - med, 10 * spread)
    out: npt.NDArray[np.float64] = np.clip(
        np.arcsinh((a - med) / spread) / np.arcsinh(top / spread), 0, 1
    )
    return out


def _sheet(
    src: Source,
    *,
    planes: npt.NDArray[np.float64],
    tile_png: Path,
    filters: list[str],
    out: Path,
) -> None:
    import matplotlib as mpl  # noqa: PLC0415 matplotlib is slow to load

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415 matplotlib is slow to load

    n = len(filters)
    fig, ax = plt.subplots(1, n + 2, figsize=(4 * (n + 2), 4.6), dpi=80)
    step = planes.shape[1] / 28
    x, y = (src.cell[1] + 0.5) * step, (src.cell[0] + 0.5) * step
    chans = [_stretch(p)[::-1] for p in planes]
    ax[0].imshow(np.asarray(Image.open(tile_png)))
    ax[0].set_title("tile as scored")
    for a, c, f, s in zip(ax[1 : n + 1], chans, filters, src.sigs, strict=True):
        a.imshow(c, cmap="gray")
        a.set_title(f"{f}, {s:.1f} MAD at the hot patch")
    for a in ax[: n + 1]:
        a.plot(x, y, "o", mfc="none", mec="cyan", ms=18)
    half = max(int(3 * step), 24)
    lo_y, lo_x = max(int(y) - half, 0), max(int(x) - half, 0)
    ax[-1].imshow(np.dstack(chans[:3])[lo_y : int(y) + half, lo_x : int(x) + half])
    ax[-1].set_title("hot spot, colour")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(
        f"#{src.rank}  {src.visit}  z {src.z:.2f}  {src.flag or 'no flag'}  "
        f"hot spot {src.hot.ra.deg:.5f} {src.hot.dec.deg:+.5f}"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _rank(args: argparse.Namespace) -> tuple[list[Source], list[Source]]:
    split = json.loads(args.split.read_text(encoding="utf-8"))
    kept, _ = kept_tiles(
        split, stars_json=args.stars, data=args.data, results=args.results
    )
    z = standardise(kept)
    cutouts = _cutouts(args.data, list(z))
    clean: list[Source] = []
    flagged: list[Source] = []
    taken: list[SkyCoords] = []
    for score, visit, key in sorted(
        ((s, v, k) for v in z for k, s in z[v].items()), reverse=True
    ):
        if len(clean) >= args.top:
            break
        cell, hot, grid = _hot_spot(args.results, args.data, visit=visit, key=key)
        if any(hot.separation(t).arcsec < SAME_SOURCE_ARCSEC for t in taken):
            continue
        taken.append(hot)
        src = Source(len(taken), visit, key, score, hot, cell)
        src.sigs = [significance(p, cell, grid) for p in _planes(cutouts[key])]
        src.flag = flag(src.sigs)
        (flagged if src.flag else clean).append(src)
    return clean, flagged


def _write(args: argparse.Namespace, sources: list[Source], n_clean: int) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    filters = list(_mosaic_record(args.data, sources[0].visit)["filters"])
    cutouts = _cutouts(args.data, sorted({s.visit for s in sources}))
    rows = []
    for src in sources:
        src.near = {} if args.no_catalogues else catalogue_note(src.hot)
        sheet = f"source{src.rank:03d}.png"
        _sheet(
            src,
            planes=_planes(cutouts[src.key]),
            tile_png=args.data / src.visit / "query" / src.key,
            filters=filters,
            out=args.out / sheet,
        )
        mads = {f"{f}_mad": round(s, 1) for f, s in zip(filters, src.sigs, strict=True)}
        rows.append({
            "rank": src.rank,
            "visit": src.visit,
            "key": src.key,
            "z": round(src.z, 3),
            "flag": src.flag,
            "ra": round(src.hot.ra.deg, 6),
            "dec": round(src.hot.dec.deg, 6),
            **mads,
            **{c.lower(): n for c, n in src.near.items()},
            "sheet": sheet,
        })
    with (args.out / "sources.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    body = "".join(
        f"<h3>#{s.rank} {html.escape(s.visit)}, z {s.z:.2f}"
        f"{', flagged: ' + s.flag if s.flag else ''}</h3>"
        f"<p>{'<br>'.join(html.escape(f'{c}: {n}') for c, n in s.near.items())}</p>"
        f"<img src='source{s.rank:03d}.png' width='100%'>"
        for s in sources
    )
    (args.out / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Vetting List</title>"
        "<body style='font-family:sans-serif;background:#111;color:#ddd;margin:16px'>"
        f"<p>{n_clean} sources, then {len(sources) - n_clean} flagged ones.</p>"
        f"{body}</body>",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m jwstdetector.vet",
        description=(__doc__ or "").split("\n\n", maxsplit=1)[0],
    )
    p.add_argument("split", type=Path)
    p.add_argument("stars", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--top", type=int, default=50, help="unflagged sources (default 50)")
    p.add_argument("--no_catalogues", action="store_true", help="skip SIMBAD and NED")
    return p


def main(argv: list[str] | None = None) -> int:
    """Write OUT/index.html, OUT/sources.csv and one sheet per source.

    Returns:
        0.
    """
    args = _parser().parse_args(argv)
    clean, flagged = _rank(args)
    _write(args, [*clean, *flagged], len(clean))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
