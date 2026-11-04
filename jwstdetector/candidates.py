"""Writes a run's candidates as a table for TOPCAT or astropy, and a DS9 region file.

    jwstdetector candidates results_smacs datasets/smacs_nircam -o candidates.fits
    jwstdetector candidates results_smacs datasets/smacs_nircam -o new.vot --top 50 --uncatalogued

The format follows the file name, so .fits, .vot, .csv and .ecsv all work. Each row
is a tile, with the sky position of its most anomalous patch, its scores, the nearest
source in every catalogue `jwstdetector match` checked, and your verdict and note.
A .reg file with a circle and the rank at every peak is written beside it.
"""  # noqa: E501 --help prints this, so the usage line stays whole

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astropy.table import Table
import numpy as np

from jwstdetector.bundle import RunBundle, read_verdicts

if TYPE_CHECKING:
    from jwstdetector.utils import AstropyTable

logger = logging.getLogger(__name__)

FORMATS = {
    ".fits": "fits",
    ".vot": "votable",
    ".xml": "votable",
    ".csv": "ascii.csv",
    ".ecsv": "ascii.ecsv",
}


def _slug(cat: str) -> str:
    return cat.lower().replace(" ", "_")


def _empty_columns(checked: list[str]) -> dict[str, list[Any]]:
    cols: dict[str, list[Any]] = {
        "rank": [],
        "tile": [],
        "mosaic": [],
        "score": [],
        "score_pass1": [],
        "ra_peak": [],
        "dec_peak": [],
        "ra_tile": [],
        "dec_tile": [],
        "uncatalogued": [],
    }
    for cat in checked:
        for field in ("name", "type", "sep"):
            cols[f"{_slug(cat)}_{field}"] = []
    cols.update({"verdict": [], "starred": [], "note": []})
    return cols


def _append_tile(
    cols: dict[str, list[Any]],
    e: dict[str, Any],
    *,
    checked: list[str],
    verdict: dict[str, Any],
) -> None:
    cols["rank"].append(e["rank"])
    cols["tile"].append(e["key"])
    cols["mosaic"].append(e["mosaic"])
    cols["score"].append(np.nan if e["score"] is None else e["score"])
    cols["score_pass1"].append(e.get("score_pass1", np.nan))
    peak = e.get("peak_sky") or [np.nan, np.nan]
    cols["ra_peak"].append(peak[0])
    cols["dec_peak"].append(peak[1])
    cols["ra_tile"].append(np.nan if e["ra"] is None else e["ra"])
    cols["dec_tile"].append(np.nan if e["dec"] is None else e["dec"])
    xm = e.get("xmatch", {})
    cols["uncatalogued"].append(bool(xm.get("uncatalogued", False)))
    for cat in checked:
        slug = _slug(cat)
        near = xm.get("nearest", {}).get(cat)
        cols[f"{slug}_name"].append(near["name"] if near else "")
        cols[f"{slug}_type"].append(near["type"] if near else "")
        cols[f"{slug}_sep"].append(near["sep"] if near else np.nan)
    cols["verdict"].append(verdict.get("verdict") or "")
    cols["starred"].append(bool(verdict.get("star", False)))
    cols["note"].append(verdict.get("note", ""))


def candidate_table(
    bundle: RunBundle, top: int | None = None, *, uncatalogued: bool = False
) -> AstropyTable:
    """Build the candidates table, one row per tile in rank order.

    Returns:
        An astropy Table with units on the sky positions and separations.
    """
    verdicts = read_verdicts(bundle.run_dir)
    index: dict[str, Any] = bundle.index
    run = index["run"]
    checked = (run.get("xmatch") or {}).get("checked", [])
    entries = [
        e
        for e in index["tiles"]
        if (top is None or e["rank"] <= top)
        and (not uncatalogued or e.get("xmatch", {}).get("uncatalogued"))
    ]
    cols = _empty_columns(checked)
    for e in entries:
        _append_tile(cols, e, checked=checked, verdict=verdicts.get(e["key"], {}))

    table: AstropyTable = Table(cols)
    for name in ("ra_peak", "dec_peak", "ra_tile", "dec_tile"):
        table[name].unit = "deg"
    for cat in checked:
        table[f"{_slug(cat)}_sep"].unit = "arcsec"
    table.meta["RUN"] = run["name"]
    table.meta["MODEL"] = run["model"] or ""
    table.meta["DETECTOR"] = run["detector"]
    if run.get("xmatch"):
        table.meta["XRADIUS"] = run["xmatch"]["radius_arcsec"]
    return table


def write_regions(table: AstropyTable, path: Path, radius_arcsec: float = 1.5) -> None:
    """Write a DS9 region file with a circle and the rank at every tile's peak."""
    lines = ["# Region file format: DS9 version 4.1", "fk5"]
    for row in table:
        if np.isfinite(row["ra_peak"]):
            colour = "cyan" if row["uncatalogued"] else "orange"
            lines.append(
                f'circle({row["ra_peak"]:.7f},{row["dec_peak"]:.7f},{radius_arcsec}") '
                f"# color={colour} text={{{row['rank']}}}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Write the candidates table and its region file.

    Returns:
        The exit code, which is always 0.

    Raises:
        SystemExit: If the output's extension isn't one of FORMATS.
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="A `jwstdetector run` output folder.")
    ap.add_argument(
        "data_root", type=Path, help="The `jwstdetector prep` folder it scored."
    )
    ap.add_argument(
        "-o", "--out", type=Path, default=None, help="RUN/candidates.fits by default."
    )
    ap.add_argument("--top", type=int, default=None, help="Only the top N by rank.")
    ap.add_argument(
        "--uncatalogued",
        action="store_true",
        help="Only tiles `jwstdetector match` found nothing near.",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    out = args.out or args.run_dir / "candidates.fits"
    fmt = FORMATS.get(out.suffix.lower())
    if fmt is None:
        msg = f"Can't tell the format of {out.name}. Use {', '.join(FORMATS)}."
        raise SystemExit(msg)
    bundle = RunBundle(args.run_dir, args.data_root, writable=False)
    table = candidate_table(bundle, top=args.top, uncatalogued=args.uncatalogued)
    table.write(out, format=fmt, overwrite=True)
    index: dict[str, Any] = bundle.index
    radius = (index["run"].get("xmatch") or {}).get("radius_arcsec", 1.5)
    write_regions(table, out.with_suffix(".reg"), radius)
    logger.info(
        "Wrote %d candidates to %s and %s",
        len(table),
        out,
        out.with_suffix(".reg").name,
    )
    return 0
