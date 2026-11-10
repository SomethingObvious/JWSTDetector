"""Cross-matches each tile's most anomalous patch with SIMBAD, NED, Gaia DR3 and local catalogues.

    jwst-peculiar match results_smacs datasets/smacs_nircam
    jwst-peculiar match results_smacs datasets/smacs_nircam --catalogue jades.fits --radius 1

Every catalogued source over each mosaic is fetched once, in cones a few arcminutes
across, and kept in RUN/catalogues/ so later runs and the viewer work offline. Each
tile then gets the sky position of its most anomalous patch, the nearest source in
every catalogue, and whether anything lies within --radius of it. Uncatalogued
means none of these catalogues has a source there. It doesn't mean nobody has seen
the object, as plenty of JWST catalogues aren't in SIMBAD or NED yet, which is what
--catalogue is for.
"""  # noqa: E501 --help prints this, so the summary and usage lines stay whole

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from astropy.coordinates import SkyCoord
from astropy.table import Table
import numpy as np
import pyvo

from jwst_peculiar.bundle import RunBundle, mosaic_wcs

if TYPE_CHECKING:
    import numpy.typing as npt

    from jwst_peculiar.utils import SkyCoords, TableCell

logger = logging.getLogger(__name__)

CONE = (
    "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f})) = 1"
)


class _Service(TypedDict):
    urls: list[str]
    adql: str
    columns: tuple[str, str | None, str | None]


# Each service lists its mirrors in order, and the columns that become name, type and
# a short note. SIMBAD's home at CDS goes down now and then, and Harvard mirrors it.
SERVICES: dict[str, _Service] = {
    "SIMBAD": {
        "urls": [
            "https://simbad.cds.unistra.fr/simbad/sim-tap",
            "https://simbad.harvard.edu/simbad/sim-tap",
        ],
        "adql": "SELECT main_id, ra, dec, otype FROM basic WHERE {cone}",
        "columns": ("main_id", "otype", None),
    },
    "NED": {
        "urls": ["https://ned.ipac.caltech.edu/tap"],
        "adql": "SELECT prefname, ra, dec, prefphytype, z "
        "FROM NEDTAP.objdir WHERE {cone}",
        "columns": ("prefname", "prefphytype", "z"),
    },
    "Gaia DR3": {
        "urls": ["https://gea.esac.esa.int/tap-server/tap"],
        "adql": "SELECT source_id, ra, dec, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source WHERE {cone}",
        "columns": ("source_id", None, "phot_g_mean_mag"),
    },
}

RA_NAMES = ("ra", "ra_icrs", "raj2000", "ra_deg", "alpha_j2000", "alpha", "ra_j2000")
DEC_NAMES = (
    "dec",
    "de_icrs",
    "dej2000",
    "dec_deg",
    "delta_j2000",
    "delta",
    "dec_j2000",
    "de",
)
ID_NAMES = ("id", "name", "source_id", "number", "objid", "object")


def _note(cat: str, value: TableCell) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    if cat not in {"NED", "Gaia DR3"}:
        return str(value)
    # Through float, as a numpy float32 isn't a float and a null comes back as NaN.
    number = float(value)
    if math.isnan(number):
        return ""
    return f"z = {number:.4g}" if cat == "NED" else f"G = {number:.2f}"


def _text(value: TableCell) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    return value.decode() if isinstance(value, bytes) else str(value).strip()


def cone_centres(
    meta: dict[str, Any], radius_arcmin: float, pad_arcsec: float = 0.0
) -> list[tuple[float, float]]:
    """Place cones of `radius_arcmin` that between them cover the whole mosaic.

    Returns:
        The cone centres as (RA, Dec) in degrees, covering everything within
        `pad_arcsec` of the mosaic as well.
    """
    h, w = meta["shape"]
    radius_px = (radius_arcmin * 60 - pad_arcsec) / meta["pixel_scale_arcsec"]
    # A square inscribed in each cone shrunk by the pad, so neighbouring cones leave no
    # gaps and a source just outside a square is still inside its cone.
    step = radius_px * math.sqrt(2)
    xs = np.arange(step / 2, w + step / 2, step)
    ys = np.arange(step / 2, h + step / 2, step)
    gx, gy = np.meshgrid(np.minimum(xs, w), np.minimum(ys, h))
    wcs: Any = mosaic_wcs(meta)
    ra, dec = wcs.all_pix2world(gx.ravel(), gy.ravel(), 0)
    return list(zip(ra.tolist(), dec.tolist(), strict=True))


def _query(
    cat: str, url: str, centres: list[tuple[float, float]], *, radius_arcmin: float
) -> list[list[Any]]:
    spec = SERVICES[cat]
    name_col, type_col, note_col = spec["columns"]
    rows: dict[str, list[Any]] = {}
    service = pyvo.dal.TAPService(url)
    for ra, dec in centres:
        cone = CONE.format(ra=ra, dec=dec, r=radius_arcmin / 60)
        result = service.run_sync(spec["adql"].format(cone=cone), maxrec=2_000_000)
        if result.status[0] == "OVERFLOW":
            # A cut-off answer would read as sky with nothing catalogued in it.
            msg = f"{cat} cut its answer short at {len(result)} rows. "
            msg += "Try a smaller --cone."
            raise RuntimeError(msg)
        for row in result.to_table():
            name = _text(row[name_col])
            if cat == "Gaia DR3":
                name = f"Gaia DR3 {name}"
            rows[name] = [
                name,
                float(row["ra"]),
                float(row["dec"]),
                _text(row[type_col]) if type_col else "",
                _note(cat, row[note_col]) if note_col else "",
            ]
    return list(rows.values())


def fetch(
    cat: str, centres: list[tuple[float, float]], radius_arcmin: float
) -> dict[str, Any]:
    """Fetch every source of one service over the cones, trying each mirror in turn.

    Returns:
        The rows and the mirror that answered, or the errors when none did.
    """
    errors: list[str] = []
    for url in SERVICES[cat]["urls"]:
        try:
            rows = _query(cat, url, centres, radius_arcmin=radius_arcmin)
        except Exception as exc:  # noqa: BLE001 any failure moves on to the next mirror
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:200]}")
            logger.warning("%s didn't answer at %s, %s", cat, url, type(exc).__name__)
            continue
        return {"ok": True, "server": url, "rows": rows}
    return {"ok": False, "error": " | ".join(errors), "rows": []}


def read_local(path: Path) -> dict[str, Any]:
    """Read any catalogue astropy can, with RA and Dec in degrees under the usual names.

    Returns:
        The catalogue's rows, in the same form `fetch` returns them.

    Raises:
        SystemExit: If no column looks like RA or Dec.
    """
    table: Any = Table.read(path)
    lower = {c.lower(): c for c in table.colnames}
    ra = next((lower[n] for n in RA_NAMES if n in lower), None)
    dec = next((lower[n] for n in DEC_NAMES if n in lower), None)
    if ra is None or dec is None:
        msg = (
            f"Couldn't find RA and Dec columns in {path}. "
            f"It has {', '.join(table.colnames[:12])}."
        )
        raise SystemExit(msg)
    ident = next((lower[n] for n in ID_NAMES if n in lower), None)
    rows: list[list[Any]] = []
    for i, row in enumerate(table):
        name = _text(row[ident]) if ident else str(i + 1)
        rows.append([f"{path.stem} {name}", float(row[ra]), float(row[dec]), "", ""])
    return {"ok": True, "server": str(path), "rows": rows}


def on_mosaic(
    meta: dict[str, Any], entry: dict[str, Any], margin_px: float = 100
) -> dict[str, Any]:
    """Keep the catalogue's rows that fall on the mosaic, give or take `margin_px`.

    Returns:
        `entry` with only those rows.
    """
    rows = entry["rows"]
    if not rows:
        return entry
    wcs: Any = mosaic_wcs(meta)
    x, y = wcs.all_world2pix([r[1] for r in rows], [r[2] for r in rows], 0)
    h, w = meta["shape"]
    keep = (
        (x > -margin_px) & (x < w + margin_px) & (y > -margin_px) & (y < h + margin_px)
    )
    return {**entry, "rows": [r for r, k in zip(rows, keep, strict=True) if k]}


def nearest(
    peaks: SkyCoords, rows: list[list[Any]]
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Find the closest catalogue row to each peak.

    Returns:
        Each peak's row index, -1 when there are no rows, and its separation in
        arcseconds.
    """
    if not rows:
        return np.full(len(peaks), -1), np.full(len(peaks), np.inf)
    arr = np.array([[r[1], r[2]] for r in rows])
    cat = SkyCoord(arr[:, 0], arr[:, 1], unit="deg")
    idx, sep, _ = peaks.match_to_catalog_sky(cat)
    return np.asarray(idx), sep.arcsec


@dataclass(frozen=True)
class _Plan:
    radius: float
    services: list[str]
    local: list[Path]
    cone_arcmin: float
    refresh: bool


def _fetch_services(
    cats: dict[str, Any],
    record: dict[str, Any],
    centres: list[tuple[float, float]],
    *,
    mosaic_id: str,
    plan: _Plan,
) -> None:
    for cat in plan.services:
        kept = cats.get(cat, {})
        if kept.get("ok") and not plan.refresh:
            continue
        logger.info("Fetching %s over %s in %d cones", cat, mosaic_id, len(centres))
        got = fetch(cat, centres, plan.cone_arcmin)
        if got["ok"]:
            logger.info("  %d sources from %s", len(got["rows"]), got["server"])
        elif kept.get("ok"):
            # A refresh that fails keeps the copy it was meant to replace.
            logger.warning(
                "%s didn't answer, so the copy from %s stays",
                cat,
                kept.get("fetched", record.get("fetched")),
            )
            continue
        got["fetched"] = datetime.now(UTC).isoformat(timespec="seconds")
        cats[cat] = got


def _update_cache(
    cache: Path, meta: dict[str, Any], *, mosaic_id: str, plan: _Plan
) -> dict[str, Any]:
    record = json.loads(cache.read_text("utf-8")) if cache.is_file() else {}
    cats: dict[str, Any] = record.get("catalogues", {})
    centres = cone_centres(meta, plan.cone_arcmin, pad_arcsec=plan.radius)
    _fetch_services(cats, record, centres, mosaic_id=mosaic_id, plan=plan)
    for path in plan.local:
        cats[Path(path).stem] = on_mosaic(meta, read_local(Path(path)))
    cache.write_text(json.dumps({"catalogues": cats}), encoding="utf-8")
    return cats


def _match_tiles(
    entries: list[dict[str, Any]],
    cats: dict[str, Any],
    *,
    ok: set[str],
    radius: float,
) -> dict[str, dict[str, Any]]:
    peaks = SkyCoord(
        [e["peak_sky"][0] for e in entries],
        [e["peak_sky"][1] for e in entries],
        unit="deg",
    )
    found = {c: nearest(peaks, cats[c]["rows"]) for c in sorted(ok)}
    tiles: dict[str, dict[str, Any]] = {}
    for i, e in enumerate(entries):
        near: dict[str, dict[str, Any]] = {}
        for c, (idx, sep) in found.items():
            if idx[i] >= 0:
                row = cats[c]["rows"][idx[i]]
                near[c] = {
                    "name": row[0],
                    "type": row[3],
                    "note": row[4],
                    "sep": round(float(sep[i]), 3),
                }
        tiles[e["key"]] = {
            "nearest": near,
            "uncatalogued": all(v["sep"] > radius for v in near.values()),
        }
    return tiles


def crossmatch(  # noqa: PLR0913 the CLI and tests pass each option by name
    run_dir: Path,
    data_root: Path,
    *,
    radius: float = 1.5,
    services: list[str] | None = None,
    local: list[Path] | None = None,
    cone_arcmin: float = 3.0,
    refresh: bool = False,
) -> dict[str, Any]:
    """Match every tile's peak against the catalogues and write RUN/crossmatch.json.

    Returns:
        What was written to crossmatch.json.
    """
    bundle = RunBundle(run_dir, data_root, writable=False)
    index: dict[str, Any] = bundle.index
    plan = _Plan(
        radius=radius,
        services=list(SERVICES) if services is None else services,
        local=local or [],
        cone_arcmin=cone_arcmin,
        refresh=refresh,
    )
    out_dir = Path(run_dir) / "catalogues"
    out_dir.mkdir(exist_ok=True)

    tiles: dict[str, dict[str, Any]] = {}
    checked_everywhere: set[str] | None = None
    failed: dict[str, str] = {}
    for mosaic_id, meta in index["mosaics"].items():
        entries = [e for e in index["tiles"] if e["mosaic"] == mosaic_id]
        if "wcs" not in meta:
            logger.warning(
                "%s has no WCS, so its %d tiles can't be matched",
                mosaic_id,
                len(entries),
            )
            continue
        cats = _update_cache(
            out_dir / f"{mosaic_id}.json", meta, mosaic_id=mosaic_id, plan=plan
        )

        # The cache keeps what earlier runs fetched, but only the catalogues asked for
        # here count.
        used = {
            c: cats[c] for c in [*plan.services, *(Path(p).stem for p in plan.local)]
        }
        ok = {c for c, v in used.items() if v["ok"]}
        checked_everywhere = (
            ok if checked_everywhere is None else checked_everywhere & ok
        )
        failed.update({c: v["error"] for c, v in used.items() if not v["ok"]})
        if not ok:
            logger.warning(
                "No catalogue answered for %s, so its tiles aren't matched", mosaic_id
            )
            continue
        tiles.update(_match_tiles(entries, cats, ok=ok, radius=radius))

    result = {
        "radius_arcsec": radius,
        "checked": sorted(checked_everywhere or []),
        "failed": failed,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tiles": tiles,
    }
    (Path(run_dir) / "crossmatch.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="A `jwst-peculiar run` output folder.")
    ap.add_argument(
        "data_root", type=Path, help="The `jwst-peculiar prep` folder it scored."
    )
    ap.add_argument(
        "--radius",
        type=float,
        default=1.5,
        help="Arcseconds from the anomalous patch within which a source counts "
        "as a match.",
    )
    ap.add_argument(
        "--services",
        nargs="*",
        default=list(SERVICES),
        choices=list(SERVICES),
        help="Online catalogues to ask. Pass none of them to work from "
        "--catalogue alone.",
    )
    ap.add_argument(
        "--catalogue",
        type=Path,
        action="append",
        default=[],
        help="A local catalogue (FITS, CSV, ECSV or VOTable) with RA and Dec in "
        "degrees. Can be given more than once.",
    )
    ap.add_argument(
        "--cone",
        type=float,
        default=3.0,
        help="Radius in arcminutes of each query cone.",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch again instead of reusing RUN/catalogues/.",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    """Cross-match a run's candidates and log how many have nothing near them.

    Returns:
        The exit code, which is always 0.

    Raises:
        SystemExit: If --radius is too wide for the cones to cover the mosaic.
    """
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.radius >= args.cone * 30:
        msg = (
            f"A match radius of {args.radius} arcsec is too wide for cones of "
            f"{args.cone} arcmin. Keep --radius under half of --cone."
        )
        raise SystemExit(msg)

    result = crossmatch(
        args.run_dir,
        args.data_root,
        radius=args.radius,
        services=args.services,
        local=args.catalogue,
        cone_arcmin=args.cone,
        refresh=args.refresh,
    )
    n = len(result["tiles"])
    free = sum(t["uncatalogued"] for t in result["tiles"].values())
    for cat, err in result["failed"].items():
        logger.warning("%s couldn't be reached, so it wasn't checked: %s", cat, err)
    logger.info(
        "%d of %d tiles have nothing in %s within %.1f arcsec of their anomalous patch",
        free,
        n,
        ", ".join(result["checked"]) or "any catalogue",
        args.radius,
    )
    return 0
