"""Cross-matches each tile's most anomalous patch with SIMBAD, NED, Gaia DR3 and local catalogues.

    jwstdetector match results_smacs datasets/smacs_nircam
    jwstdetector match results_smacs datasets/smacs_nircam --catalogue jades.fits --radius 1

Every catalogued source over each mosaic is fetched once, in cones a few arcminutes
across, and kept in RUN/catalogues/ so later runs and the viewer work offline. Each
tile then gets the sky position of its most anomalous patch, the nearest source in
every catalogue, and whether anything lies within --radius of it. Uncatalogued
means none of these catalogues has a source there. It doesn't mean nobody has seen
the object, as plenty of JWST catalogues aren't in SIMBAD or NED yet, which is what
--catalogue is for.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CONE = "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra:.7f}, {dec:.7f}, {r:.7f})) = 1"

# Each service lists its mirrors in order, and the columns that become name, type and
# a short note. SIMBAD's home at CDS goes down now and then, and Harvard mirrors it.
SERVICES = {
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
        "adql": "SELECT prefname, ra, dec, prefphytype, z FROM NEDTAP.objdir WHERE {cone}",
        "columns": ("prefname", "prefphytype", "z"),
    },
    "Gaia DR3": {
        "urls": ["https://gea.esac.esa.int/tap-server/tap"],
        "adql": "SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE {cone}",
        "columns": ("source_id", None, "phot_g_mean_mag"),
    },
}

RA_NAMES = ("ra", "ra_icrs", "raj2000", "ra_deg", "alpha_j2000", "alpha", "ra_j2000")
DEC_NAMES = ("dec", "de_icrs", "dej2000", "dec_deg", "delta_j2000", "delta", "dec_j2000", "de")
ID_NAMES = ("id", "name", "source_id", "number", "objid", "object")


def _note(cat: str, value) -> str:
    if value is None or np.ma.is_masked(value) or (isinstance(value, float) and math.isnan(value)):
        return ""
    if cat == "NED":
        return f"z = {float(value):.4g}"
    if cat == "Gaia DR3":
        return f"G = {float(value):.2f}"
    return str(value)


def _text(value) -> str:
    if value is None or np.ma.is_masked(value):
        return ""
    return value.decode() if isinstance(value, bytes) else str(value).strip()


def cone_centres(
    meta: dict, radius_arcmin: float, pad_arcsec: float = 0.0
) -> list[tuple[float, float]]:
    """Centres of cones of `radius_arcmin` that between them cover the whole mosaic, and
    everything within `pad_arcsec` of it."""
    from jwstdetector.bundle import mosaic_wcs

    h, w = meta["shape"]
    radius_px = (radius_arcmin * 60 - pad_arcsec) / meta["pixel_scale_arcsec"]
    # A square inscribed in each cone shrunk by the pad, so neighbouring cones leave no
    # gaps and a source just outside a square is still inside its cone.
    step = radius_px * math.sqrt(2)
    xs = np.arange(step / 2, w + step / 2, step)
    ys = np.arange(step / 2, h + step / 2, step)
    gx, gy = np.meshgrid(np.minimum(xs, w), np.minimum(ys, h))
    ra, dec = mosaic_wcs(meta).all_pix2world(gx.ravel(), gy.ravel(), 0)
    return list(zip(ra.tolist(), dec.tolist(), strict=True))


def fetch(cat: str, centres: list[tuple[float, float]], radius_arcmin: float) -> dict:
    """Every source of one service over the cones, with the mirror that answered."""
    import pyvo

    spec = SERVICES[cat]
    name_col, type_col, note_col = spec["columns"]
    errors = []
    for url in spec["urls"]:
        rows: dict[str, list] = {}
        try:
            service = pyvo.dal.TAPService(url)
            for ra, dec in centres:
                cone = CONE.format(ra=ra, dec=dec, r=radius_arcmin / 60)
                result = service.run_sync(spec["adql"].format(cone=cone), maxrec=2_000_000)
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
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:200]}")
            logger.warning("%s didn't answer at %s, %s", cat, url, type(exc).__name__)
            continue
        return {"ok": True, "server": url, "rows": list(rows.values())}
    return {"ok": False, "error": " | ".join(errors), "rows": []}


def read_local(path: Path) -> dict:
    """A catalogue file astropy can read, with RA and Dec in degrees under any usual name."""
    from astropy.table import Table

    table = Table.read(path)
    lower = {c.lower(): c for c in table.colnames}
    ra = next((lower[n] for n in RA_NAMES if n in lower), None)
    dec = next((lower[n] for n in DEC_NAMES if n in lower), None)
    if ra is None or dec is None:
        raise SystemExit(
            f"Couldn't find RA and Dec columns in {path}. It has {', '.join(table.colnames[:12])}."
        )
    ident = next((lower[n] for n in ID_NAMES if n in lower), None)
    rows = []
    for i, row in enumerate(table):
        name = _text(row[ident]) if ident else str(i + 1)
        rows.append([f"{path.stem} {name}", float(row[ra]), float(row[dec]), "", ""])
    return {"ok": True, "server": str(path), "rows": rows}


def on_mosaic(meta: dict, entry: dict, margin_px: float = 100) -> dict:
    """The catalogue's rows that fall on the mosaic, give or take `margin_px`."""
    from jwstdetector.bundle import mosaic_wcs

    rows = entry["rows"]
    if not rows:
        return entry
    x, y = mosaic_wcs(meta).all_world2pix([r[1] for r in rows], [r[2] for r in rows], 0)
    h, w = meta["shape"]
    keep = (x > -margin_px) & (x < w + margin_px) & (y > -margin_px) & (y < h + margin_px)
    return {**entry, "rows": [r for r, k in zip(rows, keep, strict=True) if k]}


def nearest(peaks, rows: list[list]) -> tuple[np.ndarray, np.ndarray]:
    """For each peak, the index of the closest row and its separation in arcseconds."""
    from astropy.coordinates import SkyCoord

    if not rows:
        return np.full(len(peaks), -1), np.full(len(peaks), np.inf)
    arr = np.array([[r[1], r[2]] for r in rows])
    cat = SkyCoord(arr[:, 0], arr[:, 1], unit="deg")
    idx, sep, _ = peaks.match_to_catalog_sky(cat)
    return np.asarray(idx), sep.arcsec


def crossmatch(
    run_dir: Path,
    data_root: Path,
    *,
    radius: float = 1.5,
    services: list[str] | None = None,
    local: list[Path] | None = None,
    cone_arcmin: float = 3.0,
    refresh: bool = False,
) -> dict:
    """Writes RUN/crossmatch.json and returns it."""
    from astropy.coordinates import SkyCoord

    from jwstdetector.bundle import RunBundle

    bundle = RunBundle(run_dir, data_root, writable=False)
    services = list(SERVICES) if services is None else services
    out_dir = Path(run_dir) / "catalogues"
    out_dir.mkdir(exist_ok=True)

    tiles: dict[str, dict] = {}
    checked_everywhere: set[str] | None = None
    failed: dict[str, str] = {}
    for mosaic_id, meta in bundle.index["mosaics"].items():
        entries = [e for e in bundle.index["tiles"] if e["mosaic"] == mosaic_id]
        if "wcs" not in meta:
            logger.warning(
                "%s has no WCS, so its %d tiles can't be matched", mosaic_id, len(entries)
            )
            continue
        cache = out_dir / f"{mosaic_id}.json"
        record = json.loads(cache.read_text("utf-8")) if cache.is_file() else {}
        cats = record.get("catalogues", {})
        centres = cone_centres(meta, cone_arcmin, pad_arcsec=radius)
        for cat in services:
            kept = cats.get(cat, {})
            if kept.get("ok") and not refresh:
                continue
            logger.info("Fetching %s over %s in %d cones", cat, mosaic_id, len(centres))
            got = fetch(cat, centres, cone_arcmin)
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
        for path in local or []:
            cats[Path(path).stem] = on_mosaic(meta, read_local(Path(path)))
        cache.write_text(json.dumps({"catalogues": cats}), encoding="utf-8")

        # The cache keeps whatever earlier runs fetched, but only this run's are used.
        used = {c: cats[c] for c in [*services, *(Path(p).stem for p in local or [])]}
        ok = {c for c, v in used.items() if v["ok"]}
        checked_everywhere = ok if checked_everywhere is None else checked_everywhere & ok
        failed.update({c: v["error"] for c, v in used.items() if not v["ok"]})
        if not ok:
            logger.warning("No catalogue answered for %s, so its tiles aren't matched", mosaic_id)
            continue
        peaks = SkyCoord(
            [e["peak_sky"][0] for e in entries], [e["peak_sky"][1] for e in entries], unit="deg"
        )
        found = {c: nearest(peaks, cats[c]["rows"]) for c in sorted(ok)}
        for i, e in enumerate(entries):
            near = {}
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

    result = {
        "radius_arcsec": radius,
        "checked": sorted(checked_everywhere or []),
        "failed": failed,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tiles": tiles,
    }
    (Path(run_dir) / "crossmatch.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path, help="A `jwstdetector run` output folder.")
    ap.add_argument("data_root", type=Path, help="The `jwstdetector prep` folder it scored.")
    ap.add_argument(
        "--radius",
        type=float,
        default=1.5,
        help="Arcseconds from the anomalous patch within which a source counts as a match.",
    )
    ap.add_argument(
        "--services",
        nargs="*",
        default=list(SERVICES),
        choices=list(SERVICES),
        help="Online catalogues to ask. Pass none of them to work from --catalogue alone.",
    )
    ap.add_argument(
        "--catalogue",
        type=Path,
        action="append",
        default=[],
        help="A local catalogue (FITS, CSV, ECSV or VOTable) with RA and Dec in degrees. "
        "Can be given more than once.",
    )
    ap.add_argument(
        "--cone", type=float, default=3.0, help="Radius in arcminutes of each query cone."
    )
    ap.add_argument(
        "--refresh", action="store_true", help="Fetch again instead of reusing RUN/catalogues/."
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
