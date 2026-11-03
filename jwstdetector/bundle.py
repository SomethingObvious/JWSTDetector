"""A run and its tiles laid out the way the web viewer reads them.

The viewer only ever asks for paths like `tiles/000012.png` or `maps/000012.f32`.
`RunBundle.read` answers those straight from the run and dataset folders, which is
what the local server does, and `export` writes every one of them to a folder that
any static host can serve.
"""

from __future__ import annotations

import csv
import html
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from jwstdetector.data import owned_patches, read_manifest, read_measurements
from jwstdetector.detection import Footprint

EMBED_DIMS = 32


def _tile_id(i: int) -> str:
    return f"{i:06d}"


def mosaic_wcs(meta: dict):
    """The celestial WCS mosaics/<id>.json records."""
    from astropy.wcs import WCS

    return WCS(meta["wcs"])


def patch_pixels(entries: list[dict], grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Mosaic pixel coordinates, 0-based, of the centre of each entry's peak patch.

    The patch grid covers the PNG with its top row first, and the PNG's top row is the
    tile's highest FITS row, so rows count down from y0 + size.
    """
    r = np.array([e["peak"][0] for e in entries], dtype=np.float64)
    c = np.array([e["peak"][1] for e in entries], dtype=np.float64)
    size = np.array([e["size"] for e in entries], dtype=np.float64)
    x = np.array([e["x0"] for e in entries]) + (c + 0.5) * size / grid[1] - 0.5
    y = np.array([e["y0"] for e in entries]) + size - (r + 0.5) * size / grid[0] - 0.5
    return x, y


def descriptor_embedding(path: Path, keys: list[str], dims: int = EMBED_DIMS) -> np.ndarray:
    """Each key's descriptor from descriptors.npz, reduced by PCA and scaled to unit length.

    A 1024-dim float per tile is 4 KB, which adds up for a browser, and the top few
    dozen components keep nearly all of what cosine similarity sees.
    """
    with np.load(path) as f:
        lookup = {str(k): i for i, k in enumerate(f["keys"])}
        x = f["vectors"].astype(np.float64)[[lookup[k] for k in keys]]
    x -= x.mean(axis=0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[: min(dims, vt.shape[0])].T
    z /= np.linalg.norm(z, axis=1, keepdims=True).clip(1e-12)
    out = np.zeros((len(keys), dims), dtype=np.float32)
    out[:, : z.shape[1]] = z
    return out


def _map_path(maps_dir: Path, key: str) -> Path:
    # Same rule as detection.score_queries, since tile stems are full of RA and Dec dots.
    return maps_dir / f"{Path(key).with_suffix('')}.npy"


@dataclass
class Tile:
    key: str
    png: Path
    map2: Path
    map1: Path | None
    cutout: Path | None


class RunBundle:
    """Joins a folder from `jwstdetector run` with the dataset `jwstdetector prep` wrote."""

    def __init__(self, run_dir: Path, data_root: Path, *, writable: bool):
        self.run_dir = Path(run_dir)
        self.data_root = Path(data_root)
        args_path = self.run_dir / "args.yaml"
        if not args_path.is_file():
            raise FileNotFoundError(f"{self.run_dir} has no args.yaml, so it isn't a run folder")
        self.args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
        self.query_dir = self.data_root / self.args.get("query_subdir", "query")
        self.writable = writable

        final = self.run_dir / "measurements_final.csv"
        scores = read_measurements(final)
        pass1_csv = self.run_dir / "pass1" / "measurements.csv"
        pass1 = read_measurements(pass1_csv) if pass1_csv.is_file() else {}
        ref_list = self.run_dir / "pass2" / "ref_list.txt"
        refs = set(ref_list.read_text(encoding="utf-8").split()) if ref_list.is_file() else set()
        manifest = read_manifest(self.data_root)
        seed = self.args.get("seed", 0)

        self.tiles: list[Tile] = []
        entries = []
        values = []
        # A tile with nothing to compare against scored NaN. It goes last, as null.
        ranked = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
        owned = {}
        # Runs from before --score_region scored every patch.
        if self.args.get("score_region", "all") == "owned" and ranked:
            first = np.load(
                _map_path(self.run_dir / "pass2" / f"anomaly_maps/seed={seed}", ranked[0])
            )
            footprints = {
                k: Footprint(m["mosaic"], m["x0"], m["y0"], m["size"])
                for k, m in manifest.items()
                if k in scores
            }
            owned = owned_patches(footprints, first.shape)
        for rank, key in enumerate(ranked, start=1):
            meta = manifest.get(key)
            if meta is None:
                raise KeyError(
                    f"{key} is scored in {final} but missing from {self.data_root}/manifest.csv. "
                    "Was the run made from a different dataset?"
                )
            map2 = _map_path(self.run_dir / "pass2" / f"anomaly_maps/seed={seed}", key)
            map1 = _map_path(self.run_dir / "pass1" / f"anomaly_maps/seed={seed}", key)
            cutout = self.data_root / meta["cutout"] if meta.get("cutout") else None
            tile = Tile(key, self.query_dir / key, map2, map1 if map1.is_file() else None, cutout)
            self.tiles.append(tile)

            grid = np.load(map2)
            values.append(grid.ravel())
            mine = owned.get(key)
            scored = np.where(mine, grid, -np.inf) if mine is not None else grid
            peak = np.unravel_index(int(np.argmax(scored)), grid.shape)
            entry = {
                "key": key,
                "rank": rank,
                "score": _finite(scores[key]),
                "in_reference": key in refs,
                "mosaic": meta["mosaic"],
                "x0": meta["x0"],
                "y0": meta["y0"],
                "size": meta["size"],
                "ra": meta["ra"],
                "dec": meta["dec"],
                "peak": [int(peak[0]), int(peak[1])],
                "has_cutout": cutout is not None and cutout.is_file(),
            }
            if _finite(pass1.get(key)) is not None:
                entry["score_pass1"] = pass1[key]
            if mine is not None and not mine.all():
                r, c = np.nonzero(mine)
                # First and last row and column the tile scored, for the plate's outline.
                entry["owned"] = [int(r.min()), int(r.max()), int(c.min()), int(c.max())]
            entries.append(entry)

        if not entries:
            raise ValueError(f"{final} has no tiles in it")
        grid_shape = list(np.load(self.tiles[0].map2).shape)
        # The overlay runs from the run's median patch to its 99.9th percentile, so a
        # colour means the same distance on every tile and quiet tiles stay dark.
        lo, hi = (float(v) for v in np.nanpercentile(np.concatenate(values), [50, 99.9]))
        mosaics = {}
        for mosaic_id in sorted({e["mosaic"] for e in entries}):
            meta = json.loads((self.data_root / "mosaics" / f"{mosaic_id}.json").read_text("utf-8"))
            meta["preview"] = f"mosaics/{mosaic_id}.png"
            mosaics[mosaic_id] = meta

        for mosaic_id, meta in mosaics.items():
            if "wcs" not in meta:
                continue
            mine = [e for e in entries if e["mosaic"] == mosaic_id]
            x, y = patch_pixels(mine, grid_shape)
            ra, dec = mosaic_wcs(meta).all_pix2world(x, y, 0)
            for e, a, d in zip(mine, ra.tolist(), dec.tolist(), strict=True):
                e["peak_sky"] = [round(a, 7), round(d, 7)]

        xmatch = None
        xmatch_path = self.run_dir / "crossmatch.json"
        if xmatch_path.is_file():
            xmatch = json.loads(xmatch_path.read_text(encoding="utf-8"))
            for e in entries:
                if e["key"] in xmatch["tiles"]:
                    e["xmatch"] = xmatch["tiles"][e["key"]]

        self.embedding = None
        desc_path = self.run_dir / "pass2" / "descriptors.npz"
        if desc_path.is_file():
            self.embedding = descriptor_embedding(desc_path, [e["key"] for e in entries])
        self._sources: dict[str, tuple] = {}

        resolution = int(self.args.get("resolution", 448))
        self.index = {
            "version": 1,
            "writable": writable,
            "run": {
                "name": self.args.get("tag") or self.run_dir.name,
                "model": self.args.get("model_name"),
                "detector": self.args.get("detector", "knn"),
                "resolution": resolution,
                "patch_size": resolution // grid_shape[0],
                "grid": grid_shape,
                "map_range": [lo, hi if hi > lo else lo + 1],
                "score_mode": self.args.get("score_mode"),
                "passes": 2 if pass1 else 1,
                "n_reference": len(refs),
                "created": datetime.fromtimestamp(final.stat().st_mtime, UTC).isoformat(),
                "args": self.args,
                "embedding_dims": EMBED_DIMS if self.embedding is not None else 0,
                "xmatch": (
                    {
                        "radius_arcsec": xmatch["radius_arcsec"],
                        "checked": xmatch["checked"],
                        "failed": sorted(xmatch["failed"]),
                    }
                    if xmatch
                    else None
                ),
            },
            "mosaics": mosaics,
            "tiles": entries,
        }

    def _tile(self, name: str) -> Tile:
        try:
            return self.tiles[int(Path(name).stem)]
        except (ValueError, IndexError):
            raise FileNotFoundError(name) from None

    def read(self, path: str) -> tuple[bytes, str]:
        """The bytes and content type for one viewer path, or FileNotFoundError."""
        folder, _, name = path.partition("/")
        if path == "index.json":
            return json.dumps(self.index).encode(), "application/json"
        if path == "verdicts.json":
            return json.dumps(read_verdicts(self.run_dir)).encode(), "application/json"
        if folder == "tiles":
            return self._tile(name).png.read_bytes(), "image/png"
        if folder in ("maps", "maps1"):
            tile = self._tile(name)
            src = tile.map2 if folder == "maps" else tile.map1
            if src is None:
                raise FileNotFoundError(path)
            return np.load(src).astype("<f4").tobytes(), "application/octet-stream"
        if folder == "cutouts":
            cutout = self._tile(name).cutout
            if cutout is None:
                raise FileNotFoundError(path)
            return cutout.read_bytes(), "application/fits"
        if path == "embedding.f32" and self.embedding is not None:
            return self.embedding.astype("<f4").tobytes(), "application/octet-stream"
        if folder == "cats" and self.index["run"]["xmatch"]:
            i = int(Path(name).stem) if Path(name).stem.isdigit() else -1
            if not 0 <= i < len(self.tiles):
                raise FileNotFoundError(path)
            return json.dumps(self.tile_sources(i)).encode(), "application/json"
        if folder == "mosaics" and name.endswith(".png") and "/" not in name:
            return (self.data_root / "mosaics" / name).read_bytes(), "image/png"
        raise FileNotFoundError(path)

    def tile_sources(self, i: int) -> list[list]:
        """Catalogued sources over tile i as [catalogue, name, type, note, u, v].

        u and v run from 0 to 1 across the tile from its top left corner, which maps
        onto the PNG and the cutout alike whatever their pixel sizes.
        """
        entry = self.index["tiles"][i]
        mosaic = entry["mosaic"]
        if mosaic not in self._sources:
            cache = self.run_dir / "catalogues" / f"{mosaic}.json"
            rows = []
            if cache.is_file() and "wcs" in self.index["mosaics"][mosaic]:
                cats = json.loads(cache.read_text(encoding="utf-8"))["catalogues"]
                rows = [[c, *r] for c, v in cats.items() if v["ok"] for r in v["rows"]]
            if rows:
                wcs = mosaic_wcs(self.index["mosaics"][mosaic])
                x, y = wcs.all_world2pix([r[2] for r in rows], [r[3] for r in rows], 0)
            else:
                x = y = np.zeros(0)
            self._sources[mosaic] = (rows, np.asarray(x), np.asarray(y))
        rows, x, y = self._sources[mosaic]
        x0, y0, size = entry["x0"], entry["y0"], entry["size"]
        u = (x - x0 + 0.5) / size
        v = (y0 + size - 0.5 - y) / size
        inside = np.flatnonzero((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1))
        return [
            [
                rows[k][0],
                rows[k][1],
                rows[k][4],
                rows[k][5],
                round(float(u[k]), 5),
                round(float(v[k]), 5),
            ]
            for k in inside
        ]

    def paths(self):
        yield "index.json"
        yield "verdicts.json"
        if self.embedding is not None:
            yield "embedding.f32"
        for i, tile in enumerate(self.tiles):
            yield f"tiles/{_tile_id(i)}.png"
            yield f"maps/{_tile_id(i)}.f32"
            if tile.map1 is not None:
                yield f"maps1/{_tile_id(i)}.f32"
            if self.index["tiles"][i]["has_cutout"]:
                yield f"cutouts/{_tile_id(i)}.fits"
            if self.index["run"]["xmatch"]:
                yield f"cats/{_tile_id(i)}.json"
        for mosaic_id in self.index["mosaics"]:
            yield f"mosaics/{mosaic_id}.png"


def _finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


VERDICTS = ("interesting", "artifact", "known", "not_anomalous")
_VERDICT_FIELDS = ["key", "verdict", "star", "note", "updated"]


def read_verdicts(run_dir: Path) -> dict:
    path = Path(run_dir) / "verdicts.csv"
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["key"]: {
                "verdict": row["verdict"] or None,
                "star": row["star"] == "1",
                "note": row["note"],
                "updated": row["updated"],
            }
            for row in csv.DictReader(f)
        }


def save_verdict(run_dir: Path, key: str, verdict: str | None, star: bool, note: str) -> str:
    """Writes one verdict into verdicts.csv beside the run and returns its timestamp.

    The file is rewritten through a temporary file and a rename, so a crash mid-write
    leaves the previous file intact rather than half of one.
    """
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError(f"Unknown verdict {verdict!r}, expected one of {VERDICTS} or null")
    run_dir = Path(run_dir)
    rows = read_verdicts(run_dir)
    updated = datetime.now(UTC).isoformat(timespec="seconds")
    if verdict is None and not star and not note:
        rows.pop(key, None)
    else:
        rows[key] = {"verdict": verdict, "star": star, "note": note, "updated": updated}

    tmp = run_dir / "verdicts.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_VERDICT_FIELDS)
        w.writeheader()
        for k in sorted(rows):
            r = rows[k]
            w.writerow({**r, "key": k, "verdict": r["verdict"] or "", "star": int(r["star"])})
    tmp.replace(run_dir / "verdicts.csv")
    return updated


def export(
    bundle: RunBundle, out_dir: Path, app_dir: Path | None = None, cutouts: int | None = None
) -> int:
    """Writes the bundle under out_dir/run/, with the viewer app beside it when given.

    With `cutouts`, only the tiles ranked that high keep their FITS cutout, and the
    rest show their PNG. Returns how many files were written.
    """
    out_dir = Path(out_dir)
    run_out = out_dir / "run"
    if run_out.exists():
        shutil.rmtree(run_out)
    if app_dir is not None:
        shutil.copytree(app_dir, out_dir, dirs_exist_ok=True)
    bundle.index["writable"] = False
    if cutouts is not None:
        for entry in bundle.index["tiles"]:
            entry["has_cutout"] = entry["has_cutout"] and entry["rank"] <= cutouts
    n = 0
    for path in bundle.paths():
        data, _ = bundle.read(path)
        dest = run_out / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        n += 1
    # GitHub Pages runs Jekyll otherwise, which is slow on a thousand files and does nothing here.
    (out_dir / ".nojekyll").touch()
    return n


def add_site_meta(bundle: RunBundle, out_dir: Path, site_url: str) -> None:
    """Makes an exported run findable once it's hosted at `site_url`.

    It names the run in index.html's title and description for crawlers that don't
    run scripts, adds the canonical URL, a social card cut from the mosaic around
    the top tile, and JSON-LD, and writes a sitemap.xml to submit to search engines. A
    robots.txt would do nothing below the domain's root, so there isn't one.
    """
    out_dir = Path(out_dir)
    url = site_url.rstrip("/") + "/"
    run = bundle.index["run"]
    tiles = bundle.index["tiles"]
    title = f"{run['name']} · JWSTDetector"
    desc = (
        f"{len(tiles)} tiles of {run['name']} ranked by JWSTDetector, unsupervised anomaly "
        "detection for James Webb Space Telescope images. Step through them with the float "
        "pixels, patch distances, RA and Dec."
    )
    _social_card(bundle, out_dir / "social.png")

    page = (out_dir / "index.html").read_text(encoding="utf-8")
    page = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", page, count=1)
    for attr, value in (
        ('name="description"', desc),
        ('property="og:title"', title),
        ('property="og:description"', desc),
    ):
        page = re.sub(
            rf'(<meta {attr} content=")[^"]*(")',
            lambda m, v=value: m.group(1) + html.escape(v) + m.group(2),
            page,
            count=1,
        )
    ld = {
        "@context": "https://schema.org",
        "@type": "WebApplication",
        "name": title,
        "url": url,
        "description": desc,
        "applicationCategory": "Science",
        "operatingSystem": "Any browser with WebGL2",
        "isAccessibleForFree": True,
        "image": url + "social.png",
    }
    head = "\n".join(
        [
            f'    <link rel="canonical" href="{url}" />',
            f'    <meta property="og:url" content="{url}" />',
            f'    <meta property="og:image" content="{url}social.png" />',
            '    <meta property="og:image:width" content="1200" />',
            '    <meta property="og:image:height" content="630" />',
            f'    <meta property="og:image:alt" content="{html.escape(run["name"])}" />',
            f'    <meta name="twitter:image" content="{url}social.png" />',
            f'    <script type="application/ld+json">{json.dumps(ld)}</script>',
            "  </head>",
        ]
    )
    page = page.replace("  </head>", head, 1)
    (out_dir / "index.html").write_text(page, encoding="utf-8")

    (out_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{url}</loc><lastmod>{run['created'][:10]}</lastmod></url>\n"
        "</urlset>\n",
        encoding="utf-8",
    )


def _lit_span(lit: np.ndarray, centre: float, length: int) -> tuple[int, int]:
    """The run of indices around `centre` lit at least a quarter as much as the median,
    over 5 at a time, or the whole axis when that run is shorter than `length`."""
    good = np.convolve(lit, np.ones(5) / 5, mode="same") >= 0.25 * np.median(lit)
    i = min(max(int(centre), 0), len(lit) - 1)
    if not good[i]:
        return 0, len(lit)
    start, end = i, i + 1
    while start > 0 and good[start - 1]:
        start -= 1
    while end < len(lit) and good[end]:
        end += 1
    return (start, end) if end - start >= length else (0, len(lit))


def _social_card(bundle: RunBundle, path: Path) -> None:
    """A 1200 by 630 crop of the top tile's mosaic preview, centred on that tile."""
    top = bundle.index["tiles"][0]
    preview = bundle.data_root / "mosaics" / f"{top['mosaic']}.png"
    with Image.open(preview) as im:
        im = im.convert("RGB")
        h_mosaic = bundle.index["mosaics"][top["mosaic"]]["shape"][0]
        scale = im.height / h_mosaic
        cx = (top["x0"] + top["size"] / 2) * scale
        cy = (h_mosaic - top["y0"] - top["size"] / 2) * scale
        # Half the mosaic's height keeps the card on the tile's own field, clear of
        # gaps between detectors, and still shows plenty of sky around it.
        h = min(im.height // 2, round(im.width * 630 / 1200))
        w = round(h * 1200 / 630)
        # Kept off blank columns and rows, like the gap between NIRCam's modules.
        lit = np.asarray(im.convert("L")) > 0
        x0, x1 = _lit_span(lit.mean(axis=0), cx, w)
        y0, y1 = _lit_span(lit.mean(axis=1), cy, h)
        left = int(min(max(cx - w / 2, x0), x1 - w))
        upper = int(min(max(cy - h / 2, y0), y1 - h))
        card = im.crop((left, upper, left + w, upper + h)).resize((1200, 630), Image.LANCZOS)
    card.save(path, optimize=True)
