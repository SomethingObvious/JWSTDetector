"""A run and its tiles laid out the way the web viewer reads them.

The viewer only ever asks for paths like `tiles/000012.png` or `maps/000012.f32`.
`RunBundle.read` answers those straight from the run and dataset folders, which is
what the local server does, and `export` writes every one of them to a folder that
any static host can serve.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import html
import json
import math
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Any, TypeAlias

from astropy.wcs import WCS
import numpy as np
from PIL import Image
import yaml

from jwstdetector.data import (
    final_pass_dir,
    owned_patches,
    read_manifest,
    read_measurements,
)
from jwstdetector.detection import Footprint
from jwstdetector.significance import read_significance

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import numpy.typing as npt

    from jwstdetector.utils import SkyWcs

    # Rows of cached catalogue sources over one mosaic, with their mosaic pixels.
    _Sources: TypeAlias = tuple[
        list[list[Any]], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]

EMBED_DIMS = 32
_JSON = "application/json"
_FLOATS = "application/octet-stream"
_TILE_FOLDERS = ("tiles", "maps", "maps1", "cutouts")


def _tile_id(i: int) -> str:
    return f"{i:06d}"


def mosaic_wcs(meta: dict[str, Any]) -> SkyWcs:
    """Build the celestial WCS that mosaics/<id>.json records.

    Returns:
        The WCS from the `wcs` header in `meta`.
    """
    return WCS(meta["wcs"])


def patch_pixels(
    entries: list[dict[str, Any]], grid: Sequence[int]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Mosaic pixel coordinates, 0-based, of the centre of each entry's peak patch.

    The patch grid covers the PNG with its top row first, and the PNG's top row is the
    tile's highest FITS row, so rows count down from y0 + size.

    Returns:
        The x and y of every entry's peak patch centre.
    """
    r = np.array([e["peak"][0] for e in entries], dtype=np.float64)
    c = np.array([e["peak"][1] for e in entries], dtype=np.float64)
    size = np.array([e["size"] for e in entries], dtype=np.float64)
    x = np.array([e["x0"] for e in entries]) + (c + 0.5) * size / grid[1] - 0.5
    y = np.array([e["y0"] for e in entries]) + size - (r + 0.5) * size / grid[0] - 0.5
    return x, y


def descriptor_embedding(
    path: Path, keys: list[str], dims: int = EMBED_DIMS
) -> npt.NDArray[np.float32]:
    """Each key's descriptor from descriptors.npz, reduced by PCA to unit length.

    A 1024-dim float per tile is 4 KB, which adds up for a browser, and the top few
    dozen components keep nearly all of what cosine similarity sees.

    Returns:
        One row of `dims` floats per key, zero padded past the components there are.
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


def map_path(maps_dir: Path, key: str) -> Path:
    """Find the saved patch-distance grid for tile `key`.

    Returns:
        The .npy path under `maps_dir`.
    """
    # Same rule as detection.score_queries, since tile stems are full of RA and Dec
    # dots.
    return maps_dir / f"{Path(key).with_suffix('')}.npy"


@dataclass
class Tile:
    """The files behind one ranked tile."""

    key: str
    png: Path
    map2: Path
    map1: Path | None
    cutout: Path | None


@dataclass
class _RunScores:
    """What a run's CSVs and the dataset's manifest say about every tile."""

    final: Path
    scores: dict[str, float]
    pass1: dict[str, float]
    refs: set[str]
    manifest: dict[str, dict[str, Any]]
    ranked: list[str]
    maps2: Path
    maps1: Path
    owned: dict[str, npt.NDArray[np.bool_]]


def _read_args(run_dir: Path) -> dict[str, Any]:
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        msg = f"{run_dir} has no args.yaml, so it isn't a run folder"
        raise FileNotFoundError(msg)
    return yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}


def _read_words(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split()) if path.is_file() else set()


def _owned_masks(
    manifest: dict[str, dict[str, Any]], scores: dict[str, float], first_map: Path
) -> dict[str, npt.NDArray[np.bool_]]:
    first = np.load(first_map)
    footprints = {
        k: Footprint(m["mosaic"], m["x0"], m["y0"], m["size"])
        for k, m in manifest.items()
        if k in scores
    }
    return owned_patches(footprints, (first.shape[0], first.shape[1]))


def _peak(
    grid: npt.NDArray[np.floating[Any]], mine: npt.NDArray[np.bool_] | None
) -> list[int]:
    scored = np.where(mine, grid, -np.inf) if mine is not None else grid
    peak = np.unravel_index(int(np.argmax(scored)), grid.shape)
    return [int(peak[0]), int(peak[1])]


def _owned_box(mine: npt.NDArray[np.bool_]) -> list[int]:
    r, c = np.nonzero(mine)
    # First and last row and column the tile scored, for the plate's outline.
    return [int(np.min(r)), int(np.max(r)), int(np.min(c)), int(np.max(c))]


def _map_range(values: list[npt.NDArray[np.floating[Any]]]) -> list[float]:
    # The overlay runs from the run's median patch to its 99.9th percentile, so a
    # colour means the same distance on every tile and quiet tiles stay dark.
    lo, hi = (float(v) for v in np.nanpercentile(np.concatenate(values), [50, 99.9]))
    return [lo, hi if hi > lo else lo + 1]


def _add_peak_sky(
    entries: list[dict[str, Any]],
    mosaics: dict[str, dict[str, Any]],
    grid: Sequence[int],
) -> None:
    for mosaic_id, meta in mosaics.items():
        if "wcs" not in meta:
            continue
        mine = [e for e in entries if e["mosaic"] == mosaic_id]
        x, y = patch_pixels(mine, grid)
        ra, dec = mosaic_wcs(meta).all_pix2world(x, y, 0)
        for e, a, d in zip(mine, ra.tolist(), dec.tolist(), strict=True):
            e["peak_sky"] = [round(a, 7), round(d, 7)]


def _add_xmatch(run_dir: Path, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    path = run_dir / "crossmatch.json"
    if not path.is_file():
        return None
    xmatch: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for e in entries:
        if e["key"] in xmatch["tiles"]:
            e["xmatch"] = xmatch["tiles"][e["key"]]
    return xmatch


def _add_significance(run_dir: Path, entries: list[dict[str, Any]]) -> int:
    significance = run_dir / "significance.csv"
    if not significance.is_file():
        return 0
    pq = read_significance(significance)
    for e in entries:
        if e["key"] in pq:
            e["p"], e["q"] = pq[e["key"]]
    with (run_dir / "calibration.csv").open(encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def _xmatch_summary(xmatch: dict[str, Any] | None) -> dict[str, Any] | None:
    if not xmatch:
        return None
    return {
        "radius_arcsec": xmatch["radius_arcsec"],
        "checked": xmatch["checked"],
        "failed": sorted(xmatch["failed"]),
    }


class RunBundle:
    """Joins a `jwstdetector run` folder with the dataset `jwstdetector prep` wrote."""

    def __init__(self, run_dir: Path, data_root: Path, *, writable: bool) -> None:
        self.run_dir = Path(run_dir)
        self.data_root = Path(data_root)
        self.args = _read_args(self.run_dir)
        self.query_dir = self.data_root / self.args.get("query_subdir", "query")
        self.writable = writable

        self.final_dir = final_pass_dir(self.run_dir)
        run = self._read_scores()
        self.tiles: list[Tile] = []
        entries, values = self._load_tiles(run)
        grid_shape = list(np.load(self.tiles[0].map2).shape)
        map_range = _map_range(values)
        mosaics = self._read_mosaics(entries)
        _add_peak_sky(entries, mosaics, grid_shape)
        xmatch = _add_xmatch(self.run_dir, entries)
        n_calibration = _add_significance(self.run_dir, entries)

        self.embedding: npt.NDArray[np.float32] | None = None
        desc_path = self.final_dir / "descriptors.npz"
        if desc_path.is_file():
            self.embedding = descriptor_embedding(
                desc_path, [e["key"] for e in entries]
            )
        self._sources: dict[str, _Sources] = {}

        resolution = int(self.args.get("resolution", 448))
        self.index: dict[str, Any] = {
            "version": 1,
            "writable": writable,
            "run": {
                "name": self.args.get("tag") or self.run_dir.name,
                "model": self.args.get("model_name"),
                "detector": self.args.get("detector", "knn"),
                "resolution": resolution,
                "patch_size": resolution // grid_shape[0],
                "grid": grid_shape,
                "map_range": map_range,
                "score_mode": self.args.get("score_mode"),
                "passes": 2 if run.pass1 else 1,
                "n_reference": len(run.refs),
                "reference_fields": [
                    Path(r).name for r in self.args.get("reference_roots") or []
                ],
                "n_calibration": n_calibration,
                "created": datetime.fromtimestamp(
                    run.final.stat().st_mtime, UTC
                ).isoformat(),
                "args": self.args,
                "embedding_dims": EMBED_DIMS if self.embedding is not None else 0,
                "xmatch": _xmatch_summary(xmatch),
            },
            "mosaics": mosaics,
            "tiles": entries,
        }

    def _read_scores(self) -> _RunScores:
        final = self.run_dir / "measurements_final.csv"
        scores = read_measurements(final)
        pass1_csv = self.run_dir / "pass1" / "measurements.csv"
        pass1 = read_measurements(pass1_csv) if pass1_csv.is_file() else {}
        refs = _read_words(self.final_dir / "ref_list.txt")
        manifest = read_manifest(self.data_root)
        seed = self.args.get("seed", 0)
        # A tile with nothing to compare against scored NaN. It goes last, as null.
        ranked = sorted(scores, key=lambda k: (scores[k] != scores[k], -scores[k]))
        maps2 = self.final_dir / f"anomaly_maps/seed={seed}"
        maps1 = self.run_dir / "pass1" / f"anomaly_maps/seed={seed}"
        owned: dict[str, npt.NDArray[np.bool_]] = {}
        # Runs from before --score_region scored every patch.
        if self.args.get("score_region", "all") == "owned" and ranked:
            owned = _owned_masks(manifest, scores, map_path(maps2, ranked[0]))
        return _RunScores(
            final, scores, pass1, refs, manifest, ranked, maps2, maps1, owned
        )

    def _load_tiles(
        self, run: _RunScores
    ) -> tuple[list[dict[str, Any]], list[npt.NDArray[np.floating[Any]]]]:
        entries: list[dict[str, Any]] = []
        values: list[npt.NDArray[np.floating[Any]]] = []
        for rank, key in enumerate(run.ranked, start=1):
            entry, flat = self._load_tile(rank, key, run)
            entries.append(entry)
            values.append(flat)
        if not entries:
            msg = f"{run.final} has no tiles in it"
            raise ValueError(msg)
        return entries, values

    def _load_tile(
        self, rank: int, key: str, run: _RunScores
    ) -> tuple[dict[str, Any], npt.NDArray[np.floating[Any]]]:
        meta = run.manifest.get(key)
        if meta is None:
            msg = (
                f"{key} is scored in {run.final} but missing from "
                f"{self.data_root}/manifest.csv. "
                "Was the run made from a different dataset?"
            )
            raise KeyError(msg)
        map1 = map_path(run.maps1, key)
        cutout = self.data_root / meta["cutout"] if meta.get("cutout") else None
        tile = Tile(
            key,
            self.query_dir / key,
            map_path(run.maps2, key),
            map1 if map1.is_file() else None,
            cutout,
        )
        self.tiles.append(tile)

        grid = np.load(tile.map2)
        mine = run.owned.get(key)
        entry = {
            "key": key,
            "rank": rank,
            "score": _finite(run.scores[key]),
            "in_reference": key in run.refs,
            "mosaic": meta["mosaic"],
            "x0": meta["x0"],
            "y0": meta["y0"],
            "size": meta["size"],
            "ra": meta["ra"],
            "dec": meta["dec"],
            "peak": _peak(grid, mine),
            "has_cutout": cutout is not None and cutout.is_file(),
        }
        if _finite(run.pass1.get(key)) is not None:
            entry["score_pass1"] = run.pass1[key]
        if mine is not None and not mine.all():
            entry["owned"] = _owned_box(mine)
        return entry, grid.ravel()

    def _read_mosaics(self, entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        mosaics: dict[str, dict[str, Any]] = {}
        for mosaic_id in sorted({e["mosaic"] for e in entries}):
            meta = json.loads(
                (self.data_root / "mosaics" / f"{mosaic_id}.json").read_text("utf-8")
            )
            meta["preview"] = f"mosaics/{mosaic_id}.png"
            mosaics[mosaic_id] = meta
        return mosaics

    def _tile(self, name: str) -> Tile:
        try:
            return self.tiles[int(Path(name).stem)]
        except (ValueError, IndexError):
            raise FileNotFoundError(name) from None

    def read(self, path: str) -> tuple[bytes, str]:
        """Read one viewer path from the run and dataset folders.

        Returns:
            The file's bytes and its content type.

        Raises:
            FileNotFoundError: The run has nothing at `path`.
        """
        folder, _, name = path.partition("/")
        if folder in _TILE_FOLDERS:
            return self._read_tile_file(path)
        if folder == "cats" and self.index["run"]["xmatch"]:
            return self._read_cats(path)
        if folder == "mosaics" and name.endswith(".png") and "/" not in name:
            return (self.data_root / "mosaics" / name).read_bytes(), "image/png"
        return self._read_run_file(path)

    def _read_run_file(self, path: str) -> tuple[bytes, str]:
        if path == "index.json":
            return json.dumps(self.index).encode(), _JSON
        if path == "verdicts.json":
            return json.dumps(read_verdicts(self.run_dir)).encode(), _JSON
        if path == "embedding.f32" and self.embedding is not None:
            return self.embedding.astype("<f4").tobytes(), _FLOATS
        raise FileNotFoundError(path)

    def _read_tile_file(self, path: str) -> tuple[bytes, str]:
        folder, _, name = path.partition("/")
        tile = self._tile(name)
        if folder == "tiles":
            return tile.png.read_bytes(), "image/png"
        src = {"maps": tile.map2, "maps1": tile.map1, "cutouts": tile.cutout}[folder]
        if src is None:
            raise FileNotFoundError(path)
        if folder == "cutouts":
            return src.read_bytes(), "application/fits"
        return np.load(src).astype("<f4").tobytes(), _FLOATS

    def _read_cats(self, path: str) -> tuple[bytes, str]:
        stem = Path(path.partition("/")[2]).stem
        i = int(stem) if stem.isdigit() else -1
        if not 0 <= i < len(self.tiles):
            raise FileNotFoundError(path)
        return json.dumps(self.tile_sources(i)).encode(), _JSON

    def _mosaic_sources(self, mosaic: str) -> _Sources:
        if mosaic not in self._sources:
            cache = self.run_dir / "catalogues" / f"{mosaic}.json"
            rows: list[list[Any]] = []
            if cache.is_file() and "wcs" in self.index["mosaics"][mosaic]:
                cats = json.loads(cache.read_text(encoding="utf-8"))["catalogues"]
                rows = [[c, *r] for c, v in cats.items() if v["ok"] for r in v["rows"]]
            if rows:
                wcs = mosaic_wcs(self.index["mosaics"][mosaic])
                x, y = wcs.all_world2pix([r[2] for r in rows], [r[3] for r in rows], 0)
            else:
                x = y = np.zeros(0)
            self._sources[mosaic] = (rows, np.asarray(x), np.asarray(y))
        return self._sources[mosaic]

    def tile_sources(self, i: int) -> list[list[Any]]:
        """Catalogued sources over tile i as [catalogue, name, type, note, u, v].

        u and v run from 0 to 1 across the tile from its top left corner, which maps
        onto the PNG and the cutout alike whatever their pixel sizes.

        Returns:
            One row per source that falls on the tile.
        """
        entry = self.index["tiles"][i]
        rows, x, y = self._mosaic_sources(entry["mosaic"])
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

    def paths(self) -> Iterator[str]:
        """List every path the viewer can ask this run for.

        Yields:
            Each path, in the order `export` writes them.
        """
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


def read_verdicts(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Read the verdicts saved beside a run.

    Returns:
        Each judged tile's verdict, star, note and when it was saved, or {} without a
        verdicts.csv.
    """
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


def save_verdict(  # noqa: PLR0917 the server passes all five positionally
    run_dir: Path,
    key: str,
    verdict: str | None,
    star: bool,  # noqa: FBT001 the server passes all five positionally
    note: str,
) -> str:
    """Writes one verdict into verdicts.csv beside the run.

    The file is rewritten through a temporary file and a rename, so a crash mid-write
    leaves the previous file intact rather than half of one.

    Returns:
        When the verdict was saved, as an ISO 8601 UTC timestamp.

    Raises:
        ValueError: `verdict` isn't one of VERDICTS or None.
    """
    if verdict is not None and verdict not in VERDICTS:
        msg = f"Unknown verdict {verdict!r}, expected one of {VERDICTS} or null"
        raise ValueError(msg)
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
            w.writerow({
                **r,
                "key": k,
                "verdict": r["verdict"] or "",
                "star": int(r["star"]),
            })
    tmp.replace(run_dir / "verdicts.csv")
    return updated


def export(
    bundle: RunBundle,
    out_dir: Path,
    app_dir: Path | None = None,
    *,
    cutouts: int | None = None,
) -> int:
    """Writes the bundle under out_dir/run/, with the viewer app beside it when given.

    With `cutouts`, only the tiles ranked that high keep their FITS cutout, and the
    rest show their PNG.

    Returns:
        How many files were written.
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
    # GitHub Pages runs Jekyll otherwise, which is slow on a thousand files and does
    # nothing here.
    (out_dir / ".nojekyll").touch()
    return n


def _set_meta(page: str, attr: str, value: str) -> str:
    escaped = html.escape(value)
    return re.sub(
        rf'(<meta {attr} content=")[^"]*(")',
        lambda m: m.group(1) + escaped + m.group(2),
        page,
        count=1,
    )


def _head_tags(url: str, name: str, ld: dict[str, Any]) -> str:
    return "\n".join([
        f'    <link rel="canonical" href="{url}" />',
        f'    <meta property="og:url" content="{url}" />',
        f'    <meta property="og:image" content="{url}social.png" />',
        '    <meta property="og:image:width" content="1200" />',
        '    <meta property="og:image:height" content="630" />',
        f'    <meta property="og:image:alt" content="{html.escape(name)}" />',
        f'    <meta name="twitter:image" content="{url}social.png" />',
        f'    <script type="application/ld+json">{json.dumps(ld)}</script>',
        "  </head>",
    ])


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
        f"{len(tiles)} tiles of {run['name']} ranked by JWSTDetector, unsupervised "
        "anomaly detection for James Webb Space Telescope images. Step through them "
        "with the float pixels, patch distances, RA and Dec."
    )
    _social_card(bundle, out_dir / "social.png")

    page = (out_dir / "index.html").read_text(encoding="utf-8")
    page = re.sub(
        r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", page, count=1
    )
    for attr, value in (
        ('name="description"', desc),
        ('property="og:title"', title),
        ('property="og:description"', desc),
    ):
        page = _set_meta(page, attr, value)
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
    page = page.replace("  </head>", _head_tags(url, run["name"], ld), 1)
    (out_dir / "index.html").write_text(page, encoding="utf-8")

    (out_dir / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"  <url><loc>{url}</loc><lastmod>{run['created'][:10]}</lastmod></url>\n"
        "</urlset>\n",
        encoding="utf-8",
    )


def lit_span(
    lit: npt.NDArray[np.floating[Any]], centre: float, length: int
) -> tuple[int, int]:
    """Find the run of indices around `centre` that is lit, 5 at a time.

    Lit means at least a quarter as bright as the median.

    Returns:
        The run's start and end, or the whole axis when that run is shorter than
        `length`.
    """
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


def _card_start(lit: npt.NDArray[np.floating[Any]], centre: float, length: int) -> int:
    lo, hi = lit_span(lit, centre, length)
    return int(min(max(centre - length / 2, lo), hi - length))


def _card_box(
    lit: npt.NDArray[np.bool_], top: dict[str, Any], h_mosaic: int
) -> tuple[int, int, int, int]:
    height, width = lit.shape
    scale = height / h_mosaic
    cx = (top["x0"] + top["size"] / 2) * scale
    cy = (h_mosaic - top["y0"] - top["size"] / 2) * scale
    # Half the mosaic's height keeps the card on the tile's own field, clear of
    # gaps between detectors, and still shows plenty of sky around it.
    h = min(height // 2, round(width * 630 / 1200))
    w = round(h * 1200 / 630)
    # Kept off blank columns and rows, like the gap between NIRCam's modules.
    left = _card_start(lit.mean(axis=0), cx, w)
    upper = _card_start(lit.mean(axis=1), cy, h)
    return left, upper, left + w, upper + h


def _social_card(bundle: RunBundle, path: Path) -> None:
    """A 1200 by 630 crop of the top tile's mosaic preview, centred on that tile."""
    top = bundle.index["tiles"][0]
    preview = bundle.data_root / "mosaics" / f"{top['mosaic']}.png"
    with Image.open(preview) as src:
        im = src.convert("RGB")
        h_mosaic = bundle.index["mosaics"][top["mosaic"]]["shape"][0]
        box = _card_box(np.asarray(im.convert("L")) > 0, top, h_mosaic)
        card = im.crop(box).resize((1200, 630), Image.Resampling.LANCZOS)
    card.save(path, optimize=True)
