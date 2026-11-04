"""Tile listing, loading and result files, shared by both runners."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import numpy.typing as npt

    from jwstdetector.backbones import Backbone
    from jwstdetector.detection import Footprint

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

K = TypeVar("K")


def list_images(root: Path, *, recursive: bool = True) -> list[Path]:
    """Find every image under `root`, sorted so a run can be repeated.

    Returns:
        The image paths.

    Raises:
        FileNotFoundError: If `root` isn't a folder or holds no images.
    """
    root = Path(root)
    if not root.is_dir():
        msg = f"There is no folder at {root}"
        raise FileNotFoundError(msg)

    walk = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in walk if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not paths:
        msg = f"There are no images under {root} (recursive={recursive})"
        raise FileNotFoundError(msg)
    return paths


def relative_key(path: Path, root: Path) -> str:
    """Return the tile's path under `root` with forward slashes, its name everywhere."""
    return path.relative_to(root).as_posix()


class TileDataset(Dataset[tuple[str, torch.Tensor]]):
    """Tiles read off disk and transformed, in the DataLoader's worker processes."""

    def __init__(
        self,
        paths: list[Path],
        root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.paths = paths
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of tiles."""
        return len(self.paths)

    @override
    def __getitem__(self, i: int) -> tuple[str, torch.Tensor]:
        """Return the tile's key and its transformed pixels."""
        path = self.paths[i]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return relative_key(path, self.root), tensor


def tile_loader(
    paths: list[Path],
    root: Path,
    backbone: Backbone,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader[tuple[str, torch.Tensor]]:
    """Return a loader that reads `paths` in order, ready for `backbone`."""
    return DataLoader(
        TileDataset(paths, root, backbone.transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=backbone.device.type == "cuda",
    )


def write_measurements_csv(
    path: Path,
    scores: Mapping[str, float],
    *,
    memorybank_sec: float,
    inference_sec: Mapping[str, float],
) -> None:
    """Write each tile's score and timings to a measurements CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sample", "AnomalyScore", "MemoryBankTimeSec", "InferenceTimeSec"])
        for key in sorted(scores):
            w.writerow([
                key,
                f"{float(scores[key]):.8f}",
                f"{memorybank_sec:.6f}",
                f"{float(inference_sec[key]):.6f}",
            ])


MANIFEST_FIELDS = ["key", "mosaic", "x0", "y0", "size", "ra", "dec", "cutout"]


def read_manifest(data_root: Path) -> dict[str, dict[str, Any]]:
    """Read manifest.csv from `jwstdetector prep`.

    Returns:
        Each tile's row keyed by its path under query/, or {} without a manifest.
    """
    path = Path(data_root) / "manifest.csv"
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[row["key"]] = {
                "mosaic": row["mosaic"],
                "x0": int(row["x0"]),
                "y0": int(row["y0"]),
                "size": int(row["size"]),
                "ra": float(row["ra"]) if row["ra"] else None,
                "dec": float(row["dec"]) if row["dec"] else None,
                "cutout": row["cutout"] or None,
            }
    return out


def final_pass_dir(run_dir: Path) -> Path:
    """Return the pass with the final scores, cross for a run against other fields."""
    cross = Path(run_dir) / "cross"
    return cross if cross.is_dir() else Path(run_dir) / "pass2"


def read_measurements(path: Path) -> dict[str, float]:
    """Return each Sample's AnomalyScore from a measurements CSV the runners wrote."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return {row["Sample"]: float(row["AnomalyScore"]) for row in csv.DictReader(f)}


def write_ref_list(path: Path, root: Path, paths: list[Path]) -> None:
    """Write the reference tiles' keys, one to a line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{relative_key(p, root)}\n" for p in paths), encoding="utf-8"
    )


def tagged_dir(out_dir: Path, tag: str | None) -> Path:
    """Return `out_dir` with `_tag` added, so results and miri give results_miri."""
    return out_dir.with_name(f"{out_dir.name}_{tag}") if tag else out_dir


def prepare_out_dir(out_dir: Path, patterns: list[str], *, overwrite: bool) -> None:
    """Make out_dir, stopping if it holds results unless `overwrite` clears them.

    Mixing a new run's args.yaml with an older run's scores is how a summary ends up
    labelling old numbers with new settings, so a folder is either empty of results
    or deliberately emptied.

    Raises:
        SystemExit: If there are results already and `overwrite` is off.
    """
    old = sorted({p for pattern in patterns for p in out_dir.glob(pattern)})
    if old and not overwrite:
        names = ", ".join(p.name for p in old[:4])
        msg = (
            f"{out_dir} already holds results ({names}). Pass --overwrite to replace "
            "them, or pick another --out_dir or --tag."
        )
        raise SystemExit(msg)
    for p in old:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def load_footprints(
    data_root: Path, query_dir: Path, paths: list[Path]
) -> dict[Path, Footprint]:
    """Read each tile's footprint from manifest.csv.

    Without one, tiles that overlap can still vouch for each other, so it says so.

    Returns:
        The footprints by tile path, or {} when the dataset has no manifest.
    """
    from jwstdetector.detection import Footprint  # noqa: PLC0415 detection imports this module

    manifest = read_manifest(data_root)
    if not manifest:
        logging.getLogger(__name__).warning(
            "%s has no manifest.csv, so overlapping tiles can match each other's "
            "pixels. Cutting the tiles again with `jwstdetector prep` writes one.",
            data_root,
        )
        return {}
    scales: dict[str, float] = {}
    out: dict[Path, Footprint] = {}
    for p in paths:
        meta = manifest.get(relative_key(p, query_dir))
        if meta is None:
            continue
        mosaic = meta["mosaic"]
        if mosaic not in scales:
            scales[mosaic] = _pixel_scale_arcsec(data_root, mosaic)
        ra: float | None = meta["ra"]
        dec: float | None = meta["dec"]
        nan = float("nan")
        if ra is None or dec is None or not np.isfinite(scales[mosaic]):
            ra, dec, radius = nan, nan, nan
        else:
            radius = meta["size"] * scales[mosaic] / 3600 / np.sqrt(2)
        out[p] = Footprint(
            mosaic,
            meta["x0"],
            meta["y0"],
            meta["size"],
            ra=ra,
            dec=dec,
            radius_deg=radius,
        )
    return out


def _pixel_scale_arcsec(data_root: Path, mosaic: str) -> float:
    """Read the mosaic's pixel scale from prep's mosaics/<id>.json.

    Returns:
        Arcseconds per pixel, or NaN when the file or the value isn't there.
    """
    path = Path(data_root) / "mosaics" / f"{mosaic}.json"
    if not path.is_file():
        return float("nan")
    scale = json.loads(path.read_text(encoding="utf-8")).get("pixel_scale_arcsec")
    return float(scale) if scale else float("nan")


class _MosaicTiles:
    """The tiles cut from one mosaic, binned by centre so neighbours are quick to find.

    `box` holds each tile's x0, y0 and size, and `fx` and `fy` the patch centres as
    fractions of a tile.
    """

    def __init__(
        self,
        box: npt.NDArray[np.float64],
        fx: npt.NDArray[np.float64],
        fy: npt.NDArray[np.float64],
    ) -> None:
        self.box, self.fx, self.fy = box, fx, fy
        self.cx = box[:, 0] + box[:, 2] / 2
        self.cy = box[:, 1] + box[:, 2] / 2
        # Overlapping tiles have centres less than one tile apart, so they share a cell
        # or sit in the next one.
        self.cell = np.max(box[:, 2])
        self.cells: dict[tuple[float, float], list[int]] = {}
        for i, key in enumerate(
            zip(self.cx // self.cell, self.cy // self.cell, strict=True)
        ):
            self.cells.setdefault(key, []).append(i)

    def _neighbours(self, i: int) -> Iterator[int]:
        ci, cj = self.cx[i] // self.cell, self.cy[i] // self.cell
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self.cells.get((ci + dx, cj + dy), ())

    def owned(self, i: int) -> npt.NDArray[np.bool_]:
        """Return the (rows, cols) patches tile `i` is the nearest centre for."""
        x0, y0, s = self.box[i]
        shape = (len(self.fy), len(self.fx))
        px = np.broadcast_to(x0 + self.fx * s, shape)
        py = np.broadcast_to((y0 + self.fy * s)[:, None], shape)
        mine = np.maximum(abs(px - self.cx[i]), abs(py - self.cy[i]))
        keep = np.ones(shape, dtype=bool)
        for j in self._neighbours(i):
            if j == i:
                continue
            inside = (px >= self.box[j, 0]) & (px < self.box[j, 0] + self.box[j, 2])
            inside &= (py >= self.box[j, 1]) & (py < self.box[j, 1] + self.box[j, 2])
            keep &= ~(
                inside & (np.maximum(abs(px - self.cx[j]), abs(py - self.cy[j])) < mine)
            )
        return keep


def owned_patches(
    footprints: Mapping[K, Footprint], grid: tuple[int, int]
) -> dict[K, npt.NDArray[np.bool_]]:
    """Work out the patches each tile scores.

    An object cut off by a tile's edge looks odd to the backbone however ordinary it
    is, and on SMACS 0723 the outermost ring of patches held the peak of 45% of the top
    20 tiles against 14% by chance. So a patch only counts for the tile whose centre
    it's nearest, by the larger of its x and y offsets, which hands every bit of sky
    that tiles share to exactly one of them and drops none. `footprints` gives each
    tile's mosaic, x0, y0 and size, in mosaic pixels.

    Returns:
        A (rows, cols) bool array per key.
    """
    rows, cols = grid
    by_mosaic: dict[str, list[K]] = {}
    for k, f in footprints.items():
        by_mosaic.setdefault(f.mosaic, []).append(k)
    # Patch centres as fractions of the tile, with PNG row 0 at the top, which is the
    # highest FITS row.
    fx = (np.arange(cols) + 0.5) / cols
    fy = 1 - (np.arange(rows) + 0.5) / rows
    out: dict[K, npt.NDArray[np.bool_]] = {}
    for keys in by_mosaic.values():
        box = np.array(
            [[footprints[k].x0, footprints[k].y0, footprints[k].size] for k in keys],
            float,
        )
        tiles = _MosaicTiles(box, fx, fy)
        for i, k in enumerate(keys):
            out[k] = tiles.owned(i)
    return out


def resolve_device(spec: str) -> torch.device:
    """Find the device a --device value names, failing early when it isn't there.

    Returns:
        The device, with "auto" picking CUDA, then MPS, then the CPU.

    Raises:
        RuntimeError: If it names CUDA or MPS and torch can't see one.
    """
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = f"--device {spec} asks for CUDA, but torch can't see a CUDA device"
        raise RuntimeError(msg)
    if device.type == "mps" and not torch.backends.mps.is_available():
        msg = f"--device {spec} asks for MPS, but it isn't available here"
        raise RuntimeError(msg)
    return device
