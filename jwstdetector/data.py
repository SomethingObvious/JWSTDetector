"""Tile listing, loading and result files, shared by both runners."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_images(root: Path, recursive: bool = True) -> list[Path]:
    """Every image under `root`, sorted so a run can be repeated."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"There is no folder at {root}")

    walk = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in walk if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"There are no images under {root} (recursive={recursive})")
    return paths


def relative_key(path: Path, root: Path) -> str:
    """The tile's path under `root` with forward slashes, which names it in every output."""
    return path.relative_to(root).as_posix()


class TileDataset(Dataset):
    """Tiles read off disk and transformed, in the DataLoader's worker processes."""

    def __init__(self, paths: list[Path], root: Path, transform):
        self.paths = paths
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        path = self.paths[i]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return relative_key(path, self.root), tensor


def tile_loader(
    paths: list[Path],
    root: Path,
    transform,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        TileDataset(paths, root, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def write_measurements_csv(path: Path, scores: dict, memorybank_sec: float, inference_sec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Sample", "AnomalyScore", "MemoryBankTimeSec", "InferenceTimeSec"])
        for key in sorted(scores):
            w.writerow(
                [
                    key,
                    f"{float(scores[key]):.8f}",
                    f"{memorybank_sec:.6f}",
                    f"{float(inference_sec[key]):.6f}",
                ]
            )


MANIFEST_FIELDS = ["key", "mosaic", "x0", "y0", "size", "ra", "dec", "cutout"]


def read_manifest(data_root: Path) -> dict[str, dict]:
    """manifest.csv from prep_data.py, keyed by tile path under query/, or {} without one."""
    path = Path(data_root) / "manifest.csv"
    if not path.is_file():
        return {}
    out = {}
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


def read_measurements(path: Path) -> dict[str, float]:
    """Sample to AnomalyScore from a measurements CSV the runners wrote."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return {row["Sample"]: float(row["AnomalyScore"]) for row in csv.DictReader(f)}


def write_ref_list(path: Path, root: Path, paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{relative_key(p, root)}\n" for p in paths), encoding="utf-8")


def tagged_dir(out_dir: Path, tag: str | None) -> Path:
    """`out_dir` with `_tag` added to its last part, so results/ and miri give results_miri."""
    return out_dir.with_name(f"{out_dir.name}_{tag}") if tag else out_dir


def prepare_out_dir(out_dir: Path, patterns: list[str], overwrite: bool) -> None:
    """Make out_dir, stopping if it already holds results unless `overwrite` clears them.

    Mixing a new run's args.yaml with an older run's scores is how a summary ends up
    labelling old numbers with new settings, so a folder is either empty of results
    or deliberately emptied.
    """
    old = sorted({p for pattern in patterns for p in out_dir.glob(pattern)})
    if old and not overwrite:
        names = ", ".join(p.name for p in old[:4])
        raise SystemExit(
            f"{out_dir} already holds results ({names}). Pass --overwrite to replace them, "
            "or pick another --out_dir or --tag."
        )
    for p in old:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def load_footprints(data_root: Path, query_dir: Path, paths: list[Path]) -> dict:
    """Each tile's footprint from manifest.csv, or {} when the dataset has none.

    Without one, tiles that overlap can still vouch for each other, so it says so.
    """
    from jwstdetector.detection import Footprint

    manifest = read_manifest(data_root)
    if not manifest:
        logging.getLogger(__name__).warning(
            "%s has no manifest.csv, so overlapping tiles can match each other's pixels. "
            "Cutting the tiles again with this version of prep_data.py writes one.",
            data_root,
        )
        return {}
    out = {}
    for p in paths:
        meta = manifest.get(relative_key(p, query_dir))
        if meta is not None:
            out[p] = Footprint(meta["mosaic"], meta["x0"], meta["y0"], meta["size"])
    return out


def resolve_device(spec: str) -> torch.device:
    """The device a --device value names, failing early when it isn't there."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec} asks for CUDA, but torch can't see a CUDA device")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"--device {spec} asks for MPS, but it isn't available here")
    return device
