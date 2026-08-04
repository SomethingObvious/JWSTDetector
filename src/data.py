"""Tile listing, loading and result IO. Shared by both runners."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def list_images(root: Path, recursive: bool = True) -> list[Path]:
    """Every image under `root`, sorted so a run is reproducible."""
    root = Path(root)
    if not root.is_dir():
        raise RuntimeError(f"Folder does not exist: {root}")

    walk = root.rglob("*") if recursive else root.iterdir()
    paths = sorted(p for p in walk if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise RuntimeError(f"No images found under {root} (recursive={recursive})")
    return paths


def relative_key(path: Path, root: Path) -> str:
    """Stable, platform-independent identifier for a tile."""
    return path.relative_to(root).as_posix()


class TileDataset(Dataset):
    """Reads tiles off disk and applies the backbone transform, in worker processes."""

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
        # Workers are only worth their startup cost if they are reused.
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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


def write_ref_list(path: Path, root: Path, paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{relative_key(p, root)}\n" for p in paths), encoding="utf-8")


def resolve_device(spec: str) -> torch.device:
    """Turn a --device string into a real device, failing early if it is not there."""
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec} but no CUDA device is visible to torch")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"--device {spec} but MPS is not available")
    return device
