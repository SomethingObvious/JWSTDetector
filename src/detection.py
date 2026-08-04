"""Patch-level nearest-neighbour anomaly detection.

Build a memory bank of patch embeddings from reference tiles, then score every
query tile by how far its patches sit from their nearest neighbours in that bank.
Derived from AnomalyDINO, with three changes that matter at JWST scale:

* Tiles run in batches through a DataLoader instead of one at a time.
* The bank is capped, so a large reference set cannot quietly ask for 30 GB.
* The nearest-neighbour search is a chunked matmul in torch. Exact, same numbers
  as a flat FAISS index, and one less dependency that is painful to install.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.backbones import Backbone, background_mask
from src.data import tile_loader
from src.scoring import aggregate_score

logger = logging.getLogger(__name__)

METRICS = ("L2", "L2_normalized")

# Cap the (queries x bank) distance matrix at roughly 256 MB of float32.
_MAX_DISTANCE_ELEMENTS = 64 << 20


def dihedral(x: torch.Tensor) -> list[torch.Tensor]:
    """The eight square symmetries of a batch of tiles.

    Exact, unlike arbitrary-angle rotation, which resamples the pixels and folds a
    reflected border into the corners. Sky has no preferred orientation, so this is
    the right augmentation group for it anyway.
    """
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out.append(r)
        out.append(torch.flip(r, dims=(3,)))
    return out


class MemoryBank:
    """Reference patch embeddings, plus an exact k-nearest-neighbour search over them."""

    def __init__(self, features: torch.Tensor, metric: str):
        if metric not in METRICS:
            raise ValueError(f"Unknown knn_metric={metric!r}, expected one of {METRICS}")
        self.metric = metric
        if _cosine(metric):
            features = torch.nn.functional.normalize(features, dim=1)
        self.features = features

    def __len__(self) -> int:
        return self.features.shape[0]

    @torch.inference_mode()
    def distances(self, queries: torch.Tensor, k: int) -> torch.Tensor:
        """(N, D) -> (N,) mean distance from each query to its k nearest references."""
        k = max(1, min(k, len(self)))
        if _cosine(self.metric):
            queries = torch.nn.functional.normalize(queries, dim=1)

        chunk = max(1, _MAX_DISTANCE_ELEMENTS // max(1, len(self)))
        out = []
        for start in range(0, queries.shape[0], chunk):
            q = queries[start : start + chunk]
            if _cosine(self.metric):
                # Unit vectors, so ||a-b||^2 / 2 == 1 - cos. Same scale as before.
                nearest = (q @ self.features.T).topk(k, dim=1).values
                out.append((1.0 - nearest).mean(dim=1))
            else:
                d = torch.cdist(q, self.features)
                out.append(d.topk(k, dim=1, largest=False).values.mean(dim=1))
        return torch.cat(out)


def _cosine(metric: str) -> bool:
    return metric == "L2_normalized"


@torch.inference_mode()
def build_memory_bank(
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    *,
    masking: bool = False,
    rotation: bool = False,
    metric: str = "L2_normalized",
    max_patches: int = 1_000_000,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 0,
) -> MemoryBank:
    """Embed the reference tiles and keep at most `max_patches` of their patches.

    Every patch of every reference tile is far more than the search needs and far
    more than fits: 425 tiles at 672 px through ViT-L, times eight augmentations, is
    roughly 32 GB of embeddings. Patches are dropped as they arrive, at a fixed rate
    worked out from the tile count, so nothing large is ever held.
    """
    n_aug = 8 if rotation else 1
    expected = len(ref_paths) * n_aug * backbone.n_patches
    keep_prob = min(1.0, max_patches / max(1, expected))
    if keep_prob < 1.0:
        logger.info(
            "reference set would yield ~%d patches; sampling %.1f%% to stay under %d",
            expected,
            100 * keep_prob,
            max_patches,
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    loader = tile_loader(
        ref_paths,
        ref_root,
        backbone.transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=backbone.device.type == "cuda",
    )

    chunks: list[torch.Tensor] = []
    for _, images in tqdm(loader, desc="Building memory bank", leave=False):
        for view in dihedral(images) if rotation else [images]:
            feats = backbone.embed(view)
            if masking:
                keep = background_mask(feats, backbone.grid_size)
                feats = feats[keep]
            else:
                feats = feats.reshape(-1, feats.shape[-1])

            if keep_prob < 1.0:
                take = torch.rand(feats.shape[0], generator=generator) < keep_prob
                feats = feats[take.to(feats.device)]
            if feats.shape[0]:
                chunks.append(feats)

    if not chunks:
        raise RuntimeError("Memory bank is empty. Masking may have rejected every reference patch.")

    features = torch.cat(chunks)
    if features.shape[0] > max_patches:
        pick = torch.randperm(features.shape[0], generator=generator)[:max_patches]
        features = features[pick.to(features.device)]

    logger.info("memory bank: %d patches x %d dims", *features.shape)
    return MemoryBank(features, metric)


@torch.inference_mode()
def score_queries(
    backbone: Backbone,
    bank: MemoryBank,
    query_paths: list[Path],
    query_root: Path,
    maps_dir: Path | None,
    *,
    masking: bool = False,
    k_neighbors: int = 1,
    score_mode: str = "top1p",
    score_top_frac: float = 0.01,
    score_quantile: float = 0.9995,
    batch_size: int = 8,
    num_workers: int = 4,
    save_patch_dists: bool = True,
    save_tiffs: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score every query tile. Returns (scores, per-tile seconds)."""
    if save_tiffs:
        import tifffile

        from src.utils import dists2map

    loader = tile_loader(
        query_paths,
        query_root,
        backbone.transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=backbone.device.type == "cuda",
    )
    grid = backbone.grid_size
    scores: dict[str, float] = {}
    seconds: dict[str, float] = {}

    for keys, images in tqdm(loader, desc="Scoring query tiles"):
        started = time.perf_counter()

        feats = backbone.embed(images)
        keep = background_mask(feats, grid) if masking else None
        flat = feats.reshape(-1, feats.shape[-1])
        valid = keep.reshape(-1) if keep is not None else None

        if valid is None:
            dists = bank.distances(flat, k_neighbors)
        else:
            dists = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
            if bool(valid.any()):
                dists[valid] = bank.distances(flat[valid], k_neighbors)

        _sync(backbone.device)
        elapsed = (time.perf_counter() - started) / len(keys)

        grids = dists.reshape(len(keys), *grid).cpu().numpy()
        keeps = keep.reshape(len(keys), *grid).cpu().numpy() if keep is not None else None

        for i, key in enumerate(keys):
            # Masked-out patches carry a distance of zero, which would drag a
            # median- or mean-based score down. Score the real patches only; the
            # saved map keeps the zeros so it still lines up with the tile.
            patch_dists = grids[i][keeps[i]] if keeps is not None else grids[i]
            scores[key] = aggregate_score(
                patch_dists,
                mode=score_mode,
                top_frac=score_top_frac,
                quantile=score_quantile,
            )
            seconds[key] = elapsed

            if maps_dir is not None and (save_patch_dists or save_tiffs):
                # Append rather than Path.with_suffix: --wcs_in_name puts the RA and
                # Dec in the filename, so the stem is full of dots and with_suffix
                # would eat everything after the last one.
                stem = maps_dir / str(Path(key).with_suffix(""))
                stem.parent.mkdir(parents=True, exist_ok=True)
                if save_patch_dists:
                    np.save(f"{stem}.npy", grids[i])
                if save_tiffs:
                    side = backbone.resolution
                    tifffile.imwrite(f"{stem}.tiff", dists2map(grids[i], (side, side)))

    return scores, seconds


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_anomaly_detection(
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    query_paths: list[Path],
    query_root: Path,
    out_dir: Path,
    *,
    masking: bool = False,
    mask_ref_images: bool = False,
    rotation: bool = False,
    knn_metric: str = "L2_normalized",
    k_neighbors: int = 1,
    max_bank_patches: int = 1_000_000,
    score_mode: str = "top1p",
    score_top_frac: float = 0.01,
    score_quantile: float = 0.9995,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 0,
    save_patch_dists: bool = True,
    save_tiffs: bool = False,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """One full pass: build the bank from `ref_paths`, score `query_paths`.

    Returns (scores, memory-bank seconds, per-tile inference seconds).

    There is no separate CUDA warmup loop any more. Building the bank runs the model
    hundreds of times first, so the kernels are already warm by the time the timed
    scoring pass starts.
    """
    started = time.perf_counter()
    bank = build_memory_bank(
        backbone,
        ref_paths,
        ref_root,
        masking=masking and mask_ref_images,
        rotation=rotation,
        metric=knn_metric,
        max_patches=max_bank_patches,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    _sync(backbone.device)
    memorybank_sec = time.perf_counter() - started

    maps_dir = Path(out_dir) / f"anomaly_maps/seed={seed}"
    scores, seconds = score_queries(
        backbone,
        bank,
        query_paths,
        query_root,
        maps_dir,
        masking=masking,
        k_neighbors=k_neighbors,
        score_mode=score_mode,
        score_top_frac=score_top_frac,
        score_quantile=score_quantile,
        batch_size=batch_size,
        num_workers=num_workers,
        save_patch_dists=save_patch_dists,
        save_tiffs=save_tiffs,
    )
    return scores, memorybank_sec, seconds
