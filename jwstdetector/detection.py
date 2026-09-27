"""Patch-level nearest-neighbour anomaly detection, adapted from AnomalyDINO.

A memory bank holds patch embeddings from the reference tiles, and each query tile is
scored by how far its patches sit from their nearest neighbours in that bank. Every
bank patch remembers which reference tile it came from, because the label-free runner
scores tiles that are also in its reference set, and a tile matched against its own
patches scores zero however strange it is.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as tnf
from tqdm import tqdm

from jwstdetector.backbones import Backbone, background_mask
from jwstdetector.data import tile_loader
from jwstdetector.scoring import aggregate_score

logger = logging.getLogger(__name__)

METRICS = ("L2", "L2_normalized")

# Caps the (queries x bank) distance matrix at about 256 MB of float32.
_MAX_DISTANCE_ELEMENTS = 64 << 20


def dihedral(x: torch.Tensor) -> list[torch.Tensor]:
    """The eight square symmetries of a batch of tiles.

    They are exact, where rotating by an arbitrary angle resamples the pixels and folds
    a reflected border into the corners, and sky has no preferred orientation anyway.
    """
    out = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out.append(r)
        out.append(torch.flip(r, dims=(3,)))
    return out


class MemoryBank:
    """Reference patch embeddings, the tile each came from, and an exact kNN search.

    It takes ownership of `features` and normalises them in place for L2_normalized,
    so a 4 GB bank doesn't briefly need 8.
    """

    def __init__(self, features: torch.Tensor, metric: str, owners: torch.Tensor | None = None):
        if metric not in METRICS:
            raise ValueError(f"Unknown knn_metric {metric!r}, expected one of {METRICS}")
        self.metric = metric
        if metric == "L2_normalized":
            features.div_(features.norm(dim=1, keepdim=True).clamp_min_(1e-12))
        self.features = features
        self.owners = owners

    def __len__(self) -> int:
        return self.features.shape[0]

    @torch.inference_mode()
    def distances(
        self, queries: torch.Tensor, k: int, owners: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Mean distance from each of the (N, D) queries to its k nearest bank patches.

        `owners` gives the reference tile each query patch belongs to, or -1, and bank
        patches from that same tile are skipped.
        """
        k = max(1, min(k, len(self)))
        cosine = self.metric == "L2_normalized"
        if cosine:
            queries = tnf.normalize(queries, dim=1)
        skip_own = owners is not None and self.owners is not None and bool((owners >= 0).any())

        # Every chunk of queries reads the whole bank, which is probably memory-bound on a
        # GPU once the bank runs to a million patches. Splitting the bank into blocks as
        # well is the thing to try if scoring gets slow.
        chunk = max(1, _MAX_DISTANCE_ELEMENTS // len(self))
        out = []
        for start in range(0, queries.shape[0], chunk):
            q = queries[start : start + chunk]
            if cosine:
                # Unit vectors, so half the squared L2 distance is `1 - cos`, which is
                # the scale AnomalyDINO reports.
                d = torch.mm(q, self.features.T).neg_().add_(1.0)
            else:
                d = torch.cdist(q, self.features)
            if skip_own:
                own = owners[start : start + chunk, None] == self.owners[None, :]
                d.masked_fill_(own, float("inf"))
            out.append(d.topk(k, dim=1, largest=False).values.mean(dim=1))
        return torch.cat(out)


def _last_of_each(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique values of `slots` and the position of the last occurrence of each."""
    s = slots.numpy()
    unique, first_from_end = np.unique(s[::-1], return_index=True)
    return torch.from_numpy(unique), torch.from_numpy(len(s) - 1 - first_from_end)


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
    """Embed the reference tiles and keep a uniform random sample of up to `max_patches` patches.

    Not all of them fit, since 425 tiles at 672 px through DINOv3 ViT-L with --rotation
    come to about 25 GB. It's a reservoir sample (Algorithm R), so the bank is allocated once
    at its final size and fills up to the cap however many patches masking leaves.
    """
    n_views = 8 if rotation else 1
    capacity = min(max_patches, len(ref_paths) * n_views * backbone.n_patches)
    generator = torch.Generator().manual_seed(seed)
    loader = tile_loader(
        ref_paths,
        ref_root,
        backbone.transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=backbone.device.type == "cuda",
    )

    features = owners = None
    filled = seen = first_tile = 0
    for _, images in tqdm(loader, desc="Building memory bank", leave=False):
        tile_ids = torch.arange(first_tile, first_tile + len(images), dtype=torch.int32)
        tile_ids = tile_ids.repeat_interleave(backbone.n_patches).to(backbone.device)
        first_tile += len(images)

        for view in dihedral(images) if rotation else [images]:
            feats = backbone.embed(view)
            ids = tile_ids
            if masking:
                keep = background_mask(feats, backbone.grid_size).reshape(-1)
                feats, ids = feats.reshape(-1, feats.shape[-1])[keep], ids[keep]
            else:
                feats = feats.reshape(-1, feats.shape[-1])
            if features is None:
                features = torch.empty((capacity, feats.shape[1]), device=feats.device)
                owners = torch.empty(capacity, dtype=torch.int32, device=feats.device)

            n = feats.shape[0]
            take = min(capacity - filled, n)
            features[filled : filled + take] = feats[:take]
            owners[filled : filled + take] = ids[:take]
            filled += take

            if take < n:
                # Patch i, counting from 0 over every patch seen, draws a slot in [0, i]
                # and replaces what's there when the slot is inside the bank.
                i = torch.arange(seen + take, seen + n, dtype=torch.float64)
                slots = torch.rand(n - take, generator=generator, dtype=torch.float64) * (i + 1)
                slots = slots.long()
                hits = torch.nonzero(slots < capacity).squeeze(1)
                # When two patches draw the same slot the later one wins, as it would
                # one at a time, and deduplicating keeps the GPU write deterministic.
                slots, last = _last_of_each(slots[hits])
                src = (hits[last] + take).to(feats.device)
                slots = slots.to(feats.device)
                features[slots] = feats[src]
                owners[slots] = ids[src]
            seen += n

    if features is None or filled == 0:
        raise RuntimeError("The memory bank is empty. Is the reference set empty?")
    logger.info(
        "Memory bank holds %d of %d reference patches, %d dims each",
        filled,
        seen,
        features.shape[1],
    )
    return MemoryBank(features[:filled], metric, owners[:filled])


@torch.inference_mode()
def score_queries(
    backbone: Backbone,
    bank: MemoryBank,
    query_paths: list[Path],
    query_root: Path,
    maps_dir: Path | None,
    *,
    query_owners: list[int] | None = None,
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
    """Score every query tile. Returns the scores and the seconds spent per tile.

    `query_owners` gives, for each query tile, its index among the reference tiles or
    -1, so a reference tile is scored against the other tiles only.
    """
    if save_tiffs:
        import tifffile

        from jwstdetector.utils import dists2map

    loader = tile_loader(
        query_paths,
        query_root,
        backbone.transform(),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=backbone.device.type == "cuda",
    )
    owners_all = torch.tensor(
        query_owners if query_owners is not None else [-1] * len(query_paths), dtype=torch.int32
    )
    grid = backbone.grid_size
    scores: dict[str, float] = {}
    seconds: dict[str, float] = {}

    start = 0
    for keys, images in tqdm(loader, desc="Scoring query tiles"):
        started = time.perf_counter()
        owners = owners_all[start : start + len(keys)]
        owners = owners.repeat_interleave(backbone.n_patches).to(backbone.device)
        start += len(keys)

        feats = backbone.embed(images)
        flat = feats.reshape(-1, feats.shape[-1])
        keep = background_mask(feats, grid) if masking else None
        if keep is None:
            dists = bank.distances(flat, k_neighbors, owners)
        else:
            valid = keep.reshape(-1)
            dists = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
            dists[valid] = bank.distances(flat[valid], k_neighbors, owners[valid])

        _sync(backbone.device)
        elapsed = (time.perf_counter() - started) / len(keys)

        grids = dists.reshape(len(keys), *grid).cpu().numpy()
        keeps = keep.reshape(len(keys), *grid).cpu().numpy() if keep is not None else None

        for i, key in enumerate(keys):
            # Masked-out patches carry a distance of 0, which would drag a median or
            # mean down, so only the real patches are scored. The saved map keeps the
            # zeros so it still lines up with the tile.
            patch_dists = grids[i][keeps[i]] if keeps is not None else grids[i]
            scores[key] = aggregate_score(
                patch_dists,
                mode=score_mode,
                top_frac=score_top_frac,
                quantile=score_quantile,
            )
            seconds[key] = elapsed

            if maps_dir is not None and (save_patch_dists or save_tiffs):
                # Appending the suffix by hand, as with_suffix(".npy") on a stem full of
                # RA and Dec dots would cut it at the last dot.
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
    """Build the bank from `ref_paths` and score `query_paths` against it.

    Returns the scores, the seconds spent on the bank and the seconds spent per tile.
    A query path that is also in `ref_paths` is scored against the other reference
    tiles only. The bank build runs the model many times before any timing
    starts, so the CUDA kernels are warm by the time the scoring is timed.
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

    ref_index = {p: i for i, p in enumerate(ref_paths)}
    scores, seconds = score_queries(
        backbone,
        bank,
        query_paths,
        query_root,
        Path(out_dir) / f"anomaly_maps/seed={seed}",
        query_owners=[ref_index.get(p, -1) for p in query_paths],
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
