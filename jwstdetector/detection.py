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
from dataclasses import dataclass
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
        self,
        queries: torch.Tensor,
        k: int,
        rows: torch.Tensor | None = None,
        excluded: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean distance from each of the (N, D) queries to its k nearest bank patches.

        `rows` gives each query patch a row of the (R, n_ref_tiles) bool `excluded`, and
        bank patches from the tiles marked there are skipped. A patch left with fewer
        than k candidates averages the ones it has, and one with none gets NaN.
        """
        k = max(1, min(k, len(self)))
        cosine = self.metric == "L2_normalized"
        if cosine:
            queries = tnf.normalize(queries, dim=1)
        else:
            bank_sq = self.features.square().sum(dim=1)
        skip = rows is not None and excluded is not None and self.owners is not None
        if skip:
            # Gathered once per call, so each chunk only indexes its own rows.
            excluded = excluded[:, self.owners.long()]

        # Every chunk of queries reads the whole bank, which is probably memory-bound on a
        # GPU once the bank runs to a million patches. Splitting the bank into blocks as
        # well is the thing to try if scoring gets slow.
        chunk = max(1, _MAX_DISTANCE_ELEMENTS // len(self))
        out = []
        for start in range(0, queries.shape[0], chunk):
            q = queries[start : start + chunk]
            # Both branches build the distances in one (chunk, bank) matrix in place.
            # torch.cdist needs several of those at once and runs out of memory on an
            # 8 GB card with a million-patch ViT-L bank.
            d = torch.mm(q, self.features.T)
            if cosine:
                # Unit vectors, so half the squared L2 distance is `1 - cos`, which is
                # the scale AnomalyDINO reports.
                d.neg_().add_(1.0)
            else:
                d.mul_(-2.0).add_(q.square().sum(dim=1, keepdim=True)).add_(bank_sq)
                d.clamp_(min=0.0).sqrt_()
            if skip:
                d.masked_fill_(excluded[rows[start : start + chunk].long()], float("inf"))
            top = d.topk(k, dim=1, largest=False).values
            finite = torch.isfinite(top)
            total = torch.where(finite, top, torch.zeros_like(top)).sum(dim=1)
            out.append(total / finite.sum(dim=1))  # 0 / 0 is NaN, which is the point
        return torch.cat(out)


@dataclass(frozen=True)
class Footprint:
    """Where a tile was cut from, in mosaic pixels."""

    mosaic: str
    x0: int
    y0: int
    size: int


def excluded_refs(
    query_paths: list[Path],
    ref_paths: list[Path],
    footprints: dict[Path, Footprint] | None = None,
) -> list[list[int]]:
    """For each query tile, the reference tiles it must not be matched against.

    That is the tile itself and every reference tile overlapping it on the sky. Tiles
    overlap by design (half a crop by default, and the crop flush with each far edge
    can share almost all of its pixels), and an object in the shared strip would
    otherwise find itself in the bank and score as ordinary.
    """
    ref_index = {p: i for i, p in enumerate(ref_paths)}
    out = [[ref_index[p]] if p in ref_index else [] for p in query_paths]
    if not footprints:
        return out

    by_mosaic: dict[str, list[int]] = {}
    for i, p in enumerate(ref_paths):
        if p in footprints:
            by_mosaic.setdefault(footprints[p].mosaic, []).append(i)
    boxes = {
        m: (
            np.array(idx),
            np.array(
                [
                    [
                        footprints[ref_paths[i]].x0,
                        footprints[ref_paths[i]].y0,
                        footprints[ref_paths[i]].size,
                    ]
                    for i in idx
                ]
            ),
        )
        for m, idx in by_mosaic.items()
    }
    for q, p in enumerate(query_paths):
        f = footprints.get(p)
        if f is None or f.mosaic not in boxes:
            continue
        idx, b = boxes[f.mosaic]
        hit = (
            (b[:, 0] < f.x0 + f.size)
            & (f.x0 < b[:, 0] + b[:, 2])
            & (b[:, 1] < f.y0 + f.size)
            & (f.y0 < b[:, 1] + b[:, 2])
        )
        out[q] = sorted(set(out[q]) | set(idx[hit].tolist()))
    return out


class SubspaceModel:
    """SubspaceAD (Lendering et al., 2026): normal patches live near a PCA subspace.

    It keeps the mean and the top principal directions that explain `variance` of the
    reference patches, and scores a patch by its squared distance to that subspace.
    Nothing is matched patch to patch, so there is no own-tile problem to work
    around, and one tile's patches barely move a covariance built from thousands.
    """

    def __init__(self, features: torch.Tensor, variance: float = 0.99):
        if not 0 < variance <= 1:
            raise ValueError(f"--subspace_variance has to be in (0, 1], got {variance}")
        x = features.double()
        self.mean = x.mean(dim=0)
        cov = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        for start in range(0, x.shape[0], 1 << 16):
            c = x[start : start + (1 << 16)] - self.mean
            cov += c.T @ c
        cov /= max(1, x.shape[0] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
        eigvals, eigvecs = eigvals.flip(0).clamp_min(0), eigvecs.flip(1)
        share = eigvals.cumsum(0) / eigvals.sum()
        self.rank = min(int((share < variance).sum().item()) + 1, share.numel())
        self.basis = eigvecs[:, : self.rank].float()
        self.mean = self.mean.float()
        logger.info(
            "Subspace keeps %d of %d directions for %.1f%% of the variance",
            self.rank,
            x.shape[1],
            100 * variance,
        )

    def __len__(self) -> int:
        return self.basis.shape[0]

    @torch.inference_mode()
    def distances(self, queries: torch.Tensor, *_unused) -> torch.Tensor:
        """Squared residual of each (N, D) query off the subspace. It takes, and ignores,
        the bank's k and exclusion arguments so score_queries can call either."""
        c = queries.float() - self.mean
        return (c.square().sum(dim=1) - (c @ self.basis).square().sum(dim=1)).clamp_min(0)


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
    bank: MemoryBank | SubspaceModel,
    query_paths: list[Path],
    query_root: Path,
    maps_dir: Path | None,
    *,
    exclude: list[list[int]] | None = None,
    n_ref: int = 0,
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

    `exclude` lists, for each query tile, the indices of the `n_ref` reference tiles it
    must not be matched against: itself, and any tile sharing pixels with it.
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
    grid = backbone.grid_size
    scores: dict[str, float] = {}
    seconds: dict[str, float] = {}

    start = 0
    for keys, images in tqdm(loader, desc="Scoring query tiles"):
        started = time.perf_counter()
        rows = excluded = None
        batch_exclude = exclude[start : start + len(keys)] if exclude else []
        if any(batch_exclude):
            excluded = torch.zeros((len(keys), max(n_ref, 1)), dtype=torch.bool)
            for i, refs in enumerate(batch_exclude):
                excluded[i, refs] = True
            excluded = excluded.to(backbone.device)
            rows = torch.arange(len(keys), device=backbone.device)
            rows = rows.repeat_interleave(backbone.n_patches)
        start += len(keys)

        feats = backbone.embed(images)
        flat = feats.reshape(-1, feats.shape[-1])
        keep = background_mask(feats, grid) if masking else None
        if keep is None:
            dists = bank.distances(flat, k_neighbors, rows, excluded)
        else:
            valid = keep.reshape(-1)
            dists = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
            sub_rows = rows[valid] if rows is not None else None
            dists[valid] = bank.distances(flat[valid], k_neighbors, sub_rows, excluded)

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
    footprints: dict[Path, Footprint] | None = None,
    detector: str = "knn",
    subspace_variance: float = 0.99,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Build the bank from `ref_paths` and score `query_paths` against it.

    Returns the scores, the seconds spent on the bank and the seconds spent per tile.
    A query path that is also in `ref_paths` is scored against the other reference
    tiles only, and with `footprints` against the ones it shares no pixels with
    either. The bank build runs the model many times before any timing starts, so
    the CUDA kernels are warm by the time the scoring is timed.
    """
    if mask_ref_images and not masking:
        logger.warning("--mask_ref_images only works with --masking, so it is ignored here")
    started = time.perf_counter()
    bank = build_memory_bank(
        backbone,
        ref_paths,
        ref_root,
        masking=masking and mask_ref_images,
        rotation=rotation,
        # The subspace is fitted to the raw features, which L2_normalized would rescale.
        metric=knn_metric if detector == "knn" else "L2",
        max_patches=max_bank_patches,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    if detector == "subspace":
        bank = SubspaceModel(bank.features, subspace_variance)
    elif detector != "knn":
        raise ValueError(f"Unknown detector {detector!r}, expected knn or subspace")
    _sync(backbone.device)
    memorybank_sec = time.perf_counter() - started

    scores, seconds = score_queries(
        backbone,
        bank,
        query_paths,
        query_root,
        Path(out_dir) / f"anomaly_maps/seed={seed}",
        exclude=excluded_refs(query_paths, ref_paths, footprints),
        n_ref=len(ref_paths),
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
