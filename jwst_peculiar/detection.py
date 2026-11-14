"""Patch-level nearest-neighbour anomaly detection, adapted from AnomalyDINO.

A memory bank holds patch embeddings from the reference tiles, and each query tile is
scored by how far its patches sit from their nearest neighbours in that bank. Every
bank patch remembers which reference tile it came from, because the label-free runner
scores tiles that are also in its reference set, and a tile matched against its own
patches scores zero however strange it is.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
import torch
from torch.nn import functional as tnf
from tqdm import tqdm

from jwst_peculiar.backbones import Backbone, background_mask
from jwst_peculiar.data import owned_patches, tile_loader
from jwst_peculiar.features import EmbedOptions, FeatureCache, embedded
from jwst_peculiar.scoring import aggregate_score

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import numpy.typing as npt

logger = logging.getLogger(__name__)

METRICS = ("L2", "L2_normalized")

# Caps the (queries x bank) distance matrix at about 1 GB of float32. The search is
# bound by writing and reading that matrix, and blocks of 256 queries against a
# million patches ran twice as fast as blocks of 64.
_MAX_DISTANCE_ELEMENTS = 256 << 20
# Nearest bank patches searched for --places matches in separate places. A patch of
# ordinary sky finds them all among its first few.
_PLACES_CANDIDATES = 64
# Bank patches the principal directions are fitted on, and the chunk they're dropped
# in.
_DROP_SAMPLE = 100_000
# A bench runs the same bank against the same tiles 7 times a seed, and only the
# planted tiles change between runs. So a process keeps its last bank, and the
# distances of every tile it has scored against it, keyed by the tile's features.
# The reference tiles must not change on disk while the process runs.
_REUSED: dict[str, Any] = {}


def dihedral(x: torch.Tensor) -> list[torch.Tensor]:
    """Take the eight square symmetries of a batch of tiles.

    They are exact, where rotating by an arbitrary angle resamples the pixels and folds
    a reflected border into the corners, and sky has no preferred orientation anyway.

    Returns:
        The eight views, each the shape of `x`.
    """
    out: list[torch.Tensor] = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        out.append(r)
        out.append(torch.flip(r, dims=(3,)))
    return out


def _mean_of_nearest(d: torch.Tensor, k: int) -> torch.Tensor:
    if k == 1:
        # amin skips the sort topk sets up, which saves about a quarter of the time.
        nearest = d.amin(dim=1)
        return torch.where(torch.isfinite(nearest), nearest, torch.nan)
    top = d.topk(k, dim=1, largest=False).values
    finite = torch.isfinite(top)
    total = torch.where(finite, top, torch.zeros_like(top)).sum(dim=1)
    return total / finite.sum(dim=1)  # 0 / 0 is NaN, which is the point


class MemoryBank:
    """Reference patch embeddings, the tile each came from, and an exact kNN search.

    It takes ownership of `features` and normalises them in place for L2_normalized,
    so a 4 GB bank doesn't briefly need 8.
    """

    def __init__(
        self, features: torch.Tensor, metric: str, owners: torch.Tensor | None = None
    ) -> None:
        if metric not in METRICS:
            msg = f"Unknown knn_metric {metric!r}, expected one of {METRICS}"
            raise ValueError(msg)
        self.metric = metric
        if metric == "L2_normalized":
            # In fp32 a chunk at a time, since the squares of a 1024-dim fp16 row can
            # pass 65504.
            for start in range(0, features.shape[0], _DROP_SAMPLE):
                part = features[start : start + _DROP_SAMPLE].float()
                features[start : start + _DROP_SAMPLE] = tnf.normalize(part, dim=1)
        self.features = features
        self.owners = owners
        # (n_ref, n_ref) bool of reference tiles that share sky, itself included. Set it
        # for distances(places > 1).
        self.near: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.dropped: torch.Tensor | None = None

    @torch.inference_mode()
    def drop_directions(self, n: int, seed: int = 0) -> None:
        """Drop the mean and top `n` principal directions from the bank and queries.

        Every patch shares the mean, and a few directions carry most of the variance,
        such as overall brightness or noise level. Under cosine distance they make
        every patch look alike and swamp the patches that differ in shape.
        """
        generator = torch.Generator().manual_seed(seed)
        pick = torch.randperm(len(self), generator=generator)[:_DROP_SAMPLE].to(
            self.features.device
        )
        sample = self.features[pick].float()
        self.mean = sample.mean(dim=0)
        _, _, v = torch.linalg.svd(sample - self.mean, full_matrices=False)
        self.dropped = v[:n].contiguous()
        for start in range(0, len(self), _DROP_SAMPLE):
            self.features[start : start + _DROP_SAMPLE] = self._project(
                self.features[start : start + _DROP_SAMPLE]
            )

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropped is None or self.mean is None:
            return x
        x = x.float() - self.mean
        x -= (x @ self.dropped.T) @ self.dropped
        return tnf.normalize(x, dim=1) if self.metric == "L2_normalized" else x

    def __len__(self) -> int:
        """Return the number of patches in the bank."""
        return self.features.shape[0]

    @torch.inference_mode()
    def distances(
        self,
        queries: torch.Tensor,
        k: int,
        *,
        rows: torch.Tensor | None = None,
        excluded: torch.Tensor | None = None,
        places: int = 1,
    ) -> torch.Tensor:
        """Find the mean distance from each of the (N, D) queries to its k nearest.

        `rows` gives each query patch a row of the (R, n_ref_tiles) bool `excluded`, and
        bank patches from the tiles marked there are skipped. A patch left with fewer
        than k candidates averages the ones it has, and one with none gets NaN.

        With `places` above 1 it's the distance to the nearest match in the `places`-th
        separate place instead, where two places are separate when their tiles share no
        sky. A rare object with a twin or two elsewhere in the reference then still
        scores high, as its twins can only fill some of the places. It needs `near`.

        Returns:
            One distance per query.

        Raises:
            ValueError: If `places` is above 1 and the bank has no owners or `near`.
        """
        if places > 1 and (self.owners is None or self.near is None):
            msg = "places > 1 needs the bank's owners and near set"
            raise ValueError(msg)
        k = max(1, min(k, len(self)))
        cosine = self.metric == "L2_normalized"
        if cosine:
            queries = tnf.normalize(queries, dim=1)
        queries = self._project(queries).to(self.features.dtype)
        bank_sq = None if cosine else self.features.square().sum(dim=1)
        if rows is None or excluded is None or self.owners is None:
            rows = excluded = None
        else:
            # Gathered once per call, so each chunk only indexes its own rows.
            excluded = excluded[:, self.owners.long()]

        chunk = max(1, _MAX_DISTANCE_ELEMENTS // len(self))
        out: list[torch.Tensor] = []
        for start in range(0, queries.shape[0], chunk):
            d = self._chunk_distances(queries[start : start + chunk], bank_sq)
            if rows is not None and excluded is not None:
                d.masked_fill_(
                    excluded[rows[start : start + chunk].long()], float("inf")
                )
            if places > 1:
                out.append(self._place_distance(d, places))
            else:
                out.append(_mean_of_nearest(d, k))
        return torch.cat(out)

    def _chunk_distances(
        self, q: torch.Tensor, bank_sq: torch.Tensor | None
    ) -> torch.Tensor:
        # Both branches build the distances in one (chunk, bank) matrix in place.
        # torch.cdist needs several of those at once and runs out of memory on an 8 GB
        # card with a million-patch ViT-L bank.
        if self.features.dtype == torch.float32:
            d = torch.mm(q, self.features.T)
        else:
            d = torch.mm(q, self.features.T, out_dtype=torch.float32)
        if bank_sq is None:
            # Unit vectors, so half the squared L2 distance is `1 - cos`, which is the
            # scale AnomalyDINO reports.
            d.neg_().add_(1.0)
        else:
            d.mul_(-2.0).add_(q.square().sum(dim=1, keepdim=True)).add_(bank_sq)
            d.clamp_(min=0.0).sqrt_()
        return d

    def _place_distance(self, d: torch.Tensor, places: int) -> torch.Tensor:
        """Find the match that fills the `places`-th separate place in each row.

        Each row of (Q, bank) distances is walked out from the nearest. A row whose
        candidates run out first gets the farthest one it looked at, which is the
        closest the true answer could be. One with no candidates gets NaN.

        Returns:
            One distance per row.

        Raises:
            ValueError: If the bank has no owners or `near`.
        """
        if self.owners is None or self.near is None:
            msg = "places > 1 needs the bank's owners and near set"
            raise ValueError(msg)
        vals, idx = d.topk(min(_PLACES_CANDIDATES, d.shape[1]), dim=1, largest=False)
        tiles = self.owners[idx].long()
        q = d.shape[0]
        blocked = torch.zeros(
            (q, self.near.shape[0]), dtype=torch.bool, device=d.device
        )
        found = torch.zeros(q, dtype=torch.long, device=d.device)
        result = torch.full((q,), float("nan"), device=d.device)
        for j in range(vals.shape[1]):
            t, v = tiles[:, j], vals[:, j]
            new = (
                torch.isfinite(v)
                & ~blocked.gather(1, t[:, None]).squeeze(1)
                & (found < places)
            )
            found += new
            result = torch.where(new & (found == places), v, result)
            blocked |= self.near[t] & new[:, None]
        last = torch.where(torch.isfinite(vals), vals, torch.nan).nanquantile(
            1.0, dim=1
        )
        return torch.where(found < places, last, result)


@dataclass(frozen=True)
class Footprint:
    """Where a tile was cut from, in mosaic pixels, and its centre on the sky."""

    mosaic: str
    x0: int
    y0: int
    size: int
    ra: float = float("nan")
    dec: float = float("nan")
    # Half the tile's diagonal, so two tiles can only overlap when their centres
    # are closer than the sum of theirs.
    radius_deg: float = float("nan")


def excluded_refs(
    query_paths: list[Path],
    ref_paths: list[Path],
    footprints: dict[Path, Footprint] | None = None,
) -> list[list[int]]:
    """List, for each query tile, the reference tiles it must not be matched against.

    That is the tile itself and every reference tile overlapping it on the sky. Tiles
    overlap by design (half a crop by default, and the crop flush with each far edge
    can share almost all of its pixels), and an object in the shared strip would
    otherwise find itself in the bank and score as ordinary.

    Returns:
        The indices into `ref_paths`, one list per query tile.
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
            np.array([
                [
                    footprints[ref_paths[i]].x0,
                    footprints[ref_paths[i]].y0,
                    footprints[ref_paths[i]].size,
                ]
                for i in idx
            ]),
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
    _exclude_on_sky(out, query_paths, ref_paths, footprints=footprints)
    return out


def _unit(
    ra: npt.NDArray[np.float64], dec: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], axis=-1)


def _exclude_on_sky(
    out: list[list[int]],
    query_paths: list[Path],
    ref_paths: list[Path],
    *,
    footprints: dict[Path, Footprint],
) -> None:
    """Add reference tiles from other mosaics that may share sky with each query.

    Separate visits or fields can overlap on the sky, as neighbouring survey
    pointings do, and pixel boxes only compare tiles of one mosaic.
    """
    refs = [
        (i, footprints[p])
        for i, p in enumerate(ref_paths)
        if p in footprints and np.isfinite(footprints[p].radius_deg)
    ]
    if not refs:
        return
    idx = np.array([i for i, _ in refs])
    mosaic = np.array([f.mosaic for _, f in refs])
    vec = _unit(np.array([f.ra for _, f in refs]), np.array([f.dec for _, f in refs]))
    radius = np.array([f.radius_deg for _, f in refs])
    for q, p in enumerate(query_paths):
        f = footprints.get(p)
        if f is None or not np.isfinite(f.radius_deg):
            continue
        sep = np.degrees(
            np.arccos(np.clip(vec @ _unit(np.array(f.ra), np.array(f.dec)), -1, 1))
        )
        hit = (mosaic != f.mosaic) & (sep < radius + f.radius_deg)
        if hit.any():
            out[q] = sorted(set(out[q]) | set(idx[hit].tolist()))


class SubspaceModel:
    """SubspaceAD (Lendering et al., 2026): normal patches live near a PCA subspace.

    It keeps the mean and the top principal directions that explain `variance` of the
    reference patches, and scores a patch by its squared distance to that subspace.
    Nothing is matched patch to patch, so there is no own-tile problem to work
    around, and one tile's patches barely move a covariance built from thousands.
    """

    def __init__(self, features: torch.Tensor, variance: float = 0.99) -> None:
        if not 0 < variance <= 1:
            msg = f"--subspace_variance has to be in (0, 1], got {variance}"
            raise ValueError(msg)
        # Only a chunk at a time goes to float64, as a million ViT-L patches are 8 GB
        # in it.
        x, step = features, 1 << 16
        starts = range(0, x.shape[0], step)
        total = torch.zeros(x.shape[1], dtype=torch.float64, device=x.device)
        for start in starts:
            total += x[start : start + step].double().sum(dim=0)
        self.mean = total / max(1, x.shape[0])
        cov = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=x.device)
        for start in starts:
            c = x[start : start + step].double() - self.mean
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
        """Return the number of dimensions the patches have."""
        return int(self.basis.shape[0])

    @torch.inference_mode()
    def distances(self, queries: torch.Tensor) -> torch.Tensor:
        """Return the squared residual of each (N, D) query off the subspace."""
        c = queries.float() - self.mean
        residual = c.square().sum(dim=1) - torch.matmul(c, self.basis).square().sum(
            dim=1
        )
        return residual.clamp_min(0)


def _last_of_each(slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the unique values of `slots` and where the last of each one is."""
    s = slots.numpy()
    unique, first_from_end = np.unique(s[::-1], return_index=True)
    return torch.from_numpy(unique), torch.from_numpy(len(s) - 1 - first_from_end)


@dataclass(frozen=True)
class BankSettings:
    """Which reference patches go in the memory bank, and how they're stored."""

    masking: bool = False
    rotation: bool = False
    metric: str = "L2_normalized"
    max_patches: int = 1_000_000
    seed: int = 0
    dtype: torch.dtype = torch.float32


class _Reservoir:
    """A uniform random sample of up to `capacity` patches, kept as they stream past.

    It's Algorithm R, so the bank is allocated once at its final size and fills up to
    the cap however many patches masking leaves.
    """

    def __init__(self, capacity: int, dtype: torch.dtype, seed: int) -> None:
        self.capacity, self.dtype = capacity, dtype
        self.generator = torch.Generator().manual_seed(seed)
        self.features: torch.Tensor | None = None
        self.owners: torch.Tensor | None = None
        self.filled = self.seen = 0

    def add(self, feats: torch.Tensor, ids: torch.Tensor) -> None:
        """Offer (n, D) patches from the tiles `ids` names to the sample."""
        if self.features is None or self.owners is None:
            self.features = torch.empty(
                (self.capacity, feats.shape[1]), device=feats.device, dtype=self.dtype
            )
            self.owners = torch.empty(
                self.capacity, dtype=torch.int32, device=feats.device
            )

        n = feats.shape[0]
        take = min(self.capacity - self.filled, n)
        self.features[self.filled : self.filled + take] = feats[:take]
        self.owners[self.filled : self.filled + take] = ids[:take]
        self.filled += take
        if take < n:
            slots, src = self._draw(n, take, feats.device)
            # Indexed assignment won't cast the way slicing does, so an fp16 bank
            # needs the fp32 features cast first.
            self.features[slots] = feats[src].to(self.dtype)
            self.owners[slots] = ids[src]
        self.seen += n

    def _draw(
        self, n: int, take: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Patch i, counting from 0 over every patch seen, draws a slot in [0, i] and
        # replaces what's there when the slot is inside the bank.
        i = torch.arange(self.seen + take, self.seen + n, dtype=torch.float64)
        slots = torch.rand(n - take, generator=self.generator, dtype=torch.float64) * (
            i + 1
        )
        slots = slots.long()
        hits = torch.nonzero(slots < self.capacity).squeeze(1)
        # When two patches draw the same slot the later one wins, as it would one at a
        # time, and deduplicating keeps the GPU write deterministic.
        slots, last = _last_of_each(slots[hits])
        src = (hits[last] + take).to(device)
        return slots.to(device), src


def _bank_batches(
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    *,
    rotation: bool,
    options: EmbedOptions,
) -> Iterator[tuple[int, Iterable[torch.Tensor]]]:
    # The eight views aren't cached, as they'd take eight times the disk.
    if not rotation:
        for keys, feats in embedded(
            backbone, ref_paths, ref_root, options=options, desc="Building memory bank"
        ):
            yield len(keys), (feats,)
        return
    loader = tile_loader(
        ref_paths,
        ref_root,
        backbone,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
    )
    for _, images in tqdm(loader, desc="Building memory bank", leave=False):
        yield len(images), (backbone.embed(v) for v in dihedral(images))


def _bank_rows(
    feats: torch.Tensor,
    tile_ids: torch.Tensor,
    grid: tuple[int, int],
    *,
    settings: BankSettings,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch's patches to bank rows, with the tile each came from.

    Returns:
        The (n, D) rows and the n tile ids, with the masked-out patches gone.
    """
    if settings.masking:
        keep = background_mask(feats, grid).reshape(-1)
        rows, ids = feats.reshape(-1, feats.shape[-1])[keep], tile_ids[keep]
    else:
        rows, ids = feats.reshape(-1, feats.shape[-1]), tile_ids
    if settings.dtype != torch.float32:
        # Unit length before the cast, as hidden states from the middle of an LVD
        # model run past what fp16 holds.
        rows = tnf.normalize(rows, dim=1)
    return rows, ids


@torch.inference_mode()
def build_memory_bank(
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    *,
    settings: BankSettings,
    options: EmbedOptions,
) -> MemoryBank:
    """Embed the reference tiles and keep a uniform random sample of their patches.

    It keeps up to `settings.max_patches`, as 425 tiles at 672 px through DINOv3
    ViT-L with --rotation come to about 25 GB.

    Returns:
        The bank, with each patch's reference tile.

    Raises:
        RuntimeError: If no patches made it into the bank.
    """
    n_views = 8 if settings.rotation else 1
    capacity = min(settings.max_patches, len(ref_paths) * n_views * backbone.n_patches)
    reservoir = _Reservoir(capacity, settings.dtype, settings.seed)

    first_tile = 0
    for n_tiles, views in _bank_batches(
        backbone, ref_paths, ref_root, rotation=settings.rotation, options=options
    ):
        tile_ids = torch.arange(first_tile, first_tile + n_tiles, dtype=torch.int32)
        tile_ids = tile_ids.repeat_interleave(backbone.n_patches).to(backbone.device)
        first_tile += n_tiles
        for feats in views:
            reservoir.add(
                *_bank_rows(feats, tile_ids, backbone.grid_size, settings=settings)
            )

    features, owners, filled = reservoir.features, reservoir.owners, reservoir.filled
    if features is None or owners is None or filled == 0:
        msg = "The memory bank is empty. Is the reference set empty?"
        raise RuntimeError(msg)
    logger.info(
        "Memory bank holds %d of %d reference patches, %d dims each",
        filled,
        reservoir.seen,
        features.shape[1],
    )
    return MemoryBank(features[:filled], settings.metric, owners[:filled])


def _kept_or_built(  # noqa: PLR0913 the same arguments as build_memory_bank, plus keep
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    *,
    settings: BankSettings,
    options: EmbedOptions,
    keep: bool,
) -> MemoryBank:
    """Return the bank this process last built from the same tiles, or build one.

    Returns:
        The bank.
    """
    # The backbone is held as well, so a new one at a recycled id never matches.
    key = (tuple(str(p) for p in ref_paths), str(ref_root), settings)
    if keep and _REUSED.get("backbone") is backbone and _REUSED.get("key") == key:
        kept: MemoryBank = _REUSED["bank"]
        return kept
    _REUSED.clear()
    if backbone.device.type == "cuda":
        torch.cuda.empty_cache()
    bank = build_memory_bank(
        backbone, ref_paths, ref_root, settings=settings, options=options
    )
    if keep:
        _REUSED.update(backbone=backbone, key=key, bank=bank, dists={})
    return bank


@dataclass(frozen=True)
class Queries:
    """The tiles to score, and what each one is kept from.

    `exclude` lists, for each tile, the indices of the `n_ref` reference tiles it must
    not be matched against: itself, and any tile sharing pixels with it. `owned` gives,
    for each tile, the (rows, cols) patches it scores, from data.owned_patches. The rest
    still go in its map.
    """

    paths: list[Path]
    root: Path
    exclude: list[list[int]] | None = None
    n_ref: int = 0
    owned: list[npt.NDArray[np.bool_] | None] | None = None

    def exclusion(
        self, start: int, n: int, backbone: Backbone
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the excluded rows for a batch, and each patch's row, or two Nones."""
        batch_exclude = self.exclude[start : start + n] if self.exclude else []
        if not any(batch_exclude):
            return None, None
        excluded = torch.zeros((n, max(self.n_ref, 1)), dtype=torch.bool)
        for i, refs in enumerate(batch_exclude):
            excluded[i, refs] = True
        rows = torch.arange(n, device=backbone.device)
        return excluded.to(backbone.device), rows.repeat_interleave(backbone.n_patches)

    def owned_in(self, start: int, n: int) -> list[npt.NDArray[np.bool_] | None]:
        """Return the patches each tile in a batch scores, None for all of them."""
        if self.owned:
            return self.owned[start : start + n]
        return [None] * n


@dataclass(frozen=True)
class ScoreSettings:
    """How query patches are searched, scored and saved."""

    masking: bool = False
    k_neighbors: int = 1
    places: int = 1
    score_mode: str = "top1p"
    score_top_frac: float = 0.01
    score_quantile: float = 0.9995
    maps_dir: Path | None = None
    save_patch_dists: bool = True
    save_tiffs: bool = False
    reuse: dict[tuple[str, str], torch.Tensor] | None = None


def _dists2map() -> Callable[[npt.NDArray[Any], tuple[int, int]], npt.NDArray[Any]]:
    from jwst_peculiar.utils import dists2map  # noqa: PLC0415 utils loads matplotlib

    return dists2map


def _descriptors(
    feats: torch.Tensor,
    dists: torch.Tensor,
    owned: list[npt.NDArray[np.bool_] | None],
) -> list[npt.NDArray[np.float32]]:
    ranked = torch.nan_to_num(dists.reshape(feats.shape[0], -1), nan=float("-inf"))
    out: list[npt.NDArray[np.float32]] = []
    for i, mine in enumerate(owned):
        row = ranked[i]
        if mine is not None:
            row = row.masked_fill(
                ~torch.from_numpy(mine.reshape(-1)).to(row.device), -torch.inf
            )
        n_top = max(1, round(0.01 * (mine.sum() if mine is not None else row.numel())))
        out.append(feats[i, row.topk(int(n_top)).indices].mean(dim=0).cpu().numpy())
    return out


class _Scoring:
    """Scores, descriptors and maps for the query tiles, a batch at a time."""

    def __init__(
        self,
        backbone: Backbone,
        bank: MemoryBank | SubspaceModel,
        settings: ScoreSettings,
    ) -> None:
        self.backbone, self.bank, self.settings = backbone, bank, settings
        self.scores: dict[str, float] = {}
        self.descriptors: dict[str, npt.NDArray[np.float32]] = {}
        self.to_pixels = _dists2map() if settings.save_tiffs else None

    def _search(
        self,
        flat: torch.Tensor,
        rows: torch.Tensor | None,
        excluded: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(self.bank, SubspaceModel):
            return self.bank.distances(flat)
        return self.bank.distances(
            flat,
            self.settings.k_neighbors,
            rows=rows,
            excluded=excluded,
            places=self.settings.places,
        )

    def _distances(
        self,
        feats: torch.Tensor,
        rows: torch.Tensor | None,
        excluded: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flat = feats.reshape(-1, feats.shape[-1])
        if not self.settings.masking:
            return self._search(flat, rows, excluded), None
        keep = background_mask(feats, self.backbone.grid_size)
        valid = keep.reshape(-1)
        dists = torch.zeros(flat.shape[0], device=flat.device, dtype=torch.float32)
        sub_rows = rows[valid] if rows is not None else None
        dists[valid] = self._search(flat[valid], sub_rows, excluded)
        return dists, keep

    def _save_maps(self, key: str, grid: npt.NDArray[np.float32]) -> None:
        maps_dir = self.settings.maps_dir
        if maps_dir is None or not (
            self.settings.save_patch_dists or self.settings.save_tiffs
        ):
            return
        # Appending the suffix by hand, as with_suffix(".npy") on a stem full of RA and
        # Dec dots would cut it at the last dot.
        stem = maps_dir / str(Path(key).with_suffix(""))
        stem.parent.mkdir(parents=True, exist_ok=True)
        if self.settings.save_patch_dists:
            np.save(f"{stem}.npy", grid)
        if self.to_pixels is not None:
            side = self.backbone.resolution
            tifffile.imwrite(f"{stem}.tiff", self.to_pixels(grid, (side, side)))

    def _reused_distances(
        self,
        keys: list[str],
        feats: torch.Tensor,
        *,
        memo: dict[tuple[str, str], torch.Tensor],
        excluded: torch.Tensor | None,
    ) -> torch.Tensor:
        """Search only the tiles this bank hasn't seen with these features.

        Returns:
            The distances of every patch in the batch.
        """
        per_tile = feats.reshape(len(keys), -1, feats.shape[-1])
        # The number of neighbours and the excluded reference tiles go in the name, as
        # the bank outlives one run and the next can ask for others.
        nearest = f"{self.settings.k_neighbors}:"
        names = [
            (
                k,
                nearest
                + ("" if excluded is None else str(excluded[i].nonzero().tolist()))
                + hashlib.blake2b(per_tile[i].cpu().numpy().tobytes()).hexdigest(),
            )
            for i, k in enumerate(keys)
        ]
        todo = [i for i, name in enumerate(names) if name not in memo]
        if todo:
            sub_excluded = rows = None
            if excluded is not None:
                sub_excluded = excluded[todo]
                rows = torch.arange(len(todo), device=feats.device).repeat_interleave(
                    per_tile.shape[1]
                )
            found = self._search(
                per_tile[todo].reshape(-1, feats.shape[-1]), rows, sub_excluded
            ).reshape(len(todo), -1)
            for j, i in enumerate(todo):
                memo[names[i]] = found[j]
        return torch.cat([memo[name] for name in names])

    def add_batch(
        self,
        keys: list[str],
        feats: torch.Tensor,
        *,
        exclusion: tuple[torch.Tensor | None, torch.Tensor | None],
        owned: list[npt.NDArray[np.bool_] | None],
    ) -> None:
        """Score a batch of tiles and save their maps."""
        excluded, rows = exclusion
        memo = self.settings.reuse
        if memo is None or self.settings.masking:
            dists, keep = self._distances(feats, rows, excluded)
        else:
            dists, keep = (
                self._reused_distances(keys, feats, memo=memo, excluded=excluded),
                None,
            )
        descriptors = _descriptors(feats, dists, owned)
        grid = self.backbone.grid_size
        grids = dists.reshape(len(keys), *grid).cpu().numpy()
        keeps = (
            keep.reshape(len(keys), *grid).cpu().numpy() if keep is not None else None
        )

        for i, key in enumerate(keys):
            # Masked-out patches carry a distance of 0, which would drag a median or
            # mean down, so only the real patches are scored. The saved map keeps the
            # zeros so it still lines up with the tile.
            scored = keeps[i] if keeps is not None else np.ones(grid, dtype=bool)
            mine = owned[i]
            if mine is not None:
                scored = scored & mine
            self.scores[key] = aggregate_score(
                grids[i][scored],
                mode=self.settings.score_mode,
                top_frac=self.settings.score_top_frac,
                quantile=self.settings.score_quantile,
            )
            self.descriptors[key] = descriptors[i]
            self._save_maps(key, grids[i])


@torch.inference_mode()
def score_queries(
    backbone: Backbone,
    bank: MemoryBank | SubspaceModel,
    queries: Queries,
    *,
    settings: ScoreSettings,
    options: EmbedOptions,
) -> tuple[dict[str, float], dict[str, float], dict[str, npt.NDArray[np.float32]]]:
    """Score every query tile, and describe each by what made it score.

    The descriptor is the mean feature of the tile's most distant 1% of patches, so it
    describes whatever made the tile score high rather than the sky around it, which
    is what similarity search and learning from verdicts want to compare.

    Returns:
        The scores, the mean seconds per tile and a descriptor per tile.
    """
    scoring = _Scoring(backbone, bank, settings)
    start = 0
    started = time.perf_counter()
    for keys, feats in embedded(
        backbone,
        queries.paths,
        queries.root,
        options=options,
        desc="Scoring query tiles",
    ):
        n = len(keys)
        scoring.add_batch(
            keys,
            feats,
            exclusion=queries.exclusion(start, n, backbone),
            owned=queries.owned_in(start, n),
        )
        start += n

    # An average, as the feature cache embeds every missing tile before the first batch
    # comes back, which leaves no honest per-batch time to give each tile.
    _sync(backbone.device)
    per_tile = (time.perf_counter() - started) / max(len(scoring.scores), 1)
    seconds = dict.fromkeys(scoring.scores, per_tile)
    return scoring.scores, seconds, scoring.descriptors


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _choose_detector(
    bank: MemoryBank, detector: str, subspace_variance: float
) -> MemoryBank | SubspaceModel:
    if detector == "subspace":
        return SubspaceModel(bank.features, subspace_variance)
    if detector != "knn":
        msg = f"Unknown detector {detector!r}, expected knn or subspace"
        raise ValueError(msg)
    return bank


def _near(
    ref_paths: list[Path], footprints: dict[Path, Footprint] | None
) -> torch.Tensor:
    near = torch.zeros((len(ref_paths), len(ref_paths)), dtype=torch.bool)
    for i, js in enumerate(excluded_refs(ref_paths, ref_paths, footprints)):
        near[i, js] = True
    return near


def _owned(
    score_region: str,
    footprints: dict[Path, Footprint] | None,
    query_paths: list[Path],
    *,
    grid: tuple[int, int],
) -> list[npt.NDArray[np.bool_] | None] | None:
    if score_region not in ("owned", "middle", "all"):
        msg = f"Unknown score_region {score_region!r}, expected owned, middle or all"
        raise ValueError(msg)
    if score_region != "all" and footprints:
        mine = owned_patches(footprints, grid, middle=score_region == "middle")
        return [mine.get(p) for p in query_paths]
    return None


def run_anomaly_detection(  # noqa: PLR0913 the CLI passes every option through by name
    backbone: Backbone,
    ref_paths: list[Path],
    ref_root: Path,
    *,
    query_paths: list[Path],
    query_root: Path,
    out_dir: Path,
    masking: bool = False,
    mask_ref_images: bool = False,
    rotation: bool = False,
    knn_metric: str = "L2_normalized",
    k_neighbors: int = 1,
    places: int = 1,
    drop_directions: int = 0,
    bank_precision: str = "fp16",
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
    cache: FeatureCache | None = None,
    score_region: str = "owned",
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Build the bank from `ref_paths` and score `query_paths` against it.

    A query path that is also in `ref_paths` is scored against the other reference
    tiles only, and with `footprints` against the ones it shares no pixels with
    either. The bank build runs the model many times before any timing starts, so
    the CUDA kernels are warm by the time the scoring is timed. Each tile's descriptor
    goes to descriptors.npz in `out_dir`. With `score_region` "owned" and footprints,
    a tile only scores the patches no overlapping tile is more central to.

    Returns:
        The scores, the seconds spent on the bank and the seconds spent per tile.
    """
    if mask_ref_images and not masking:
        logger.warning(
            "--mask_ref_images only works with --masking, so it is ignored here"
        )
    options = EmbedOptions(batch_size=batch_size, num_workers=num_workers, cache=cache)
    started = time.perf_counter()
    bank = _kept_or_built(
        backbone,
        ref_paths,
        ref_root,
        settings=BankSettings(
            masking=masking and mask_ref_images,
            rotation=rotation,
            # The subspace is fitted to the raw features, which L2_normalized would
            # rescale.
            metric=knn_metric if detector == "knn" else "L2",
            max_patches=max_bank_patches,
            seed=seed,
            # fp16 halves the bank and runs the search on tensor cores. Only cosine
            # distances take it, as the squares in plain L2 lose too much.
            dtype=torch.float16
            if bank_precision == "fp16"
            and detector == "knn"
            and knn_metric == "L2_normalized"
            and backbone.device.type == "cuda"
            else torch.float32,
        ),
        options=options,
        # Only a bank nothing changes after it's built can be kept.
        keep=detector == "knn" and not drop_directions and places == 1,
    )
    model = _choose_detector(bank, detector, subspace_variance)
    if isinstance(model, MemoryBank) and drop_directions:
        model.drop_directions(drop_directions, seed)
    if isinstance(model, MemoryBank) and places > 1:
        model.near = _near(ref_paths, footprints).to(model.features.device)
    _sync(backbone.device)
    memorybank_sec = time.perf_counter() - started

    owned = _owned(score_region, footprints, query_paths, grid=backbone.grid_size)
    queries = Queries(
        query_paths,
        query_root,
        exclude=excluded_refs(query_paths, ref_paths, footprints),
        n_ref=len(ref_paths),
        owned=owned,
    )
    scores, seconds, descriptors = score_queries(
        backbone,
        model,
        queries,
        settings=ScoreSettings(
            masking=masking,
            k_neighbors=k_neighbors,
            places=places if detector == "knn" else 1,
            score_mode=score_mode,
            score_top_frac=score_top_frac,
            score_quantile=score_quantile,
            maps_dir=Path(out_dir) / f"anomaly_maps/seed={seed}",
            save_patch_dists=save_patch_dists,
            save_tiffs=save_tiffs,
            # None unless the bank was kept above.
            reuse=_REUSED.get("dists"),
        ),
        options=options,
    )
    keys = sorted(descriptors)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(out_dir) / "descriptors.npz",
        keys=np.array(keys),
        vectors=np.stack([descriptors[k] for k in keys]).astype(np.float32),
    )
    return scores, memorybank_sec, seconds
