"""Patch embeddings cached on disk, so each tile goes through the backbone once.

The two-pass run scores every tile twice and builds each pass's memory bank from
tiles it also scores, so without the cache a tile can be embedded four times. A
cached tile is named by a hash of its bytes and the backbone settings. Editing a tile
or changing the model just misses the cache, and the benchmark's copy of a dataset
reuses every tile it didn't plant a source in.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

from jwst_peculiar.data import relative_key, tile_loader

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from jwst_peculiar.backbones import Backbone

# float16 tops out at 65504, and a feature past that is stored as float32 instead.
_HALF_MAX = 6e4
# What a torch model's from_pretrained reads weights from, sharded or whole.
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _local_weights(model_name: str) -> list[str]:
    """Hash the weights of a checkpoint folder, as its name doesn't say what's in it.

    A fine-tune exported again to the same folder would otherwise find the features
    its old weights made. A Hub name gives nothing, so its cached tiles keep their
    names.

    Returns:
        One digest per weight file from_pretrained can load, in name order.
    """
    out: list[str] = []
    for path in sorted(Path(model_name).glob("*")):
        if path.suffix not in _WEIGHT_SUFFIXES:
            continue
        stat = path.stat()
        out.append(_digest(path, stat.st_size, stat.st_mtime_ns))
    return out


@functools.cache
def _digest(path: Path, _size: int, _mtime_ns: int) -> str:
    # A bench opens the cache once a run, and 1.2 GB takes a couple of seconds to
    # hash, so a file that hasn't changed is hashed once a process.
    with path.open("rb") as f:
        return hashlib.file_digest(f, "blake2b").hexdigest()


class FeatureCache:
    """One .npy of (n_patches, dims) per tile under `root`.

    About 1.6 MB a tile for ViT-L at 448 px.
    """

    def __init__(self, root: Path, backbone: Backbone) -> None:
        self.root = Path(root)
        self._salt = "|".join(
            str(v)
            for v in (
                backbone.name,
                backbone.resolution,
                backbone.layers,
                backbone.dtype,
                backbone.compiled,
                backbone.fast_accumulate,
                tuple(backbone.mean),
                tuple(backbone.std),
                *_local_weights(backbone.name),
            )
        ).encode()

    def path(self, tile: Path) -> Path:
        """Return where a tile's features go, named for its bytes and the model."""
        digest = hashlib.sha1(self._salt, usedforsecurity=False)
        digest.update(Path(tile).read_bytes())
        name = digest.hexdigest()
        return self.root / name[:2] / f"{name}.npy"

    @staticmethod
    def save(path: Path, feats: npt.NDArray[Any]) -> None:
        """Save one tile's features, as float16 when they fit."""
        if np.abs(feats).max() < _HALF_MAX:
            feats = feats.astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside the target and renamed, so an interrupted run never leaves a
        # half-written file that a later run would trust.
        tmp = path.with_name(path.stem + ".partial.npy")
        np.save(tmp, feats)
        tmp.replace(path)


@dataclass(frozen=True)
class EmbedOptions:
    """How tiles are read and embedded."""

    batch_size: int = 8
    num_workers: int = 4
    cache: FeatureCache | None = None


def embedded(
    backbone: Backbone,
    paths: list[Path],
    root: Path,
    *,
    options: EmbedOptions,
    desc: str = "Embedding tiles",
) -> Iterator[tuple[list[str], torch.Tensor]]:
    """Embed the tiles a batch at a time, in `paths` order.

    Yields:
        Each batch's tile keys and (B, n_patches, D) float32 features.
    """
    batch_size = max(options.batch_size, backbone.min_batch)
    cache = options.cache
    if cache is None:
        loader = tile_loader(
            paths,
            root,
            backbone,
            batch_size=batch_size,
            num_workers=options.num_workers,
        )
        for keys, images in tqdm(loader, desc=desc, leave=False):
            yield list(keys), backbone.embed(images)
        return

    files = [cache.path(p) for p in paths]
    missing = [i for i, f in enumerate(files) if not f.is_file()]
    if missing:
        loader = tile_loader(
            [paths[i] for i in missing],
            root,
            backbone,
            batch_size=batch_size,
            num_workers=options.num_workers,
        )
        slots = iter(missing)
        for _, images in tqdm(
            loader, desc=f"{desc} ({len(missing)} not cached)", leave=False
        ):
            for feats in backbone.embed(images).cpu().numpy():
                cache.save(files[next(slots)], feats)

    for start in range(0, len(paths), batch_size):
        idx = range(start, min(start + batch_size, len(paths)))
        stack = np.stack([np.load(files[i]) for i in idx]).astype(np.float32)
        yield (
            [relative_key(paths[i], root) for i in idx],
            torch.from_numpy(stack).to(backbone.device),
        )
