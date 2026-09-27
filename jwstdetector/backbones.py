"""Loading DINO backbones and reading patch embeddings out of them.

Everything loads through transformers from the Hugging Face Hub. DINOv3 is the one
worth reaching for, as its dense features hold up well (Gram anchoring) and its
SAT-493M checkpoints were trained on overhead imagery, which a JWST mosaic tile looks
a lot more like than it does a web photo. DINOv3 weights are gated, though, so accept
the licence on the model page and run `hf auth login` once. DINOv2 isn't gated.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as tnf

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The torch.hub names from the DINOv2 repo still work and map to the same weights on the Hub.
_DINOV2_HUB_NAME = re.compile(r"dinov2_vit([sblg])14(_reg)?$")
_DINOV2_SIZES = {"s": "small", "b": "base", "l": "large", "g": "giant"}


def hub_repo(model_name: str) -> str:
    """The Hugging Face repo id or local folder for a --model_name."""
    if "/" in model_name or "\\" in model_name:
        return model_name
    m = _DINOV2_HUB_NAME.match(model_name)
    if m:
        size = _DINOV2_SIZES[m.group(1)]
        return f"facebook/dinov2-with-registers-{size}" if m.group(2) else f"facebook/dinov2-{size}"
    if model_name.startswith(("dinov2-", "dinov3-")):
        return f"facebook/{model_name}"
    raise ValueError(
        f"Unknown model name {model_name!r}. Use a dinov3-* or dinov2_* name, or a Hub repo id."
    )


@dataclass(frozen=True)
class TileTransform:
    """RGB tile to a normalised (3, R, R) tensor.

    It's a class rather than a closure so DataLoader workers can pickle it on Windows.
    """

    resolution: int
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        size = (self.resolution, self.resolution)
        if img.size != size:
            img = img.resize(size, Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        return (x - mean) / std


@dataclass
class Backbone:
    """A loaded model and what the pipeline needs to feed it and read it."""

    name: str
    model: torch.nn.Module
    forward_patches: Callable[[torch.Tensor], torch.Tensor]
    patch_size: int
    resolution: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    device: torch.device
    autocast_dtype: torch.dtype | None

    @property
    def grid_size(self) -> tuple[int, int]:
        n = self.resolution // self.patch_size
        return (n, n)

    @property
    def n_patches(self) -> int:
        h, w = self.grid_size
        return h * w

    def transform(self) -> TileTransform:
        return TileTransform(self.resolution, self.mean, self.std)

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Patch embeddings of a (B, 3, R, R) batch, as (B, n_patches, D) float32 on the device."""
        batch = batch.to(self.device, non_blocking=True)
        if self.autocast_dtype is None:
            return self.forward_patches(batch).float()
        with torch.autocast(self.device.type, dtype=self.autocast_dtype):
            return self.forward_patches(batch).float()


def _snap_resolution(resolution: int, patch_size: int) -> int:
    snapped = max(patch_size, (resolution // patch_size) * patch_size)
    if snapped != resolution:
        logger.warning(
            "Resolution %d isn't a multiple of the %d px patch, so using %d",
            resolution,
            patch_size,
            snapped,
        )
    return snapped


def _normalisation(repo: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    # Read straight from the processor config, since the satellite checkpoints have
    # their own constants and the DINOv3 processor class won't load without torchvision.
    from huggingface_hub import hf_hub_download

    path = Path(repo) / "preprocessor_config.json"
    if not path.is_file():
        path = Path(hf_hub_download(repo, "preprocessor_config.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    mean = tuple(config.get("image_mean", IMAGENET_MEAN))
    std = tuple(config.get("image_std", IMAGENET_STD))
    return mean, std


def get_backbone(
    model_name: str,
    device: str | torch.device,
    resolution: int = 448,
    *,
    autocast: bool = True,
) -> Backbone:
    """Load a backbone by name, dinov3-*, dinov2_*, a Hub repo id or a local checkpoint folder."""
    from transformers import AutoModel

    device = torch.device(device)
    repo = hub_repo(model_name)
    try:
        model = AutoModel.from_pretrained(repo).eval().to(device)
    except OSError as exc:
        raise RuntimeError(
            f"Couldn't load {repo}. If it's a DINOv3 model, the weights are gated, so accept "
            f"the licence on https://huggingface.co/{repo} and run `hf auth login`, or use "
            "dinov2_vitl14, which isn't gated."
        ) from exc
    mean, std = _normalisation(repo)

    # The CLS token and any register tokens sit in front of the patch tokens.
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        return model(pixel_values=x).last_hidden_state[:, n_prefix:, :]

    patch_size = model.config.patch_size
    patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)

    dtype = None
    if autocast and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    backbone = Backbone(
        name=model_name,
        model=model,
        forward_patches=forward_patches,
        patch_size=patch_size,
        resolution=_snap_resolution(resolution, patch_size),
        mean=mean,
        std=std,
        device=device,
        autocast_dtype=dtype,
    )
    logger.info(
        "Loaded %s on %s, %d px patches at %d px, a %dx%d grid, autocast %s",
        repo,
        device,
        backbone.patch_size,
        backbone.resolution,
        *backbone.grid_size,
        dtype,
    )
    return backbone


def _first_components(feats: torch.Tensor, q: int) -> torch.Tensor:
    """(B, N, D) patch embeddings projected onto their top `q` principal components."""
    centred = feats - feats.mean(dim=1, keepdim=True)
    # pca_lowrank wants a little headroom above the rank it's asked for.
    rank = min(q + 4, centred.shape[1], centred.shape[2])
    _, _, v = torch.pca_lowrank(centred, q=rank, center=False)
    return centred @ v[..., :q]


@torch.inference_mode()
def background_mask(
    feats: torch.Tensor,
    grid_size: tuple[int, int],
    sigma: float = 0.5,
    *,
    border: float = 0.2,
    kernel: int = 3,
) -> torch.Tensor:
    """A (B, N) bool mask of the patches in a (B, N, D) batch that are worth scoring.

    It thresholds the first principal component of the patch embeddings and keeps
    whichever sign covers the centre of the tile, since the component's sign is
    arbitrary. The cut is `sigma` standard deviations of that component rather than
    AnomalyDINO's fixed 10, which was tuned to DINOv2's feature scale and masks
    out the whole tile on a backbone with smaller embeddings, DINOv3 included.
    """
    b = feats.shape[0]
    h, w = grid_size
    pc1 = _first_components(feats, 1).squeeze(-1)  # (B, N), already mean-centred
    cut = sigma * pc1.std(dim=1, keepdim=True)

    mask = (pc1 > cut).view(b, h, w)
    y0, y1 = int(h * border), int(h * (1 - border))
    x0, x1 = int(w * border), int(w * (1 - border))
    centre = mask[:, y0:y1, x0:x1]
    flipped = (-pc1 > cut).view(b, h, w)
    keep_flipped = centre.flatten(1).float().mean(1) <= 0.35
    mask = torch.where(keep_flipped[:, None, None], flipped, mask)

    # Dilate, then close, which fills pinholes and grows the mask a little.
    m = mask.unsqueeze(1).float()
    pad = kernel // 2
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = -tnf.max_pool2d(-m, kernel, stride=1, padding=pad)
    mask = (m.squeeze(1) > 0.5).flatten(1)

    # A tile of plain sky has no foreground at all. Scoring all of it beats dropping
    # it, which could leave the memory bank empty.
    return torch.where(mask.any(dim=1, keepdim=True), mask, torch.ones_like(mask))


@torch.inference_mode()
def embedding_rgb(feats: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Top 3 principal components of (N, D) patch embeddings as an (H, W, 3) image in [0, 1]."""
    reduced = _first_components(feats.unsqueeze(0), 3).squeeze(0)
    lo, hi = reduced.min(), reduced.max()
    reduced = (reduced - lo) / (hi - lo + 1e-12)
    return reduced.view(*grid_size, 3)
