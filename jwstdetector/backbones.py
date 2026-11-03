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
    layers: str = "last"

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
        if Path(repo).is_dir():
            logger.warning("%s has no preprocessor_config.json, so using ImageNet's", repo)
            return IMAGENET_MEAN, IMAGENET_STD
        path = Path(hf_hub_download(repo, "preprocessor_config.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    mean = tuple(config.get("image_mean", IMAGENET_MEAN))
    std = tuple(config.get("image_std", IMAGENET_STD))
    return mean, std


def parse_layers(spec: str, depth: int) -> list[int] | None:
    """Which hidden states to average for --layers, or None for the final output.

    "mid" is the band SubspaceAD found best on DINOv2-G (layers 22 to 28 of 40),
    scaled to this model's depth. "a-b" takes layers a to b counted from 1, and a
    comma list takes those layers.
    """
    if spec == "last":
        return None
    if spec == "mid":
        return list(range(round(depth * 22 / 40), round(depth * 28 / 40) + 1))
    try:
        if "-" in spec:
            a, b = (int(v) for v in spec.split("-"))
            picked = list(range(a, b + 1))
        else:
            picked = [int(v) for v in spec.split(",")]
    except ValueError:
        raise ValueError(
            f"--layers takes last, mid, a range like 14-20 or a list, not {spec!r}"
        ) from None
    if not picked or min(picked) < 1 or max(picked) > depth:
        raise ValueError(f"--layers {spec} is outside this model's 1 to {depth}")
    return picked


def get_backbone(
    model_name: str,
    device: str | torch.device,
    resolution: int = 448,
    *,
    autocast: bool = True,
    layers: str = "last",
) -> Backbone:
    """Load a backbone by name, dinov3-*, dinov2_*, a Hub repo id or a local checkpoint folder.

    "pixels" is no model at all, just each 16 px patch's raw RGB values, which runs the
    same search on pixels so the benchmark can show what the features add.
    """
    device = torch.device(device)
    if model_name == "pixels":
        size = 16

        def raw(x: torch.Tensor) -> torch.Tensor:
            return tnf.unfold(x, size, stride=size).transpose(1, 2)

        return Backbone(
            name=model_name,
            model=torch.nn.Identity(),
            forward_patches=raw,
            patch_size=size,
            resolution=_snap_resolution(resolution, size),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            device=device,
            autocast_dtype=None,
            layers=layers,
        )

    from transformers import AutoModel

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

    picked = parse_layers(layers, int(model.config.num_hidden_layers))

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        if picked is None:
            return model(pixel_values=x).last_hidden_state[:, n_prefix:, :]
        hidden = model(pixel_values=x, output_hidden_states=True).hidden_states
        return torch.stack([hidden[i][:, n_prefix:, :] for i in picked]).mean(dim=0)

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
        layers=layers,
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
    """(B, N, D) patch embeddings projected onto their top `q` principal components.

    It's an exact SVD. torch.pca_lowrank starts from a random matrix drawn from the
    global generator, so a tile's mask used to depend on how many draws came before it.
    """
    centred = feats.float() - feats.float().mean(dim=1, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return centred @ vh[..., :q, :].transpose(-1, -2)


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

    It thresholds the first principal component of the patch embeddings on both
    sides and keeps the side that crowds into the centre of the tile more than it
    covers the tile overall, which is where a tile's subject sits. Both sides are
    judged the same way, so the component's arbitrary sign can't change the answer.
    The cut is `sigma` standard deviations of that component rather than AnomalyDINO's
    fixed 10, which was tuned to DINOv2's feature scale and masks out the whole tile
    on a backbone with smaller embeddings, DINOv3 included.
    """
    b = feats.shape[0]
    h, w = grid_size
    pc1 = _first_components(feats, 1).squeeze(-1)  # (B, N), already mean-centred
    cut = sigma * pc1.std(dim=1, keepdim=True)

    y0, y1 = int(h * border), int(h * (1 - border))
    x0, x1 = int(w * border), int(w * (1 - border))

    def lean(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        overall = m.flatten(1).float().mean(1)
        return m[:, y0:y1, x0:x1].flatten(1).float().mean(1) - overall, overall

    plus, minus = (pc1 > cut).view(b, h, w), (-pc1 > cut).view(b, h, w)
    (lean_plus, area_plus), (lean_minus, area_minus) = lean(plus), lean(minus)
    # On an even lean the smaller side wins, as a tile is mostly sky.
    take_minus = (lean_minus > lean_plus) | ((lean_minus == lean_plus) & (area_minus < area_plus))
    mask = torch.where(take_minus[:, None, None], minus, plus)

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
