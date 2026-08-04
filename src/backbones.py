"""Patch-token extractors.

Two families, one interface: images in, one embedding per patch out.

  dinov2_*  torch.hub (facebookresearch/dinov2), patch size 14
  dinov3-*  transformers AutoModel, patch size 16

DINOv3 is the one worth reaching for. Its dense features hold up under long
training thanks to Gram anchoring, which is exactly the property patch-level
nearest-neighbour scoring depends on, and the SAT-493M checkpoints were trained
on overhead imagery rather than web photos. A JWST mosaic tile looks a great
deal more like a satellite scene than like a picture of a dog.

DINOv3 weights are gated on the Hub: accept the licence and run `hf auth login`
once. The image processor is only consulted for its normalisation constants,
which differ between the web and satellite checkpoints, so they are never
hardcoded here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.nn import functional as tnf
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Backbone:
    """A loaded model plus everything the pipeline needs to feed and read it."""

    name: str
    model: torch.nn.Module
    forward_patches: Callable[[torch.Tensor], torch.Tensor]
    patch_size: int
    resolution: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
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

    def transform(self):
        """Tile -> normalised square tensor. Tiles are square, so no aspect juggling."""
        return transforms.Compose(
            [
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, 3, R, R) -> (B, n_patches, D) float32 on the model device."""
        batch = batch.to(self.device, non_blocking=True)
        if self.autocast_dtype is None:
            return self.forward_patches(batch).float()
        with torch.autocast(self.device.type, dtype=self.autocast_dtype):
            return self.forward_patches(batch).float()


def _snap_resolution(resolution: int, patch_size: int) -> int:
    snapped = max(patch_size, (resolution // patch_size) * patch_size)
    if snapped != resolution:
        logger.warning(
            "resolution %d is not a multiple of patch size %d; using %d",
            resolution,
            patch_size,
            snapped,
        )
    return snapped


def _first_int(value) -> int:
    return int(value[0]) if isinstance(value, (list, tuple)) else int(value)


def _load_dinov2(name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", name).eval().to(device)

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        return model.forward_features(x)["x_norm_patchtokens"]

    return model, forward_patches, int(model.patch_size), IMAGENET_MEAN, IMAGENET_STD


def _load_dinov3(name: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "DINOv3 needs `transformers`. Install it, then `hf auth login` "
            "once because the facebook/dinov3-* weights are gated."
        ) from exc

    repo = name if "/" in name else f"facebook/{name}"
    model = AutoModel.from_pretrained(repo).eval().to(device)
    processor = AutoImageProcessor.from_pretrained(repo)

    # CLS token plus however many registers this checkpoint carries sit in front
    # of the patch tokens.
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        return model(pixel_values=x).last_hidden_state[:, n_prefix:, :]

    return (
        model,
        forward_patches,
        _first_int(model.config.patch_size),
        tuple(processor.image_mean),
        tuple(processor.image_std),
    )


def get_backbone(
    model_name: str,
    device: str | torch.device,
    resolution: int = 448,
    *,
    autocast: bool = True,
) -> Backbone:
    """Load a backbone by name. `dinov3-...` goes through transformers, `dinov2_...` through hub."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is cuda but torch.cuda.is_available() is False")

    if model_name.startswith("dinov3"):
        loaded = _load_dinov3(model_name, device)
    elif model_name.startswith("dinov2"):
        loaded = _load_dinov2(model_name, device)
    else:
        raise ValueError(f"Unknown model name: {model_name!r} (expected dinov2_* or dinov3-*)")

    model, forward_patches, patch_size, mean, std = loaded

    dtype = None
    if autocast and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    backbone = Backbone(
        name=model_name,
        model=model,
        forward_patches=forward_patches,
        patch_size=patch_size,
        resolution=_snap_resolution(resolution, patch_size),
        mean=tuple(mean),
        std=tuple(std),
        device=device,
        autocast_dtype=dtype,
    )
    logger.info(
        "loaded %s on %s: patch=%d resolution=%d grid=%dx%d autocast=%s",
        model_name,
        device,
        backbone.patch_size,
        backbone.resolution,
        *backbone.grid_size,
        dtype,
    )
    return backbone


# ---------------------------------------------------------------------------
# Background masking
# ---------------------------------------------------------------------------
def _first_components(feats: torch.Tensor, q: int) -> torch.Tensor:
    """Project (B, N, D) onto its top `q` principal components -> (B, N, q)."""
    centred = feats - feats.mean(dim=1, keepdim=True)
    # pca_lowrank wants a little headroom above the rank it is asked for.
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
    """(B, N, D) -> (B, N) bool, True where a patch is foreground and worth scoring.

    Threshold the first principal component of the patch embeddings, then pick the
    sign that keeps the centre of the tile, since the component's sign is arbitrary.

    The cut sits `sigma` standard deviations above the mean of that component rather
    than at a fixed value. AnomalyDINO's absolute threshold of 10 was calibrated on
    DINOv2's feature scale and masks away the entire image on a backbone whose
    embeddings happen to be smaller, DINOv3 included.
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

    # Dilate, then close: fill pinholes and grow the mask slightly.
    m = mask.unsqueeze(1).float()
    pad = kernel // 2
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = -tnf.max_pool2d(-m, kernel, stride=1, padding=pad)
    mask = (m.squeeze(1) > 0.5).flatten(1)

    # A tile of uniform sky has no foreground to find. Score all of it rather than
    # dropping it, which would otherwise leave the run with an empty memory bank.
    return torch.where(mask.any(dim=1, keepdim=True), mask, torch.ones_like(mask))


@torch.inference_mode()
def embedding_rgb(feats: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Top-3 principal components of (N, D) patch tokens as an (H, W, 3) image in [0, 1]."""
    reduced = _first_components(feats.unsqueeze(0), 3).squeeze(0)
    lo, hi = reduced.min(), reduced.max()
    reduced = (reduced - lo) / (hi - lo + 1e-12)
    return reduced.view(*grid_size, 3)
