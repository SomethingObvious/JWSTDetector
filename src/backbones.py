"""Loading DINO backbones and reading patch embeddings out of them.

DINOv2 comes from torch.hub and DINOv3 through transformers. DINOv3 is the one worth
reaching for, as its dense features hold up well (Gram anchoring) and its SAT-493M
checkpoints were trained on overhead imagery, which a JWST mosaic tile looks a lot
more like than it does a web photo. DINOv3 weights are gated, though, so accept the
licence on the model page and run `hf auth login` once.
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
    """A loaded model and what the pipeline needs to feed it and read it."""

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
        """RGB tile to a normalised (3, R, R) tensor."""
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
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "DINOv3 needs transformers. Install it and run `hf auth login` "
            "once, as the facebook/dinov3-* weights are gated."
        ) from exc

    repo = name if "/" in name else f"facebook/{name}"
    model = AutoModel.from_pretrained(repo).eval().to(device)
    # Only the processor's mean and std get used, as the satellite checkpoints have their own.
    processor = AutoImageProcessor.from_pretrained(repo)

    # The CLS token and any register tokens sit in front of the patch tokens.
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
    """Load a backbone by name, dinov3-* through transformers or dinov2_* through torch.hub."""
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The device is cuda, but torch can't see a CUDA GPU.")

    if model_name.startswith("dinov3"):
        loaded = _load_dinov3(model_name, device)
    elif model_name.startswith("dinov2"):
        loaded = _load_dinov2(model_name, device)
    else:
        raise ValueError(f"Unknown model name {model_name!r}. Use a dinov3-* or dinov2_* name.")

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
        "Loaded %s on %s, %d px patches at %d px, a %dx%d grid, autocast %s",
        model_name,
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
