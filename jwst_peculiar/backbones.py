"""Loading DINO backbones and reading patch embeddings out of them.

Everything loads through transformers from the Hugging Face Hub. DINOv3 is the one
worth reaching for, as its dense features hold up well (Gram anchoring) and its
SAT-493M checkpoints were trained on overhead imagery, which a JWST mosaic tile looks
a lot more like than it does a web photo. DINOv3 weights are gated, though, so accept
the licence on the model page and run `hf auth login` once. DINOv2 isn't gated.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.util
import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as tnf

from jwst_peculiar.streaming import StreamedDinov3, fits_on_gpu

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

PRECISIONS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# The last model get_backbone loaded, kept so that a process scoring many runs, like
# the benchmark, loads and compiles it once. Holding more than one would also hold
# every compiled graph on the GPU.
_loaded: dict[_Request, Backbone] = {}

# transformers ships no type information the type checkers can follow, so its models
# are Any to them. Naming them still says what each value is.
HubModel: TypeAlias = Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# The torch.hub names from the DINOv2 repo still work and map to the same weights on
# the Hub.
_DINOV2_HUB_NAME = re.compile(r"dinov2_vit([sblg])14(_reg)?$")
_DINOV2_SIZES = {"s": "small", "b": "base", "l": "large", "g": "giant"}


def hub_repo(model_name: str) -> str:
    """Return the Hugging Face repo id or local folder for a --model_name.

    Raises:
        ValueError: If the name isn't one it knows.
    """
    if "/" in model_name or "\\" in model_name:
        return model_name
    m = _DINOV2_HUB_NAME.match(model_name)
    if m:
        size = _DINOV2_SIZES[m.group(1)]
        return (
            f"facebook/dinov2-with-registers-{size}"
            if m.group(2)
            else f"facebook/dinov2-{size}"
        )
    if model_name.startswith(("dinov2-", "dinov3-")):
        return f"facebook/{model_name}"
    msg = (
        f"Unknown model name {model_name!r}. Use a dinov3-* or dinov2_* name, or a Hub "
        "repo id."
    )
    raise ValueError(msg)


@dataclass(frozen=True)
class TileTransform:
    """RGB tile to a normalised (3, R, R) tensor.

    It's a class rather than a closure so DataLoader workers can pickle it on Windows.
    """

    resolution: int
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Return the tile resized to the model's resolution and normalised."""
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
    dtype: torch.dtype = torch.float32
    layers: str = "last"
    compiled: bool = False
    fast_accumulate: bool = False
    streamed: bool = False
    min_batch: int = 1
    _traced_batch: int | None = None

    @property
    def grid_size(self) -> tuple[int, int]:
        """Patches down and across a tile."""
        n = self.resolution // self.patch_size
        return (n, n)

    @property
    def n_patches(self) -> int:
        """Patches in a tile."""
        h, w = self.grid_size
        return h * w

    def transform(self) -> TileTransform:
        """Return the transform that turns a tile into this model's input."""
        return TileTransform(self.resolution, self.mean, self.std)

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Embed the patches of a (B, 3, R, R) batch.

        Returns:
            The (B, n_patches, D) embeddings, float32 on the device.

        Raises:
            FloatingPointError: If the model overflowed and gave inf or NaN.
        """
        n = batch.shape[0]
        batch = batch.to(self.device, dtype=self.dtype, non_blocking=True)
        if self.compiled:
            if self._traced_batch is None:
                self._traced_batch = n
            elif n < self._traced_batch:
                # A compiled model traces again for every new shape, which takes longer
                # than running a short last batch padded to the usual size.
                pad = batch.new_zeros((self._traced_batch - n, *batch.shape[1:]))
                batch = torch.cat([batch, pad])
        feats = self.forward_patches(batch)[:n].float()
        if not torch.isfinite(feats).all():
            msg = (
                f"{self.name} overflowed {str(self.dtype).removeprefix('torch.')} and "
                "returned inf or NaN features. Run it with --precision bf16 or fp32."
            )
            raise FloatingPointError(msg)
        return feats


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
    from huggingface_hub import hf_hub_download  # noqa: PLC0415 huggingface_hub is slow to import

    path = Path(repo) / "preprocessor_config.json"
    if not path.is_file():
        if Path(repo).is_dir():
            logger.warning(
                "%s has no preprocessor_config.json, so using ImageNet's", repo
            )
            return IMAGENET_MEAN, IMAGENET_STD
        path = Path(hf_hub_download(repo, "preprocessor_config.json"))
    config = json.loads(path.read_text(encoding="utf-8"))
    mean = tuple(config.get("image_mean", IMAGENET_MEAN))
    std = tuple(config.get("image_std", IMAGENET_STD))
    return mean, std


def parse_layers(spec: str, depth: int) -> list[int] | None:
    """Work out which hidden states to average for --layers.

    "mid" is the band SubspaceAD found best on DINOv2-G (layers 22 to 28 of 40),
    scaled to this model's depth. "a-b" takes layers a to b counted from 1, and a
    comma list takes those layers.

    Returns:
        The layers counted from 1, or None for the final output.

    Raises:
        ValueError: If `spec` doesn't parse or reaches past the model's depth.
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
        msg = f"--layers takes last, mid, a range like 14-20 or a list, not {spec!r}"
        raise ValueError(msg) from None
    if not picked or min(picked) < 1 or max(picked) > depth:
        msg = f"--layers {spec} is outside this model's 1 to {depth}"
        raise ValueError(msg)
    if len(set(picked)) != len(picked):
        # The streamed model counts a layer once, where the in-memory one would
        # weight it twice.
        msg = f"--layers {spec} names a layer more than once"
        raise ValueError(msg)
    return picked


def _checkpoint_folder(repo: str) -> Path:
    """Return the folder holding a repo's config and safetensors, downloading them."""
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download  # noqa: PLC0415 huggingface_hub is slow to import

    return Path(snapshot_download(repo, allow_patterns=["*.json", "*.safetensors"]))


def _pixel_backbone(resolution: int, device: torch.device, layers: str) -> Backbone:
    size = 16

    def raw(x: torch.Tensor) -> torch.Tensor:
        return tnf.unfold(x, size, stride=size).transpose(1, 2)

    return Backbone(
        name="pixels",
        model=torch.nn.Identity(),
        forward_patches=raw,
        patch_size=size,
        resolution=_snap_resolution(resolution, size),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        device=device,
        layers=layers,
    )


@dataclass(frozen=True)
class _Request:
    """What get_backbone was asked for, and the key it caches the model under."""

    model_name: str
    device: str
    resolution: int
    precision: str
    compile_model: bool | None
    fast_accumulate: bool
    layers: str
    stream: bool | None


def get_backbone(  # noqa: PLR0913 the CLI passes every option through by name
    model_name: str,
    device: str | torch.device,
    resolution: int = 448,
    *,
    precision: str = "fp16",
    compile_model: bool | None = None,
    fast_accumulate: bool = False,
    layers: str = "last",
    stream: bool | None = None,
) -> Backbone:
    """Load a backbone by name, a Hub repo id or a local checkpoint folder.

    The name can be dinov3-* or dinov2_*. On CUDA the weights and activations are cast
    to `precision`, and the model is compiled when Triton is installed unless
    `compile_model` says otherwise. fp16 keeps the patch distances closer to fp32 than
    bf16 does at the same speed, as DINOv3's activations fit its range and it carries
    three more bits of mantissa. `fast_accumulate` sums fp16 matmuls in fp16, which
    GeForce cards run at twice the rate of an fp32 sum. The CPU always runs fp32.

    A DINOv3 model too big for the GPU, like the 7B one on an 8 GB card, is run one
    layer at a time from the checkpoint on disk instead, and `stream` forces that
    either way.

    "pixels" is no model at all, just each 16 px patch's raw RGB values, which runs the
    same search on pixels so the benchmark can show what the features add.

    Returns:
        The loaded backbone, the same one as last time when nothing changed.
    """
    device = torch.device(device)
    if model_name == "pixels":
        return _pixel_backbone(resolution, device, layers)

    request = _Request(
        model_name,
        str(device),
        resolution,
        precision,
        compile_model,
        fast_accumulate,
        layers,
        stream,
    )
    if request in _loaded:
        return _loaded[request]
    if _loaded:
        _loaded.clear()
        torch.compiler.reset()
        gc.collect()
        torch.cuda.empty_cache()
    backbone = _load_backbone(request, device)
    _loaded[request] = backbone
    return backbone


def _precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    if precision not in PRECISIONS:
        msg = f"--precision is fp16, bf16 or fp32, not {precision!r}"
        raise ValueError(msg)
    if device.type != "cuda":
        precision = "fp32"
    elif precision == "bf16" and not torch.cuda.is_bf16_supported():
        precision = "fp16"
    return PRECISIONS[precision]


def _load_model(
    repo: str, dtype: torch.dtype, device: torch.device, *, stream: bool | None
) -> tuple[HubModel, bool]:
    from transformers import AutoModel  # noqa: PLC0415 transformers is slow to import

    try:
        folder = _checkpoint_folder(repo)
        if stream is None:
            stream = "dinov3" in repo.lower() and not fits_on_gpu(folder, dtype, device)
        if stream:
            model = StreamedDinov3(folder, dtype, device)
        else:
            model = AutoModel.from_pretrained(repo, dtype=dtype).eval().to(device)
    except OSError as exc:
        msg = (
            f"Couldn't load {repo}. If it's a DINOv3 model, the weights are gated, so "
            f"accept the licence on https://huggingface.co/{repo} and run "
            "`hf auth login`, or use dinov2_vitl14, which isn't gated."
        )
        raise RuntimeError(msg) from exc
    return model, stream


def _patch_reader(
    model: HubModel, layers: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    # The CLS token and any register tokens sit in front of the patch tokens.
    n_prefix = 1 + int(getattr(model.config, "num_register_tokens", 0))

    picked = parse_layers(layers, int(model.config.num_hidden_layers))

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        if isinstance(model, StreamedDinov3):
            out: torch.Tensor = model(x, picked)
            return out[:, n_prefix:, :]
        if picked is None:
            last: torch.Tensor = model(pixel_values=x).last_hidden_state
            return last[:, n_prefix:, :]
        hidden = model(pixel_values=x, output_hidden_states=True).hidden_states
        return torch.stack([hidden[i][:, n_prefix:, :] for i in picked]).mean(dim=0)

    patch_size = model.config.patch_size
    patch_size = int(
        patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
    )
    return forward_patches, patch_size


def _overflows_fp16(
    forward_patches: Callable[[torch.Tensor], torch.Tensor],
    *,
    resolution: int,
    patch_size: int,
    device: torch.device,
) -> bool:
    with torch.inference_mode():
        size = _snap_resolution(resolution, patch_size)
        probe = forward_patches(
            torch.randn(1, 3, size, size, device=device, dtype=torch.float16)
        )
    # The LVD checkpoints carry activations past fp16's 65504 on any input at all,
    # noise and black tiles included, so one probe settles it.
    return not torch.isfinite(probe).all()


def _load_backbone(request: _Request, device: torch.device) -> Backbone:
    dtype = _precision_dtype(request.precision, device)
    repo = hub_repo(request.model_name)
    model, stream = _load_model(repo, dtype, device, stream=request.stream)
    mean, std = _normalisation(repo)
    forward_patches, patch_size = _patch_reader(model, request.layers)

    if dtype == torch.float16 and _overflows_fp16(
        forward_patches,
        resolution=request.resolution,
        patch_size=patch_size,
        device=device,
    ):
        if _precision_dtype("bf16", device) == dtype:
            msg = f"{repo} overflows fp16, and {device} has no bf16 to run it in"
            raise RuntimeError(msg)
        logger.warning("%s overflows fp16, so running it in bf16", repo)
        del model, forward_patches
        gc.collect()
        torch.cuda.empty_cache()
        return get_backbone(
            request.model_name,
            device,
            request.resolution,
            precision="bf16",
            compile_model=request.compile_model,
            layers=request.layers,
            stream=stream,
        )

    compile_model = request.compile_model
    if compile_model is None:
        compile_model = (
            device.type == "cuda"
            and not stream
            and importlib.util.find_spec("triton") is not None
        )
    if compile_model:
        model.compile(dynamic=False)
    fast_accumulate = request.fast_accumulate and dtype == torch.float16
    # Set either way, or a model loaded later in this process would inherit it.
    torch.backends.cuda.matmul.allow_fp16_accumulation = fast_accumulate

    backbone = Backbone(
        name=request.model_name,
        model=model,
        forward_patches=forward_patches,
        patch_size=patch_size,
        resolution=_snap_resolution(request.resolution, patch_size),
        mean=mean,
        std=std,
        device=device,
        dtype=dtype,
        layers=request.layers,
        compiled=compile_model,
        fast_accumulate=fast_accumulate,
        streamed=stream,
        # Every streamed weight crosses the bus once per batch, so a batch has to be
        # big enough for the layer to run longer than the next one takes to read.
        min_batch=128 if stream else 1,
    )
    _log_loaded(backbone, repo)
    return backbone


def _log_loaded(backbone: Backbone, repo: str) -> None:
    precision = next(k for k, v in PRECISIONS.items() if v == backbone.dtype)
    logger.info(
        "Loaded %s on %s in %s%s%s, %d px patches at %d px, a %dx%d grid",
        repo,
        backbone.device,
        precision,
        ", compiled"
        if backbone.compiled
        else ", a layer at a time from disk"
        if backbone.streamed
        else "",
        " with fp16 sums" if backbone.fast_accumulate else "",
        backbone.patch_size,
        backbone.resolution,
        *backbone.grid_size,
    )


def _first_components(feats: torch.Tensor, q: int) -> torch.Tensor:
    """Project (B, N, D) patch embeddings onto their top `q` principal components.

    It's an exact SVD, as torch.pca_lowrank draws from the global generator and would
    make a tile's mask depend on how many draws came before it.

    Returns:
        The (B, N, q) projections.
    """
    centred = feats.float() - feats.float().mean(dim=1, keepdim=True)
    _, _, vh = torch.linalg.svd(centred, full_matrices=False)
    return torch.matmul(centred, vh[..., :q, :].transpose(-1, -2))


def _centre_lean(m: torch.Tensor, border: float) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = m.shape[1:]
    y0, y1 = int(h * border), int(h * (1 - border))
    x0, x1 = int(w * border), int(w * (1 - border))
    overall = m.flatten(1).float().mean(1)
    return m[:, y0:y1, x0:x1].flatten(1).float().mean(1) - overall, overall


def _subject_side(
    plus: torch.Tensor, minus: torch.Tensor, border: float
) -> torch.Tensor:
    lean_plus, area_plus = _centre_lean(plus, border)
    lean_minus, area_minus = _centre_lean(minus, border)
    # On an even lean the smaller side wins, as a tile is mostly sky.
    take_minus = (lean_minus > lean_plus) | (
        (lean_minus == lean_plus) & (area_minus < area_plus)
    )
    return torch.where(take_minus[:, None, None], minus, plus)


def _dilate_and_close(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    # Dilate, then close, which fills pinholes and grows the mask a little.
    m = mask.unsqueeze(1).float()
    pad = kernel // 2
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = tnf.max_pool2d(m, kernel, stride=1, padding=pad)
    m = -tnf.max_pool2d(-m, kernel, stride=1, padding=pad)
    # Pooling a mask of zeros and ones leaves only zeros and ones.
    return (m.squeeze(1) > 0).flatten(1)


@torch.inference_mode()
def background_mask(
    feats: torch.Tensor,
    grid_size: tuple[int, int],
    sigma: float = 0.5,
    *,
    border: float = 0.2,
    kernel: int = 3,
) -> torch.Tensor:
    """Mask the patches in a (B, N, D) batch that are worth scoring.

    It thresholds the first principal component of the patch embeddings on both
    sides and keeps the side that crowds into the centre of the tile more than it
    covers the tile overall, which is where a tile's subject sits. Both sides are
    judged the same way, so the component's arbitrary sign can't change the answer.
    The cut is `sigma` standard deviations of that component rather than AnomalyDINO's
    fixed 10, which was tuned to DINOv2's feature scale and masks out the whole tile
    on a backbone with smaller embeddings, DINOv3 included.

    Returns:
        A (B, N) bool mask.
    """
    shape = (feats.shape[0], *grid_size)
    pc1 = _first_components(feats, 1).squeeze(-1)  # (B, N), already mean-centred
    cut = sigma * pc1.std(dim=1, keepdim=True)
    plus, minus = (pc1 > cut).view(shape), (-pc1 > cut).view(shape)
    mask = _dilate_and_close(_subject_side(plus, minus, border), kernel)

    # A tile of plain sky has no foreground at all. Scoring all of it beats dropping
    # it, which could leave the memory bank empty.
    return torch.where(mask.any(dim=1, keepdim=True), mask, torch.ones_like(mask))


@torch.inference_mode()
def embedding_rgb(feats: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
    """Return the top 3 principal components of (N, D) embeddings as an RGB image.

    The image is (H, W, 3), scaled to [0, 1].
    """
    reduced = _first_components(feats.unsqueeze(0), 3).squeeze(0)
    lo, hi = reduced.min(), reduced.max()
    reduced = (reduced - lo) / (hi - lo + 1e-12)
    return reduced.view(*grid_size, 3)
