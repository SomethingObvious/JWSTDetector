"""DINOv3 run a layer at a time from weights on disk, for models bigger than the GPU.

The 7B model is 13.4 GB in fp16, so it can't sit on an 8 GB card, and keeping it in
RAM takes more than most laptops have free. Instead the safetensors shards are memory
mapped, and each layer is read, cast and copied to the GPU just before it runs, while
a thread reads the next one. A whole chunk of tiles goes through each layer before the
next is loaded, so every weight crosses the bus once per chunk rather than once per
tile, and with chunks of a few dozen tiles the GPU is busy most of the time.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, cast

from safetensors import safe_open
import torch
from typing_extensions import override

if TYPE_CHECKING:
    from concurrent.futures import Future

    from transformers.models.dinov3_vit.configuration_dinov3_vit import (
        DINOv3ViTConfig,
    )

    _Event: TypeAlias = torch.cuda.Event | torch.Event

logger = logging.getLogger(__name__)


class _Shards:
    """Tensors by name across a sharded safetensors checkpoint, memory mapped."""

    def __init__(self, folder: Path) -> None:
        index = folder / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            single = folder / "model.safetensors"
            with safe_open(str(single), framework="pt") as f:
                weight_map = dict.fromkeys(f.keys(), single.name)
        self._files = {
            name: safe_open(str(folder / name), framework="pt")
            for name in set(weight_map.values())
        }
        self._where = weight_map

    def names(self) -> list[str]:
        return list(self._where)

    def get(self, name: str) -> torch.Tensor:
        tensor: torch.Tensor = self._files[self._where[name]].get_tensor(name)  # type: ignore[no-untyped-call]  # safetensors leaves this unannotated
        return tensor

    def numel(self, name: str) -> int:
        # From the header, without reading the tensor.
        shape: list[int] = self._files[self._where[name]].get_slice(name).get_shape()  # type: ignore[no-untyped-call]  # safetensors leaves this unannotated
        return math.prod(shape)


class StreamedDinov3(torch.nn.Module):
    """A DINOv3 ViT whose layers live on disk and visit the GPU one at a time.

    The forward pass takes any number of tiles and runs them through each layer in
    micro-batches, so callers should hand it chunks of tens of tiles. It returns the
    same last_hidden_state as the transformers model, and with `layers` the mean of
    those hidden states instead.
    """

    def __init__(
        self,
        folder: Path,
        dtype: torch.dtype,
        device: torch.device,
        *,
        micro_batch: int = 8,
    ) -> None:
        super().__init__()
        from transformers import AutoConfig  # noqa: PLC0415 transformers is slow to import
        from transformers.models.dinov3_vit import (  # noqa: PLC0415 transformers is slow to import
            modeling_dinov3_vit as m,
        )

        self.config = cast(
            "DINOv3ViTConfig",
            AutoConfig.from_pretrained(folder, attn_implementation="sdpa"),
        )
        self.dtype_, self.device_, self.micro_batch = dtype, device, micro_batch
        self.shards = _Shards(Path(folder))
        names = self.shards.names()
        prefix = "model." if any(n.startswith("model.layer.") for n in names) else ""
        self._layer_prefix = f"{prefix}layer."

        self.embeddings = m.DINOv3ViTEmbeddings(self.config)
        self.rope = m.DINOv3ViTRopePositionEmbedding(self.config)
        self.norm = torch.nn.LayerNorm(
            self.config.hidden_size, eps=self.config.layer_norm_eps
        )
        for module, start in ((self.embeddings, "embeddings."), (self.norm, "norm.")):
            state = {
                n[len(start) :]: self.shards.get(n)
                for n in names
                if n.startswith(start)
            }
            module.load_state_dict(state, strict=True)
        self.embeddings.to(device, dtype)
        self.norm.to(device, dtype)
        self.rope.to(device)

        # Two layers on the GPU, so one can take the next weights while the other runs.
        self.slots = [
            m.DINOv3ViTLayer(self.config).to(device, dtype).eval() for _ in range(2)
        ]
        self.n_layers = int(self.config.num_hidden_layers)
        self._keys = list(self.slots[0].state_dict())
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._copy_stream = (
            torch.cuda.Stream(device)  # type: ignore[no-untyped-call]  # torch leaves Stream unannotated
            if device.type == "cuda"
            else None
        )
        # Training mode would jitter the RoPE coordinates.
        self.eval()

    def _read(self, i: int) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for k in self._keys:
            t = self.shards.get(f"{self._layer_prefix}{i}.{k}").to(self.dtype_)
            state[k] = t.pin_memory() if self._copy_stream is not None else t
        return state

    def _load(
        self,
        slot: torch.nn.Module,
        state: dict[str, torch.Tensor],
        free: _Event | None,
    ) -> _Event | None:
        params = slot.state_dict()
        if self._copy_stream is None:
            for k, t in state.items():
                params[k].copy_(t)
            return None
        with torch.cuda.stream(self._copy_stream):
            if free is not None:
                # The slot still holds the layer before last until its work has run.
                self._copy_stream.wait_event(free)
            for k, t in state.items():
                params[k].copy_(t, non_blocking=True)
            done = torch.cuda.Event()  # type: ignore[no-untyped-call]  # torch leaves Event unannotated
            done.record(self._copy_stream)
        return done

    def _next_read(self, i: int) -> Future[dict[str, torch.Tensor]] | None:
        return self._pool.submit(self._read, i) if i < self.n_layers else None

    @override
    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor, picked: list[int] | None = None
    ) -> torch.Tensor:
        """Run a chunk of tiles through every layer, loading each in turn.

        Returns:
            The normed last hidden state, or the mean of the `picked` layers' states.
        """
        mb = self.micro_batch
        x = pixel_values.to(self.device_, self.dtype_)
        h = torch.cat([self.embeddings(x[s : s + mb]) for s in range(0, len(x), mb)])
        pos = self.rope(x[:1])
        picked = picked or []
        total = torch.zeros_like(h) if picked else None
        compute = (
            torch.cuda.current_stream(self.device_)
            if self._copy_stream is not None
            else None
        )

        loaded = [self._load(self.slots[0], self._read(0), None), None]
        freed: list[_Event | None] = [None, None]
        pending = self._next_read(1)
        for i in range(self.n_layers):
            ready = loaded[i % 2]
            if compute is not None and ready is not None:
                compute.wait_event(ready)
            for s in range(0, len(h), mb):
                h[s : s + mb] = self.slots[i % 2](
                    h[s : s + mb], position_embeddings=pos
                )
            if compute is not None:
                freed[i % 2] = compute.record_event()
            if total is not None and i + 1 in picked:
                total += h
            if pending is not None:
                nxt = (i + 1) % 2
                loaded[nxt] = self._load(self.slots[nxt], pending.result(), freed[nxt])
                pending = self._next_read(i + 2)
        if total is not None:
            return total / len(picked)
        return torch.cat([self.norm(h[s : s + mb]) for s in range(0, len(h), mb)])


def fits_on_gpu(
    folder: Path, dtype: torch.dtype, device: torch.device, *, headroom: float = 0.7
) -> bool:
    """Check whether a checkpoint's weights fit on the GPU.

    Returns:
        True when the weights at `dtype` fit in `headroom` of the free GPU memory,
        and always off CUDA.
    """
    if device.type != "cuda":
        return True
    shards = _Shards(Path(folder))
    n = sum(shards.numel(name) for name in shards.names())
    free, _ = torch.cuda.mem_get_info(device)
    return n * torch.finfo(dtype).bits // 8 < headroom * free
