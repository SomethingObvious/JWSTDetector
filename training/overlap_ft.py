"""Fine-tunes DINOv3 ViT-L SAT with LoRA so the same sky gives the same patch features.

    overlap_ft.py --roots DIR [DIR ...] --out DIR --steps 2000

Tiles are cut every 160 px, so two neighbouring tiles share half their pixels exactly,
and after the resize to 448 px that half is 14 whole patches. A patch of sky therefore
turns up at two places in two tiles with different surroundings, and each view also
gets its own 90 degree turn or flip. The loss pulls the two features of every shared
patch together, and a second term keeps each feature close to the frozen model's on
the same input, which stops the features collapsing and keeps what the kNN relies on.
What the training removes is what the kNN shouldn't see: where a patch falls in its
tile, its orientation and the tile edge.

Lupton colour clips each filter's sky at 0, so 60 to 80 percent of a channel is 0 and
the sky's noise can't be measured from the PNGs (a quantile fit gave 45 to 185 counts
on tiles whose 84th percentile is 30 to 70). With `--noise S`, a tile that has a FITS
cutout is drawn again from its flux instead, after the noise equalisation and before
the stretch, with Gaussian noise of U(0, S) times its own sky sigma added to each
filter, the same way the bench renders a planted source. Each view gets its own draw,
so the loss also pulls apart what the sky's noise pattern does to a feature.

With `--stretch J`, each view of a tile drawn from flux also gets each filter's range
above the sky multiplied by its own factor, log-uniform between 1/J and J. One bright
star pushed a COSMOS-Web visit's range 4 to 12 times above the others', and every tile
in it scored like an outlier, so this teaches the features to ignore the stretch.

The export folds the LoRA into an fp32 copy of the original weights, saved as a folder
that `--model_name <folder>` loads.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

REPO = "facebook/dinov3-vitl16-pretrain-sat493m"
N_PREFIX = 5  # the class token and 4 registers
RES, GRID, TILE, STRIDE = 448, 28, 320, 160
HALF = GRID * STRIDE // TILE  # 14 patches


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.on = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.on:
            out = out + (x @ self.a.T.to(x.dtype)) @ self.b.T.to(x.dtype) * self.scale
        return out

    def delta(self) -> torch.Tensor:
        return (self.b.float() @ self.a.float()) * self.scale


TARGETS = ("attention.q_proj", "attention.v_proj", "mlp.up_proj", "mlp.down_proj")


def build(r: int, alpha: float) -> tuple[nn.Module, list[LoRA]]:
    model = AutoModel.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    loras = []
    for layer in model.model.layer:
        for name in TARGETS:
            parent_name, attr = name.rsplit(".", 1)
            parent = layer.get_submodule(parent_name)
            lora = LoRA(getattr(parent, attr), r, alpha)
            setattr(parent, attr, lora)
            loras.append(lora)
        # Stochastic depth would make the frozen pass random too.
        layer.drop_path = nn.Identity()
    # Only the LoRA weights need gradients, which the reentrant version can't handle.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, loras


def seed_worker(seed: int, worker: int) -> None:
    random.seed(seed * 100 + worker)
    np.random.seed(seed * 100 + worker)


def pairs_of(root: Path) -> list[tuple[Path, Path, str]]:
    """Every pair of tiles a stride apart, across or up, in one prepped dataset."""
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    at = {(r["mosaic"], int(r["x0"]), int(r["y0"])): r["key"] for r in rows if int(r["size"]) == TILE}
    out = []
    for (m, x, y), key in at.items():
        if (m, x + STRIDE, y) in at:
            out.append((root / "query" / key, root / "query" / at[m, x + STRIDE, y], "across"))
        if (m, x, y + STRIDE) in at:
            out.append((root / "query" / key, root / "query" / at[m, x, y + STRIDE], "up"))
    return out


def shared(direction: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Patch indices in A and in B of the half the two tiles share."""
    idx = torch.arange(GRID * GRID).reshape(GRID, GRID)
    if direction == "across":  # B starts 160 px to the right, so A's right half is B's left
        return idx[:, HALF:].reshape(-1), idx[:, :HALF].reshape(-1)
    # B sits 160 px higher on the sky, and PNG rows run down, so A's top is B's bottom.
    return idx[:HALF, :].reshape(-1), idx[HALF:, :].reshape(-1)


def sky_sigma(x: np.ndarray) -> float:
    """Sigma of the sky in one plane, clipping the sources above it three times."""
    v = x[np.isfinite(x) & (x != 0)]
    for _ in range(3):
        med = float(np.median(v))
        sigma = 1.4826 * float(np.median(np.abs(v - med)))
        v = v[np.abs(v - med) < 3 * sigma]
    return sigma


def cutouts(roots: list[Path]) -> dict[Path, tuple[Path, dict]]:
    """The FITS cutout and render record of every tile that has one."""
    out = {}
    for root in roots:
        with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("cutout") and (root / r["cutout"]).exists()]
        records = {}
        for r in rows:
            if r["mosaic"] not in records:
                text = (root / "mosaics" / f"{r['mosaic']}.json").read_text(encoding="utf-8")
                records[r["mosaic"]] = json.loads(text)["render"]
            out[root / "query" / r["key"]] = (root / r["cutout"], records[r["mosaic"]])
    return out


def noisy(cutout: Path, render: dict, scale: float, rng: np.random.Generator,
          stretch: float = 1.0) -> Image.Image:
    from astropy.io import fits

    from jwst_peculiar.prep import equalise, render_cutout

    with fits.open(cutout, memmap=False) as hdul:
        planes = np.asarray(hdul[0].data, dtype=np.float32).reshape(-1, *hdul[0].data.shape[-2:])
        factor = hdul["NOISEEQ"].data if "NOISEEQ" in hdul else None
    skies = [v["sky"] for v in render["levels"]] if render["kind"] == "lupton" else [render["sky"]]
    out = []
    for i, (p, sky) in enumerate(zip(planes, skies, strict=True)):
        p = equalise(p, sky, None if factor is None else factor[i] if factor.ndim == 3 else factor)
        ok = np.isfinite(p) & (p != 0)
        sigma = scale * sky_sigma(p)
        out.append(np.where(ok, p + rng.normal(0, sigma, p.shape).astype(np.float32), p))
    if stretch > 1 and render["kind"] == "lupton":
        span = np.log(stretch)
        levels = [{"sky": v["sky"], "hi": v["sky"] + (v["hi"] - v["sky"]) * float(np.exp(rng.uniform(-span, span)))}
                  for v in render["levels"]]
        render = {**render, "levels": levels}
    tile = render_cutout(out if len(out) > 1 else out[0], render, None)
    return Image.fromarray(tile)


class Pairs(torch.utils.data.Dataset):
    def __init__(self, pairs, mean, std, epoch: int, flux=None, noise: float = 0.0,
                 stretch: float = 1.0) -> None:
        self.pairs, self.epoch = pairs, epoch
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.flux, self.noise, self.stretch = flux or {}, noise, stretch
        self.redraw = noise > 0 or stretch > 1
        # Only the control fields kept their cutouts, so half the draws come from them,
        # or the noise would reach about a quarter of the pairs.
        self.with_flux = [p for p in pairs if p[0] in self.flux and p[1] in self.flux]
        self.without = [p for p in pairs if not (p[0] in self.flux and p[1] in self.flux)]

    def __len__(self) -> int:
        return self.epoch

    def view(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self.redraw and path in self.flux:
            rng = np.random.default_rng(random.getrandbits(64))
            img = noisy(*self.flux[path], random.uniform(0, self.noise), rng, self.stretch).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        img = img.resize((RES, RES), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - self.mean) / self.std
        grid = torch.arange(GRID * GRID).reshape(GRID, GRID)
        k, flip = random.randrange(4), random.random() < 0.5
        x, grid = torch.rot90(x, k, (1, 2)), torch.rot90(grid, k, (0, 1))
        if flip:
            x, grid = x.flip(2), grid.flip(1)
        # Where each original patch ends up among the output tokens.
        where = torch.argsort(grid.reshape(-1))
        return x, where

    def __getitem__(self, i: int):
        if self.redraw and self.with_flux and self.without:
            a, b, direction = random.choice(self.with_flux if random.random() < 0.5 else self.without)
        else:
            a, b, direction = random.choice(self.pairs)
        xa, wa = self.view(a)
        xb, wb = self.view(b)
        ia, ib = shared(direction)
        return xa, xb, wa[ia], wb[ib]


def export(loras_by_name: dict[str, LoRA], out: Path) -> None:
    model = AutoModel.from_pretrained(REPO, torch_dtype=torch.float32)
    sd = model.state_dict()
    with torch.no_grad():
        for name, lora in loras_by_name.items():
            sd[name + ".weight"] += lora.delta().cpu()
    model.load_state_dict(sd)
    model.save_pretrained(out)
    shutil.copy(hf_hub_download(REPO, "preprocessor_config.json"), out / "preprocessor_config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--export_at", type=int, nargs="+", default=[500, 1000, 2000])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--stretch", type=float, default=1.0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    pairs = [p for root in args.roots for p in pairs_of(root)]
    tiles = sorted({p for a, b, _ in pairs for p in (a, b)})
    flux = cutouts(args.roots) if args.noise or args.stretch > 1 else {}
    print(f"{len(pairs)} pairs from {len(tiles)} tiles in {len(args.roots)} datasets, "
          f"{sum(t in flux for t in tiles)} drawn from flux with noise up to {args.noise} sigma "
          f"and stretch within {args.stretch}x", flush=True)

    model, loras = build(args.rank, 2.0 * args.rank)
    model.cuda().train()
    # In training mode DINOv3 rescales its RoPE positions at random (pos_embed_rescale
    # 2), which scoring never does, so the frozen pass wouldn't match it either.
    model.rope_embeddings.eval()
    names = {n: m for n, m in model.named_modules() if isinstance(m, LoRA)}
    params = [p for lora in loras for p in (lora.a, lora.b)]
    print(f"LoRA rank {args.rank} on {len(loras)} layers, {sum(p.numel() for p in params) / 1e6:.1f}M "
          "parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    ckpt, done = args.out / "ckpt.pt", 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cuda", weights_only=True)
        with torch.no_grad():
            for lora, (a, b) in zip(loras, state["loras"], strict=True):
                lora.a.copy_(a)
                lora.b.copy_(b)
        opt.load_state_dict(state["opt"])
        done = state["step"]
        print(f"resumed from step {done}", flush=True)

    def save(step: int) -> None:
        tmp = ckpt.with_suffix(".tmp")
        torch.save({"loras": [(lora.a.detach(), lora.b.detach()) for lora in loras],
                    "opt": opt.state_dict(), "step": step}, tmp)
        tmp.replace(ckpt)

    pre = json.loads(Path(hf_hub_download(REPO, "preprocessor_config.json")).read_text(encoding="utf-8"))
    data = Pairs(pairs, pre["image_mean"], pre["image_std"], (args.steps - done) * args.batch, flux, args.noise,
                 args.stretch)
    # A new seed after a resume, so the workers don't replay the draws they already made.
    loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch, num_workers=args.workers, persistent_workers=args.workers > 0,
        worker_init_fn=functools.partial(seed_worker, args.seed + done),
    )

    def patches(x: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=x).last_hidden_state[:, N_PREFIX:]
        return F.normalize(out.float(), dim=-1)

    start = time.time()
    for step, (xa, xb, wa, wb) in enumerate(loader, done + 1):
        lr = args.lr * min(1.0, step / 100) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        x = torch.cat([xa, xb]).cuda(non_blocking=True)
        for lora in loras:
            lora.on = False
        with torch.no_grad():
            teacher = patches(x)
        for lora in loras:
            lora.on = True
        student = patches(x)
        n = xa.shape[0]
        sa = torch.gather(student[:n], 1, wa.cuda()[..., None].expand(-1, -1, student.shape[-1]))
        sb = torch.gather(student[n:], 1, wb.cuda()[..., None].expand(-1, -1, student.shape[-1]))
        ta = torch.gather(teacher[:n], 1, wa.cuda()[..., None].expand(-1, -1, teacher.shape[-1]))
        tb = torch.gather(teacher[n:], 1, wb.cuda()[..., None].expand(-1, -1, teacher.shape[-1]))
        inv = (1 - (sa * sb).sum(-1)).mean()
        dist = (1 - (student * teacher).sum(-1)).mean()
        loss = inv + args.distill * dist
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 25 == 0 or step == 1:
            inv, dist = inv.detach(), dist.detach()
            frozen_gap = float((1 - (ta * tb).sum(-1)).mean())
            rate = (step - done) * args.batch * 2 / (time.time() - start)
            print(f"step {step} shared-patch gap {float(inv):.4f} (frozen model {frozen_gap:.4f}) "
                  f"distance from frozen {float(dist):.4f} lr {lr:.2e} {rate:.1f} img/s "
                  f"mem {torch.cuda.max_memory_allocated() / 1e9:.1f}G", flush=True)
        if step in args.export_at:
            export(names, args.out / f"step{step}")
            print(f"exported step {step}", flush=True)
        if step % 100 == 0 or step in args.export_at:
            save(step)
        if step >= args.steps:
            break
    print("training done", flush=True)


if __name__ == "__main__":
    main()
