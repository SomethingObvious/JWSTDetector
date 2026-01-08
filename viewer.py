#!/usr/bin/env python3
"""
viewer.py

Viewer for anomaly scores that:
  - Reads samples_sorted_by_anomaly_score.csv (or any CSV with Sample + AnomalyScore cols)
  - Loads the corresponding PNG tile
  - Loads the corresponding anomaly MAP **from .npy ONLY** (raw patch-distance grid)
  - Displays PNG + NPY patch grid side-by-side (NPY shown blocky via interpolation='nearest')
  - Saves a render to viewer_renders/ only if it's new
  - Opens a Qt popup window (QtAgg) and maximizes it (taskbar stays visible)

Controls (terminal):
  - Enter or any text + Enter: next
  - q / quit / exit + Enter: quit

Example:
  python viewer.py \
    --maps_dir /path/to/pass2/anomaly_maps/seed=0 \
    --png_dir  /path/to/dataset/query \
    --csv_path /path/to/samples_sorted_by_anomaly_score.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

# Force Qt backend BEFORE importing pyplot. [web:108]
os.environ.setdefault("MPLBACKEND", "QtAgg")
import matplotlib
matplotlib.use("QtAgg", force=True)

import matplotlib.pyplot as plt  # noqa: E402


def _slugify(s: str) -> str:
    s = s.replace("\\", "/").strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9._/-]+", "_", s)
    s = s.replace("/", "__")
    return s or "sample"


def _read_sorted_samples(csv_path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"CSV has no header: {csv_path}")

        cols = {c.lower(): c for c in r.fieldnames}
        sample_col = cols.get("sample")
        score_col = cols.get("anomalyscore") or cols.get("anomaly_score") or cols.get("score")
        if not sample_col or not score_col:
            raise RuntimeError(f"Need columns Sample + AnomalyScore, got: {r.fieldnames}")

        for row in r:
            s = (row.get(sample_col) or "").strip()
            if not s:
                continue
            try:
                score = float(row[score_col])
            except Exception:
                continue
            rows.append((s, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _load_png(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def _load_npy_map_only(maps_dir: Path, sample_rel: str) -> tuple[np.ndarray, Path]:
    """
    Loads ONLY .npy maps (raw patch-distance grid saved by detection). [file:16]
    No TIFF fallback by design; if missing, fail loudly so you notice path issues.
    """
    rel = Path(sample_rel)
    stem_rel = rel.with_suffix("")  # keep subfolders, drop extension
    npy_path = maps_dir / (str(stem_rel) + ".npy")
    if npy_path.exists():
        return np.asarray(np.load(str(npy_path))), npy_path

    # Optional fallback: search by basename (still only .npy)
    base = stem_rel.name
    hits = list(maps_dir.rglob(base + ".npy"))
    if hits:
        p = hits[0]
        return np.asarray(np.load(str(p))), p

    raise FileNotFoundError(f"NPY map not found for sample={sample_rel} under {maps_dir}")


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(np.min(a))
    amax = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def _maximize(fig) -> None:
    """Maximize window (keeps taskbar visible)."""
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is None:
        return
    try:
        mgr.window.showMaximized()
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps_dir", required=True, type=str, help="Directory containing .npy anomaly maps.")
    ap.add_argument("--png_dir", required=True, type=str, help="Directory containing PNG tiles (mirrors CSV Sample paths).")
    ap.add_argument("--csv_path", required=True, type=str, help="Path to samples_sorted_by_anomaly_score.csv.")
    ap.add_argument("--top", default=500, type=int, help="Max samples to step through.")
    ap.add_argument("--render_dir", default="viewer_renders", type=str, help="Where to save side-by-side renders.")
    ap.add_argument("--render_dpi", default=150, type=int, help="DPI for saved renders.")
    ap.add_argument("--no_maximize", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument(
        "--map_display",
        choices=["patchgrid", "upsampled"],
        default="patchgrid",
        help="patchgrid=show raw .npy grid (blocky). upsampled=resize to PNG size (still nearest).",
    )
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir).resolve()
    png_dir = Path(args.png_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    render_dir = Path(args.render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_sorted_samples(csv_path)

    print(f"Loaded {len(samples)} samples from: {csv_path}")
    print("Controls: Enter=next, any text=next, q/quit/exit=stop")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")

    plt.ion()

    shown = 0
    for rank, (sample_rel, score) in enumerate(samples):
        if shown >= args.top:
            break

        png_path = png_dir / sample_rel
        if not png_path.exists():
            print(f"[skip] missing PNG: {png_path}")
            continue

        try:
            amap, amap_path = _load_npy_map_only(maps_dir, sample_rel)
        except Exception as e:
            print(f"[skip] npy map load failed for {sample_rel}: {e}")
            continue

        img = _load_png(png_path)
        h, w = img.shape[:2]

        # Make it obvious what you loaded:
        amap = np.asarray(amap)
        print(f"MAP LOADED: {amap_path}  shape={amap.shape}  min={amap.min():.4g} max={amap.max():.4g}")

        # For visualization only:
        amap_disp = _normalize01(amap)

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        fig.canvas.manager.set_window_title("AnomalyDINO Viewer (NPY)")

        axs[0].imshow(img)
        axs[0].set_title("PNG")
        axs[0].axis("off")

        if args.map_display == "patchgrid":
            # Show the raw patch grid; nearest prevents smoothing when scaled up. [web:227]
            axs[1].imshow(amap_disp, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"NPY patch grid {amap.shape} (nearest)")
        else:
            # Upsample to image size but still force nearest to remain blocky.
            im = Image.fromarray((255.0 * amap_disp).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
            axs[1].imshow(np.asarray(im) / 255.0, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"NPY upsampled to {w}x{h} (nearest)")

        axs[1].axis("off")

        slug = _slugify(sample_rel)
        out_render = render_dir / f"{rank:05d}_{slug}.png"

        fig.suptitle(f"Rank={rank}  Score={score:.6f} | {sample_rel}", fontsize=12)
        fig.text(
            0.01,
            0.01,
            f"PNG: {png_path}\nNPY: {amap_path}\nNPY shape: {amap.shape}\nOUT: {out_render}",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        plt.tight_layout()

        # Save render only if new.
        if not out_render.exists():
            fig.savefig(out_render, dpi=args.render_dpi)
            print(f"[saved] {out_render}")
        else:
            print(f"[exists] {out_render}")

        # Popup window
        plt.show(block=False)
        plt.pause(0.001)
        if not args.no_maximize:
            _maximize(fig)

        cmd = input().strip().lower()
        plt.close(fig)

        if cmd in ("q", "quit", "exit"):
            break

        shown += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
