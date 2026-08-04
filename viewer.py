#!/usr/bin/env python3
"""Steps through the highest-scoring tiles, each beside its patch-distance map.

It reads samples_sorted_by_anomaly_score.csv, or any CSV with Sample and AnomalyScore
columns, and shows each PNG tile next to the raw .npy grid of patch distances in a
maximised window. A render of each one goes to viewer_renders/ unless it's already
there. Press Enter in the terminal for the next tile, or type q and Enter to stop.

Usage:
  python viewer.py \
    --maps_dir /path/to/pass2/anomaly_maps/seed=0 \
    --png_dir  /path/to/dataset/query \
    --csv_path /path/to/samples_sorted_by_anomaly_score.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

# The backend has to be picked before pyplot is imported. Qt is an optional extra
# (pip install -e '.[viewer]'), so without it this falls back to whatever matplotlib
# would have chosen rather than failing on the import.
os.environ.setdefault("MPLBACKEND", "QtAgg")
import matplotlib

try:
    matplotlib.use("QtAgg", force=True)
except ImportError:
    os.environ.pop("MPLBACKEND", None)
    matplotlib.use(matplotlib.rcParamsDefault["backend"], force=True)

import matplotlib.pyplot as plt


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
            raise RuntimeError(f"{csv_path} has no header row")

        cols = {c.lower(): c for c in r.fieldnames}
        sample_col = cols.get("sample")
        score_col = cols.get("anomalyscore") or cols.get("anomaly_score") or cols.get("score")
        if not sample_col or not score_col:
            raise RuntimeError(f"The CSV needs Sample and AnomalyScore columns, and it has {r.fieldnames}")

        for row in r:
            s = (row.get(sample_col) or "").strip()
            if not s:
                continue
            try:
                score = float(row[score_col])
            except (ValueError, TypeError):
                continue
            rows.append((s, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _load_png(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def _load_npy_map_only(maps_dir: Path, sample_rel: str) -> tuple[np.ndarray, Path]:
    """The saved .npy patch-distance grid for a sample, and the path it came from.

    There is no TIFF fallback on purpose, so a wrong --maps_dir fails loudly rather
    than quietly showing a smoothed map.
    """
    rel = Path(sample_rel)
    stem_rel = rel.with_suffix("")
    npy_path = maps_dir / (str(stem_rel) + ".npy")
    if npy_path.exists():
        return np.asarray(np.load(str(npy_path))), npy_path

    # The same file name anywhere under maps_dir, in case the subfolders don't match.
    base = stem_rel.name
    hits = list(maps_dir.rglob(base + ".npy"))
    if hits:
        p = hits[0]
        return np.asarray(np.load(str(p))), p

    raise FileNotFoundError(f"There is no .npy map for {sample_rel} under {maps_dir}")


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(np.min(a))
    amax = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def _maximize(fig) -> None:
    """Maximise the window, which unlike full screen leaves the taskbar showing."""
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is None:
        return
    with contextlib.suppress(Exception):
        mgr.window.showMaximized()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maps_dir", required=True, type=str, help="Folder holding the .npy anomaly maps."
    )
    ap.add_argument(
        "--png_dir",
        required=True,
        type=str,
        help="Folder holding the PNG tiles, laid out like the CSV's Sample paths.",
    )
    ap.add_argument(
        "--csv_path", required=True, type=str, help="Path to samples_sorted_by_anomaly_score.csv."
    )
    ap.add_argument("--top", default=500, type=int, help="Most tiles to step through.")
    ap.add_argument(
        "--render_dir",
        default="viewer_renders",
        type=str,
        help="Where the side-by-side renders go.",
    )
    ap.add_argument("--render_dpi", default=150, type=int, help="DPI of the saved renders.")
    ap.add_argument("--no_maximize", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument(
        "--map_display",
        choices=["patchgrid", "upsampled"],
        default="patchgrid",
        help="patchgrid shows the raw .npy grid, and upsampled resizes it to the tile's size.",
    )
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir).resolve()
    png_dir = Path(args.png_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    render_dir = Path(args.render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_sorted_samples(csv_path)

    print(f"Loaded {len(samples)} tiles from {csv_path}")
    print("Press Enter for the next tile, or type q and Enter to stop.")
    print(f"Matplotlib is using the {matplotlib.get_backend()} backend")

    plt.ion()

    shown = 0
    for rank, (sample_rel, score) in enumerate(samples):
        if shown >= args.top:
            break

        png_path = png_dir / sample_rel
        if not png_path.exists():
            print(f"[skip] There is no PNG at {png_path}")
            continue

        try:
            amap, amap_path = _load_npy_map_only(maps_dir, sample_rel)
        except Exception as e:
            print(f"[skip] Couldn't load the map for {sample_rel}. {e}")
            continue

        img = _load_png(png_path)
        h, w = img.shape[:2]

        amap = np.asarray(amap)
        print(
            f"Map {amap_path}, shape {amap.shape}, min {amap.min():.4g}, max {amap.max():.4g}"
        )

        # Normalised for display only. The raw values are in the line printed above.
        amap_disp = _normalize01(amap)

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        fig.canvas.manager.set_window_title("AnomalyDINO Viewer")

        axs[0].imshow(img)
        axs[0].set_title("PNG")
        axs[0].axis("off")

        if args.map_display == "patchgrid":
            # Nearest keeps each patch a hard-edged block when matplotlib scales it up.
            axs[1].imshow(amap_disp, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"NPY Patch Grid {amap.shape}")
        else:
            # Still nearest, so the patches stay blocky at the tile's size.
            im = Image.fromarray((255.0 * amap_disp).astype(np.uint8)).resize(
                (w, h), resample=Image.NEAREST
            )
            axs[1].imshow(np.asarray(im) / 255.0, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"NPY Upsampled to {w}x{h}")

        axs[1].axis("off")

        slug = _slugify(sample_rel)
        out_render = render_dir / f"{rank:05d}_{slug}.png"

        fig.suptitle(f"Rank {rank}, Score {score:.6f}, {sample_rel}", fontsize=12)
        fig.text(
            0.01,
            0.01,
            f"PNG: {png_path}\nNPY: {amap_path}\nNPY shape: {amap.shape}\nRender: {out_render}",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        plt.tight_layout()

        if not out_render.exists():
            fig.savefig(out_render, dpi=args.render_dpi)
            print(f"[saved] {out_render}")
        else:
            print(f"[skip] {out_render} is already there")

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
