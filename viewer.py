#!/usr/bin/env python3
"""Steps through ranked tiles, showing each PNG beside its raw patch-distance grid.

It reads samples_sorted_by_anomaly_score.csv, or any CSV with Sample and AnomalyScore
columns, and loads the .npy map saved for each tile. Every view is also saved to
--render_dir unless a render for it is already there. It opens a Qt window, so it
needs PyQt and a display. Press Enter in the terminal for the next tile and q to stop.

    python viewer.py --maps_dir results/pass2/anomaly_maps/seed=0 \\
        --png_dir datasets/miri/query --csv_path samples_sorted_by_anomaly_score.csv
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

# The backend has to be set before pyplot is imported, and only Qt has showMaximized.
os.environ.setdefault("MPLBACKEND", "QtAgg")
import matplotlib

matplotlib.use("QtAgg", force=True)

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
            raise RuntimeError(f"The CSV needs Sample and AnomalyScore columns, not {r.fieldnames}")

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
    """The .npy patch-distance grid saved for a tile.

    There is no TIFF fallback, on purpose. A missing .npy usually means --maps_dir
    points at the wrong run, and quietly showing something else would hide that.
    """
    rel = Path(sample_rel)
    stem_rel = rel.with_suffix("")
    npy_path = maps_dir / (str(stem_rel) + ".npy")
    if npy_path.exists():
        return np.asarray(np.load(str(npy_path))), npy_path

    # The CSV may name tiles relative to a different root than the maps.
    base = stem_rel.name
    hits = list(maps_dir.rglob(base + ".npy"))
    if hits:
        p = hits[0]
        return np.asarray(np.load(str(p))), p

    raise FileNotFoundError(f"There is no map for {sample_rel} under {maps_dir}")


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a), nan=0.0, posinf=0.0, neginf=0.0)
    amin = float(np.min(a))
    amax = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - amin) / (amax - amin)).astype(np.float32)


def _maximize(fig) -> None:
    """Maximize rather than go full screen, which would hide the taskbar."""
    mgr = getattr(fig.canvas, "manager", None)
    if mgr is None:
        return
    with contextlib.suppress(Exception):
        mgr.window.showMaximized()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maps_dir", required=True, type=str, help="Folder holding the .npy maps."
    )
    ap.add_argument(
        "--png_dir",
        required=True,
        type=str,
        help="Folder the CSV's Sample paths are under.",
    )
    ap.add_argument(
        "--csv_path", required=True, type=str, help="samples_sorted_by_anomaly_score.csv"
    )
    ap.add_argument("--top", default=500, type=int, help="Most tiles to step through.")
    ap.add_argument(
        "--render_dir",
        default="viewer_renders",
        type=str,
        help="Where renders are saved.",
    )
    ap.add_argument("--render_dpi", default=150, type=int, help="DPI for saved renders.")
    ap.add_argument("--no_maximize", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument(
        "--map_display",
        choices=["patchgrid", "upsampled"],
        default="patchgrid",
        help="Show the raw patch grid, or blow it up to the tile size. Both stay blocky.",
    )
    args = ap.parse_args()

    maps_dir = Path(args.maps_dir).resolve()
    png_dir = Path(args.png_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    render_dir = Path(args.render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_sorted_samples(csv_path)

    print(f"Loaded {len(samples)} tiles from {csv_path}")
    print("Press Enter for the next tile, or q then Enter to stop.")
    print(f"Drawing with the {matplotlib.get_backend()} backend")

    plt.ion()

    shown = 0
    for rank, (sample_rel, score) in enumerate(samples):
        if shown >= args.top:
            break

        png_path = png_dir / sample_rel
        if not png_path.exists():
            print(f"Skipping {png_path}, as it doesn't exist")
            continue

        try:
            amap, amap_path = _load_npy_map_only(maps_dir, sample_rel)
        except Exception as e:
            print(f"Skipping {sample_rel}. {e}")
            continue

        img = _load_png(png_path)
        h, w = img.shape[:2]

        amap = np.asarray(amap)
        print(
            f"{amap_path} {amap.shape} from {amap.min():.4g} to {amap.max():.4g}"
        )

        amap_disp = _normalize01(amap)

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        fig.canvas.manager.set_window_title("JWSTDetector Viewer")

        axs[0].imshow(img)
        axs[0].set_title("Tile")
        axs[0].axis("off")

        if args.map_display == "patchgrid":
            # nearest keeps each patch a solid block rather than smearing it into the next.
            axs[1].imshow(amap_disp, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"Patch Distances {amap.shape}")
        else:
            im = Image.fromarray((255.0 * amap_disp).astype(np.uint8)).resize(
                (w, h), resample=Image.NEAREST
            )
            axs[1].imshow(np.asarray(im) / 255.0, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"Patch Distances at {w}x{h}")

        axs[1].axis("off")

        slug = _slugify(sample_rel)
        out_render = render_dir / f"{rank:05d}_{slug}.png"

        fig.suptitle(f"Rank {rank}, score {score:.6f}, {sample_rel}", fontsize=12)
        fig.text(
            0.01,
            0.01,
            f"PNG: {png_path}\nNPY: {amap_path}\nNPY shape: {amap.shape}\nOUT: {out_render}",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        plt.tight_layout()

        if not out_render.exists():
            fig.savefig(out_render, dpi=args.render_dpi)
            print(f"Saved {out_render}")
        else:
            print(f"Kept the render already at {out_render}")

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
