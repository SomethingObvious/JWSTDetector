#!/usr/bin/env python3
"""Steps through ranked tiles, showing each PNG beside its raw patch-distance grid.

It reads samples_sorted_by_anomaly_score.csv, or any CSV with Sample and AnomalyScore
columns, and loads the .npy map saved for each tile. Every view is also saved to
--render_dir unless a render for it is already there. Press Enter in the terminal for
the next tile and q to stop.

    python viewer.py --maps_dir results/pass2/anomaly_maps/seed=0 \\
        --png_dir datasets/miri/query --csv_path samples_sorted_by_anomaly_score.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _slugify(s: str) -> str:
    s = s.replace("\\", "/").strip().strip("/")
    s = re.sub(r"[^A-Za-z0-9._/-]+", "_", s)
    return s.replace("/", "__") or "sample"


def _read_sorted_samples(csv_path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"{csv_path} has no header row")

        cols = {c.lower(): c for c in r.fieldnames}
        sample_col = cols.get("sample")
        score_col = cols.get("anomalyscore") or cols.get("anomaly_score") or cols.get("score")
        if not sample_col or not score_col:
            raise ValueError(
                f"{csv_path} needs Sample and AnomalyScore columns, not {r.fieldnames}"
            )

        for row in r:
            sample = (row.get(sample_col) or "").strip()
            try:
                score = float(row[score_col])
            except (ValueError, TypeError):
                continue
            if sample:
                rows.append((sample, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _load_map(maps_dir: Path, sample_rel: str) -> tuple[np.ndarray, Path]:
    """The .npy patch-distance grid saved for a tile.

    There is no TIFF fallback, on purpose. A missing .npy usually means --maps_dir
    points at the wrong run, and quietly showing something else would hide that.
    """
    stem = Path(sample_rel).with_suffix("")
    path = maps_dir / f"{stem}.npy"
    if not path.exists():
        # The CSV may name tiles relative to a different root than the maps.
        hits = list(maps_dir.rglob(f"{stem.name}.npy"))
        if not hits:
            raise FileNotFoundError(f"There is no map for {sample_rel} under {maps_dir}")
        path = hits[0]
    return np.load(path), path


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _maximize(fig) -> None:
    # Only Qt windows have showMaximized, and a window that stays small is fine.
    with contextlib.suppress(AttributeError):
        fig.canvas.manager.window.showMaximized()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--maps_dir", required=True, help="Folder holding the .npy maps.")
    ap.add_argument("--png_dir", required=True, help="Folder the CSV's Sample paths are under.")
    ap.add_argument("--csv_path", required=True, help="samples_sorted_by_anomaly_score.csv")
    ap.add_argument("--top", default=500, type=int, help="Most tiles to step through.")
    ap.add_argument("--render_dir", default="viewer_renders", help="Where renders are saved.")
    ap.add_argument("--render_dpi", default=150, type=int)
    ap.add_argument("--no_maximize", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument(
        "--map_display",
        choices=["patchgrid", "upsampled"],
        default="patchgrid",
        help="Show the raw patch grid, or blow it up to the tile size. Both stay blocky.",
    )
    args = ap.parse_args(argv)

    maps_dir = Path(args.maps_dir).resolve()
    png_dir = Path(args.png_dir).resolve()
    csv_path = Path(args.csv_path).resolve()
    render_dir = Path(args.render_dir).resolve()
    render_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_sorted_samples(csv_path)
    print(
        f"Loaded {len(samples)} tiles from {csv_path}, showing them in {matplotlib.get_backend()}"
    )
    print("Press Enter for the next tile, or q then Enter to stop.")

    plt.ion()
    shown = 0
    for rank, (sample_rel, score) in enumerate(samples):
        if shown >= args.top:
            break

        png_path = png_dir / sample_rel
        if not png_path.exists():
            print(f"Skipping {sample_rel}, as {png_path} doesn't exist")
            continue
        try:
            amap, amap_path = _load_map(maps_dir, sample_rel)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {sample_rel}. {e}")
            continue

        with Image.open(png_path) as im:
            img = np.asarray(im.convert("RGB"))
        h, w = img.shape[:2]
        print(f"{amap_path} {amap.shape} from {amap.min():.4g} to {amap.max():.4g}")
        amap_disp = _normalize01(amap)

        fig, axs = plt.subplots(1, 2, figsize=(16, 9))
        with contextlib.suppress(AttributeError):
            fig.canvas.manager.set_window_title("JWSTDetector Viewer")

        axs[0].imshow(img)
        axs[0].set_title("Tile")
        if args.map_display == "patchgrid":
            axs[1].imshow(amap_disp, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"Patch Distances {amap.shape}")
        else:
            big = Image.fromarray((255.0 * amap_disp).astype(np.uint8)).resize(
                (w, h), resample=Image.Resampling.NEAREST
            )
            axs[1].imshow(np.asarray(big) / 255.0, cmap="magma", interpolation="nearest")
            axs[1].set_title(f"Patch Distances at {w}x{h}")
        for ax in axs:
            ax.axis("off")

        out_render = render_dir / f"{rank:05d}_{_slugify(sample_rel)}.png"
        fig.suptitle(f"Rank {rank}, score {score:.6f}, {sample_rel}", fontsize=12)
        fig.text(
            0.01,
            0.01,
            f"PNG: {png_path}\nNPY: {amap_path}\nOUT: {out_render}",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        plt.tight_layout()
        if not out_render.exists():
            fig.savefig(out_render, dpi=args.render_dpi)

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
