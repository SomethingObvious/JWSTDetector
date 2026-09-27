"""Turning patch distances into something you can look at."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def _to_pixels(grid: np.ndarray, shape: tuple[int, int], order: int) -> np.ndarray:
    # grid_mode lines each patch up with the pixels it covers. Without it scipy pins
    # the first and last patch centres to the image corners, which is up to half a
    # patch off at the edges.
    factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
    return zoom(grid, factors, order=order, grid_mode=True, mode="nearest")


def dists2map(dists: np.ndarray, img_shape: tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """The patch-distance grid scaled up to pixels and smoothed."""
    grid = np.asarray(dists, dtype=np.float32)
    return gaussian_filter(_to_pixels(grid, img_shape[:2], order=1), sigma=sigma)


def plot_reference_masks(backbone, paths: list[Path], root: Path, out_path: Path, limit: int = 8):
    """Draw a few reference tiles beside their embedding PCA and background mask.

    It's only worth it while tuning --masking, which otherwise stays invisible until
    the scores start looking wrong.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from src.backbones import background_mask, embedding_rgb
    from src.data import TileDataset, relative_key

    paths = paths[:limit]
    dataset = TileDataset(paths, root, backbone.transform())
    grid = backbone.grid_size

    fig, axs = plt.subplots(len(paths), 3, figsize=(9, 3 * len(paths)), squeeze=False)
    for row, path in enumerate(paths):
        _, tensor = dataset[row]
        feats = backbone.embed(tensor.unsqueeze(0))
        mask = background_mask(feats, grid)[0].reshape(grid).cpu().numpy()
        rgb = embedding_rgb(feats[0], grid).cpu().numpy()

        with Image.open(path) as img:
            tile = np.asarray(img.convert("RGB"))

        axs[row][0].imshow(tile)
        axs[row][0].set_title(relative_key(path, root), fontsize=8)
        axs[row][1].imshow(rgb)
        axs[row][1].set_title("PCA of Patch Embeddings", fontsize=8)
        axs[row][2].imshow(tile)
        axs[row][2].imshow(_to_pixels(mask.astype(np.float32), tile.shape[:2], order=0), alpha=0.5)
        axs[row][2].set_title("Background Mask", fontsize=8)
        for ax in axs[row]:
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
