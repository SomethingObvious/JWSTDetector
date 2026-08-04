"""Small helpers for turning patch distances into something you can look at."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def dists2map(dists: np.ndarray, img_shape: tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """Blow the patch-distance grid up to pixel space and smooth it."""
    h, w = img_shape[:2]
    grid = np.asarray(dists, dtype=np.float32)
    scaled = zoom(grid, (h / grid.shape[0], w / grid.shape[1]), order=1)
    return gaussian_filter(scaled, sigma=sigma)


def plot_reference_masks(backbone, paths: list[Path], root: Path, out_path: Path, limit: int = 8):
    """Render a few reference tiles beside their PCA embedding and background mask.

    Only worth running when you are tuning --masking, which is otherwise invisible
    until you notice the scores looking wrong.
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
        axs[row][1].set_title("PCA of patch embeddings", fontsize=8)
        scale = (tile.shape[0] / grid[0], tile.shape[1] / grid[1])
        axs[row][2].imshow(tile)
        axs[row][2].imshow(zoom(mask.astype(np.float32), scale, order=0), alpha=0.5)
        axs[row][2].set_title("background mask", fontsize=8)
        for ax in axs[row]:
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
