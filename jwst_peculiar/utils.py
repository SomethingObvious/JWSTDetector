"""Turning patch distances into something you can look at, and the astropy aliases.

Import the aliases under `if TYPE_CHECKING:`, since this module loads torch and
matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

from jwst_peculiar.backbones import background_mask, embedding_rgb
from jwst_peculiar.data import TileDataset, relative_key

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from jwst_peculiar.backbones import Backbone

# astropy ships no type information, so its objects are Any to the type checkers.
# Naming them still says what each value is.
FitsHeader: TypeAlias = Any
FitsHDUList: TypeAlias = Any
SkyWcs: TypeAlias = Any
SkyCoords: TypeAlias = Any
AstropyTable: TypeAlias = Any
TableCell: TypeAlias = Any


def _to_pixels(
    grid: npt.NDArray[np.float32], shape: tuple[int, int], order: Literal[0, 1]
) -> npt.NDArray[np.float32]:
    # grid_mode lines each patch up with the pixels it covers. Without it scipy pins
    # the first and last patch centres to the image corners, which is up to half a
    # patch off at the edges.
    factors = (shape[0] / grid.shape[0], shape[1] / grid.shape[1])
    return zoom(grid, factors, order=order, grid_mode=True, mode="nearest")


def dists2map(
    dists: npt.NDArray[Any], img_shape: tuple[int, int], sigma: float = 4.0
) -> npt.NDArray[np.float32]:
    """Scale the patch-distance grid up to pixels and smooth it.

    Returns:
        The smoothed map, `img_shape` pixels in size.
    """
    grid = np.asarray(dists, dtype=np.float32)
    return gaussian_filter(_to_pixels(grid, img_shape[:2], order=1), sigma=sigma)


def plot_reference_masks(  # noqa: PLR0917 bootstrap and reference pass all four positionally
    backbone: Backbone, paths: list[Path], root: Path, out_path: Path, limit: int = 8
) -> None:
    """Draw a few reference tiles beside their embedding PCA and background mask.

    Only worth it while tuning --masking, whose effect is otherwise invisible until
    the scores look wrong.
    """
    mpl.use("Agg")

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
        axs[row][2].imshow(
            _to_pixels(mask.astype(np.float32), tile.shape[:2], order=0), alpha=0.5
        )
        axs[row][2].set_title("Background Mask", fontsize=8)
        for ax in axs[row]:
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
