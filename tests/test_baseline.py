from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from jwstdetector import bootstrap
from jwstdetector.data import read_measurements

if TYPE_CHECKING:
    from pathlib import Path


def test_isolation_forest_puts_the_odd_tile_first(tmp_path: Path) -> None:
    query = tmp_path / "data" / "query"
    query.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(40):
        tile = rng.integers(40, 70, (64, 64, 3), dtype=np.uint8)
        if i == 7:
            tile[28:36, 8:56] = (250, 120, 40)
        Image.fromarray(tile).save(query / f"t{i:03d}.png")
    out = tmp_path / "out"
    bootstrap.main([
        "--data_root",
        str(tmp_path / "data"),
        "--out_dir",
        str(out),
        "--detector",
        "iforest",
    ])
    scores = read_measurements(out / "measurements_final.csv")
    assert len(scores) == 40
    assert max(scores, key=scores.__getitem__) == "t007.png"
