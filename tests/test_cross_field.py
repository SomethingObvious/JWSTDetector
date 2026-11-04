from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from jwstdetector import bootstrap

if TYPE_CHECKING:
    from pathlib import Path

SIZE = 64


def _field(root: Path, n: int, *, seed: int, odd: bool = False) -> None:
    query = root / "query"
    query.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        tile = rng.integers(40, 70, (SIZE, SIZE, 3), dtype=np.uint8)
        if odd and i == 0:
            tile[24:40, :, :] = 255
        Image.fromarray(tile).save(query / f"t{i:03d}.png")


def test_cross_field_calibrates_against_the_other_fields(tmp_path: Path) -> None:
    _field(tmp_path / "target", 6, seed=1, odd=True)
    for j in range(3):
        _field(tmp_path / f"ctrl{j}", 8, seed=10 + j)
    out = tmp_path / "out"
    bootstrap.main(
        [
            "--data_root", str(tmp_path / "target"),
            "--reference_roots", *(str(tmp_path / f"ctrl{j}") for j in range(3)),
            "--out_dir", str(out),
            "--model_name", "pixels",
            "--knn_metric", "L2",
            "--resolution", str(SIZE),
            "--device", "cpu",
            "--num_workers", "0",
            "--feature_cache", "none",
        ]
    )  # fmt: skip
    with (out / "significance.csv").open(newline="", encoding="utf-8") as f:
        rows = {
            r["Sample"]: (float(r["PValue"]), float(r["QValue"]))
            for r in csv.DictReader(f)
        }
    assert set(rows) == {f"t{i:03d}.png" for i in range(6)}
    # The bar beats all 24 calibration tiles, which is the smallest p there is.
    assert rows["t000.png"][0] == 1 / 25, rows["t000.png"]
    assert min(rows, key=lambda k: rows[k][0]) == "t000.png"
    with (out / "calibration.csv").open(newline="", encoding="utf-8") as f:
        fields = [r["Field"] for r in csv.DictReader(f)]
    assert sorted(set(fields)) == ["ctrl0", "ctrl1", "ctrl2"]
    assert len(fields) == 24
    assert (out / "measurements_final.csv").is_file()
