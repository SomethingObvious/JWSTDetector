# pyright: reportPrivateUsage=false
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from jwst_peculiar import bootstrap
from jwst_peculiar.detection import Footprint

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


def test_no_calibrate_ranks_without_p_values(tmp_path: Path) -> None:
    _field(tmp_path / "target", 6, seed=1, odd=True)
    for j in range(2):
        _field(tmp_path / f"ctrl{j}", 8, seed=10 + j)
    out = tmp_path / "out"
    bootstrap.main(
        [
            "--data_root", str(tmp_path / "target"),
            "--reference_roots", *(str(tmp_path / f"ctrl{j}") for j in range(2)),
            "--no-calibrate",
            "--out_dir", str(out),
            "--model_name", "pixels",
            "--knn_metric", "L2",
            "--resolution", str(SIZE),
            "--device", "cpu",
            "--num_workers", "0",
            "--feature_cache", "none",
        ]
    )  # fmt: skip
    with (out / "measurements_final.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert max(rows, key=lambda r: float(r["AnomalyScore"]))["Sample"] == "t000.png"
    assert not (out / "significance.csv").exists()
    assert not (out / "calibration.csv").exists()


def _grid(name: str, x0: int, y0: int, *, ra0: float = 34.3) -> dict[Path, Footprint]:
    """Nine tiles of a mosaic whose pixels are 0.04 arcsec, from one corner.

    Returns:
        Their footprints by a made-up tile path.
    """
    scale = 0.04 / 3600
    return {
        Path(f"{name}_{x0}_{y0}_{i}.png"): Footprint(
            name,
            x0 + 320 * (i % 3),
            y0 + 320 * (i // 3),
            320,
            ra=ra0 - (x0 + 320 * (i % 3)) * scale,
            dec=-5.2 + (y0 + 320 * (i // 3)) * scale,
        )
        for i in range(9)
    }


def test_blocks_of_one_mosaic_can_share_its_name() -> None:
    a = bootstrap._Field("b00", bootstrap._Tiles([], Path()), _grid("m", 0, 0))
    b = bootstrap._Field("b01", bootstrap._Tiles([], Path()), _grid("m", 1920, 0))
    bootstrap._check_mosaic_names([a, b], _grid("m", 0, 1920))


def test_two_mosaics_with_one_name_still_stop() -> None:
    a = bootstrap._Field("a", bootstrap._Tiles([], Path()), _grid("m", 0, 0))
    b = bootstrap._Field("b", bootstrap._Tiles([], Path()), _grid("m", 0, 0, ra0=150.1))
    with pytest.raises(SystemExit, match="different grids"):
        bootstrap._check_mosaic_names([a, b], {})


@pytest.mark.parametrize(
    "flags",
    [
        ["--score_top_frac", "5"],
        ["--subspace_variance", "1"],
        ["--drop_directions", "-1"],
        ["--detector", "iforest", "--reference_roots", "a", "b"],
    ],
)
def test_out_of_range_options_stop_the_run(flags: list[str]) -> None:
    with pytest.raises(SystemExit, match=r"has to be|on its own"):
        bootstrap.main(["--data_root", "nowhere", *flags])
