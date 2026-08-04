"""End-to-end check of the detector with a stand-in backbone.

Run with ``python test_detection.py``. Needs torch but no GPU, no network and no
downloaded weights: a deterministic pooling-and-projection module stands in for
DINO so the memory bank, the nearest-neighbour search, masking, augmentation and
the map writing all run for real.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.backbones import Backbone
from src.detection import MemoryBank, dihedral, run_anomaly_detection

PATCH, RESOLUTION, DIM = 16, 128, 32
GRID = RESOLUTION // PATCH
ANOMALOUS = "A5_cx000264_cy000264_ra150.999999_decm2.197645_i00000099.png"


def _stub_backbone(seed: int = 0) -> Backbone:
    """Average each patch down to its mean colour, then project it. Content-sensitive
    and deterministic, which is all the detector needs from a backbone."""
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(3, DIM, generator=generator)

    def forward_patches(x: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(x, PATCH)  # (B, 3, GRID, GRID)
        return pooled.flatten(2).transpose(1, 2) @ projection

    return Backbone(
        name="stub",
        model=torch.nn.Identity(),
        forward_patches=forward_patches,
        patch_size=PATCH,
        resolution=RESOLUTION,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        device=torch.device("cpu"),
        autocast_dtype=None,
    )


def _write_tiles(root: Path, n_normal: int = 12) -> list[str]:
    """A field of ordinary tiles plus one with a bright bar across it.

    Names carry an RA/Dec the way prep_data --wcs_in_name writes them, so the dots in
    the stem stay in the test's way and the map filenames have to survive them.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for i in range(n_normal):
        tile = rng.integers(40, 70, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
        name = f"A5_cx000064_cy000064_ra150.10{i:04d}_decm2.197645_i{i:08d}.png"
        Image.fromarray(tile).save(root / name)
        names.append(name)

    odd = rng.integers(40, 70, (RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    odd[48:80, :, :] = 255
    Image.fromarray(odd).save(root / ANOMALOUS)
    names.append(ANOMALOUS)
    return names


def test_memory_bank_metrics() -> None:
    bank_features = torch.eye(4, 8)
    for metric, expected in (("L2_normalized", 0.0), ("L2", 0.0)):
        bank = MemoryBank(bank_features.clone(), metric)
        d = bank.distances(bank_features.clone(), k=1)
        assert torch.allclose(d, torch.full((4,), expected), atol=1e-5), (metric, d)

    # A vector that is in the bank must sit closer than one that is not.
    bank = MemoryBank(bank_features.clone(), "L2_normalized")
    far = torch.zeros(1, 8)
    far[0, 5] = 1.0
    assert bank.distances(far, k=1).item() > bank.distances(bank_features[:1], k=1).item()


def test_memory_bank_chunking_matches_single_pass() -> None:
    # The search chunks its query set; chunk boundaries must not change the answer.
    import src.detection as detection

    torch.manual_seed(0)
    bank = MemoryBank(torch.randn(64, 16), "L2")
    queries = torch.randn(200, 16)
    full = bank.distances(queries, k=3)

    original = detection._MAX_DISTANCE_ELEMENTS
    try:
        detection._MAX_DISTANCE_ELEMENTS = 64 * 7  # forces several ragged chunks
        chunked = bank.distances(queries, k=3)
    finally:
        detection._MAX_DISTANCE_ELEMENTS = original
    assert torch.allclose(full, chunked, atol=1e-6)


def test_dihedral_is_lossless() -> None:
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    views = dihedral(x)
    assert len(views) == 8
    assert all(v.shape == x.shape for v in views)
    # Exact symmetries: every view is a permutation of the same pixels.
    for v in views:
        assert torch.equal(v.flatten().sort().values, x.flatten().sort().values)
    # And they are all distinct, so the bank actually gains eight views.
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8


def _run(tmp: Path, **kwargs):
    query_dir = tmp / "query"
    names = _write_tiles(query_dir)
    ref_paths = [query_dir / n for n in names if n != ANOMALOUS]
    query_paths = sorted(query_dir.glob("*.png"))

    scores, memorybank_sec, seconds = run_anomaly_detection(
        _stub_backbone(),
        ref_paths,
        query_dir,
        query_paths,
        query_dir,
        tmp / "out",
        num_workers=0,
        batch_size=4,
        **kwargs,
    )
    assert set(scores) == {p.name for p in query_paths}
    assert set(seconds) == set(scores)
    assert memorybank_sec >= 0.0
    return scores


def test_end_to_end_ranks_the_odd_tile_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scores = _run(tmp)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]

        # One map per tile, named for the tile. Path.with_suffix would have chopped
        # these stems at the last dot of the RA and silently collided them.
        maps_dir = tmp / "out/anomaly_maps/seed=0"
        maps = sorted(maps_dir.glob("*.npy"))
        assert len(maps) == len(scores), [p.name for p in maps]
        for key in scores:
            assert (maps_dir / f"{key[:-4]}.npy").exists(), key
        assert np.load(maps[0]).shape == (GRID, GRID)


def test_end_to_end_options() -> None:
    # Augmentation, masking, the L2 metric, k>1 and TIFF output all have to survive
    # a real pass, not just import cleanly.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scores = _run(
            tmp,
            rotation=True,
            masking=True,
            mask_ref_images=True,
            knn_metric="L2",
            k_neighbors=3,
            score_mode="peak_minus_median",
            save_tiffs=True,
        )
        assert all(np.isfinite(v) for v in scores.values())
        assert (tmp / f"out/anomaly_maps/seed=0/{ANOMALOUS[:-4]}.tiff").exists()


def test_memory_bank_cap_is_respected() -> None:
    from src.detection import build_memory_bank

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        query_dir = tmp / "query"
        _write_tiles(query_dir)
        paths = sorted(query_dir.glob("*.png"))

        uncapped = build_memory_bank(
            _stub_backbone(), paths, query_dir, num_workers=0, max_patches=10**9
        )
        assert len(uncapped) == len(paths) * GRID * GRID

        cap = 100
        capped = build_memory_bank(
            _stub_backbone(), paths, query_dir, num_workers=0, max_patches=cap, seed=1
        )
        assert len(capped) <= cap, len(capped)
        assert len(capped) > 0


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
