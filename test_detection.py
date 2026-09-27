"""Checks for the memory bank, the search and a whole scoring pass, with a stand-in backbone.

Run it with `python test_detection.py`, or with pytest. It needs torch but no GPU,
network or weights, as a small deterministic module takes DINO's place.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import jwstdetector.detection as detection
from jwstdetector.backbones import Backbone, hub_repo
from jwstdetector.detection import MemoryBank, build_memory_bank, dihedral, run_anomaly_detection

PATCH, RESOLUTION, DIM = 16, 128, 32
GRID = RESOLUTION // PATCH
ANOMALOUS = "A5_cx000264_cy000264_ra150.999999_decm2.197645_i00000099.png"


def _stub_backbone(seed: int = 0) -> Backbone:
    """Each patch averaged down to its mean colour and projected, which is content-sensitive
    and deterministic, and that's all the detector needs from a backbone."""
    projection = torch.randn(3, DIM, generator=torch.Generator().manual_seed(seed))

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

    The names carry an RA and Dec the way prep_data --wcs_in_name writes them, so the
    map file names have to survive the dots in the stem.
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


def _run(tmp: Path, include_odd_in_reference: bool = False, **kwargs):
    query_dir = tmp / "query"
    names = _write_tiles(query_dir)
    ref_paths = [query_dir / n for n in names if include_odd_in_reference or n != ANOMALOUS]
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


def test_hub_names() -> None:
    assert hub_repo("dinov2_vitl14") == "facebook/dinov2-large"
    assert hub_repo("dinov2_vits14_reg") == "facebook/dinov2-with-registers-small"
    assert hub_repo("dinov3-vitl16-pretrain-sat493m") == "facebook/dinov3-vitl16-pretrain-sat493m"
    assert hub_repo("someone/else") == "someone/else"


def test_memory_bank_metrics() -> None:
    bank_features = torch.eye(4, 8)
    for metric in ("L2_normalized", "L2"):
        bank = MemoryBank(bank_features.clone(), metric)
        d = bank.distances(bank_features.clone(), k=1)
        assert torch.allclose(d, torch.zeros(4), atol=1e-5), (metric, d)

    bank = MemoryBank(bank_features.clone(), "L2_normalized")
    far = torch.zeros(1, 8)
    far[0, 5] = 1.0
    assert torch.allclose(bank.distances(far, k=1), torch.ones(1))


def test_distances_skip_the_querys_own_tile() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    bank = MemoryBank(features.clone(), "L2_normalized", torch.tensor([0, 1, 2], dtype=torch.int32))
    query = features[:1].clone()
    # Its own patch is an exact match. Skipped, the nearest is [0.6, 0.8] at 1 - 0.6.
    assert torch.allclose(bank.distances(query, 1), torch.zeros(1))
    own = bank.distances(query, 1, torch.tensor([0], dtype=torch.int32))
    assert torch.allclose(own, torch.tensor([0.4])), own
    assert torch.allclose(
        bank.distances(query, 1, torch.tensor([-1], dtype=torch.int32)), torch.zeros(1)
    )


def test_chunking_matches_a_single_pass() -> None:
    torch.manual_seed(0)
    bank = MemoryBank(torch.randn(64, 16), "L2", torch.arange(64, dtype=torch.int32) % 5)
    queries = torch.randn(200, 16)
    owners = torch.arange(200, dtype=torch.int32) % 7 - 1
    full = bank.distances(queries, k=3, owners=owners)

    original = detection._MAX_DISTANCE_ELEMENTS
    try:
        detection._MAX_DISTANCE_ELEMENTS = 64 * 7  # several ragged chunks
        chunked = bank.distances(queries, k=3, owners=owners)
    finally:
        detection._MAX_DISTANCE_ELEMENTS = original
    assert torch.allclose(full, chunked, atol=1e-6)


def test_dihedral_is_lossless() -> None:
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    views = dihedral(x)
    assert len(views) == 8
    assert all(v.shape == x.shape for v in views)
    for v in views:
        assert torch.equal(v.flatten().sort().values, x.flatten().sort().values)
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8


def test_end_to_end_ranks_the_odd_tile_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scores = _run(tmp)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]

        # One map per tile, named for the tile, which with_suffix(".npy") would have
        # cut at the last dot of the RA.
        maps_dir = tmp / "out/anomaly_maps/seed=0"
        maps = sorted(maps_dir.glob("*.npy"))
        assert len(maps) == len(scores), [p.name for p in maps]
        for key in scores:
            assert (maps_dir / f"{key[:-4]}.npy").exists(), key
        assert np.load(maps[0]).shape == (GRID, GRID)


def test_reference_tile_is_scored_against_the_others() -> None:
    # The label-free runner scores tiles that are also in its reference. Matched
    # against its own patches, the odd tile would score 0 and sink to the bottom.
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        outside = _run(Path(a))
        inside = _run(Path(b), include_odd_in_reference=True)
        ranked = sorted(inside, key=inside.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]
        assert abs(inside[ANOMALOUS] - outside[ANOMALOUS]) < 1e-6
        assert min(inside.values()) > 0.0


def test_end_to_end_options() -> None:
    # Augmentation, masking, the L2 metric, k > 1 and TIFF output all have to get
    # through a real pass.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scores = _run(
            tmp,
            include_odd_in_reference=True,
            rotation=True,
            masking=True,
            mask_ref_images=True,
            knn_metric="L2",
            k_neighbors=3,
            score_mode="peak_minus_median",
            save_tiffs=True,
        )
        assert all(np.isfinite(v) for v in scores.values())
        tiff = tmp / f"out/anomaly_maps/seed=0/{ANOMALOUS[:-4]}.tiff"
        assert tiff.exists()


def _bank(max_patches: int, seed: int = 0, **kwargs) -> MemoryBank:
    with tempfile.TemporaryDirectory() as td:
        query_dir = Path(td) / "query"
        _write_tiles(query_dir)
        paths = sorted(query_dir.glob("*.png"))
        return build_memory_bank(
            _stub_backbone(), paths, query_dir, num_workers=0, max_patches=max_patches,
            seed=seed, **kwargs,
        )  # fmt: skip


def test_bank_holds_every_patch_under_the_cap() -> None:
    bank = _bank(10**9)
    assert len(bank) == 13 * GRID * GRID
    assert bank.owners.tolist() == [i for i in range(13) for _ in range(GRID * GRID)]


def test_bank_sample_fills_the_cap_evenly_and_repeats() -> None:
    cap = 13 * GRID * GRID // 2
    bank = _bank(cap)
    assert len(bank) == cap
    per_tile = np.bincount(bank.owners.numpy(), minlength=13)
    # Each tile should give about 32 of its 64 patches, the last tile as much as the first.
    assert per_tile.min() >= 20 and per_tile.max() <= 44, per_tile

    again = _bank(cap)
    assert torch.equal(bank.features, again.features)
    assert torch.equal(bank.owners, again.owners)
    assert not torch.equal(bank.owners, _bank(cap, seed=1).owners)


def test_masked_bank_still_fills_the_cap() -> None:
    # Masking leaves fewer patches than the tile count suggests, and the bank should
    # still fill all the way to the cap.
    everything = len(_bank(10**9, masking=True))
    assert everything < 13 * GRID * GRID
    assert len(_bank(everything // 2, masking=True)) == everything // 2


def main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
