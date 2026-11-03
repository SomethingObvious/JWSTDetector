"""Checks for the memory bank, the search, the feature cache and whole scoring passes.

They need torch but no GPU, network or weights, as a small deterministic module
takes DINO's place.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import jwstdetector.detection as detection
from jwstdetector.backbones import Backbone, background_mask, hub_repo, parse_layers
from jwstdetector.detection import (
    Footprint,
    MemoryBank,
    SubspaceModel,
    build_memory_bank,
    dihedral,
    excluded_refs,
    run_anomaly_detection,
)

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


def test_distances_skip_excluded_tiles() -> None:
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    bank = MemoryBank(features.clone(), "L2_normalized", torch.tensor([0, 1, 2], dtype=torch.int32))
    query = features[:1].clone()
    rows = torch.zeros(1, dtype=torch.long)
    # Its own patch is an exact match. Skipped, the nearest is [0.6, 0.8] at 1 - 0.6.
    assert torch.allclose(bank.distances(query, 1), torch.zeros(1))
    own = bank.distances(query, 1, rows, torch.tensor([[True, False, False]]))
    assert torch.allclose(own, torch.tensor([0.4])), own
    # Tile 2 overlaps it as well, which leaves only [0, 1] at 1 - 0.
    both = bank.distances(query, 1, rows, torch.tensor([[True, False, True]]))
    assert torch.allclose(both, torch.tensor([1.0])), both
    none = bank.distances(query, 1, rows, torch.tensor([[False, False, False]]))
    assert torch.allclose(none, torch.zeros(1))


def test_too_few_candidates_average_what_is_left() -> None:
    bank = MemoryBank(torch.eye(3), "L2_normalized", torch.tensor([0, 0, 1], dtype=torch.int32))
    query = torch.tensor([[1.0, 0.0, 0.0]])
    rows = torch.zeros(1, dtype=torch.long)
    # k=3 with tile 0 excluded leaves one patch, at 1 - 0. It used to be inf, then 0.
    one_left = bank.distances(query, 3, rows, torch.tensor([[True, False]]))
    assert torch.allclose(one_left, torch.ones(1)), one_left
    nothing = bank.distances(query, 3, rows, torch.tensor([[True, True]]))
    assert torch.isnan(nothing).all(), nothing


def test_l2_matches_cdist() -> None:
    torch.manual_seed(0)
    bank_features = torch.randn(50, 8)
    queries = torch.randn(30, 8)
    expected = torch.cdist(queries, bank_features).topk(4, dim=1, largest=False).values.mean(1)
    got = MemoryBank(bank_features.clone(), "L2").distances(queries, 4)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_chunking_matches_a_single_pass() -> None:
    torch.manual_seed(0)
    bank = MemoryBank(torch.randn(64, 16), "L2", torch.arange(64, dtype=torch.int32) % 5)
    queries = torch.randn(200, 16)
    rows = torch.arange(200) % 7
    excluded = torch.rand(7, 5, generator=torch.Generator().manual_seed(1)) < 0.4
    full = bank.distances(queries, 3, rows, excluded)

    original = detection._MAX_DISTANCE_ELEMENTS
    try:
        detection._MAX_DISTANCE_ELEMENTS = 64 * 7  # several ragged chunks
        chunked = bank.distances(queries, 3, rows, excluded)
    finally:
        detection._MAX_DISTANCE_ELEMENTS = original
    assert torch.allclose(full, chunked, atol=1e-6, equal_nan=True)


def test_excluded_refs_follow_the_footprints() -> None:
    q = [Path(f"q{i}") for i in range(4)]
    refs = [q[0], q[2], Path("r")]
    feet = {
        q[0]: Footprint("A", 0, 0, 100),
        q[1]: Footprint("A", 90, 0, 100),  # shares a 10 px strip with q0
        q[2]: Footprint("A", 300, 0, 100),
        q[3]: Footprint("B", 0, 0, 100),  # same pixels, other mosaic
        Path("r"): Footprint("A", 50, 50, 100),
    }
    assert excluded_refs(q, refs) == [[0], [], [1], []]
    assert excluded_refs(q, refs, feet) == [[0, 2], [0, 2], [1], []]


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
        assert len(scores) == 13
        assert all(np.isfinite(v) for v in scores.values()), scores
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


def test_background_mask_ignores_sign_and_random_state() -> None:
    # A bright square off the centre: the side picked can't depend on the PCA sign or
    # on how many random numbers were drawn before.
    torch.manual_seed(0)
    grid = (12, 12)
    feats = torch.randn(1, 144, 32) * 0.1
    subject = torch.zeros(12, 12, dtype=torch.bool)
    subject[3:8, 2:9] = True
    feats[0, subject.flatten()] += torch.randn(32) * 3
    first = background_mask(feats, grid)
    torch.rand(1000)
    assert torch.equal(background_mask(feats, grid), first)
    assert torch.equal(background_mask(-feats, grid), first)
    assert first[0].reshape(grid)[subject].float().mean() > 0.9


def test_subspace_scores_the_off_plane_patch() -> None:
    torch.manual_seed(0)
    # Normal patches span the first 3 of 10 dimensions, with a little noise.
    normal = torch.zeros(2000, 10)
    normal[:, :3] = torch.randn(2000, 3) * 5
    normal += torch.randn(2000, 10) * 0.01
    model = SubspaceModel(normal, variance=0.99)
    assert model.rank == 3
    on = torch.tensor([[4.0, -2.0, 1.0] + [0.0] * 7])
    off = torch.tensor([[0.0] * 9 + [2.0]])
    d = model.distances(torch.cat([on, off]), 5, None, None)
    assert d[0] < 1e-2, d
    assert abs(d[1].item() - 4.0) < 0.05, d


def test_end_to_end_subspace() -> None:
    with tempfile.TemporaryDirectory() as td:
        scores = _run(Path(td), detector="subspace")
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]


def test_parse_layers() -> None:
    assert parse_layers("last", 24) is None
    assert parse_layers("mid", 40) == list(range(22, 29))
    assert parse_layers("mid", 24) == [13, 14, 15, 16, 17]
    assert parse_layers("14-16", 24) == [14, 15, 16]
    assert parse_layers("3,5", 24) == [3, 5]
    for bad in ("0-3", "20-30", "x"):
        try:
            parse_layers(bad, 24)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_subset_size() -> None:
    from jwstdetector.bootstrap import subset_size

    assert subset_size(1000, 0.1, 500, 5000) == 500
    assert subset_size(100_000, 0.1, 500, 5000) == 5000
    assert subset_size(10_000, 0.1, 500, 5000) == 1000
    assert subset_size(3, 0.1, 500, 5000) == 1
    assert subset_size(60, 0.1, 500, 5000) == 30


def test_cached_features_give_the_same_scores_and_embed_once() -> None:
    from jwstdetector.features import FeatureCache

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        plain = _run(tmp / "plain")
        backbone = _stub_backbone()
        calls = []
        forward = backbone.forward_patches
        backbone.forward_patches = lambda x: calls.append(len(x)) or forward(x)
        cache = FeatureCache(tmp / "cache", backbone)
        query_dir = tmp / "query"
        names = _write_tiles(query_dir)
        paths = sorted(query_dir.glob("*.png"))
        refs = [query_dir / n for n in names if n != ANOMALOUS]
        opts = {"num_workers": 0, "batch_size": 4, "cache": cache}
        first, *_ = run_anomaly_detection(
            backbone, refs, query_dir, paths, query_dir, tmp / "a", **opts
        )
        embedded_once = sum(calls)
        second, *_ = run_anomaly_detection(
            backbone, refs, query_dir, paths, query_dir, tmp / "b", **opts
        )

        # The bank and the scoring both read the cache, so each tile went through once.
        assert embedded_once == len(paths)
        assert sum(calls) == embedded_once
        assert len(list((tmp / "cache").rglob("*.npy"))) == len(paths)
        for key, value in plain.items():
            assert abs(first[key] - value) < 1e-3 * max(1.0, abs(value))
            assert first[key] == second[key]

        with np.load(tmp / "a" / "descriptors.npz") as f:
            keys, vectors = f["keys"].tolist(), f["vectors"].astype(np.float32)
        assert sorted(keys) == sorted(p.name for p in paths)
        assert vectors.shape == (len(paths), DIM)
        # The odd tile's descriptor describes its bright bar, so it sits apart from the rest.
        odd = vectors[keys.index(ANOMALOUS)]
        others = np.delete(vectors, keys.index(ANOMALOUS), 0)
        spread = np.linalg.norm(others - others.mean(0), axis=1).max()
        assert np.linalg.norm(others - odd, axis=1).min() > 2 * spread


def test_overlapping_tiles_share_out_their_patches() -> None:
    from types import SimpleNamespace

    from jwstdetector.data import owned_patches

    # 320 px tiles at a 160 px stride, and one more flush against the far edge.
    fp = {
        (x, y): SimpleNamespace(mosaic="m", x0=x, y0=y, size=320)
        for x in range(0, 641, 160)
        for y in (0, 160, 320)
    }
    fp["flush"] = SimpleNamespace(mosaic="m", x0=630, y0=320, size=320)
    owned = owned_patches(fp, (28, 28))
    rows, cols = np.nonzero(owned[(320, 160)])
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (7, 20, 7, 20)
    # PNG rows run down and FITS rows up, so the bottom tile keeps its bottom rows.
    rows, _ = np.nonzero(owned[(320, 0)])
    assert (rows.min(), rows.max()) == (7, 27)
    # Every bit of covered sky is within one patch of a patch some tile scores.
    patch = 320 / 28
    centres = np.array(
        [
            (f.x0 + (c + 0.5) * patch, f.y0 + 320 - (r + 0.5) * patch)
            for k, f in fp.items()
            for r, c in zip(*np.nonzero(owned[k]), strict=True)
        ]
    )
    xs, ys = np.meshgrid(np.arange(0.5, 950, 7.0), np.arange(0.5, 640, 7.0))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    gap = np.abs(points[:, None, :] - centres[None, :, :]).max(axis=2).min(axis=1)
    assert gap.max() <= patch


def test_pixel_backbone_returns_raw_patches() -> None:
    from jwstdetector.backbones import get_backbone

    b = get_backbone("pixels", "cpu", resolution=32)
    x = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    out = b.embed(x)
    assert out.shape == (1, 4, 3 * 16 * 16)
    assert torch.equal(out[0, 1].reshape(3, 16, 16), x[0, :, :16, 16:])
