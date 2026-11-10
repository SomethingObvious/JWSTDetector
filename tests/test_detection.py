"""Checks for the memory bank, the search, the feature cache and whole scoring passes.

They need torch but no GPU, network or weights, as a small deterministic module
takes DINO's place.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from PIL import Image
import torch
from typing_extensions import Unpack

from jwst_peculiar import detection
from jwst_peculiar.backbones import Backbone, background_mask, hub_repo, parse_layers
from jwst_peculiar.detection import (
    BankSettings,
    Footprint,
    MemoryBank,
    SubspaceModel,
    build_memory_bank,
    dihedral,
    excluded_refs,
    run_anomaly_detection,
)
from jwst_peculiar.features import EmbedOptions, FeatureCache

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest

PATCH, RESOLUTION, DIM = 16, 128, 32
GRID = RESOLUTION // PATCH
ANOMALOUS = "A5_cx000264_cy000264_ra150.999999_decm2.197645_i00000099.png"


def _stub_backbone(seed: int = 0) -> Backbone:
    # Each patch's mean colour, projected. Deterministic and content-sensitive, which
    # is all the detector needs from a backbone.
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
    )


def _write_tiles(root: Path, n_normal: int = 12) -> list[str]:
    # A field of ordinary tiles plus one with a bright bar across it. The names carry
    # an RA and Dec the way prep_data --wcs_in_name writes them, so the map file names
    # have to survive the dots in the stem.
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names: list[str] = []
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


class _RunOptions(TypedDict, total=False):
    places: int
    drop_directions: int
    rotation: bool
    masking: bool
    mask_ref_images: bool
    knn_metric: str
    k_neighbors: int
    score_mode: str
    save_tiffs: bool
    detector: str


def _run(
    tmp: Path,
    *,
    include_odd_in_reference: bool = False,
    **kwargs: Unpack[_RunOptions],
) -> dict[str, float]:
    query_dir = tmp / "query"
    names = _write_tiles(query_dir)
    ref_paths = [
        query_dir / n for n in names if include_odd_in_reference or n != ANOMALOUS
    ]
    query_paths = sorted(query_dir.glob("*.png"))

    scores, memorybank_sec, seconds = run_anomaly_detection(
        _stub_backbone(),
        ref_paths,
        query_dir,
        query_paths=query_paths,
        query_root=query_dir,
        out_dir=tmp / "out",
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
    assert (
        hub_repo("dinov3-vitl16-pretrain-sat493m")
        == "facebook/dinov3-vitl16-pretrain-sat493m"
    )
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
    bank = MemoryBank(
        features.clone(), "L2_normalized", torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    query = features[:1].clone()
    rows = torch.zeros(1, dtype=torch.long)
    # Its own patch is an exact match. Skipped, the nearest is [0.6, 0.8] at 1 - 0.6.
    assert torch.allclose(bank.distances(query, 1), torch.zeros(1))
    own = bank.distances(
        query, 1, rows=rows, excluded=torch.tensor([[True, False, False]])
    )
    assert torch.allclose(own, torch.tensor([0.4])), own
    # Tile 2 overlaps it as well, which leaves only [0, 1] at 1 - 0.
    both = bank.distances(
        query, 1, rows=rows, excluded=torch.tensor([[True, False, True]])
    )
    assert torch.allclose(both, torch.tensor([1.0])), both
    none = bank.distances(
        query, 1, rows=rows, excluded=torch.tensor([[False, False, False]])
    )
    assert torch.allclose(none, torch.zeros(1))


def test_too_few_candidates_average_what_is_left() -> None:
    bank = MemoryBank(
        torch.eye(3), "L2_normalized", torch.tensor([0, 0, 1], dtype=torch.int32)
    )
    query = torch.tensor([[1.0, 0.0, 0.0]])
    rows = torch.zeros(1, dtype=torch.long)
    # k=3 with tile 0 excluded leaves one patch, at 1 - 0.
    one_left = bank.distances(
        query, 3, rows=rows, excluded=torch.tensor([[True, False]])
    )
    assert torch.allclose(one_left, torch.ones(1)), one_left
    nothing = bank.distances(query, 3, rows=rows, excluded=torch.tensor([[True, True]]))
    assert torch.isnan(nothing).all(), nothing


def test_twins_only_fill_their_own_places() -> None:
    # Five tiles of plain sky, and the rare patch in tiles 0 and 1 alone.
    sky, rare = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    features = torch.tensor([
        sky,
        rare,
        sky,
        rare,
        sky,
        sky,
        sky,
        sky,
        sky,
        [0.0, 0.0, 1.0],
    ])
    owners = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.int32)
    bank = MemoryBank(features, "L2_normalized", owners)
    bank.near = torch.eye(5, dtype=torch.bool)
    query = torch.tensor([rare])
    for places, expected in ((1, 0.0), (2, 0.0), (3, 1.0)):
        got = bank.distances(query, 1, places=places)
        assert torch.allclose(got, torch.tensor([expected])), (places, got)
    # Plain sky has a match in all five places, so it stays ordinary.
    assert torch.allclose(
        bank.distances(torch.tensor([sky]), 1, places=5), torch.zeros(1)
    )
    # Tiles 0 and 1 overlap, so their copies of the rare patch are one place.
    bank.near[0, 1] = bank.near[1, 0] = True
    got = bank.distances(query, 1, places=2)
    assert torch.allclose(got, torch.ones(1)), got


def test_places_beyond_the_candidates_give_the_farthest_seen() -> None:
    bank = MemoryBank(
        torch.eye(3), "L2_normalized", torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    bank.near = torch.eye(3, dtype=torch.bool)
    got = bank.distances(torch.tensor([[1.0, 0.0, 0.0]]), 1, places=4)
    assert torch.allclose(got, torch.ones(1)), got


def test_l2_matches_cdist() -> None:
    torch.manual_seed(0)
    bank_features = torch.randn(50, 8)
    queries = torch.randn(30, 8)
    expected = (
        torch.cdist(queries, bank_features).topk(4, dim=1, largest=False).values.mean(1)
    )
    got = MemoryBank(bank_features.clone(), "L2").distances(queries, 4)
    assert torch.allclose(got, expected, atol=1e-4), (got - expected).abs().max()


def test_chunking_matches_a_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    bank = MemoryBank(
        torch.randn(64, 16), "L2", torch.arange(64, dtype=torch.int32) % 5
    )
    queries = torch.randn(200, 16)
    rows = torch.arange(200) % 7
    excluded = torch.rand(7, 5, generator=torch.Generator().manual_seed(1)) < 0.4
    full = bank.distances(queries, 3, rows=rows, excluded=excluded)

    # Several ragged chunks.
    monkeypatch.setattr(detection, "_MAX_DISTANCE_ELEMENTS", 64 * 7)
    chunked = bank.distances(queries, 3, rows=rows, excluded=excluded)
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


def test_excluded_refs_cross_mosaics_on_the_sky() -> None:
    # Two survey visits that overlap: tiles 20 arcsec across, half-diagonal 14.1.
    r = 20 / 3600 / 2**0.5
    q = [Path("v1a"), Path("v1b")]
    refs = [Path("v2a"), Path("v2b"), Path("v3")]
    feet = {
        q[0]: Footprint("v1", 0, 0, 320, ra=10.0, dec=0.0, radius_deg=r),
        q[1]: Footprint("v1", 320, 0, 320, ra=10.1, dec=0.0, radius_deg=r),
        # 25 arcsec from v1a, close enough that the corners can meet.
        refs[0]: Footprint("v2", 0, 0, 320, ra=10.0 + 25 / 3600, dec=0.0, radius_deg=r),
        # 30 arcsec off, past the two half-diagonals.
        refs[1]: Footprint("v2", 320, 0, 320, ra=10.0, dec=30 / 3600, radius_deg=r),
        # No sky position, so only its own mosaic's pixels count.
        refs[2]: Footprint("v3", 0, 0, 320),
    }
    assert excluded_refs(q, refs, feet) == [[0], []]


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


def test_end_to_end_places_still_ranks_the_odd_tile_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        scores = _run(Path(td), include_odd_in_reference=True, places=3)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]


def test_end_to_end_with_directions_dropped_still_ranks_the_odd_tile_first() -> None:
    with tempfile.TemporaryDirectory() as td:
        scores = _run(Path(td), drop_directions=2)
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked[0] == ANOMALOUS, ranked[:3]


def test_rerun_reuses_bank_and_unchanged_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    # A bench scores the same bank and tiles once per magnitude, and planting a source
    # changes only some tiles. The second run has to give the same scores without
    # building the bank again, and search only the tile that changed.
    monkeypatch.setattr(detection, "_REUSED", {})
    builds: list[int] = []
    searched: list[int] = []
    build = detection.build_memory_bank
    search = MemoryBank.distances

    def counting_build(*args: object, **kwargs: object) -> MemoryBank:
        builds.append(1)
        return build(*args, **kwargs)  # type: ignore[arg-type]

    def counting_search(
        self: MemoryBank, queries: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        searched.append(queries.shape[0])
        return search(self, queries, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(detection, "build_memory_bank", counting_build)
    monkeypatch.setattr(MemoryBank, "distances", counting_search)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        query_dir = tmp / "query"
        names = _write_tiles(query_dir)
        refs = [query_dir / n for n in names if n != ANOMALOUS]
        backbone = _stub_backbone()

        def run() -> dict[str, float]:
            return run_anomaly_detection(
                backbone,
                refs,
                query_dir,
                query_paths=sorted(query_dir.glob("*.png")),
                query_root=query_dir,
                out_dir=tmp / "out",
                num_workers=0,
                batch_size=4,
            )[0]

        first = run()
        tile = np.asarray(Image.open(query_dir / ANOMALOUS)).copy()
        tile[48:80, :, :] = 55
        Image.fromarray(tile).save(query_dir / ANOMALOUS)
        searched.clear()
        second = run()

    assert len(builds) == 1
    assert sum(searched) == GRID * GRID
    assert second[ANOMALOUS] != first[ANOMALOUS]
    del first[ANOMALOUS], second[ANOMALOUS]
    assert second == first


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


def _bank(
    max_patches: int,
    seed: int = 0,
    *,
    masking: bool = False,
    dtype: torch.dtype = torch.float32,
) -> MemoryBank:
    with tempfile.TemporaryDirectory() as td:
        query_dir = Path(td) / "query"
        _write_tiles(query_dir)
        paths = sorted(query_dir.glob("*.png"))
        return build_memory_bank(
            _stub_backbone(),
            paths,
            query_dir,
            settings=BankSettings(
                max_patches=max_patches, seed=seed, masking=masking, dtype=dtype
            ),
            options=EmbedOptions(num_workers=0),
        )


def _owners(bank: MemoryBank) -> torch.Tensor:
    assert bank.owners is not None
    return bank.owners


def test_bank_holds_every_patch_under_the_cap() -> None:
    bank = _bank(10**9)
    assert len(bank) == 13 * GRID * GRID
    assert _owners(bank).tolist() == [i for i in range(13) for _ in range(GRID * GRID)]


def test_bank_sample_fills_the_cap_evenly_and_repeats() -> None:
    cap = 13 * GRID * GRID // 2
    bank = _bank(cap)
    assert len(bank) == cap
    per_tile = np.bincount(_owners(bank).numpy(), minlength=13)
    # About 32 of each tile's 64 patches, the last tile as many as the first.
    assert np.min(per_tile) >= 20, per_tile
    assert np.max(per_tile) <= 44, per_tile

    again = _bank(cap)
    assert torch.equal(bank.features, again.features)
    assert torch.equal(_owners(bank), _owners(again))
    assert not torch.equal(_owners(bank), _owners(_bank(cap, seed=1)))


def test_fp16_bank_samples_past_the_cap() -> None:
    cap = 13 * GRID * GRID // 2
    half = _bank(cap, dtype=torch.float16)
    full = _bank(cap)
    assert half.features.dtype == torch.float16
    assert torch.equal(_owners(half), _owners(full))


def test_masked_bank_still_fills_the_cap() -> None:
    # Masking leaves fewer patches than the tile count suggests.
    everything = len(_bank(10**9, masking=True))
    assert everything < 13 * GRID * GRID
    assert len(_bank(everything // 2, masking=True)) == everything // 2


def test_background_mask_ignores_sign_and_random_state() -> None:
    # A bright square off centre. The side picked can't depend on the PCA sign or on
    # how many random numbers were drawn before.
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
    d = model.distances(torch.cat([on, off]))
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
    from jwst_peculiar.bootstrap import subset_size

    assert subset_size(1000, 0.1, minimum=500, maximum=5000) == 500
    assert subset_size(100_000, 0.1, minimum=500, maximum=5000) == 5000
    assert subset_size(10_000, 0.1, minimum=500, maximum=5000) == 1000
    assert subset_size(3, 0.1, minimum=500, maximum=5000) == 1
    assert subset_size(60, 0.1, minimum=500, maximum=5000) == 30


def _cached_detector(
    tmp: Path,
) -> tuple[Callable[[Path], dict[str, float]], list[int], list[Path]]:
    # A run on a feature cache, the batch sizes the backbone saw and the query tiles.
    backbone = _stub_backbone()
    calls: list[int] = []
    forward = backbone.forward_patches

    def counted(x: torch.Tensor) -> torch.Tensor:
        calls.append(len(x))
        return forward(x)

    backbone.forward_patches = counted
    cache = FeatureCache(tmp / "cache", backbone)
    query_dir = tmp / "query"
    names = _write_tiles(query_dir)
    paths = sorted(query_dir.glob("*.png"))
    refs = [query_dir / n for n in names if n != ANOMALOUS]

    def run(out_dir: Path) -> dict[str, float]:
        scores, *_ = run_anomaly_detection(
            backbone,
            refs,
            query_dir,
            query_paths=paths,
            query_root=query_dir,
            out_dir=out_dir,
            num_workers=0,
            batch_size=4,
            cache=cache,
        )
        return scores

    return run, calls, paths


def _check_descriptors(path: Path, paths: list[Path]) -> None:
    with np.load(path) as f:
        keys, vectors = f["keys"].tolist(), f["vectors"].astype(np.float32)
    assert sorted(keys) == sorted(p.name for p in paths)
    assert vectors.shape == (len(paths), DIM)
    # The odd tile's bright bar sets its descriptor apart from the rest.
    odd = vectors[keys.index(ANOMALOUS)]
    others = np.delete(vectors, keys.index(ANOMALOUS), 0)
    spread = np.linalg.norm(others - others.mean(0), axis=1).max()
    assert np.linalg.norm(others - odd, axis=1).min() > 2 * spread


def test_cached_features_give_the_same_scores_and_embed_once() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        plain = _run(tmp / "plain")
        run, calls, paths = _cached_detector(tmp)
        first = run(tmp / "a")
        embedded_once = sum(calls)
        second = run(tmp / "b")

        # The bank and the scoring both read the cache, so each tile went through once.
        assert embedded_once == len(paths)
        assert sum(calls) == embedded_once
        assert len(list((tmp / "cache").rglob("*.npy"))) == len(paths)
        for key, value in plain.items():
            assert abs(first[key] - value) < 1e-3 * max(1.0, abs(value))
            assert first[key] == second[key]

        _check_descriptors(tmp / "a" / "descriptors.npz", paths)


def test_overlapping_tiles_share_out_their_patches() -> None:
    from jwst_peculiar.data import owned_patches

    # 320 px tiles at a 160 px stride, and one more flush against the far edge.
    fp: dict[tuple[int, int] | str, Footprint] = {
        (x, y): Footprint("m", x, y, 320)
        for x in range(0, 641, 160)
        for y in (0, 160, 320)
    }
    fp["flush"] = Footprint("m", 630, 320, 320)
    owned = owned_patches(fp, (28, 28))
    rows, cols = np.nonzero(owned[(320, 160)])
    assert [int(f(v)) for v in (rows, cols) for f in (np.min, np.max)] == [7, 20, 7, 20]
    # PNG rows run down and FITS rows up, so the bottom tile keeps its bottom rows.
    rows, _ = np.nonzero(owned[(320, 0)])
    assert [int(np.min(rows)), int(np.max(rows))] == [7, 27]
    # Every bit of covered sky is within one patch of a patch some tile scores.
    patch = 320 / 28
    centres = np.array([
        (f.x0 + (c + 0.5) * patch, f.y0 + 320 - (r + 0.5) * patch)
        for k, f in fp.items()
        for r, c in zip(*np.nonzero(owned[k]), strict=True)
    ])
    xs, ys = np.meshgrid(np.arange(0.5, 950, 7.0), np.arange(0.5, 640, 7.0))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    gap = np.abs(points[:, None, :] - centres[None, :, :]).max(axis=2).min(axis=1)
    assert gap.max() <= patch

    # The bottom tile has no neighbour below, and its middle stops where a tile with
    # every neighbour stops.
    middle = owned_patches(fp, (28, 28), middle=True)
    rows, _ = np.nonzero(middle[(320, 0)])
    assert [int(np.min(rows)), int(np.max(rows))] == [7, 20]
    assert np.array_equal(middle[(320, 160)], owned[(320, 160)])


def test_pixel_backbone_returns_raw_patches() -> None:
    from jwst_peculiar.backbones import get_backbone

    b = get_backbone("pixels", "cpu", resolution=32)
    x = torch.arange(3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    out = b.embed(x)
    assert out.shape == (1, 4, 3 * 16 * 16)
    assert torch.equal(out[0, 1].reshape(3, 16, 16), x[0, :, :16, 16:])


def test_dropping_directions_lets_a_small_difference_count() -> None:
    # Every patch points mostly along axis 0, so under cosine distance a patch that
    # differs on axis 1 instead of axis 2 looks the same as the rest.
    rng = np.random.default_rng(0)
    bank = np.zeros((400, 8), dtype=np.float32)
    bank[:, 0] = 50 + rng.normal(0, 5, 400)
    bank[:, 2] = 1.0 + rng.normal(0, 0.01, 400)
    odd = torch.tensor([[50.0, 1.0, 0.0, 0, 0, 0, 0, 0]])
    plain = detection.MemoryBank(torch.from_numpy(bank.copy()), "L2_normalized")
    dropped = detection.MemoryBank(torch.from_numpy(bank.copy()), "L2_normalized")
    dropped.drop_directions(1)
    assert plain.distances(odd, 1).item() < 1e-3
    assert dropped.distances(odd, 1).item() > 0.5
