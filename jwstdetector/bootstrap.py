"""Label-free anomaly scoring for JWST tiles.

Pass 1 draws a random reference set from the query tiles and scores every tile
against it. Pass 2 rebuilds the reference from the tiles pass 1 found most ordinary
and scores again. A tile that is in the reference is scored against the other
reference tiles, never its own patches.

With --reference_roots, the reference is other fields cut the same way instead, and
the tiles are scored once against all of them. Each of those fields is also scored
against the rest, which gives the calibration for a p-value and a q-value per tile.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import numpy as np
import yaml

from jwstdetector.backbones import get_backbone
from jwstdetector.baseline import isolation_scores
from jwstdetector.data import (
    list_images,
    load_footprints,
    prepare_out_dir,
    relative_key,
    resolve_device,
    tagged_dir,
    write_measurements_csv,
    write_ref_list,
)
from jwstdetector.detection import METRICS, run_anomaly_detection
from jwstdetector.features import FeatureCache
from jwstdetector.scoring import SCORE_MODES
from jwstdetector.seeding import set_seed
from jwstdetector.significance import write_significance

if TYPE_CHECKING:
    from jwstdetector.backbones import Backbone
    from jwstdetector.detection import Footprint

logger = logging.getLogger(__name__)

# Everything a run writes, so a folder holding an older run's results is caught.
_OUTPUTS = [
    "args.yaml",
    "measurements_final.csv",
    "pass1",
    "pass2",
    "cross",
    "calibration",
    "calibration.csv",
    "significance.csv",
    "reference_samples.png",
]
_MIN_REFERENCE_FIELDS = 2


def subset_size(n: int, frac: float, *, minimum: int, maximum: int | None) -> int:
    """How many tiles to keep, a fraction of `n` clamped to [minimum, maximum].

    It never takes more than half the tiles. With every tile in the reference, pass 2
    can't leave the strange ones out, and it just repeats pass 1.

    Returns:
        The number of tiles, at least 1.
    """
    k = max(minimum, round(n * frac))
    if maximum is not None:
        k = min(k, maximum)
    return max(1, min(k, n // 2))


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data_root", required=True, help="Folder holding the query folder."
    )
    p.add_argument("--query_subdir", default="query")
    p.add_argument(
        "--reference_roots",
        nargs="+",
        default=None,
        help="Prep folders of two or more other fields, cut with the same filters and "
        "stretch, to use as the reference in place of this field's own tiles. Each "
        "tile then also gets a calibrated p-value and q-value.",
    )
    p.add_argument(
        "--recursive_query", default=True, action=argparse.BooleanOptionalAction
    )


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model_name",
        default="dinov3-vitl16-pretrain-sat493m",
        help="A dinov3-* name (gated), a dinov2_* name, or a Hugging Face repo id.",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=448,
        help="Rounded down to a multiple of the patch size.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--num_workers", type=int, default=4, help="Processes decoding the PNGs."
    )
    p.add_argument(
        "--precision",
        default="fp16",
        choices=("fp16", "bf16", "fp32"),
        help="The backbone's weights and activations on CUDA. fp16 keeps the patch "
        "distances closest to fp32's at bf16's speed. Distances are always fp32.",
    )
    p.add_argument(
        "--compile",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Compile the backbone, which needs Triton and runs about 1.7 times as "
        "fast. On by default when Triton is installed.",
    )
    p.add_argument(
        "--fast_accumulate",
        action="store_true",
        help="Sum fp16 matmuls in fp16, about a third faster on GeForce cards. It "
        "moves the patch distances a little more than plain fp16 does.",
    )


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--detector",
        default="knn",
        choices=("knn", "subspace", "iforest"),
        help="knn is AnomalyDINO's nearest-patch distance, subspace is SubspaceAD's "
        "PCA residual, and iforest is the baseline, an isolation forest on hand-built "
        "tile features that uses no model.",
    )
    p.add_argument(
        "--subspace_variance",
        type=float,
        default=0.99,
        help="Share of the reference variance the subspace keeps.",
    )
    p.add_argument(
        "--layers",
        default="last",
        help="Patch features from the last layer, the mid band SubspaceAD uses, a "
        "range like 14-20, or a list.",
    )
    p.add_argument("--knn_metric", default="L2_normalized", choices=METRICS)
    p.add_argument(
        "--score_region",
        default="owned",
        choices=("owned", "all"),
        help="Score each tile on the patches no overlapping tile is more central to, "
        "which keeps objects cut off by a tile's edge from crowding the top, or on all "
        "of them.",
    )
    p.add_argument("--k_neighbors", type=int, default=1)
    p.add_argument(
        "--places",
        type=int,
        default=1,
        help="Separate places on the sky that need a close match before a patch counts "
        "as ordinary, so a rare object's few twins can't vouch for it. kNN only.",
    )
    p.add_argument(
        "--drop_directions",
        type=int,
        default=0,
        help="Principal directions of the reference patches to remove before the kNN "
        "search.",
    )
    p.add_argument(
        "--bank_precision",
        default="fp16",
        choices=("fp16", "fp32"),
        help="How the memory bank stores patches on a GPU. fp16 holds twice as many in "
        "the same memory and searches them about three times as fast. Cosine distance "
        "only.",
    )
    p.add_argument(
        "--max_bank_patches",
        type=int,
        default=1_000_000,
        help="Most patches the memory bank holds, about 2 GB in fp16 at 1024 dims.",
    )

    p.add_argument("--score_mode", default="top1p", choices=SCORE_MODES)
    p.add_argument("--score_top_frac", type=float, default=0.01, help="For topk_mean.")
    p.add_argument("--score_quantile", type=float, default=0.9995, help="For quantile.")


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--masking", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--mask_ref_images", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--rotation",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Add the eight square symmetries of each reference tile to the bank.",
    )
    p.add_argument(
        "--save_examples",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Draw a few reference tiles with their background masks.",
    )
    p.add_argument(
        "--save_patch_dists", default=True, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--feature_cache",
        default=None,
        help="Where each tile's patch embeddings are kept between runs, "
        "DATA_ROOT/.features by default. About 1.6 MB a tile for ViT-L at 448 px. Pass "
        "none to embed on the fly.",
    )
    p.add_argument("--save_tiffs", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1 or mps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Put cuDNN in deterministic mode, which is slower.",
    )

    p.add_argument("--init_ref_frac", type=float, default=0.1)
    p.add_argument("--init_ref_min", type=int, default=500)
    p.add_argument("--init_ref_max", type=int, default=5000)

    p.add_argument("--bootstrap_keep_frac", type=float, default=0.2)
    p.add_argument("--bootstrap_keep_min", type=int, default=500)
    p.add_argument("--bootstrap_keep_max", type=int, default=5000)

    p.add_argument("--out_dir", default="results_query_only")
    p.add_argument(
        "--tag", default=None, help="Appended to --out_dir after an underscore."
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the results already in the output folder instead of stopping.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the options of `jwstdetector run`.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_input_args(p)
    _add_model_args(p)
    _add_detector_args(p)
    _add_run_args(p)
    return p.parse_args(argv)


@dataclass(frozen=True)
class _Tiles:
    """Some tiles, and the folder their keys are relative to."""

    paths: list[Path]
    root: Path


@dataclass(frozen=True)
class _Field:
    """One of the --reference_roots fields."""

    name: str
    tiles: _Tiles
    footprints: dict[Path, Footprint]


@dataclass(frozen=True)
class _Scorer:
    """Scores tiles against a reference with the run's settings."""

    args: argparse.Namespace
    backbone: Backbone
    cache: FeatureCache | None
    # The reference fields add theirs to this, so it covers every tile scored.
    footprints: dict[Path, Footprint]
    queries: _Tiles

    def score(
        self,
        ref: _Tiles,
        pass_dir: Path,
        *,
        queries: _Tiles | None = None,
        save_maps: bool | None = None,
    ) -> dict[str, float]:
        """Score `queries`, the run's query tiles unless given, against `ref`.

        The reference list, the scores and the patch maps go in `pass_dir`.

        Returns:
            The score per tile key.
        """
        args = self.args
        if queries is None:
            queries = self.queries
        maps: bool = args.save_patch_dists if save_maps is None else save_maps
        pass_dir.mkdir(parents=True, exist_ok=True)
        write_ref_list(pass_dir / "ref_list.txt", ref.root, ref.paths)
        scores, memorybank_sec, seconds = run_anomaly_detection(
            self.backbone,
            ref.paths,
            ref.root,
            query_paths=queries.paths,
            query_root=queries.root,
            out_dir=pass_dir,
            masking=args.masking,
            mask_ref_images=args.mask_ref_images,
            rotation=args.rotation,
            knn_metric=args.knn_metric,
            score_region=args.score_region,
            k_neighbors=args.k_neighbors,
            places=args.places,
            drop_directions=args.drop_directions,
            bank_precision=args.bank_precision,
            max_bank_patches=args.max_bank_patches,
            score_mode=args.score_mode,
            score_top_frac=args.score_top_frac,
            score_quantile=args.score_quantile,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            save_patch_dists=maps,
            save_tiffs=args.save_tiffs and maps,
            detector=args.detector,
            subspace_variance=args.subspace_variance,
            footprints=self.footprints,
            cache=self.cache,
        )
        write_measurements_csv(
            pass_dir / "measurements.csv",
            scores,
            memorybank_sec=memorybank_sec,
            inference_sec=seconds,
        )
        unscored = sorted(k for k, v in scores.items() if math.isnan(v))
        if unscored:
            logger.warning(
                "%d tiles had nothing left to compare with and scored NaN, such as %s. "
                "A bigger reference fixes that.",
                len(unscored),
                unscored[0],
            )
        return scores


def _setup(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # huggingface_hub logs every HTTP request it makes at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    set_seed(args.seed, deterministic=args.deterministic)


def _load_backbone(args: argparse.Namespace, out_dir: Path) -> Backbone:
    backbone = get_backbone(
        args.model_name,
        resolve_device(args.device),
        resolution=args.resolution,
        precision=args.precision,
        compile_model=args.compile,
        fast_accumulate=args.fast_accumulate,
        layers=args.layers,
    )
    # What the model actually ran as, which can differ from what was asked, such as an
    # LVD checkpoint that overflows fp16 and runs in bf16 instead.
    ran = {
        "dtype": str(backbone.dtype).removeprefix("torch."),
        "compiled": backbone.compiled,
        "streamed": backbone.streamed,
        "fast_accumulate": backbone.fast_accumulate,
    }
    (out_dir / "args.yaml").write_text(
        yaml.safe_dump({**vars(args), "ran": ran}), encoding="utf-8"
    )
    return backbone


def _feature_cache(args: argparse.Namespace, backbone: Backbone) -> FeatureCache | None:
    if (args.feature_cache or "").lower() == "none":
        return None
    cache_dir = Path(args.feature_cache or Path(args.data_root) / ".features")
    cache = FeatureCache(cache_dir, backbone)
    logger.info("Patch embeddings are cached in %s", cache_dir)
    return cache


def main(argv: list[str] | None = None) -> int:
    """Run `jwstdetector run`.

    Returns:
        The exit code, which is always 0.

    Raises:
        SystemExit: If there's no query folder under --data_root.
    """
    args = parse_args(argv)
    _setup(args)

    query_dir = Path(args.data_root).resolve() / args.query_subdir
    if not query_dir.is_dir():
        msg = f"There is no query folder at {query_dir}"
        raise SystemExit(msg)

    out_dir = tagged_dir(Path(args.out_dir), args.tag)
    prepare_out_dir(out_dir, _OUTPUTS, overwrite=args.overwrite)
    (out_dir / "args.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")
    if args.detector == "iforest":
        return _isolation_forest(args, query_dir, out_dir)

    backbone = _load_backbone(args, out_dir)
    cache = _feature_cache(args, backbone)
    query_paths = list_images(query_dir, recursive=args.recursive_query)
    logger.info("Found %d query tiles under %s", len(query_paths), query_dir)
    scorer = _Scorer(
        args,
        backbone,
        cache,
        load_footprints(Path(args.data_root), query_dir, query_paths),
        _Tiles(query_paths, query_dir),
    )
    if args.reference_roots:
        return _cross_field(scorer, out_dir)
    return _two_passes(scorer, out_dir)


def _two_passes(scorer: _Scorer, out_dir: Path) -> int:
    args, queries = scorer.args, scorer.queries
    rng = np.random.default_rng(args.seed)
    n = len(queries.paths)
    k1 = subset_size(
        n, args.init_ref_frac, minimum=args.init_ref_min, maximum=args.init_ref_max
    )
    ref1 = [
        queries.paths[i] for i in sorted(rng.choice(n, size=k1, replace=False).tolist())
    ]
    logger.info("Pass 1 uses %d random tiles of %d as the reference", k1, n)

    if args.save_examples:
        from jwstdetector.utils import plot_reference_masks  # noqa: PLC0415 matplotlib is slow to load

        plot_reference_masks(
            scorer.backbone, ref1, queries.root, out_dir / "reference_samples.png"
        )

    scores1 = scorer.score(_Tiles(ref1, queries.root), out_dir / "pass1")

    # NaN sorts last, since a tile nothing could be compared with isn't known to be
    # normal.
    by_score = sorted(scores1, key=lambda k: (math.isnan(scores1[k]), scores1[k]))
    k2 = subset_size(
        len(by_score),
        args.bootstrap_keep_frac,
        minimum=args.bootstrap_keep_min,
        maximum=args.bootstrap_keep_max,
    )
    lookup = {relative_key(p, queries.root): p for p in queries.paths}
    ref2 = [lookup[key] for key in by_score[:k2]]
    logger.info("Pass 2 uses the %d lowest-scoring tiles as the reference", len(ref2))

    scorer.score(_Tiles(ref2, queries.root), out_dir / "pass2")

    shutil.copyfile(
        out_dir / "pass2/measurements.csv", out_dir / "measurements_final.csv"
    )
    logger.info("Final scores are in %s", out_dir / "measurements_final.csv")
    return 0


def _isolation_forest(args: argparse.Namespace, query_dir: Path, out_dir: Path) -> int:
    paths = list_images(query_dir, recursive=args.recursive_query)
    scores, seconds = isolation_scores(paths, query_dir, seed=args.seed)
    write_measurements_csv(
        out_dir / "pass2" / "measurements.csv",
        scores,
        memorybank_sec=0.0,
        inference_sec=seconds,
    )
    shutil.copyfile(
        out_dir / "pass2/measurements.csv", out_dir / "measurements_final.csv"
    )
    logger.info("Scored %d tiles with the isolation forest", len(scores))
    return 0


def _reference_fields(args: argparse.Namespace) -> list[_Field]:
    """Load the --reference_roots fields.

    Returns:
        The fields, in the order given.

    Raises:
        SystemExit: If there are fewer than two, or one has no query folder.
    """
    if len(args.reference_roots) < _MIN_REFERENCE_FIELDS:
        msg = (
            "--reference_roots needs two or more fields, as each one is calibrated "
            "against the others."
        )
        raise SystemExit(msg)
    fields: list[_Field] = []
    for r in args.reference_roots:
        qdir = Path(r).resolve() / args.query_subdir
        if not qdir.is_dir():
            msg = f"There is no query folder at {qdir}"
            raise SystemExit(msg)
        paths = list_images(qdir, recursive=args.recursive_query)
        fp = load_footprints(Path(r), qdir, paths)
        fields.append(_Field(Path(r).resolve().name, _Tiles(paths, qdir), fp))
    return fields


def _check_mosaic_names(fields: list[_Field], target: dict[Path, Footprint]) -> None:
    """Stop if two fields, or the target and a field, share a mosaic name.

    Footprints match tiles to each other by mosaic name, so a name two fields share
    would make their tiles exclude each other.

    Raises:
        SystemExit: If a mosaic name is in more than one field.
    """
    seen: dict[str, str] = {}
    for field in fields:
        for f in field.footprints.values():
            if seen.setdefault(f.mosaic, field.name) != field.name:
                msg = (
                    f"Mosaic {f.mosaic} is in both {seen[f.mosaic]} and {field.name}. "
                    "Cut one of them again with a different --tiles name."
                )
                raise SystemExit(msg)
    clash = {f.mosaic for f in target.values()} & seen.keys()
    if clash:
        msg = f"Mosaic {min(clash)} is in the target and a reference field."
        raise SystemExit(msg)


def _cross_field(scorer: _Scorer, out_dir: Path) -> int:
    """Scores the tiles against other fields, and each other field against the rest.

    Returns:
        The exit code, which is always 0.
    """
    fields = _reference_fields(scorer.args)
    for field in fields:
        scorer.footprints.update(field.footprints)
    queries = set(scorer.queries.paths)
    target = {p: f for p, f in scorer.footprints.items() if p in queries}
    _check_mosaic_names(fields, target)

    ref_root = Path(os.path.commonpath([field.tiles.root for field in fields]))
    everything = [p for field in fields for p in field.tiles.paths]
    logger.info(
        "Scoring against %d tiles from %d other fields", len(everything), len(fields)
    )
    scores = scorer.score(_Tiles(everything, ref_root), out_dir / "cross")
    calibration = _calibrate(scorer, fields, ref_root=ref_root, out_dir=out_dir)

    found = write_significance(
        out_dir / "significance.csv", scores, np.array([v for *_, v in calibration])
    )
    shutil.copyfile(
        out_dir / "cross/measurements.csv", out_dir / "measurements_final.csv"
    )
    # With few calibration tiles even the strangest tile can't reach a small q, and
    # saying how small p can go keeps a null result from reading as a clean field.
    logger.info(
        "%d of %d tiles have a q-value of 0.1 or less, against %d calibration tiles "
        "(the smallest p-value they allow is %.2g)",
        found,
        len(scores),
        len(calibration),
        1 / (len(calibration) + 1),
    )
    return 0


def _calibrate(
    scorer: _Scorer, fields: list[_Field], *, ref_root: Path, out_dir: Path
) -> list[tuple[str, str, float]]:
    """Score each field against the others, and write calibration.csv.

    Returns:
        The field, tile key and score of every calibration tile.
    """
    calibration: list[tuple[str, str, float]] = []
    for field in fields:
        others = [p for o in fields if o.name != field.name for p in o.tiles.paths]
        logger.info(
            "Calibrating on %s against the other %d fields", field.name, len(fields) - 1
        )
        cal = scorer.score(
            _Tiles(others, ref_root),
            out_dir / "calibration" / field.name,
            queries=field.tiles,
            save_maps=False,
        )
        calibration += [(field.name, k, v) for k, v in sorted(cal.items())]
    with (out_dir / "calibration.csv").open("w", encoding="utf-8") as f:
        f.write("Field,Sample,AnomalyScore\n")
        f.writelines(f"{n},{k},{v:.8f}\n" for n, k, v in calibration)
    return calibration


if __name__ == "__main__":
    raise SystemExit(main())
