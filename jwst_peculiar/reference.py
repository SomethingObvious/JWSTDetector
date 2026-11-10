"""AnomalyDINO scoring with a train/ folder of tiles known to be normal and a test/ folder to score.

Use this when you already know which tiles are normal. When you don't, which is the
usual case for JWST, use `jwst-peculiar run`.
"""  # noqa: E501 --help prints this, so the summary line stays whole

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml

from jwst_peculiar.backbones import get_backbone
from jwst_peculiar.data import (
    list_images,
    prepare_out_dir,
    resolve_device,
    tagged_dir,
    write_measurements_csv,
    write_ref_list,
)
from jwst_peculiar.detection import METRICS, run_anomaly_detection
from jwst_peculiar.scoring import SCORE_MODES
from jwst_peculiar.seeding import set_seed

if TYPE_CHECKING:
    from jwst_peculiar.backbones import Backbone

logger = logging.getLogger(__name__)


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data_root", required=True, help="Folder holding train/ and test/."
    )
    p.add_argument("--model_name", default="dinov3-vitl16-pretrain-sat493m")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
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
        choices=("knn", "subspace"),
        help="knn is AnomalyDINO's nearest-patch distance, subspace is SubspaceAD's "
        "PCA residual.",
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
    p.add_argument("--k_neighbors", type=int, default=1)
    p.add_argument("--max_bank_patches", type=int, default=1_000_000)

    p.add_argument("--score_mode", default="top1p", choices=SCORE_MODES)
    p.add_argument("--score_top_frac", type=float, default=0.01)
    p.add_argument("--score_quantile", type=float, default=0.9995)


def _add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1],
        help="How many reference tiles to draw from train/, or -1 for all of them.",
    )
    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--just_seed", type=int, default=None)

    p.add_argument("--masking", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--mask_ref_images", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument("--rotation", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--save_examples", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--save_patch_dists", default=True, action=argparse.BooleanOptionalAction
    )
    p.add_argument("--save_tiffs", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", default="auto")
    p.add_argument(
        "--deterministic", default=False, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--results_dir", default=None, help="Write here instead of results_single/."
    )
    p.add_argument("--tag", default=None)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the results already in the output folder instead of stopping.",
    )
    p.add_argument(
        "--recursive_ref", default=True, action=argparse.BooleanOptionalAction
    )
    p.add_argument(
        "--recursive_query", default=True, action=argparse.BooleanOptionalAction
    )
    p.add_argument("--verbose", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the options of `jwst-peculiar score`.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_model_args(p)
    _add_detector_args(p)
    _add_run_args(p)
    return p.parse_args(argv)


@dataclass(frozen=True)
class _Run:
    """What every shot and seed of one run shares."""

    args: argparse.Namespace
    backbone: Backbone
    train_dir: Path
    test_dir: Path
    ref_candidates: list[Path]
    query_paths: list[Path]

    def results_dir(self, shot: int) -> Path:
        """Pick the folder for one shot count and check it holds no older results.

        Returns:
            The folder, with args.yaml already written to it.
        """
        args = self.args
        if not args.results_dir:
            # A local checkpoint path would otherwise put its own folders, or a drive
            # letter, into the results path.
            model = Path(args.model_name.rstrip("/\\")).name
            base = Path(
                f"results_single/{model}_{self.backbone.resolution}/{shot}-shot"
            )
        elif len(args.shots) > 1:
            # Otherwise each shot count would overwrite the one before it.
            base = Path(args.results_dir) / f"{shot}-shot"
        else:
            base = Path(args.results_dir)
        results_dir = tagged_dir(base, args.tag)
        prepare_out_dir(
            results_dir,
            ["args.yaml", "measurements_seed=*.csv", "seed=*"],
            overwrite=args.overwrite,
        )
        (results_dir / "args.yaml").write_text(
            yaml.safe_dump(vars(args)), encoding="utf-8"
        )
        return results_dir

    def score(self, results_dir: Path, *, shot: int, seed: int) -> None:
        """Draw `shot` reference tiles with `seed`, then score test/ against them."""
        args = self.args
        logger.info("Running %d-shot with seed %d", shot, seed)
        set_seed(seed, deterministic=args.deterministic)

        if shot == -1:
            ref_paths = self.ref_candidates
        else:
            k = max(1, min(shot, len(self.ref_candidates)))
            idx = np.random.default_rng(seed).choice(
                len(self.ref_candidates), size=k, replace=False
            )
            ref_paths = [self.ref_candidates[i] for i in sorted(idx.tolist())]

        run_dir = results_dir / f"seed={seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_ref_list(run_dir / "ref_list.txt", self.train_dir, ref_paths)

        if args.save_examples:
            from jwst_peculiar.utils import plot_reference_masks  # noqa: PLC0415 matplotlib is slow to load

            plot_reference_masks(
                self.backbone,
                ref_paths,
                self.train_dir,
                run_dir / "reference_samples.png",
            )

        scores, memorybank_sec, seconds = run_anomaly_detection(
            self.backbone,
            ref_paths,
            self.train_dir,
            query_paths=self.query_paths,
            query_root=self.test_dir,
            out_dir=run_dir,
            masking=args.masking,
            mask_ref_images=args.mask_ref_images,
            rotation=args.rotation,
            knn_metric=args.knn_metric,
            k_neighbors=args.k_neighbors,
            max_bank_patches=args.max_bank_patches,
            score_mode=args.score_mode,
            score_top_frac=args.score_top_frac,
            score_quantile=args.score_quantile,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=seed,
            save_patch_dists=args.save_patch_dists,
            save_tiffs=args.save_tiffs,
            detector=args.detector,
            subspace_variance=args.subspace_variance,
        )

        for path in (
            run_dir / "measurements.csv",
            results_dir / f"measurements_seed={seed}.csv",
        ):
            write_measurements_csv(
                path, scores, memorybank_sec=memorybank_sec, inference_sec=seconds
            )


def main(argv: list[str] | None = None) -> int:
    """Run `jwst-peculiar score`.

    Returns:
        The exit code, which is always 0.

    Raises:
        SystemExit: If --data_root has no train/ or test/ folder.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # huggingface_hub logs every HTTP request it makes at INFO.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    data_root = Path(args.data_root).resolve()
    train_dir, test_dir = data_root / "train", data_root / "test"
    for d in (train_dir, test_dir):
        if not d.is_dir():
            msg = f"There is no folder at {d}"
            raise SystemExit(msg)

    backbone = get_backbone(
        args.model_name,
        resolve_device(args.device),
        resolution=args.resolution,
        precision=args.precision,
        compile_model=args.compile,
        fast_accumulate=args.fast_accumulate,
        layers=args.layers,
    )
    run = _Run(
        args,
        backbone,
        train_dir,
        test_dir,
        ref_candidates=list_images(train_dir, recursive=args.recursive_ref),
        query_paths=list_images(test_dir, recursive=args.recursive_query),
    )
    seeds = (
        [args.just_seed] if args.just_seed is not None else list(range(args.num_seeds))
    )
    for shot in args.shots:
        results_dir = run.results_dir(shot)
        for seed in seeds:
            run.score(results_dir, shot=shot, seed=seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
