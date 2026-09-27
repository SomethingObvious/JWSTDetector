#!/usr/bin/env python3
"""AnomalyDINO scoring with a train/ folder of tiles known to be normal and a test/ folder to score.

Use this when you already know which tiles are normal. When you don't, which is the
usual case for JWST, use run_query_bootstrap.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

from jwstdetector.backbones import get_backbone
from jwstdetector.data import list_images, resolve_device, write_measurements_csv, write_ref_list
from jwstdetector.detection import METRICS, run_anomaly_detection
from jwstdetector.scoring import SCORE_MODES
from jwstdetector.seeding import set_seed

logger = logging.getLogger("run_anomalydino")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("--data_root", required=True, help="Folder holding train/ and test/.")
    p.add_argument("--model_name", default="dinov3-vitl16-pretrain-sat493m")
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--autocast", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--knn_metric", default="L2_normalized", choices=METRICS)
    p.add_argument("--k_neighbors", type=int, default=1)
    p.add_argument("--max_bank_patches", type=int, default=1_000_000)

    p.add_argument("--score_mode", default="top1p", choices=SCORE_MODES)
    p.add_argument("--score_top_frac", type=float, default=0.01)
    p.add_argument("--score_quantile", type=float, default=0.9995)

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
    p.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--rotation", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_examples", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_patch_dists", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_tiffs", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", default="auto")
    p.add_argument("--deterministic", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--results_dir", default=None, help="Write here instead of results_single/.")
    p.add_argument("--tag", default=None)
    p.add_argument("--recursive_ref", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--recursive_query", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
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
            raise SystemExit(f"There is no folder at {d}")

    backbone = get_backbone(
        args.model_name,
        resolve_device(args.device),
        resolution=args.resolution,
        autocast=args.autocast,
    )

    ref_candidates = list_images(train_dir, recursive=args.recursive_ref)
    query_paths = list_images(test_dir, recursive=args.recursive_query)
    seeds = [args.just_seed] if args.just_seed is not None else list(range(args.num_seeds))

    for shot in args.shots:
        if not args.results_dir:
            base = Path(f"results_single/{args.model_name}_{backbone.resolution}/{shot}-shot")
        elif len(args.shots) > 1:
            # Otherwise each shot count would overwrite the one before it.
            base = Path(args.results_dir) / f"{shot}-shot"
        else:
            base = Path(args.results_dir)
        results_dir = Path(f"{base}_{args.tag}" if args.tag else base)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "args.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

        for seed in seeds:
            logger.info("Running %d-shot with seed %d", shot, seed)
            set_seed(seed, deterministic=args.deterministic)

            if shot == -1:
                ref_paths = ref_candidates
            else:
                k = max(1, min(shot, len(ref_candidates)))
                idx = np.random.default_rng(seed).choice(len(ref_candidates), size=k, replace=False)
                ref_paths = [ref_candidates[i] for i in sorted(idx.tolist())]

            run_dir = results_dir / f"seed={seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_ref_list(run_dir / "ref_list.txt", train_dir, ref_paths)

            if args.save_examples:
                from jwstdetector.utils import plot_reference_masks

                plot_reference_masks(
                    backbone, ref_paths, train_dir, run_dir / "reference_samples.png"
                )

            scores, memorybank_sec, seconds = run_anomaly_detection(
                backbone,
                ref_paths,
                train_dir,
                query_paths,
                test_dir,
                run_dir,
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
            )

            write_measurements_csv(run_dir / "measurements.csv", scores, memorybank_sec, seconds)
            # summarize_results.py finds per-seed runs by this file name.
            write_measurements_csv(
                results_dir / f"measurements_seed={seed}.csv", scores, memorybank_sec, seconds
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
