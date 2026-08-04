#!/usr/bin/env python3
"""Label-free anomaly scoring for JWST tiles.

Pass 1 draws a random reference subset from the query set and scores everything
against it. Pass 2 rebuilds the reference from the lowest-scoring tiles, the ones
pass 1 judged most ordinary, and scores again. Nothing here needs labels, which is
the point: JWST tiles do not come with any.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import yaml

from src.backbones import get_backbone
from src.data import (
    list_images,
    relative_key,
    resolve_device,
    write_measurements_csv,
    write_ref_list,
)
from src.detection import METRICS, run_anomaly_detection
from src.scoring import SCORE_MODES
from src.seeding import set_seed

logger = logging.getLogger("run_query_bootstrap")


def subset_size(n: int, frac: float, minimum: int, maximum: int | None) -> int:
    """How many tiles to keep: a fraction of the set, clamped both ways."""
    k = max(minimum, round(n * frac))
    if maximum is not None:
        k = min(k, maximum)
    return max(1, min(k, n))


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("--data_root", required=True, help="Folder containing the query/ subdirectory.")
    p.add_argument("--query_subdir", default="query")
    p.add_argument("--recursive_query", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument(
        "--model_name",
        default="dinov3-vitl16-pretrain-sat493m",
        help="dinov3-* (transformers, gated weights) or dinov2_* (torch.hub).",
    )
    p.add_argument("--resolution", type=int, default=448, help="Snapped down to a patch multiple.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4, help="Tile decoding workers.")
    p.add_argument(
        "--autocast",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the backbone in bf16/fp16 on CUDA. Distances stay in fp32.",
    )

    p.add_argument("--knn_metric", default="L2_normalized", choices=METRICS)
    p.add_argument("--k_neighbors", type=int, default=1)
    p.add_argument(
        "--max_bank_patches",
        type=int,
        default=1_000_000,
        help="Cap on memory-bank patches. ~4 GB of fp32 at 1024 dims.",
    )

    p.add_argument("--score_mode", default="top1p", choices=SCORE_MODES)
    p.add_argument("--score_top_frac", type=float, default=0.01, help="For --score_mode topk_mean.")
    p.add_argument("--score_quantile", type=float, default=0.9995, help="For score_mode quantile.")

    p.add_argument("--masking", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction)
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
        help="Render a few reference tiles with their background masks.",
    )
    p.add_argument("--save_patch_dists", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_tiffs", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1, mps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--deterministic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Force cuDNN deterministic mode. Slower, so off by default.",
    )

    p.add_argument("--init_ref_frac", type=float, default=0.1)
    p.add_argument("--init_ref_min", type=int, default=500)
    p.add_argument("--init_ref_max", type=int, default=5000)

    p.add_argument("--bootstrap_keep_frac", type=float, default=0.2)
    p.add_argument("--bootstrap_keep_min", type=int, default=500)
    p.add_argument("--bootstrap_keep_max", type=int, default=5000)

    p.add_argument("--out_dir", default="results_query_only")
    p.add_argument("--tag", default=None)
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    set_seed(args.seed, deterministic=args.deterministic)

    query_dir = Path(args.data_root).resolve() / args.query_subdir
    if not query_dir.is_dir():
        raise SystemExit(f"Missing query folder: {query_dir}")

    out_dir = Path(f"{args.out_dir}_{args.tag}" if args.tag else args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.yaml").write_text(yaml.safe_dump(vars(args)), encoding="utf-8")

    backbone = get_backbone(
        args.model_name,
        resolve_device(args.device),
        resolution=args.resolution,
        autocast=args.autocast,
    )

    query_paths = list_images(query_dir, recursive=args.recursive_query)
    logger.info("%d query tiles under %s", len(query_paths), query_dir)
    rng = np.random.default_rng(args.seed)

    def detect(ref_paths: list[Path], pass_dir: Path):
        pass_dir.mkdir(parents=True, exist_ok=True)
        write_ref_list(pass_dir / "ref_list.txt", query_dir, ref_paths)
        scores, memorybank_sec, seconds = run_anomaly_detection(
            backbone,
            ref_paths,
            query_dir,
            query_paths,
            query_dir,
            pass_dir,
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
            seed=args.seed,
            save_patch_dists=args.save_patch_dists,
            save_tiffs=args.save_tiffs,
        )
        write_measurements_csv(pass_dir / "measurements.csv", scores, memorybank_sec, seconds)
        return scores

    # Pass 1: a random slice of the query set stands in for "normal".
    n = len(query_paths)
    k1 = subset_size(n, args.init_ref_frac, args.init_ref_min, args.init_ref_max)
    ref1 = [query_paths[i] for i in sorted(rng.choice(n, size=k1, replace=False).tolist())]
    logger.info("pass 1: %d random reference tiles of %d", k1, n)

    if args.save_examples:
        from src.utils import plot_reference_masks

        plot_reference_masks(backbone, ref1, query_dir, out_dir / "reference_samples.png")

    scores1 = detect(ref1, out_dir / "pass1")

    # Pass 2: rebuild the reference from whatever pass 1 called most ordinary.
    by_score = sorted(scores1, key=scores1.__getitem__)
    k2 = subset_size(
        len(by_score), args.bootstrap_keep_frac, args.bootstrap_keep_min, args.bootstrap_keep_max
    )
    lookup = {relative_key(p, query_dir): p for p in query_paths}
    ref2 = [lookup[key] for key in by_score[:k2]]
    logger.info("pass 2: %d lowest-scoring tiles as the reference", len(ref2))

    detect(ref2, out_dir / "pass2")

    shutil.copyfile(out_dir / "pass2/measurements.csv", out_dir / "measurements_final.csv")
    logger.info("done: %s", out_dir / "measurements_final.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
