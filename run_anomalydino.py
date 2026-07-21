import argparse
import csv
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import yaml
from tqdm import trange

from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.seeding import set_seed

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

logger = logging.getLogger("run_anomalydino")


def find_first_image_under(root_dir: str):
    for p in Path(root_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            return str(p)
    return None


def maybe_set_cuda_visible_devices(device: str):
    # Keep behavior consistent with older runner: cuda:0 -> CUDA_VISIBLE_DEVICES=0
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx


def list_images(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        raise RuntimeError(f"Folder does not exist: {folder}")

    if recursive:
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"No images found in folder={folder} (recursive={recursive})")
    return paths


def make_ref_dir(tmp_dir: Path, base_dir: Path, ref_paths: list[Path]) -> Path:
    """
    Create a temporary reference folder containing symlinks/copies to selected ref images.
    Preserves relative paths under base_dir to avoid filename collisions.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p in ref_paths:
        rel = p.relative_to(base_dir)
        dst = tmp_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        try:
            dst.symlink_to(p)
        except Exception:
            shutil.copyfile(p, dst)

    return tmp_dir


def write_measurements_csv(path: Path, scores: dict, time_mb: float, inf_times: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        # Use the "Sec" suffix to match the query-only runner + summarizer flexibility.
        w.writerow(["Sample", "AnomalyScore", "MemoryBankTimeSec", "InferenceTimeSec"])
        for k in sorted(scores.keys()):
            w.writerow(
                [
                    k,
                    f"{float(scores[k]):.8f}",
                    f"{float(time_mb):.6f}",
                    f"{float(inf_times[k]):.6f}",
                ]
            )


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description=(
            "Classic AnomalyDINO scoring with an explicit train/ reference folder and a test/ "
            "folder to score, sweeping over --shots and seeds. Needs a CUDA GPU."
        )
    )

    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root containing train/ (reference images) and test/ (images to score).",
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vits14",
        help="Backbone model name (e.g., dinov2_vits14).",
    )

    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--knn_metric", type=str, default="L2_normalized")
    p.add_argument("--k_neighbors", type=int, default=1)

    p.add_argument(
        "--faiss_on_cpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Run FAISS kNN search on CPU (otherwise GPU FAISS is used when available).",
    )

    p.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1],
        help="Number of reference samples to use from train/. Use -1 for all reference images.",
    )

    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--just_seed", type=int, default=None)

    p.add_argument(
        "--save_examples",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save example outputs during detection (if supported).",
    )

    p.add_argument(
        "--save_tiffs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save anomaly maps as TIFFs (if supported).",
    )

    p.add_argument(
        "--save_patch_dists",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save patch-distance artifacts (debug; if supported).",
    )

    p.add_argument(
        "--masking",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Enable background masking (if supported).",
    )

    # Keep int for backward-compat with old CLI, but interpret as bool.
    p.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotation augmentation flag for reference images (0/1).",
    )

    p.add_argument(
        "--mask_ref_images",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Mask reference images too (if supported).",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup_iters", type=int, default=25)
    p.add_argument(
        "--deterministic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Force cuDNN deterministic mode (slower; off by default to preserve speed/numerics).",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")

    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Optional override for output directory. If omitted, uses results_single/...",
    )
    p.add_argument("--tag", type=str, default=None)

    # Detection traversal
    p.add_argument("--recursive_ref", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--recursive_query", default=True, action=argparse.BooleanOptionalAction)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    data_root = Path(args.data_root).resolve()
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if not train_dir.is_dir():
        raise RuntimeError(f"Missing train/ directory at: {train_dir}")
    if not test_dir.is_dir():
        raise RuntimeError(f"Missing test/ directory at: {test_dir}")

    maybe_set_cuda_visible_devices(args.device)

    # Match your other scripts: get_model(..., "cuda", ...)
    model = get_model(args.model_name, "cuda", smaller_edge_size=args.resolution)

    # Warmup image: prefer train/, else any image anywhere under data_root.
    warmup_img = find_first_image_under(str(train_dir)) or find_first_image_under(str(data_root))
    if warmup_img is None:
        raise RuntimeError(f"Could not find any image under {data_root} for warmup.")

    for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
        img_tensor, _ = model.prepare_image(warmup_img)
        _ = model.extract_features(img_tensor)

    # Enumerate candidates once.
    ref_candidates = list_images(train_dir, recursive=args.recursive_ref)

    seeds = [args.just_seed] if args.just_seed is not None else list(range(args.num_seeds))

    for shot in list(args.shots):
        if args.results_dir is not None:
            results_dir = Path(args.results_dir)
        else:
            results_dir = Path(f"results_single/{args.model_name}_{args.resolution}/{shot}-shot")

        if args.tag:
            results_dir = Path(str(results_dir) + "_" + args.tag)

        results_dir.mkdir(parents=True, exist_ok=True)

        with (results_dir / "args.yaml").open("w") as f:
            yaml.safe_dump(vars(args), f)

        for seed in seeds:
            logger.info("=========== Shot = %s, Seed = %s ===========", shot, seed)
            set_seed(seed, deterministic=args.deterministic)

            # Determine reference subset for this (shot, seed)
            if shot == -1:
                ref_paths = ref_candidates
            else:
                k = max(1, min(int(shot), len(ref_candidates)))
                rng = np.random.default_rng(seed)
                idx = sorted(rng.choice(len(ref_candidates), size=k, replace=False).tolist())
                ref_paths = [ref_candidates[i] for i in idx]

            run_dir = results_dir / f"seed={seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            with (run_dir / "ref_list.txt").open("w", encoding="utf-8") as f:
                for p in ref_paths:
                    f.write(str(p.relative_to(train_dir)).replace("\\", "/") + "\n")

            with tempfile.TemporaryDirectory(prefix="ref_train_shot_") as tmp:
                ref_dir = make_ref_dir(Path(tmp), train_dir, ref_paths)

                anomaly_scores, time_memorybank, inference_times = run_anomaly_detection(
                    model=model,
                    ref_dir=str(ref_dir),
                    query_dir=str(test_dir),
                    plots_dir=str(run_dir),
                    recursive_ref=True,  # temp ref_dir is flat-ish but safe to recurse
                    recursive_query=args.recursive_query,
                    save_examples=args.save_examples,
                    masking=args.masking,
                    mask_ref_images=args.mask_ref_images,
                    rotation=bool(args.rotation),
                    knn_metric=args.knn_metric,
                    knn_neighbors=args.k_neighbors,
                    faiss_on_cpu=args.faiss_on_cpu,
                    seed=seed,
                    save_patch_dists=args.save_patch_dists,
                    save_tiffs=args.save_tiffs,
                )

            measurements_file = run_dir / "measurements.csv"
            write_measurements_csv(
                measurements_file, anomaly_scores, time_memorybank, inference_times
            )

            # Copy into results_dir under a seed-tagged name so summarize_results.py finds it.
            try:
                shutil.copyfile(measurements_file, results_dir / f"measurements_seed={seed}.csv")
            except OSError as e:
                logger.warning("could not write compatibility measurements file: %s", e)

            logger.info("Finished AD, outputs saved to %s", run_dir)

    logger.info("Finished all runs!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
