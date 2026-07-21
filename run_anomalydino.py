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
    # The model always goes on "cuda", so --device cuda:1 works by hiding the other GPUs.
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx


def list_images(folder: Path, recursive: bool) -> list[Path]:
    if not folder.is_dir():
        raise RuntimeError(f"There is no folder at {folder}")

    if recursive:
        paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"There are no images in {folder} (recursive={recursive})")
    return paths


def make_ref_dir(tmp_dir: Path, base_dir: Path, ref_paths: list[Path]) -> Path:
    """Link or copy the chosen tiles into tmp_dir.

    Their paths under base_dir come along, so two tiles with the same name can't collide.
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
        # The same columns run_query_bootstrap.py writes, so summarize_results.py reads both.
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
            "AnomalyDINO scoring with a train/ folder of tiles known to be normal and a test/ "
            "folder to score, for each of --shots and every seed. It needs a CUDA GPU."
        )
    )

    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Folder holding train/ and test/.",
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vits14",
        help="A DINOv2 name like dinov2_vitl14, or a torchvision ViT like vit_b_16.",
    )

    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--knn_metric", type=str, default="L2_normalized")
    p.add_argument("--k_neighbors", type=int, default=1)

    p.add_argument(
        "--faiss_on_cpu",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Search on the CPU. Without it the GPU index is used, which needs faiss-gpu.",
    )

    p.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[1],
        help="How many reference tiles to draw from train/, or -1 for all of them.",
    )

    p.add_argument("--num_seeds", type=int, default=1)
    p.add_argument("--just_seed", type=int, default=None)

    p.add_argument(
        "--save_examples",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Plot up to 32 reference tiles beside their embedding PCA and background mask.",
    )

    p.add_argument(
        "--save_tiffs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save a smoothed full-size anomaly map of each tile as a TIFF.",
    )

    p.add_argument(
        "--save_patch_dists",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Save each tile's raw patch-distance grid as the .npy viewer.py reads.",
    )

    p.add_argument(
        "--masking",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Score only foreground patches. Only the DINOv2 backbones mask anything.",
    )

    # An int rather than a flag, so older commands with --rotation 1 still work.
    p.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="1 puts each reference tile in the bank at eight rotations, 45 degrees apart.",
    )

    p.add_argument(
        "--mask_ref_images",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Mask the reference tiles as well when --masking is on.",
    )

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup_iters", type=int, default=25)
    p.add_argument(
        "--deterministic",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Put cuDNN in deterministic mode, which is slower.",
    )
    p.add_argument("--verbose", action="store_true", help="Debug-level logging.")

    p.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Write here instead of results_single/.",
    )
    p.add_argument("--tag", type=str, default=None)

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
        raise RuntimeError(f"There is no train/ folder at {train_dir}")
    if not test_dir.is_dir():
        raise RuntimeError(f"There is no test/ folder at {test_dir}")

    maybe_set_cuda_visible_devices(args.device)

    model = get_model(args.model_name, "cuda", smaller_edge_size=args.resolution)

    # Untimed passes, so the CUDA kernels are warm by the time scoring is timed.
    warmup_img = find_first_image_under(str(train_dir)) or find_first_image_under(str(data_root))
    if warmup_img is None:
        raise RuntimeError(f"There are no images under {data_root}")

    for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
        img_tensor, _ = model.prepare_image(warmup_img)
        _ = model.extract_features(img_tensor)

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
            logger.info("Running %s-shot with seed %s", shot, seed)
            set_seed(seed, deterministic=args.deterministic)

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
                    recursive_ref=True,  # make_ref_dir keeps subfolders
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

            # summarize_results.py finds per-seed runs by this file name.
            try:
                shutil.copyfile(measurements_file, results_dir / f"measurements_seed={seed}.csv")
            except OSError as e:
                logger.warning("Couldn't copy the measurements to a per-seed file. %s", e)

            logger.info("Results are in %s", run_dir)

    logger.info("Finished every run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
