import sys
from pathlib import Path

# Allow running from any working directory by adding the repo root (script's folder) to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import csv
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import yaml  # type: ignore
from tqdm import trange  # type: ignore

from src.backbones import get_model
from src.detection import run_anomaly_detection

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def maybe_set_cuda_visible_devices(device: str):
    # Keep behavior consistent with older runner: cuda:0 -> CUDA_VISIBLE_DEVICES=0
    if device and device.startswith("cuda:"):
        idx = device.split(":", 1)[1]
        if idx.isdigit():
            os.environ["CUDA_VISIBLE_DEVICES"] = idx


def find_first_image_under(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(IMG_EXTS):
                return os.path.join(dirpath, fn)
    return None


def list_images(query_dir: Path, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in query_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in query_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

    paths = sorted(paths)
    if not paths:
        raise RuntimeError(f"No images found in query_dir={query_dir} (recursive={recursive})")
    return paths


def write_measurements_csv(path: Path, scores: dict, time_mb: float, inf_times: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Sample", "AnomalyScore", "MemoryBankTimeSec", "InferenceTimeSec"])
        for k in sorted(scores.keys()):
            w.writerow([k, f"{float(scores[k]):.8f}", f"{float(time_mb):.6f}", f"{float(inf_times[k]):.6f}"])


def make_ref_dir(tmp_dir: Path, ref_paths: list[Path]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for p in ref_paths:
        dst = tmp_dir / p.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(p)
        except Exception:
            shutil.copyfile(p, dst)
    return tmp_dir


# ---- memmap additions start ----
import bisect


class _MemmapDictView(dict):
    """
    Minimal dict-like wrapper backed by:
      - sorted_keys: list[str] (sorted ascending)
      - values: np.memmap (aligned by index to sorted_keys)

    Supports .keys() and __getitem__ so existing CSV writer stays unchanged.
    """

    def __init__(self, sorted_keys: list[str], values: np.memmap):
        self._keys = sorted_keys
        self._values = values

    def keys(self):
        return self._keys

    def __getitem__(self, key: str):
        i = bisect.bisect_left(self._keys, key)
        if i >= len(self._keys) or self._keys[i] != key:
            raise KeyError(key)
        return self._values[i]


def _dict_to_memmap(
    out_path: Path,
    d: dict,
    *,
    dtype: np.dtype = np.float32,
) -> tuple[list[str], np.memmap]:
    """
    Convert dict[str, number] -> (sorted_keys, memmap array of values aligned to sorted_keys).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(d.keys())
    mm = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=(len(keys),))
    for i, k in enumerate(keys):
        mm[i] = float(d[k])
    mm.flush()
    return keys, mm


# ---- memmap additions end ----


def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, required=True, help="Folder containing query/ only.")
    p.add_argument("--query_subdir", type=str, default="query")
    p.add_argument("--recursive_query", default=True, action=argparse.BooleanOptionalAction)

    p.add_argument("--model_name", type=str, default="dinov2_vits14")
    p.add_argument("--resolution", type=int, default=448)

    p.add_argument("--knn_metric", type=str, default="L2_normalized")
    p.add_argument("--k_neighbors", type=int, default=1)
    p.add_argument("--faiss_on_cpu", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--masking", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--mask_ref_images", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--rotation", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--save_examples", default=False, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_patch_dists", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--save_tiffs", default=False, action=argparse.BooleanOptionalAction)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup_iters", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)

    # Pass 1: random ref from query
    p.add_argument("--init_ref_frac", type=float, default=0.1)
    p.add_argument("--init_ref_min", type=int, default=500)
    p.add_argument("--init_ref_max", type=int, default=5000)

    # Pass 2: bootstrap ref from lowest scoring query images
    p.add_argument("--bootstrap_keep_frac", type=float, default=0.2)
    p.add_argument("--bootstrap_keep_min", type=int, default=500)
    p.add_argument("--bootstrap_keep_max", type=int, default=5000)

    p.add_argument("--out_dir", type=str, default="results_query_only")
    p.add_argument("--tag", type=str, default=None)

    # ---- memmap additions start ----
    p.add_argument(
        "--memmap_scores",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Store score vectors (scores + inference times) as disk-backed np.memmap arrays to reduce RAM usage after each pass.",
    )
    # ---- memmap additions end ----

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    maybe_set_cuda_visible_devices(args.device)

    data_root = Path(args.data_root).resolve()
    query_dir = data_root / args.query_subdir
    if not query_dir.is_dir():
        raise RuntimeError(f"Missing query folder: {query_dir}")

    out_dir = Path(args.out_dir)
    if args.tag:
        out_dir = Path(str(out_dir) + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "args.yaml", "w") as f:
        yaml.safe_dump(vars(args), f)

    # Build model
    model = get_model(args.model_name, "cuda", smaller_edge_size=args.resolution)

    # Warmup
    warm_img = find_first_image_under(str(query_dir))
    if warm_img is None:
        raise RuntimeError(f"No images found under {query_dir} for warmup.")

    for _ in trange(args.warmup_iters, desc="CUDA warmup", leave=False):
        x, _ = model.prepare_image(warm_img)
        _ = model.extract_features(x)

    all_query_paths = list_images(query_dir, recursive=args.recursive_query)
    n = len(all_query_paths)
    rng = np.random.default_rng(args.seed)

    # ---- Pass 1: random reference subset from query ----
    k1 = max(args.init_ref_min, int(round(n * args.init_ref_frac)))
    if args.init_ref_max is not None:
        k1 = min(k1, args.init_ref_max)
    k1 = max(1, min(k1, n))

    ref1_idx = sorted(rng.choice(n, size=k1, replace=False).tolist())
    ref1_paths = [all_query_paths[i] for i in ref1_idx]

    pass1_dir = out_dir / "pass1"
    pass1_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ref_pass1_") as tmp:
        ref_dir1 = make_ref_dir(Path(tmp), ref1_paths)
        scores1, tmb1, tinf1 = run_anomaly_detection(
            model=model,
            ref_dir=str(ref_dir1),
            query_dir=str(query_dir),
            plots_dir=str(pass1_dir),
            recursive_ref=True,
            recursive_query=args.recursive_query,
            save_examples=args.save_examples,
            masking=args.masking,
            mask_ref_images=args.mask_ref_images,
            rotation=args.rotation,
            knn_metric=args.knn_metric,
            knn_neighbors=args.k_neighbors,
            faiss_on_cpu=args.faiss_on_cpu,
            seed=args.seed,
            save_patch_dists=args.save_patch_dists,
            save_tiffs=args.save_tiffs,
        )

    # ---- memmap additions start ----
    if args.memmap_scores:
        keys1, scores1_mm = _dict_to_memmap(pass1_dir / "measurements.memmap", scores1, dtype=np.float32)
        keys1_t, tinf1_mm = _dict_to_memmap(pass1_dir / "inf_times.memmap", tinf1, dtype=np.float32)

        scores1_view = _MemmapDictView(keys1, scores1_mm)
        tinf1_view = _MemmapDictView(keys1_t, tinf1_mm)

        del scores1
        del tinf1

        write_measurements_csv(pass1_dir / "measurements.csv", scores1_view, tmb1, tinf1_view)

        # For bootstrap, we need sorted-by-score order (lowest = most normal).
        order1 = np.argsort(np.asarray(scores1_mm), kind="stable")
        sorted_items = [(keys1[i], float(scores1_mm[i])) for i in order1]
    else:
        write_measurements_csv(pass1_dir / "measurements.csv", scores1, tmb1, tinf1)
        sorted_items = sorted(scores1.items(), key=lambda kv: float(kv[1]))  # lowest = most normal
    # ---- memmap additions end ----

    with open(pass1_dir / "ref_list.txt", "w", encoding="utf-8") as f:
        for p in ref1_paths:
            f.write(str(p.relative_to(query_dir)).replace("\\", "/") + "\n")

    # ---- Pass 2: bootstrap reference from lowest scoring query images ----
    k2 = max(args.bootstrap_keep_min, int(round(len(sorted_items) * args.bootstrap_keep_frac)))
    if args.bootstrap_keep_max is not None:
        k2 = min(k2, args.bootstrap_keep_max)
    k2 = max(1, min(k2, len(sorted_items)))

    # Build a map from query relative-path string -> actual Path
    rel_to_path: dict[str, Path] = {}
    for p in all_query_paths:
        rel = str(p.relative_to(query_dir)).replace("\\", "/")
        rel_to_path[rel] = p

    ref2_rels = [k for k, _ in sorted_items[:k2]]
    ref2_paths = [rel_to_path[k] for k in ref2_rels if k in rel_to_path]

    pass2_dir = out_dir / "pass2"
    pass2_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ref_pass2_") as tmp:
        ref_dir2 = make_ref_dir(Path(tmp), ref2_paths)
        scores2, tmb2, tinf2 = run_anomaly_detection(
            model=model,
            ref_dir=str(ref_dir2),
            query_dir=str(query_dir),
            plots_dir=str(pass2_dir),
            recursive_ref=True,
            recursive_query=args.recursive_query,
            save_examples=args.save_examples,
            masking=args.masking,
            mask_ref_images=args.mask_ref_images,
            rotation=args.rotation,
            knn_metric=args.knn_metric,
            knn_neighbors=args.k_neighbors,
            faiss_on_cpu=args.faiss_on_cpu,
            seed=args.seed,
            save_patch_dists=args.save_patch_dists,
            save_tiffs=args.save_tiffs,
        )

    # ---- memmap additions start ----
    if args.memmap_scores:
        keys2, scores2_mm = _dict_to_memmap(pass2_dir / "measurements.memmap", scores2, dtype=np.float32)
        keys2_t, tinf2_mm = _dict_to_memmap(pass2_dir / "inf_times.memmap", tinf2, dtype=np.float32)

        scores2_view = _MemmapDictView(keys2, scores2_mm)
        tinf2_view = _MemmapDictView(keys2_t, tinf2_mm)

        del scores2
        del tinf2

        write_measurements_csv(pass2_dir / "measurements.csv", scores2_view, tmb2, tinf2_view)
    else:
        write_measurements_csv(pass2_dir / "measurements.csv", scores2, tmb2, tinf2)
    # ---- memmap additions end ----

    with open(pass2_dir / "ref_list.txt", "w", encoding="utf-8") as f:
        for p in ref2_paths:
            f.write(str(p.relative_to(query_dir)).replace("\\", "/") + "\n")

    shutil.copyfile(pass2_dir / "measurements.csv", out_dir / "measurements_final.csv")
    print(f"Done. Final measurements: {out_dir / 'measurements_final.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
