import os
import time

import cv2
import faiss
import numpy as np
import tifffile as tiff
import torch
from tqdm import tqdm

from src.post_eval import mean_top1p
from src.utils import augment_image, dists2map, plot_ref_images


_IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _list_images(folder: str, recursive: bool) -> list[str]:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder does not exist: {folder}")

    rel_paths: list[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for fn in sorted(filenames):
                if fn.lower().endswith(_IMG_EXTS):
                    full = os.path.join(dirpath, fn)
                    rel = os.path.relpath(full, folder)
                    rel_paths.append(rel)
    else:
        for fn in sorted(os.listdir(folder)):
            full = os.path.join(folder, fn)
            if os.path.isfile(full) and fn.lower().endswith(_IMG_EXTS):
                rel_paths.append(fn)

    if not rel_paths:
        raise RuntimeError(f"No images found under: {folder} (recursive={recursive})")
    return rel_paths


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _aggregate_score(
    d_flat: np.ndarray,
    mode: str = "peak_minus_median",
    *,
    top_frac: float = 0.01,
    quantile: float = 0.9995,
    eps: float = 1e-8,
) -> float:
    """
    Aggregates per-patch distances into an image-level anomaly score.

    Modes:
      - "top1p": repo default (mean of top 1% patches) via mean_top1p()
      - "max": max patch distance (very needle-sensitive)
      - "quantile": e.g. 0.9995 quantile (needle-sensitive but more stable than max)
      - "topk_mean": mean of top top_frac patches
      - "peak_minus_median": max(d) - median(d) (needle-ness vs global elevation)
      - "peak_z": (max-mean)/std (needle-ness normalized by spread)
    """
    d = np.asarray(d_flat, dtype=np.float32).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0

    mode = (mode or "top1p").lower()

    if mode == "top1p":
        return float(mean_top1p(d))
    if mode == "max":
        return float(np.max(d))
    if mode == "quantile":
        q = float(np.clip(quantile, 0.0, 1.0))
        return float(np.quantile(d, q))
    if mode == "topk_mean":
        f = float(np.clip(top_frac, 1e-6, 1.0))
        m = max(1, int(round(d.size * f)))
        top = np.partition(d, -m)[-m:]
        return float(np.mean(top))
    if mode == "peak_minus_median":
        return float(np.max(d) - np.median(d))
    if mode == "peak_z":
        mu = float(np.mean(d))
        sd = float(np.std(d))
        return float((np.max(d) - mu) / (sd + eps))

    raise ValueError(f"Unknown score_mode={mode!r}")


def run_anomaly_detection(
    model,
    ref_dir: str,
    query_dir: str,
    plots_dir: str,
    *,
    recursive_ref: bool = True,
    recursive_query: bool = True,
    save_examples: bool = False,
    masking=None,
    mask_ref_images: bool = False,
    rotation: bool = False,
    knn_metric: str = "L2_normalized",
    knn_neighbors: int = 1,
    faiss_on_cpu: bool = False,
    seed: int = 0,
    save_patch_dists: bool = True,
    save_tiffs: bool = False,
    # --- new scoring knobs ---
    score_mode: str = "top1p",
    score_top_frac: float = 0.01,
    score_quantile: float = 0.9995,
):
    """
    Query-only anomaly detection.
    Builds a memory bank from images in ref_dir and scores all images in query_dir.

    Returns:
      anomaly_scores: dict[str, float] keyed by query relative path
      time_memorybank: float seconds
      inference_times: dict[str, float] keyed by query relative path
    """
    assert knn_metric in ("L2", "L2_normalized")
    assert knn_neighbors >= 1

    ref_dir = os.path.abspath(ref_dir)
    query_dir = os.path.abspath(query_dir)
    plots_dir = os.path.abspath(plots_dir)

    ref_files = _list_images(ref_dir, recursive_ref)
    query_files = _list_images(query_dir, recursive_query)

    # ----------------------------
    # Build memory bank (ref_dir)
    # ----------------------------
    features_ref_chunks = []
    images_ref = []
    masks_ref = []
    vis_background = []

    with torch.inference_mode():
        start_time = time.time()

        for img_ref_rel in tqdm(ref_files, desc="Building memory bank", leave=False):
            img_ref_path = os.path.join(ref_dir, img_ref_rel)
            image_ref = cv2.cvtColor(cv2.imread(img_ref_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            if rotation:
                ref_aug_list = augment_image(image_ref)
            else:
                ref_aug_list = [image_ref]

            for image_ref_aug in ref_aug_list:
                image_ref_tensor, grid_size1 = model.prepare_image(image_ref_aug)
                features_ref_i = model.extract_features(image_ref_tensor)

                # Mask reference tokens if requested
                mask_ref = model.compute_background_mask(
                    features_ref_i,
                    grid_size1,
                    threshold=10,
                    masking_type=(mask_ref_images and masking),
                )

                features_ref_chunks.append(features_ref_i[mask_ref])

                if save_examples:
                    images_ref.append(image_ref_aug)
                    vis_img_bg = model.get_embedding_visualization(features_ref_i, grid_size1, mask_ref)
                    masks_ref.append(mask_ref)
                    vis_background.append(vis_img_bg)

        features_ref = np.concatenate(features_ref_chunks, axis=0).astype("float32")

        # FAISS index
        if faiss_on_cpu:
            knn_index = faiss.IndexFlatL2(features_ref.shape[1])
        else:
            res = faiss.StandardGpuResources()
            knn_index = faiss.GpuIndexFlatL2(res, features_ref.shape[1])

        if knn_metric == "L2_normalized":
            faiss.normalize_L2(features_ref)

        knn_index.add(features_ref)
        time_memorybank = time.time() - start_time

    # Optional reference inspection plots
    if save_examples:
        os.makedirs(os.path.join(plots_dir, "ref_examples"), exist_ok=True)
        plot_ref_images(
            images_ref,
            masks_ref,
            vis_background,
            grid_size1,
            os.path.join(plots_dir, "ref_examples") + os.sep,
            title="Reference Images",
            img_names=ref_files,
        )

    # ----------------------------
    # Score query_dir
    # ----------------------------
    inference_times: dict[str, float] = {}
    anomaly_scores: dict[str, float] = {}

    maps_root = os.path.join(plots_dir, f"anomaly_maps/seed={seed}")
    if save_patch_dists or save_tiffs:
        os.makedirs(maps_root, exist_ok=True)

    with torch.inference_mode():
        for img_q_rel in tqdm(query_files, desc="Scoring query images", leave=True):
            start_time = time.time()

            image_test_path = os.path.join(query_dir, img_q_rel)
            image_test = cv2.cvtColor(cv2.imread(image_test_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            image_tensor2, grid_size2 = model.prepare_image(image_test)
            features2 = model.extract_features(image_tensor2)

            if masking:
                mask2 = model.compute_background_mask(features2, grid_size2, threshold=10, masking_type=masking)
            else:
                mask2 = np.ones(features2.shape[0], dtype=bool)

            features2_masked = features2[mask2].astype("float32", copy=False)

            # kNN distances
            if knn_metric == "L2":
                # FAISS returns squared L2 distances for IndexFlatL2/GpuIndexFlatL2
                distances, _ = knn_index.search(features2_masked, k=knn_neighbors)
                if knn_neighbors > 1:
                    distances = distances.mean(axis=1)
                distances = np.sqrt(distances)
            else:
                # L2 between unit vectors: ||a-b||^2 = 2 - 2cos => (||a-b||^2)/2 = 1 - cos
                faiss.normalize_L2(features2_masked)
                distances, _ = knn_index.search(features2_masked, k=knn_neighbors)
                if knn_neighbors > 1:
                    distances = distances.mean(axis=1)
                distances = distances / 2.0

            output_distances = np.zeros_like(mask2, dtype=np.float32)
            output_distances[mask2] = distances.squeeze().astype(np.float32, copy=False)
            d_masked = output_distances.reshape(grid_size2)

            if torch.cuda.is_available() and (not faiss_on_cpu):
                torch.cuda.synchronize()

            inf_time = time.time() - start_time
            inference_times[img_q_rel] = float(inf_time)

            anomaly_scores[img_q_rel] = _aggregate_score(
                output_distances,
                mode=score_mode,
                top_frac=score_top_frac,
                quantile=score_quantile,
            )

            # Save maps preserving subfolder structure under query/
            stem = os.path.splitext(img_q_rel)[0]

            if save_tiffs:
                out_tiff = os.path.join(maps_root, stem + ".tiff")
                _ensure_parent_dir(out_tiff)
                anomaly_map = dists2map(d_masked, image_test.shape)
                tiff.imwrite(out_tiff, anomaly_map)

            if save_patch_dists:
                out_npy = os.path.join(maps_root, stem + ".npy")
                _ensure_parent_dir(out_npy)
                np.save(out_npy, d_masked)

    return anomaly_scores, time_memorybank, inference_times
