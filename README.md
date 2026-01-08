# JWSTDetectpr (JWST helpers + runners + AnomalyDINO) 

Utilities and CLI scripts to (1) preprocess JWST mosaics into PNG tiles and (2) run DINOv2-based anomaly detection in either a standard `train/` vs `test/` setting or a query-only “bootstrap reference from query” setting.
Example end-to-end commands (prep → run → summarize → view) are included in `instructions.txt`.

## Features
- FITS → tiled PNG dataset generation with streaming/memmap-friendly processing (one mosaic at a time) to limit peak RAM.
- Two inference modes:
  - Standard: build a reference memory bank from `train/` and score `test/`.
  - Query-only bootstrap: pick an initial reference subset from `query/`, score all `query/`, then bootstrap a better reference set from the lowest-scoring (most “normal”) samples and re-score.
- Outputs include per-sample anomaly scores (`measurements.csv`) and optional saved artifacts (anomaly maps as TIFFs and/or per-patch distance arrays).

## Installation
Create a virtual environment and install dependencies matching what the scripts import (notably PyTorch, NumPy, FAISS, OpenCV, tifffile, tqdm, PyYAML, Pandas, plus Astropy + Pillow for FITS preprocessing).
`instructions.txt` assumes an existing `.venv/` (activated via `source .venv/bin/activate`).

## Usage
### 1) Prepare data (FITS → PNG tiles)
`prep_data.py` writes tiles under an output root (e.g., `datasets/miri/query/`) and supports MIRI and NIRCam modes.

#### MIRI (big data example)
python prep_data.py \
  --sensor miri \
  --indir rawfits \
  --out datasets/miri \
  --pattern "*i2d.fits" \
  --sci_ext SCI \
  --wht_ext WHT \
  --tile_size 448 --upscale 1 --stride 420 \
  --min_wht_frac 0.80 \
  --mean_thresh 5 --var_thresh 50 \
  --p_lo 1 --p_hi 99.8 --asinh_q 10.0 \
  --wcs_in_name

#### MIRI (small data example)
python prep_data.py \
  --sensor miri \
  --indir rawfits \
  --out datasets/miri \
  --pattern "*sci.fits" \
  --sci_ext SCI \
  --wht_ext WHT \
  --tile_size 672 --upscale 1 --stride 644 \
  --min_wht_frac 0.5 \
  --mean_thresh 5 --var_thresh 50 \
  --p_lo 1 --p_hi 99.8 --asinh_q 10.0 \
  --wcs_in_name

### 2A) Query-only bootstrap anomaly detection
`run_query_bootstrap.py` expects a dataset root containing a `query/` subfolder of images (PNG/JPG/TIFF/etc.).

#### Small data example (from `instructions.txt`)
python run_query_bootstrap.py \
  --data_root datasets/miri \
  --query_subdir query \
  --recursive_query \
  --model_name dinov2_vitl14 \
  --resolution 672 \
  --knn_metric L2_normalized \
  --k_neighbors 20 \
  --no-faiss_on_cpu \
  --masking \
  --mask_ref_images \
  --rotation \
  --save_examples \
  --save_patch_dists \
  --save_tiffs \
  --device cuda:0 \
  --warmup_iters 100 \
  --seed 0 \
  --init_ref_frac 0.2 \
  --init_ref_min 100 \
  --init_ref_max 425 \
  --bootstrap_keep_frac 0.1 \
  --bootstrap_keep_min 100 \
  --bootstrap_keep_max 425 \
  --out_dir results_query_only \
  --tag miri_672_big

#### Big data example (with memmapped score storage)
python run_query_bootstrap.py \
  --data_root datasets/miri \
  --query_subdir query \
  --recursive_query \
  --model_name dinov2_vits14 \
  --resolution 488 \
  --knn_metric L2_normalized \
  --k_neighbors 2 \
  --no-faiss_on_cpu \
  --masking \
  --mask_ref_images \
  --rotation \
  --save_examples \
  --save_patch_dists \
  --save_tiffs \
  --device cuda:0 \
  --warmup_iters 200 \
  --seed 0 \
  --init_ref_frac 0.2 \
  --init_ref_min 1000 \
  --init_ref_max 2000 \
  --bootstrap_keep_frac 0.1 \
  --bootstrap_keep_min 1000 \
  --bootstrap_keep_max 2000 \
  --out_dir results_query_only \
  --tag miri_jwst_488_maxq \
  --memmap_scores

### 2B) Standard train/test anomaly detection
`run_anomalydino.py` expects `data_root/train/` (reference) and `data_root/test/` (to score), and can optionally create compatibility links `train/good -> train` and `test/query -> test`.

## Outputs & utilities
- Scoring outputs include `measurements.csv` with columns `Sample, AnomalyScore, MemoryBankTimeSec, InferenceTimeSec`.
- Query-only bootstrap writes `pass1/` and `pass2/` folders and copies the final scores to `measurements_final.csv`.
- `summarize_results.py` aggregates runs under a `--results_root` into summary CSVs (including a `samples_sorted_by_anomaly_score.csv`).
- `viewer.py` can be used to inspect anomaly maps alongside the ranked CSV (example invocation in `instructions.txt`).
