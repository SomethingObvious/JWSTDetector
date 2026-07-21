# JWSTDetector

Unsupervised anomaly detection on JWST imagery. It tiles FITS mosaics into PNGs, embeds each
tile with a DINOv2 backbone, and ranks tiles by how far they sit from a reference set of "normal"
tiles. The point is to surface unusual cutouts (artifacts, rare morphologies) for a human to
review. The detector is derived from AnomalyDINO, adapted for the query-only case where there are
no labels.

## How it works

Both runners share one detector (`src/detection.py`): it builds a FAISS memory bank of patch
embeddings from reference images, then scores each query image by its nearest-neighbour patch
distances.

- `run_query_bootstrap.py` needs no labels. Pass 1 picks a random reference subset from the query
  set and scores everything. Pass 2 rebuilds the reference from the lowest-scoring (most "normal")
  tiles and re-scores. This is the main entry point for JWST data.
- `run_anomalydino.py` is the classic setup: a `train/` reference folder and a `test/` folder to
  score.

## Requirements

- Python 3.11+ and a CUDA GPU. The runners place the model on `cuda`, so scoring needs a GPU.
- `pip install -r requirements.txt`.
- The first run downloads DINOv2 weights through `torch.hub`, so it needs network access once.
- FAISS: `requirements.txt` pins `faiss-cpu`. The example commands use the GPU index
  (`--no-faiss_on_cpu`); install `faiss-gpu` for that, or pass `--faiss_on_cpu` to use the CPU
  index with the pinned build.
- No API keys.

`prep_data.py`, `summarize_results.py`, and `viewer.py` run on CPU. `viewer.py` also needs a Qt
backend and a display.

Every entry point shows a `tqdm` progress bar on its long loops and logs status through the
standard library `logging` module (to stderr). Pass `--verbose` for debug-level output. No new
dependencies are needed for any of this: `tqdm` and `astropy` were already in `requirements.txt`,
and logging, argparse, and seeding are all standard library.

## Usage

`instructions.txt` has full copy-paste commands. The short version:

### 1. FITS to PNG tiles

```
python prep_data.py --sensor miri --indir rawfits --out datasets/miri \
  --pattern "*sci.fits" --sci_ext SCI --wht_ext WHT \
  --tile_size 672 --upscale 1 --stride 644 --min_wht_frac 0.5 \
  --p_lo 1 --p_hi 99.8 --asinh_q 10.0 --wcs_in_name
```

Tiles land in `datasets/miri/query/`. MIRI expects paired `*sci.fits`/`*wht.fits` files; the WHT
map is used to drop tiles with too little coverage. NIRCam mode (`--sensor nircam`) builds RGB
tiles from filter mosaics instead. Put the RA/Dec of each tile centre in its filename with
`--wcs_in_name`.

When you are tuning the filtering thresholds, `--dry-run` does all the cutting and filtering but
writes no PNGs, so you can see how many tiles a mosaic would yield without touching disk.
`--limit N` stops after the first `N` mosaics for a quick trial. Run `python prep_data.py --help`
for the full argument list.

### 2. Score tiles (query-only bootstrap)

```
python run_query_bootstrap.py --data_root datasets/miri --query_subdir query \
  --recursive_query --model_name dinov2_vitl14 --resolution 672 \
  --k_neighbors 20 --masking --mask_ref_images --rotation --save_patch_dists \
  --device cuda:0 --seed 0 --no-faiss_on_cpu \
  --init_ref_frac 0.2 --init_ref_min 100 --init_ref_max 425 \
  --bootstrap_keep_frac 0.1 --bootstrap_keep_min 100 --bootstrap_keep_max 425 \
  --out_dir results_query_only --tag miri_672
```

This writes `pass1/` and `pass2/` under the output directory, copies the final scores to
`measurements_final.csv`, and saves per-tile patch-distance maps under `anomaly_maps/seed=<n>/`
(`.npy`, plus `.tiff` with `--save_tiffs`). Add `--memmap_scores` to keep score vectors on disk
for large query sets. Each `measurements.csv` has columns
`Sample, AnomalyScore, MemoryBankTimeSec, InferenceTimeSec`.

`--seed` fixes the reference-subset draw and seeds `random`, NumPy, and torch so a run repeats.
For bit-for-bit repeatability add `--deterministic`, which puts cuDNN in deterministic mode; it is
off by default because it can slow the GPU path. `run_anomalydino.py` takes the same two flags.

### 3. Summarize and view

```
python summarize_results.py --results_root results_query_only_miri_672 --outdir results_summary

python viewer.py \
  --maps_dir results_query_only_miri_672/pass2/anomaly_maps/seed=0 \
  --png_dir datasets/miri/query \
  --csv_path results_summary/results_query_only_miri_672/samples_sorted_by_anomaly_score.csv
```

`summarize_results.py` collects every run under `--results_root` (any folder with an `args.yaml`)
into summary CSVs, including `samples_sorted_by_anomaly_score.csv`. `viewer.py` steps through the
ranked tiles, showing each PNG next to its patch-distance map (Enter for the next one, `q` to
quit).

`unzipper.py <dir>` decompresses gzipped FITS in a directory (COSMOS-Web WHT mosaics ship as
`.gz`).

## Layout

- `prep_data.py` — FITS mosaics to PNG tiles (MIRI and NIRCam).
- `run_query_bootstrap.py`, `run_anomalydino.py` — anomaly scoring.
- `summarize_results.py` — aggregate runs into CSVs.
- `viewer.py` — inspect ranked tiles and their anomaly maps.
- `src/` — backbones (`backbones.py`), the detector (`detection.py`), shared helpers
  (`utils.py`), and the seeding helper (`seeding.py`).

The CPU-side helpers (tile maths, filename builder, seeding) have a small framework-free check in
`test_helpers.py`; run it with `python test_helpers.py`. It needs only numpy, astropy, and Pillow,
not the GPU stack.

## Limitations

- Scoring needs a CUDA GPU; there is no CPU fallback for the model.
- Scores are relative to the reference set you build. If a query set is mostly artifacts, "normal"
  is whatever the majority looks like, so tune the reference fractions (`--init_ref_*`,
  `--bootstrap_keep_*`) to your data.
- No ground-truth evaluation. This fork drops the MVTec/VisA metric code because JWST tiles have no
  labels; the ranking is meant for human review.
- `prep_data.py` assumes specific FITS layouts: MIRI `*sci.fits`/`*wht.fits` pairs, and a NIRCam
  filename pattern with filter/tile fields.

## Credit

The detector and the DINOv2 wrappers are adapted from AnomalyDINO.
