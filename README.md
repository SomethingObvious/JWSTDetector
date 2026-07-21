# JWSTDetector

JWSTDetector is unsupervised anomaly detection for James Webb Space Telescope (JWST) images, so it looks for strange things without any labels. It cuts FITS mosaics into PNG tiles, embeds every tile with a DINOv2 backbone, and ranks the tiles by how far their patches sit from a reference set of ordinary ones, so a person can look through the top of the list for artifacts, odd morphologies or whatever else the reference doesn't account for. The detector is adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO).

## Install

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt torchvision scikit-learn
```

It needs Python 3.11 or newer and a CUDA GPU, as the runners always put the model on `cuda`. The DINOv2 weights come down through `torch.hub` on the first run, so that one needs a network connection. `viewer.py` opens a Qt window, so it also needs PyQt and a display.

`requirements.txt` pins `faiss-cpu`, but the runners use the FAISS GPU index unless you pass `--faiss_on_cpu`. So either pass that, or install a GPU build of FAISS in its place.

## Usage

```
python prep_data.py --sensor miri --indir rawfits --out datasets/miri --pattern "*sci.fits" \
  --tile_size 672 --upscale 1 --stride 644 --min_wht_frac 0.5 --wcs_in_name

python run_query_bootstrap.py --data_root datasets/miri --model_name dinov2_vitl14 \
  --resolution 672 --k_neighbors 20 --rotation --faiss_on_cpu --out_dir results --tag miri

python summarize_results.py --results_root results_miri --outdir results_summary
python viewer.py --maps_dir results_miri/pass2/anomaly_maps/seed=0 --png_dir datasets/miri/query \
  --csv_path results_summary/results_miri/samples_sorted_by_anomaly_score.csv
```

`prep_data.py` drops a tile when too little of it has a positive weight (`--min_wht_frac`, which wants a `*wht.fits` beside each `*sci.fits`) and when the stretched tile comes out too dark or too flat (`--mean_thresh` and `--var_thresh`). Each MIRI tile gets its own asinh stretch. `--dry-run` shows what the thresholds would keep without writing anything. With `--wcs_in_name` each file name carries the RA and Dec of the tile centre, so a candidate can be found again on the sky. NIRCam mode (`--sensor nircam`) makes colour tiles from three COSMOS-Web filter mosaics, and `unzipper.py` expands the gzipped weight maps those come with.

`run_query_bootstrap.py` scores every tile against a random reference drawn from the tiles themselves, then rebuilds the reference from the lowest scores and scores again. The final ranking is written to `measurements_final.csv`, with a patch-distance grid per tile under `pass2/anomaly_maps/`. `run_anomalydino.py` is for when you already have a `train/` folder of normal tiles. `instructions.txt` has fuller example commands.

## What It Won't Do

Scores are only relative to the reference, so if most tiles share an artifact, that artifact counts as normal. There is no ground truth or metric here either, just a ranking to read.

A tile in the reference set is also scored against a bank holding its own patches, which makes it look more ordinary than it is (a lot more so with `--k_neighbors 1`). Overlapping tiles can vouch for each other the same way, since an object in the overlap is in both tiles. `prep_data.py` only knows JWST i2d files, MIRI `*sci.fits` and `*wht.fits` pairs, and the COSMOS-Web NIRCam naming.

## Tests

```
python test_helpers.py
```

It runs without a GPU, and without torch or FAISS installed.
