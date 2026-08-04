# JWSTDetector

JWSTDetector is unsupervised anomaly detection for James Webb Space Telescope (JWST) images, so it looks for strange things without any labels. It cuts FITS mosaics into PNG tiles, embeds every tile with a DINO backbone, and ranks the tiles by how far their patches sit from a reference set of ordinary ones, so a person can look through the top of the list for artifacts, odd morphologies or whatever else the reference doesn't account for. The detector is adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025).

## Install

```
pip install -e .
pip install -e '.[viewer]'    # only for the Qt viewer window
```

It needs Python 3.11 or newer. A CUDA GPU makes scoring a lot faster, but `--device cpu` works too.

The default backbone, `dinov3-vitl16-pretrain-sat493m`, was trained on satellite imagery, which is a lot closer to a mosaic tile than web photos are. DINOv3 weights are gated, though, so accept the licence on its Hugging Face page and run `hf auth login` once. `dinov2_vitl14` isn't gated, but it comes through torch.hub, which needs the network the first time to fetch the DINOv2 repo.

## Usage

```
python prep_data.py --sensor miri --indir rawfits --out datasets/miri --pattern "*sci.fits" \
  --tile_size 672 --upscale 1 --stride 644 --min_wht_frac 0.5 --wcs_in_name

python run_query_bootstrap.py --data_root datasets/miri --model_name dinov3-vitl16-pretrain-sat493m \
  --resolution 672 --batch_size 16 --k_neighbors 20 --rotation --out_dir results --tag miri

python summarize_results.py --results_root results_miri --outdir results_summary
python viewer.py --maps_dir results_miri/pass2/anomaly_maps/seed=0 --png_dir datasets/miri/query \
  --csv_path results_summary/results_miri/samples_sorted_by_anomaly_score.csv
```

`prep_data.py` drops a tile when too little of it is covered (`--min_wht_frac`, which is 0 and so off by default) or when too few of its pixels sit `--source_sigma` above the mosaic's own noise (`--min_source_frac`). It stretches every tile of a mosaic the same way so the tiles stay comparable. `--dry-run` shows what the thresholds would keep without writing anything. With `--wcs_in_name` each file name carries the RA and Dec of the tile centre, which is how you find a candidate again. NIRCam mode (`--sensor nircam`) makes colour tiles from three filter mosaics, and the COSMOS-Web weight maps come gzipped, so run `python unzipper.py <folder>` on them first.

`run_query_bootstrap.py` scores every tile against a random reference drawn from the tiles themselves, then rebuilds the reference from the lowest scores and scores again. The final ranking is written to `measurements_final.csv`, with a patch-distance map per tile under `pass2/anomaly_maps/`. `--score_mode peak_minus_median` suits one small bright oddity on otherwise ordinary sky better than the default `top1p`. `run_anomalydino.py` is for when you already have a `train/` folder of normal tiles. `instructions.txt` has fuller example commands.

## What It Won't Do

Scores are only relative to the reference, so if most tiles share an artifact, that artifact counts as normal, and `--init_ref_*` and `--bootstrap_keep_*` may need tuning to the data. There is no ground truth or metric here, just a ranking to read. `prep_data.py` only knows MIRI `*sci.fits` and `*wht.fits` pairs and the COSMOS-Web NIRCam naming.

The shared stretch has a cost too, as one very bright object can skew the percentiles for a whole mosaic. Each mosaic logs its `sky`, `sigma` and `stretch`, so check that line and lower `--p_hi` if the range looks off.

## Tests

```
python test_helpers.py
python test_detection.py
```

Neither needs a GPU or any downloaded weights.
