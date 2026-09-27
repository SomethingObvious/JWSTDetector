# JWSTDetector

JWSTDetector looks for strange things in JWST images without any labels. It cuts FITS mosaics into PNG tiles, embeds every tile with a DINO backbone, and ranks the tiles by how far their patches sit from a reference set of ordinary ones, so a person can look through the top of the list for artifacts, odd morphologies or whatever else the reference doesn't account for. The detector is adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025).

## Install

```
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate
pip install -e .              # add '.[viewer]' for a Qt viewer window
```

It needs Python 3.11 or newer. A CUDA GPU makes scoring a lot faster, but `--device cpu` works too. Install the CUDA build of torch from pytorch.org first if pip picks the CPU one. The scripts share the `jwstdetector` package that sits beside them, so they also run from the repo root without the install, as long as the dependencies are there.

The default backbone, `dinov3-vitl16-pretrain-sat493m`, was trained on satellite imagery, which is a lot closer to a mosaic tile than web photos are. DINOv3 weights are gated, though, so accept the licence on its Hugging Face page and run `hf auth login` once. `dinov2_vitl14` isn't gated, and any Hub repo id or local checkpoint folder works as well.

## Usage

```
python prep_data.py --sensor miri --indir rawfits --out datasets/miri --pattern "*i2d.fits" \
  --tile_size 672 --upscale 1 --stride 644 --wcs_in_name

python run_query_bootstrap.py --data_root datasets/miri --model_name dinov3-vitl16-pretrain-sat493m \
  --resolution 672 --batch_size 16 --k_neighbors 20 --rotation --out_dir results --tag miri

python summarize_results.py --results_root results_miri --outdir results_summary
python viewer.py --maps_dir results_miri/pass2/anomaly_maps/seed=0 --png_dir datasets/miri/query \
  --csv_path results_summary/results_miri/samples_sorted_by_anomaly_score.csv
```

`prep_data.py` keeps a tile when all of it is covered (`--min_wht_frac`, 1 by default) and when enough pixels sit `--source_sigma` above the mosaic's own noise (`--min_source_frac`), and it stretches every tile of a mosaic the same way so tiles stay comparable. `--dry-run` shows what the thresholds would keep without writing anything. With `--wcs_in_name` each file name carries the RA and Dec of the tile centre, which is how you find a candidate again. NIRCam mode (`--sensor nircam`) makes colour tiles from three filter mosaics.

`run_query_bootstrap.py` scores every tile against a random reference drawn from the tiles themselves, then rebuilds the reference from the lowest scores and scores again. The final ranking is written to `measurements_final.csv`, with a patch-distance map per tile under `pass2/anomaly_maps/`. `run_anomalydino.py` is for when you already have a `train/` folder of normal tiles. `instructions.txt` has fuller example commands.

## What It Won't Do

Scores are only relative to the reference, so if most tiles share an artifact, that artifact counts as normal. There is no ground truth or metric here, just a ranking to read. Overlapping tiles can also vouch for each other, since an object in the overlap is in both tiles and one of them may be in the reference. `prep_data.py` only knows JWST i2d files, MIRI `*sci.fits` and `*wht.fits` pairs, and the COSMOS-Web NIRCam naming.

Tiles on the edge of the coverage beat everything else to the top of the ranking, which is why a tile has to be covered all the way by default. On the SMACS 0723 MIRI mosaic a tile missing only half a percent of its pixels still came second of 53 at `--min_wht_frac 0.95`, and all ten of its worst patches touched the hole. It does mean that sky within about a tile of the footprint edge never gets looked at, which was around 40% of that small mosaic with 224 px crops and all of it with 672 px ones. So a small mosaic needs smaller tiles, or a lower `--min_wht_frac` and edges near the top of the ranking. MIRI mosaics without a weight map need `--min_wht_frac 0`.

## Tests

```
python test_helpers.py
python test_detection.py
```

Neither needs a GPU or any downloaded weights.
