# JWSTDetector

JWSTDetector is unsupervised anomaly detection for James Webb Space Telescope (JWST) images, so it looks for strange things without any labels. It cuts FITS mosaics into tiles, embeds every tile with a DINOv3 backbone, ranks the tiles by how far their patches sit from ordinary ones, and opens the ranking in a web viewer with the real pixels, the patch distances and the sky coordinates side by side. The default detector is adapted from [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025) and the other from [SubspaceAD](https://arxiv.org/abs/2602.23013) (Lendering et al., 2026).

## Install

```
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate
pip install -e .
```

It needs Python 3.11 or newer. A CUDA GPU makes scoring a lot faster, but `--device cpu` works too, and if pip picks the CPU build of torch, install the CUDA one from pytorch.org first. The viewer comes built, and only changing it needs Node (`npm ci && npm run build` in `web/`).

The default backbone, `dinov3-vitl16-pretrain-sat493m`, was trained on satellite imagery, which is a lot closer to a mosaic tile than web photos are. Its weights are gated, so accept the licence on its Hugging Face page and run `hf auth login` once. `dinov2_vitl14` isn't gated, and any Hub repo id or local checkpoint folder works too.

## Usage

```
python prep_data.py --sensor nircam --indir rawfits/smacs0723 --out datasets/smacs \
  --pattern "jw02736-o001_t001_nircam_clear-{filter}_i2d.fits" --tiles smacs0723 \
  --rgb_filters f444w f356w f277w --tile_size 320 --upscale 1 --stride 160 \
  --wcs_in_name --save_cutouts

python run_query_bootstrap.py --data_root datasets/smacs --out_dir results --tag smacs

python viewer.py results_smacs datasets/smacs
```

`prep_data.py` keeps a tile when all of it is covered (`--min_wht_frac`) and enough of it sits `--source_sigma` above the noise (`--min_source_frac`), and `--dry-run` shows what those would keep. `--save_cutouts` also writes each tile's float pixels and WCS as a FITS cutout, which is what the viewer measures from. NIRCam mode makes colour tiles from three filters, reprojecting them onto one grid and keeping a tile only where all three have data. MIRI mode takes JWST i2d files or `*sci.fits` and `*wht.fits` pairs.

`run_query_bootstrap.py` scores every tile against a random reference drawn from the tiles themselves, then rebuilds the reference from the lowest scores and scores again. The ranking is in `measurements_final.csv`, with a patch-distance map per tile under `pass2/anomaly_maps/`. A tile is never matched against a reference tile it overlaps, using the footprints in `manifest.csv`. `run_anomalydino.py` is for when you already have a `train/` folder of normal tiles, and `instructions.txt` has fuller commands.

## The Viewer

The ranked list filters by verdict, mosaic, score or text and sorts by rank, score, change since pass 1, RA or Dec. The tile is drawn on the GPU at full float precision, with five stretches, percentile, ZScale or manual limits, and the patch distances from either pass or the change between them as patches or contours. The cursor readout gives the pixel, RA and Dec, galactic coordinates, each filter's value in its unit and the patch distance with its percentile. There's also the tile's place in the mosaic, the FITS header, and links to ESASky, Aladin, SIMBAD and NED.

`j` and `k` step, `1` to `4` mark a tile as interesting, an artifact, a known object or not anomalous, `s` stars it, `n` adds a note and `?` lists every key. Verdicts are saved to `verdicts.csv` in the run folder, and the Verdict block exports them.

`python viewer.py results_smacs datasets/smacs --export site/ --cutouts 24` writes a static site instead, which is how the demo is made. Visitors' verdicts stay in their browser, and `--cutouts` keeps the 1.2 MB cutouts for the top tiles only. Add `--site_url` with the address it'll be hosted at and it also gets a canonical link, a sitemap and a social card cut from the mosaic around the top tile.

## Detectors

`--detector knn`, the default, scores a patch by its distance to the nearest reference patch. `--detector subspace` fits a PCA to the reference patches and scores a patch by how far it sits off the subspace that holds `--subspace_variance` of them. `--layers mid` averages the middle band of the backbone's layers, which SubspaceAD found works best.

On SMACS 0723 they find different things. kNN puts the cluster core first, with its intracluster light and lensed arcs, and the subspace detector puts the two brightest stars and their diffraction spikes first. Their rankings barely agree (Spearman -0.09).

## Benchmark

`benchmark.py` plants a faint point source, arc, ring or streak in 12 random tiles of a copy of the dataset, runs the bootstrap and reports where each one ranked. Arguments after `--` go to the runner, as in `python benchmark.py --data_root datasets/smacs --out_dir bench -- --detector subspace --layers mid`.

On the 288 SMACS 0723 tiles, over 5 seeds of 12 sources at strength 0.15 (the peak a source adds on the 0 to 255 scale), the top 5% being the top 14 tiles:

| Planted | kNN median rank | kNN in the top 5% | Subspace median rank | Subspace in the top 5% |
|---|---|---|---|---|
| Ring | 6 | 8 of 15 | 76 | 0 of 15 |
| Streak | 23 | 6 of 15 | 101 | 1 of 15 |
| Arc | 48 | 1 of 15 | 104 | 2 of 15 |
| Point | 120 | 0 of 15 | 169 | 1 of 15 |
| All | 46.5 | 15 of 60 | 86.5 | 4 of 60 |

kNN put 4 of the 15 rings at rank 1 and does better than the subspace detector on every kind. Neither finds a lone point source, which just looks like one more star. A single seed of 12 moves around quite a bit, so it's worth running a few before trusting a change. Each run takes about a minute on a laptop RTX 4070.

## What It Won't Do

Scores are only relative to the reference, so if most tiles share an artifact, that artifact counts as normal. The benchmark uses planted sources, so it measures what the detector can see, not what it will find.

Tiles on the edge of the coverage beat everything else to the top of the ranking, which is why a tile has to be fully covered by default. On the SMACS 0723 MIRI mosaic a tile missing half a percent of its pixels still came second of 53, with all ten of its worst patches touching the hole. That leaves the sky within about a tile of the footprint edge unexamined, so a small mosaic needs smaller tiles or a lower `--min_wht_frac`. Mosaics without a weight map need `--min_wht_frac 0`.

## Tests

```
python test_helpers.py
python test_detection.py
cd web && npm test
```

None of them needs a GPU or downloaded weights.

## Licence

It's under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You can use it, change it and share it, forks included, for anything noncommercial, like research or study. Commercial use isn't allowed.
