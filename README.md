# jwst-peculiar

[![The viewer on a tile of Webb's First Deep Field ranked 11th, with patch-distance contours around a lensed galaxy at redshift 6.27, catalogue markers on every known source and the cursor readout in RA, Dec and MJy/sr](.github/readme/viewer.jpg)](https://somethingobvious.github.io/jwst-peculiar/)

jwst-peculiar looks for strange things in James Webb Space Telescope (JWST) images without any labels. It cuts FITS mosaics into tiles, embeds them with a DINOv3 backbone, ranks every tile by how far its patches sit from ordinary sky, cross-matches the oddest spot in each with SIMBAD, NED and Gaia DR3, and opens the ranking in a web viewer that learns from your verdicts. It reads any FITS image with a WCS, so HST and Euclid work too.

The [demo](https://somethingobvious.github.io/jwst-peculiar/) is a real run on Webb's First Deep Field, the SMACS 0723 cluster in NIRCam F444W, F356W and F277W (program 2736 on [MAST](https://mast.stsci.edu/)), cut into 288 tiles.

## What It Finds

![Four tiles kNN ranks highly on SMACS 0723, with an anomalous patch of each boxed: lensed arcs, the cluster core, a lensed galaxy at redshift 6.27 and the rainbow of a diffraction spike](.github/readme/kinds.jpg)

These are ranks 2, 3, 11 and 12. The top of the ranking is mostly lensed arcs, the light of the cluster core and diffraction spikes, which come out as rainbows because the three filters' PSFs differ in width. 234 of the 288 tiles have nothing in any catalogue within 1.5 arcseconds of their oddest patch, but on a field this deep that mostly means galaxies too faint for them, so it's a filter to use with verdicts rather than a list of discoveries.

## How Well It Works

![Share of planted points, arcs, rings and streaks found in the top 5% against AB magnitude, for kNN, the subspace detector, kNN on raw pixels and an isolation forest](docs/completeness.png)

`jwst-peculiar bench` plants sources of known AB magnitude in the FITS cutouts, blurred by STPSF's model of each filter's PSF. On SMACS 0723, kNN finds half the planted rings down to AB 26.2 and 9 of 15 arcs at AB 26 and 27, though fewer bright arcs, for a reason [the benchmark page](docs/benchmark.md#why-knn-finds-faint-arcs-more-often-than-bright-ones) explains. The same search on raw pixels finds 14 of 315 arcs, rings and streaks to kNN's 133. A LoRA fine-tune finds 1,475 planted sources to the default's 1,359 on 10 PRIMER-UDS visits it never saw (p = 0.008), and [the model page](docs/model.md) has where it doesn't help.

Two searches were written down and pushed before any of their tiles were scored, and both came back empty. COSMOS-Web tested 1,345 tiles against 39,298 calibration tiles, and an injection check found that one badly rendered visit had capped what it could see. MINERVA tested 514 tiles against 3,838, which turned out too few for any one tile to pass alone, so that empty result says little. [search.md](docs/search.md) and [minerva.md](docs/minerva.md) have the plans, what changed after scoring and the results.

## How It Works

The default detector scores each patch by its cosine distance to the nearest patch of a reference set, after [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025). With no labels, the reference comes from the tiles themselves in two passes, and a tile is never compared with one it overlaps. Each patch counts only for the tile whose centre it's nearest, and prep weights the flux by depth before rendering, as cut-off objects and shallow sky otherwise took over the top of the ranking. [The method](docs/method.md) has the measurements.

## Install

```
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate
pip install -e .
```

It needs Python 3.11 or newer. A CUDA GPU makes scoring a lot faster, but `--device cpu` works too. The default backbone, `dinov3-vitl16-pretrain-sat493m`, is gated, so accept the licence on its Hugging Face page and run `hf auth login` once. `dinov2_vitl14` isn't gated.

## Usage

```
jwst-peculiar prep --files hst_10092_acs_wfc_f814w_drz.fits --out datasets/hst \
  --tile_size 320 --upscale 1 --stride 160 --wcs_in_name --save_cutouts
jwst-peculiar run --data_root datasets/hst --out_dir results --tag hst
jwst-peculiar match results_hst datasets/hst
jwst-peculiar view results_hst datasets/hst
```

[Usage](docs/usage.md) covers every command, the JWST colour presets and the thresholds worth knowing.

## What It Won't Do

Scores are only relative to the reference, so an artifact most tiles share counts as normal, and a rare thing can hide another like it. A lone point source looks like one more star and isn't found. Tiles that aren't fully covered are dropped, which leaves the sky within about a tile of the footprint's edge unexamined.

## Tests

```
pip install -e .[dev]
pytest
cd web && npm ci && npm test
```

Neither needs a GPU or downloaded weights, and CI runs both on every push to main.

## Licence

It's under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You can use it, change it and share it for anything noncommercial like research or study. Commercial use isn't allowed.
