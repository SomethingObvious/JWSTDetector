# JWSTDetector

[![The viewer on a tile of Webb's First Deep Field ranked 11th, with patch-distance contours around a lensed galaxy at redshift 6.27, catalogue markers on every known source and the cursor readout in RA, Dec and MJy/sr](.github/readme/viewer.jpg)](https://somethingobvious.github.io/JWSTDetector/)

JWSTDetector looks for strange things in James Webb Space Telescope (JWST) images without any labels. It cuts FITS mosaics into tiles, embeds them with a DINOv3 backbone, ranks every tile by how far its patches sit from ordinary sky, cross-matches the most anomalous spot in each with SIMBAD, NED and Gaia DR3, and opens the ranking in a web viewer that learns from your verdicts. It reads any FITS image with a WCS, from JWST, HST, Euclid or anything else.

The [demo](https://somethingobvious.github.io/JWSTDetector/) is a real run on Webb's First Deep Field, the SMACS 0723 cluster in NIRCam F444W, F356W and F277W (program 2736 on [MAST](https://mast.stsci.edu/)), cut into 288 tiles. The screenshot is its 11th tile, where NED lists the lensed galaxy beside the peak at a redshift of 6.27.

## What It Finds

![Four tiles kNN ranks highly on SMACS 0723, with the most anomalous patch of each boxed: lensed arcs, the cluster core, a lensed galaxy at redshift 6.27 and the rainbow of a diffraction spike](.github/readme/kinds.jpg)

These are ranks 2, 3, 11 and 12, as prep cut them. The top of kNN's ranking is lensed arcs, the intracluster light of the core and the diffraction spikes of bright stars, which come out as rainbows because the three filters' PSFs differ in width. The subspace detector finds different things, and the two rankings barely agree (Spearman 0.20, with 3 tiles in both top 20s).

230 of the 288 tiles have nothing in any of the three catalogues within 1.5 arcseconds of their most anomalous patch. On a field this deep that mostly means galaxies too faint for them, so it's a filter to combine with verdicts rather than a list of discoveries.

## Benchmark

![Share of planted points, arcs, rings and streaks found in the top 5% against AB magnitude, for kNN, the subspace detector, kNN on raw pixels and an isolation forest](docs/completeness.png)

`jwstdetector bench` plants points, arcs, rings and streaks of known AB magnitude in the FITS cutouts, in the mosaic's own units and blurred by STPSF's model of each filter's PSF, and renders them back exactly as prep would have. On SMACS 0723, kNN finds half the planted arcs down to AB 27.2 and half the rings to 26.2. The same kNN search on raw pixels does about as well on rings and streaks but finds far fewer arcs, and an isolation forest on hand-built features stays at chance.

`jwstdetector recall` checks the 85 known lensed images in the field. Every detector ranks them above sky at the same distance from the cluster core more often than chance, kNN at 64 where 50 is chance, though no detector clearly beats the others there. [The benchmark page](docs/benchmark.md) has the design, the numbers, the backbones compared and the caveats, including why kNN finds faint arcs more often than bright ones.

## How It Works

The default detector scores each patch by its cosine distance to the nearest patch of a reference set, after [AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025), and the other by its residual off a PCA subspace, after [SubspaceAD](https://arxiv.org/abs/2602.23013) (Lendering et al., 2026). With no labels, the reference comes from the tiles themselves in two passes, and a tile is never compared with one it overlaps.

Two systematics took over the top of the SMACS ranking before they were fixed at the source. Objects cut off by a tile's edge put 45% of the top 20 peaks on the outermost ring of patches, so each patch now counts only for the tile whose centre it is nearest. Sky covered by one exposure in place of three took the top four places, so prep now weights the flux by depth before rendering, as for a detection image. [The method](docs/method.md) has the equations, the measurements behind both fixes and what each part assumes.

`--reference_roots` scores against other fields instead and gives each tile a calibrated p-value and false discovery rate, though anything that differs between the fields ranks first, so they have to match closely.

Each tile also gets a descriptor built from its anomalous patches only. Sorting by Learned in the viewer spreads your verdicts over those descriptors, so marking one diffraction spike as an artifact sinks the tiles that look like it.

## Install

```
python -m venv .venv
.venv\Scripts\activate        # or source .venv/bin/activate
pip install -e .
```

It needs Python 3.11 or newer. A CUDA GPU makes scoring a lot faster, but `--device cpu` works too, and if pip picks the CPU build of torch, install the CUDA one from pytorch.org first. The viewer comes built, and only changing it needs Node.

The default backbone, `dinov3-vitl16-pretrain-sat493m`, was trained on satellite imagery, which is a lot closer to a mosaic tile than web photos are. Its weights are gated, so accept the licence on its Hugging Face page and run `hf auth login` once. `dinov2_vitl14` isn't gated.

## Usage

```
jwstdetector prep --files hst_10092_acs_wfc_f814w_drz.fits --out datasets/hst \
  --tile_size 320 --stride 160 --wcs_in_name --save_cutouts
jwstdetector run --data_root datasets/hst --out_dir results --tag hst
jwstdetector match results_hst datasets/hst
jwstdetector view results_hst datasets/hst
```

`jwstdetector candidates` writes the ranking as a FITS or VOTable table with DS9 regions, and `jwstdetector bench` plants sources of known magnitude. [Usage](docs/usage.md) covers every command, the JWST colour presets and the thresholds worth knowing.

## What It Won't Do

Scores are only relative to the reference, so an artifact most tiles share counts as normal, and a rare thing can hide another like it if that one ends up in the reference. A lone point source looks like one more star and is never found. Tiles that aren't fully covered are dropped, which leaves the sky within about a tile of the footprint edge unexamined, so a small mosaic needs smaller tiles or a lower `--min_wht_frac`.

## Tests

```
pip install -e .[dev]
pytest
cd web && npm ci && npm test
```

Neither needs a GPU or downloaded weights, and CI runs both on every push to main.

## Licence

It's under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). You can use it, change it and share it, forks included, for anything noncommercial, like research or study. Commercial use isn't allowed.
