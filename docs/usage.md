# Usage

Everything runs through one command, `jwstdetector`, with a subcommand for each step. Each one takes `--help`.

```
jwstdetector prep        cut FITS mosaics into tiles
jwstdetector run         score every tile, in two passes
jwstdetector match       cross-match each tile's peak with SIMBAD, NED and Gaia DR3
jwstdetector candidates  write the ranking as a FITS or VOTable table and DS9 regions
jwstdetector view        open the ranking in the web viewer, or export it as a site
jwstdetector bench       plant sources of known magnitude and measure what is found
```

## Cutting Tiles

### Any FITS Image

The default sensor reads anything with a celestial WCS, one grey band per file. The image can be in the primary HDU or an extension (`--sci_ext`), and a weight map is picked up from a `WHT` extension or a file beside it.

```
# An HST drizzled image, counts per second with PHOTFNU in the header.
jwstdetector prep --files hst_10092_acs_wfc_f814w_drz.fits --out datasets/hst \
  --tile_size 320 --stride 160 --wcs_in_name --save_cutouts

# A folder of mosaics with weight maps called <name>_wht.fits.
jwstdetector prep --indir rawfits/euclid --pattern "*.fits" --wht_pattern "{stem}_wht.fits" \
  --out datasets/euclid --tile_size 320 --wcs_in_name --save_cutouts

# Colour from three files of one field, red first. Without a WCS they need the same grid.
jwstdetector prep --rgb f160w.fits f125w.fits f814w.fits --out datasets/hst_rgb \
  --tile_size 320 --wcs_in_name --save_cutouts
```

A mosaic without a weight map gets a warning, as tiles on the edge of its coverage can't be dropped and will crowd the top of the ranking. `--min_wht_frac 0` then keeps everything.

With a weight map, prep also evens out the noise before rendering. It scales the flux above the sky by the square root of each pixel's weight over the mosaic's median, capped at 1, because sky covered by fewer exposures otherwise renders noisier and fills the top of the ranking. [The method](method.md#even-noise) has the details, and `--no-noise_eq` turns it off.

Prep records what the headers say about each band in `mosaics/<id>.json`: the filter, the telescope, the pivot wavelength, how many janskys one unit of pixel value is (from `BUNIT`, `PHOTFNU` or an AB zero point such as `MAGZERO`), and the diffraction-limited PSF width for JWST, HST, Euclid, Roman, Spitzer and WISE. The benchmark needs those, and `jwstdetector bench --jy_per_unit` and `--psf_fwhm` cover headers that don't give them.

### JWST Presets

`--sensor miri` takes JWST i2d files or `*sci.fits` and `*wht.fits` pairs. `--sensor nircam` makes colour tiles from three NIRCam filters of one field, reprojecting them onto one grid and keeping a tile only where all three have data. This is how the demo was cut.

```
jwstdetector prep --sensor nircam --indir rawfits/smacs0723 --out datasets/smacs_nircam \
  --pattern "jw02736-o001_t001_nircam_clear-{filter}_i2d.fits" --tiles smacs0723 \
  --rgb_filters f444w f356w f277w --tile_size 320 --upscale 1 --stride 160 \
  --wcs_in_name --save_cutouts
```

For a lot of MIRI mosaics, smaller tiles work better. For a few, bigger ones do.

```
jwstdetector prep --sensor miri --indir rawfits --out datasets/miri --pattern "*i2d.fits" \
  --tile_size 448 --upscale 1 --stride 420 --min_wht_frac 1 \
  --source_sigma 5 --min_source_frac 2e-4 --stretch_scope mosaic --wcs_in_name --save_cutouts
```

### Choosing the Thresholds

A tile is kept when all of it is covered (`--min_wht_frac`) and enough of it sits `--source_sigma` above the noise (`--min_source_frac`). Both are measured against each mosaic's own sky level and noise, so the same values work across units and depths. `--dry-run` cuts and filters without writing anything, which is the quick way to see what a threshold keeps.

`--save_cutouts` also writes each tile's float pixels and WCS as a FITS cutout. The viewer measures from those, and the benchmark plants its sources in them, so it is worth always passing.

## Scoring

```
jwstdetector run --data_root datasets/smacs_nircam --out_dir results_smacs --tag knn
```

This scores every tile against a random reference drawn from the tiles themselves, then rebuilds the reference from the lowest scores and scores again. A tile is never compared with a reference tile it overlaps. The ranking is in `measurements_final.csv`, with a patch-distance map per tile under `pass2/anomaly_maps/` and each tile's descriptor in `pass2/descriptors.npz`.

Embeddings are cached under `DATA_ROOT/.features`, keyed by each tile's bytes and the backbone settings, so the second pass and any later run with the same backbone only embed what changed. `--feature_cache none` turns that off.

A few options matter more than the rest.

```
--detector subspace --layers mid   SubspaceAD instead of kNN
--model_name dinov2_vitl14         an ungated backbone
--resolution 672                   more patches per tile, about twice as slow
--k_neighbors 20                   smoother kNN scores
--rotation                         every rotation and flip of the reference tiles, slow
--score_mode peak_minus_median     one small oddity rather than a tile odd all over
```

`jwstdetector run --help` lists everything else, like the reference sizes for each pass.

## Catalogues

```
jwstdetector match results_smacs_knn datasets/smacs_nircam
jwstdetector match results_smacs_knn datasets/smacs_nircam --catalogue jades.fits --radius 1
```

Every source SIMBAD, NED and Gaia DR3 know over each mosaic is fetched once over TAP, in cones 3 arcminutes in radius, and kept in `RUN/catalogues/`. Each tile then gets the sky position of its most anomalous patch and the nearest source in each catalogue, and it counts as uncatalogued when none of them has anything within `--radius` arcseconds. `--catalogue` adds a local FITS, VOTable or CSV catalogue, with its RA and Dec columns found by name.

When a service doesn't answer, its other mirrors are tried, and a later `--refresh` that fails keeps the copy it already had. The viewer shows the matches and marks every catalogued source on the tile.

## Candidate Tables

```
jwstdetector candidates results_smacs_knn datasets/smacs_nircam -o candidates.fits
jwstdetector candidates results_smacs_knn datasets/smacs_nircam -o new.vot --top 50 --uncatalogued
```

The format follows the extension (`.fits`, `.vot`, `.csv` or `.ecsv`). Each row is a tile, with the peak's RA and Dec, both scores, the nearest source in every checked catalogue and its separation, and your verdict, star and note. A DS9 region file is written beside it, with a circle at each peak.

## The Viewer

```
jwstdetector view results_smacs_knn datasets/smacs_nircam
jwstdetector view results_smacs_knn datasets/smacs_nircam --export site/ --cutouts 24 \
  --site_url https://example.org/demo/
```

The first serves the viewer on localhost and saves verdicts to `verdicts.csv` in the run folder. The second writes a static site any host can serve, where verdicts stay in each visitor's browser and can be exported as CSV. `--cutouts` keeps the FITS cutouts for the top tiles only, as they're about 1.2 MB each.

`j` and `k` step through the ranking, `1` to `4` mark a tile as interesting, an artifact, a known object or not anomalous, `s` stars it, `n` adds a note, `c` toggles the catalogue markers and `?` lists every key.

Sorting the index by Learned ranks tiles by their anomaly percentile times what your verdicts say about tiles like them, which [the method](method.md#learning-from-verdicts) explains. With no verdicts it is the anomaly ranking unchanged.

## Benchmark

```
jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench --mags 22 23 24 25 26 27 28 --seeds 5
```

It needs a dataset cut with `--save_cutouts`. Arguments after `--` go to every `jwstdetector run`, and [the benchmark page](benchmark.md) has what it measures and the SMACS 0723 results.

## Without a Query Set

`python -m jwstdetector.reference` is for when you already have a `train/` folder of normal tiles to use as the reference instead of the bootstrap.
