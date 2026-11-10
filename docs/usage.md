# Usage

Each step is a subcommand of `jwst-peculiar`, and each takes `--help`.

```
jwst-peculiar prep        cut FITS mosaics into tiles
jwst-peculiar levels      measure one stretch per filter for a survey
jwst-peculiar run         score every tile, in two passes
jwst-peculiar score       score tiles against tiles known to be normal
jwst-peculiar match       cross-match each tile's peak with SIMBAD, NED and Gaia DR3
jwst-peculiar candidates  write the ranking as a FITS or VOTable table and DS9 regions
jwst-peculiar view        open the ranking in the web viewer, or export it as a site
jwst-peculiar bench       plant sources of known magnitude and measure what is found
jwst-peculiar recall      measure how a run ranks objects known to be in the field
```

## Cutting Tiles

### Any FITS Image

The default sensor reads anything with a celestial WCS, one grey band per file. The image can be in the primary HDU or an extension (`--sci_ext`), and a weight map is picked up from a `WHT` extension or a file beside it.

```
# An HST drizzled image, counts per second with PHOTFNU in the header.
jwst-peculiar prep --files hst_10092_acs_wfc_f814w_drz.fits --out datasets/hst \
  --tile_size 320 --stride 160 --wcs_in_name --save_cutouts

# A folder of mosaics with weight maps called <name>_wht.fits.
jwst-peculiar prep --indir rawfits/euclid --pattern "*.fits" --wht_pattern "{stem}_wht.fits" \
  --out datasets/euclid --tile_size 320 --wcs_in_name --save_cutouts

# Colour from three files of one field, red first. Without a WCS they need the same grid.
jwst-peculiar prep --rgb f160w.fits f125w.fits f814w.fits --out datasets/hst_rgb \
  --tile_size 320 --wcs_in_name --save_cutouts
```

A mosaic without a weight map gets a warning, as tiles on the edge of its coverage can't be dropped and will crowd the top of the ranking. `--min_wht_frac 0` then keeps everything, and it reads no weight map at all, so the noise isn't evened out either.

With a weight map, prep also evens out the noise before rendering, scaling the flux above the sky by the square root of each pixel's weight over the mosaic's median, capped at 1. Otherwise sky covered by fewer exposures renders noisier and fills the top of the ranking. `--no-noise_eq` turns it off, and [the method](method.md#even-noise) has the details.

Prep records what the headers say about each band in `mosaics/<id>.json`: the filter, the telescope, the pivot wavelength, how many janskys one unit of pixel value is (from `BUNIT`, `PHOTFNU` or an AB zero point such as `MAGZERO`), and the diffraction-limited PSF width for JWST, HST, Euclid and Roman. The benchmark needs those, and `jwst-peculiar bench --jy_per_unit` and `--psf_fwhm` cover headers that don't give them.

### JWST Presets

`--sensor miri` takes JWST i2d files or `*sci.fits` and `*wht.fits` pairs. `--sensor nircam` makes colour tiles from three NIRCam filters of one field, reprojecting them onto one grid and keeping a tile only where all three have data. This is how the demo was cut.

```
jwst-peculiar prep --sensor nircam --indir rawfits/smacs0723 --out datasets/smacs_nircam \
  --pattern "jw02736-o001_t001_nircam_clear-{filter}_i2d.fits" --tiles smacs0723 \
  --rgb_filters f444w f356w f277w --tile_size 320 --upscale 1 --stride 160 \
  --wcs_in_name --save_cutouts
```

For a lot of MIRI mosaics, smaller tiles work better. For a few, bigger ones do.

```
jwst-peculiar prep --sensor miri --indir rawfits --out datasets/miri --pattern "*i2d.fits" \
  --tile_size 448 --upscale 1 --stride 420 --min_wht_frac 1 \
  --source_sigma 5 --min_source_frac 2e-4 --stretch_scope mosaic --wcs_in_name --save_cutouts
```

### One Stretch for a Survey

Each mosaic's colour stretch comes from its own 99.8th percentile, which is fine for one field but not for a survey. A bright star in one visit pushes that percentile up and the whole visit comes out darker than the rest, and in COSMOS-Web that one visit made most of the strangest tiles the search had to beat ([search.md](search.md#the-result)). `jwst-peculiar levels` takes the median range of each filter over datasets prep already cut, and `--levels` gives every mosaic that range above its own sky.

```
jwst-peculiar levels datasets/cosmosweb/* --out levels.json
jwst-peculiar prep --sensor nircam ... --levels levels.json
```

The median needs a fair number of visits to settle, so the usual way is to cut the first batch of a survey with its own stretch, write the levels from them, and cut everything again with the file. It's for colour mosaics only. Visits of quite different depth (like a deep field and a shallow one in the same programme) probably want a file each, since the noise sits at a different place in the same range.

### Choosing the Thresholds

A tile is kept when all of it is covered (`--min_wht_frac`) and enough of it sits `--source_sigma` above the noise (`--min_source_frac`). Both are measured against each mosaic's own sky level and noise, so the same values work across units and depths. `--dry-run` cuts and filters without writing anything, which is the quick way to see what a threshold keeps.

`--save_cutouts` also writes each tile's float pixels and WCS as a FITS cutout. The viewer measures from those and the benchmark plants its sources in them, so it's worth always passing.

## Scoring

```
jwst-peculiar run --data_root datasets/smacs_nircam --out_dir results_smacs --tag knn
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
--places 3                         a patch needs close matches in 3 separate places
--detector iforest                 the isolation forest baseline, no backbone
--bank_precision fp32              an fp32 memory bank, slower with the same ranking
```

`jwst-peculiar run --help` lists everything else, like the reference sizes for each pass. A backbone too big for the GPU, like `dinov3-vit7b16-pretrain-sat493m` on 8 GB, is streamed from disk a layer at a time, and `args.yaml` records the precision and whether it was streamed.

### Another Field as Reference

```
jwst-peculiar run --data_root datasets/smacs_nircam --out_dir results_cross \
  --reference_roots datasets/abell370_nircam datasets/macs0416_nircam datasets/ceers1_nircam
```

This scores every tile against two or more other fields in place of the bootstrap, and each of those fields against the rest to calibrate a p-value and q-value per tile. They're in `significance.csv`, and the viewer shows them and filters by q. The fields need the same filters, stretch and a similar depth, and their mosaics different `--tiles` names. [The method](method.md#another-field-as-reference) has what the p-values assume.

## Catalogues

```
jwst-peculiar match results_smacs_knn datasets/smacs_nircam
jwst-peculiar match results_smacs_knn datasets/smacs_nircam --catalogue jades.fits --radius 1
```

Every source SIMBAD, NED and Gaia DR3 know over each mosaic is fetched once over TAP, in cones 3 arcminutes in radius, and kept in `RUN/catalogues/`. Each tile then gets the sky position of its most anomalous patch and the nearest source in each catalogue, and it counts as uncatalogued when none of them has anything within `--radius` arcseconds. `--catalogue` adds a local FITS, VOTable or CSV catalogue, with its RA and Dec columns found by name.

When a service doesn't answer, its other mirrors are tried, and a later `--refresh` that fails keeps the copy it already had. The viewer shows the matches and marks every catalogued source on the tile.

## Candidate Tables

```
jwst-peculiar candidates results_smacs_knn datasets/smacs_nircam -o candidates.fits
jwst-peculiar candidates results_smacs_knn datasets/smacs_nircam -o new.vot --top 50 --uncatalogued
```

The format follows the extension (`.fits`, `.vot`, `.csv` or `.ecsv`). Each row is a tile, with the peak's RA and Dec, both scores, the nearest source in every checked catalogue and its separation, and your verdict, star and note. A DS9 region file is written beside it, with a circle at each peak.

## The Viewer

```
jwst-peculiar view results_smacs_knn datasets/smacs_nircam
jwst-peculiar view results_smacs_knn datasets/smacs_nircam --export site/ --cutouts 24 \
  --site_url https://example.org/demo/
```

The first serves the viewer on localhost and saves verdicts to `verdicts.csv` in the run folder. The second writes a static site any host can serve, where verdicts stay in each visitor's browser and can be exported as CSV. The verdicts and notes already in the run folder's `verdicts.csv` go into the site, so look through them before publishing it. `--cutouts` keeps the FITS cutouts for the top tiles only, as they're about 1.2 MB each. The viewer calls a run by its `--tag`, and `--name "Webb's First Deep Field"` gives it a better one for the page title and link previews. Browsers still file their verdicts under the tag, so a new name doesn't lose them.

`j` and `k` step through the ranking, `1` to `4` mark a tile as interesting, an artifact, a known object or not anomalous, `s` stars it, `n` adds a note, `c` toggles the catalogue markers and `?` lists every key.

Sorting the index by Learned ranks tiles by their anomaly percentile times what your verdicts say about tiles like them, which [the method](method.md#learning-from-verdicts) explains. With no verdicts it is the anomaly ranking unchanged.

## Benchmark

```
jwst-peculiar bench --data_root datasets/smacs_nircam --out_dir bench --mags 22 23 24 25 26 27 28 --seeds 5
```

It needs a dataset cut with `--save_cutouts`. `--detectors` picks from knn, subspace, pixels and iforest, and `--psf` from webb, gaussian and auto, which takes STPSF's model for JWST when it's installed (`pip install -e .[webb]`). Arguments after `--` go to every `jwst-peculiar run`, and [the benchmark page](benchmark.md) has what it measures and the SMACS 0723 results.

## Known Objects

```
jwst-peculiar recall results_smacs datasets/smacs_nircam --objects lensed.csv \
  --centre 110.8270 -73.4541
```

The objects come from a CSV with `ra` and `dec` in degrees and an optional `id` like 1.2, where the part before the dot groups the images of one lensed galaxy. It reports where each object's patch ranks among every patch the run scored, and `--centre` also compares each one only with patches at the same distance from that point, which keeps a cluster core's general oddness from counting. `recall.csv` has every object and `recall.json` the summary.

## Tiles Known to Be Normal

`jwst-peculiar score --data_root DIR` is for when you already have a `train/` folder of normal tiles to use as the reference instead of the bootstrap. It scores the tiles in `test/` beside it.
