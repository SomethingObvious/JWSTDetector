# Benchmark

There is no labelled set of JWST anomalies to score against, so `jwstdetector bench` plants sources of known brightness in real tiles and measures where they land in the ranking. Everything here is from the 288 SMACS 0723 tiles, the same ones as the demo.

## What Is Planted

Each run picks 12 tiles and adds one source to each, three points, three arcs, three rings and three streaks. The arcs, rings and streaks are half a pixel wide before the PSF, like a thin lensed arc or a satellite trail, and a ring or arc has a radius of 6 to 15% of the tile. A source gets a flat, red or blue spectrum ($f_\nu \propto \lambda^{\beta}$ with $\beta$ of 0, 2 or -2) and a total AB magnitude, which is its flux inside the tile at the filters' geometric mean wavelength.

The source goes into the tile's float FITS cutout, in the mosaic's own units (MJy/sr here, from `BUNIT`), blurred by a Gaussian of each filter's diffraction-limited width. The tile is then rendered back to its PNG with the exact stretch and noise weighting prep used, which reproduces every unplanted tile bit for bit. A seed fixes the tiles, shapes, positions and colours for every magnitude, so the curves differ by brightness alone. There are 5 seeds, so each point below is 15 sources of a kind, with a 95% Wilson interval.

A source counts as found when its tile ranks in the top 5%, the top 14 of 288. Real anomalies in the field compete with the planted ones, which is why it isn't first place, and the dashed line is the 5% a tile would reach by chance.

## Results

![Found in the top 5% against AB magnitude for points, arcs, rings and streaks, for kNN, the subspace detector and kNN on raw pixels](completeness.png)

| | kNN | Subspace | kNN on pixels |
|---|---|---|---|
| Arc, magnitude found half the time | 27.6 | 24.2 | never |
| Ring, magnitude found half the time | 26.4 | 25.5 | never |
| All kinds found at AB 22 | 30 of 60 | 37 of 60 | 11 of 60 |
| All kinds found at AB 26 | 26 of 60 | 11 of 60 | 5 of 60 |
| All kinds found at AB 28 | 11 of 60 | 6 of 60 | 5 of 60 |

kNN on DINOv3 features keeps finding arcs 3.4 magnitudes fainter than the subspace detector does and rings 0.9 fainter, though the subspace detector finds more of the brightest ones. The pixel baseline is the same kNN search on raw 16 by 16 patches, and it finds little beyond the brightest arcs, so nearly all of what kNN finds comes from the backbone rather than the search.

Neither detector finds point sources, at any brightness. A planted point is a blurred dot like the hundreds of faint galaxies and stars in every tile, and its rate stays flat at whatever share of the chosen tiles was in the top 5% before anything was planted. kNN finds about half the streaks at AB 22 and a quarter from AB 25 on, while the subspace detector has lost them by AB 24.

## Why kNN's Curves Are Flat When Bright

A brighter source ought to be found more often, but kNN finds bright arcs no more often than faint ones, and the cause is the bootstrap rather than the features. Each run plants three sources of each kind at the same magnitude, and when one of them ends up in pass 2's reference, the others match it and look ordinary. Over all the runs, a bright arc, ring or streak (AB 22 to 24) was found 68% of the time when no planted twin of it was in pass 2's reference and 23% of the time when one was, over 113 and 22 sources.

Planting one source of each kind per run instead, over 10 seeds, kNN found every ring at AB 22, with a median rank of 1, and 8 of the 10 arcs. A real field can hold several of a rare thing too, like the lensed arcs around a cluster, so the same thing happens to real objects whenever the reference comes from the field itself.

## What It Doesn't Show

- The PSF is a Gaussian, with no Airy rings or diffraction spikes, so a planted point is a little easier to tell from a star than a real one would be.
- Only the planted tile is changed. The overlapping tiles that also hold that bit of sky don't see the source, and part of a big ring can fall outside the area its tile scores.
- The stretch is fixed per mosaic, so at AB 22 the core of a ring or arc saturates the rendered tile.
- These are recalls for sources added to SMACS 0723. A field with a different depth, crowding or mix of real anomalies will give different curves, and `jwstdetector bench` on that field is the way to find out.

## Running It

```
jwstdetector bench --data_root datasets/smacs_nircam --out_dir bench --mags 22 23 24 25 26 27 28 --seeds 5
```

It writes `injections.csv` with every planted source's rank, `summary.csv` with recall and its interval per detector, kind and magnitude, and `completeness.png`. The 105 runs here took 47 minutes on a laptop RTX 4070, as each embeds only the tiles that changed.
