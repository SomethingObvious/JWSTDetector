# Searching a Survey

This page fixes how the COSMOS-Web search is run and judged, before any of its tiles has a score. The split is in [search-split.json](search-split.json), the bright stars in [search-stars.json](search-stars.json), and `python -m jwstdetector.survey` runs every rule below. Commit e4f1424 has the first version of this page, with 46 calibration visits and none of the tile rules, and no tile had a score when either version was written.

## Why a Whole-Survey Search Can't Name a Tile

A conformal p-value can't go below $1/(n+1)$ with $n$ calibration tiles, and Benjamini-Hochberg at level $\alpha$ over $N$ tested tiles needs a p of at most $\alpha/N$ to pass a single one. So one tile can only pass when $n + 1 \ge N/\alpha$. When every tile of a survey is both tested and in the calibration for the others, $n$ and $N$ grow together, so it can't happen however many visits there are.

PRIMER shows it. Its 21 visits were scored in two folds against a bank of 7 visits each, 4,770 tiles in all with about 2,950 calibration tiles each, and nothing reaches $q \le 0.1$. Two tiles score above every one of their calibration tiles, where 1.6 are expected by chance. The first is blank sky. The second sits on the diffraction spikes of a saturated star, which also accounts for the cluster of low p-values in that visit. BH would need 17 tiles at the lowest possible p before it passed any of them, so PRIMER can say nothing in it stands out from chance, but not that any one tile does.

## The Split

COSMOS-Web (program 1727) has 148 visits with F150W, F277W and F444W at 1031 s. 60 of them, shuffled with seed 0, were split again with seed 0 into 4 search visits, 10 bank visits and 46 calibration visits, and the other 88 joined the calibration. Every visit was equally likely to be searched, so the search and calibration visits are exchangeable.

With $\alpha = 0.1$, a searched fraction $f$ of the scored tiles leaves $`n \approx (1 - f) N / f`$, and one tile can pass once $f \le 1/11$. The search is 4 of 138 scored visits, or 3%, which leaves room for a tile with two calibration tiles above it to pass, and for several tiles to pass together.

## Which Tiles Count

Each rule depends only on where a tile is, never on its score, and applies to search and calibration tiles alike.

1. Only tiles centred on a 320 px grid count, so no two of them share a pixel.
2. A tile goes if it may share sky with a tile of any other visit, meaning their centres are less than a tile's diagonal apart. Neighbouring visits overlap at their edges, and an object seen twice would otherwise sit in the calibration as its own twin.
3. A tile goes if a Gaia DR3 star brighter than G = 16 is within half a tile's diagonal plus $`1.5'' \times 10^{0.2 (19 - G)}`$, which is 6 arcseconds at G = 16 and 2.4 arcminutes at G = 8. Diffraction spikes are the commonest rare thing in this kind of sky. On PRIMER, which this search doesn't use, the rule drops 6.6% of the tiles and 32% of the top 2%. The rule only changes what the search can find, not whether its p-values are valid.

The kNN bank is the 10 bank visits, scored with `--reference_roots` and `--no-calibrate`, and the reference tiles that share sky with the tile being scored are left out.

## The Model

The search uses DINOv3 ViT-L SAT unless one of two changes beats it on the PRIMER-UDS bench, which never touches COSMOS-Web. The changes are a fine-tune on COSMOS-Web tiles (its step picked on SMACS 0723), and subtracting each grid cell's mean patch feature (the input-independent position term of Yang et al. 2024). A change wins if it puts more planted sources in the top 5% over AB 22 to 28 than the default, in total and on at least 4 of the 5 seeds. A winner is then rebuilt from the 10 bank visits alone, since a model that saw search or calibration tiles in training would treat them differently, and it has to win on the UDS bench again before it scores anything.

## The Rehearsal

The one approximation is that tiles in a visit aren't independent of each other, as the calibration pools them (Dunn, Wasserman and Ramdas 2023 cover this two-level case). So before the search is read, the same procedure runs 1,000 times with 4 calibration visits drawn at random as the search, where every detection is false. BH at $\alpha$ keeps the share of those draws with any detection at about $\alpha$. If more than 15% of draws have one, the search names nothing.

## What Counts as a Detection

A search tile with $q \le 0.1$ is a detection, as long as the rehearsal passed. Every other tile is a candidate for a look and nothing more. A detection gets looked at in each filter and matched against SIMBAD, NED and VizieR before anyone calls it anything, since rare is not the same as new. The p-values come from the same formula as in [the method](method.md#another-field-as-reference), and BH over them keeps the false discovery rate at $\alpha$ (Bates et al. 2023).

Every visit is prepped with:

```sh
jwstdetector prep --sensor nircam --rgb_filters f444w f277w f150w --tile_size 320 \
    --upscale 1 --stride 160 --wcs_in_name \
    --pattern "{tile}_nircam_clear-{filter}_i2d.fits" ...
```

and the search and the first 60 visits also with `--save_cutouts`, for looking at what's found.

## References

- Bates, Candès, Lei, Romano, Sesia. Testing for Outliers with Conformal p-values. Annals of Statistics 51, 149, 2023.
- Dunn, Wasserman, Ramdas. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. Journal of the American Statistical Association, 2023.
- Yang et al. Denoising Vision Transformers. ECCV 2024.
