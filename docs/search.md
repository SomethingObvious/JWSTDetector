# Searching a Survey

This page fixes how the COSMOS-Web search will be run and judged, before any of its tiles has a score. The split it uses is in [search-split.json](search-split.json), and the commit that added both is the timestamp.

## Why a Whole-Survey Search Can't Name a Tile

A conformal p-value can't go below $1/(n+1)$ with $n$ calibration tiles, and Benjamini-Hochberg at level $\alpha$ over $N$ tested tiles needs a p of at most $\alpha/N$ to pass a single one. So one tile can only pass when $n + 1 \ge N/\alpha$. When every tile of a survey is both tested and in the calibration for the others, $n$ and $N$ grow together, so it can't happen however many visits there are.

PRIMER shows it. Its 21 visits were scored in two folds against a bank of 7 visits each, 4,770 tiles in all with about 2,950 calibration tiles each, and nothing reaches $q \le 0.1$. Two tiles score above every one of their calibration tiles, where 1.6 are expected by chance. The first is blank sky. The second sits on the diffraction spikes of a saturated star, which also accounts for the cluster of low p-values in that visit. BH would need 17 tiles at the lowest possible p before it passed any of them, so PRIMER can say nothing in it stands out from chance, but not that any one tile does.

## The COSMOS-Web Search

With $\alpha = 0.1$, a searched fraction $f$ of the scored tiles leaves $`n \approx (1 - f) N / f`$, and one tile can pass once $f \le 1/11$. Of 60 COSMOS-Web visits (program 1727, the 1031 s ones with F150W, F277W and F444W), shuffled with seed 0:

- 4 are searched and 46 calibrate, so the search is 8% of the scored tiles.
- 10 build the kNN bank, with `--reference_roots` and `--no-calibrate` on the default model.

Only tiles whose centres sit on a 320 px grid count, so no two of them share a pixel. That leaves about 400 search tiles against about 4,700 calibration tiles, where one tile needs $p \le 2.5 \times 10^{-4}$ and the lowest possible p is about $2.1 \times 10^{-4}$.

A search tile with $q \le 0.1$ is a detection. Every other tile is a candidate for a look and nothing more. A detection gets looked at in each filter and matched against SIMBAD, NED and VizieR before anyone calls it anything, since rare is not the same as new.

The guarantee assumes the visits are exchangeable, which the random split gives. Tiles inside a visit aren't independent of each other, though, so it's an approximation that pools them (Dunn, Wasserman and Ramdas 2023 work through this two-level case). The p-values come from the same formula as in [the method](method.md#another-field-as-reference), and BH over them keeps the false discovery rate at $\alpha$ (Bates et al. 2023). The calibration and search visits are all prepped with:

```sh
jwstdetector prep --sensor nircam --rgb_filters f444w f277w f150w --tile_size 320 \
    --upscale 1 --stride 160 --wcs_in_name --save_cutouts \
    --pattern "{tile}_nircam_clear-{filter}_i2d.fits" ...
```

## References

- Bates, Candès, Lei, Romano, Sesia. Testing for Outliers with Conformal p-values. Annals of Statistics 51, 149, 2023.
- Dunn, Wasserman, Ramdas. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. Journal of the American Statistical Association, 2023.
