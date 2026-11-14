# Searching a Survey

This page fixes how the COSMOS-Web search is run and judged, before any of its tiles has a score. The split is in [search-split.json](search-split.json), the bright stars in [search-stars.json](search-stars.json), and `python -m jwst_peculiar.survey` runs every rule below. Commit ba55ee8 has the first version of this page, with 46 calibration visits and none of the tile rules, and 13a7b26 the second, which only counted tiles on a 320 px grid. No tile had a score when any version was written. Four commit messages from 2026-11-03 on were reworded on 2026-11-10, which gave every later commit a new hash. On 2026-11-13 the same commits were rewritten again to take out a code comment and move one test's made-up sky position, which changed the hashes once more. The hashes on these pages are the newest ones. Neither rewrite changed a date or any file these pages describe, and GitHub may still serve the commits under their old hashes.

## Why a Whole-Survey Search Can't Name a Tile

A conformal p-value can't go below $1/(n+1)$ with $n$ calibration tiles, and Benjamini-Hochberg at level $\alpha$ over $N$ tested tiles needs a p of at most $\alpha/N$ to pass a single one. So one tile can only pass when $n + 1 \ge N/\alpha$. When every tile of a survey is both tested and in the calibration for the others, $n$ and $N$ grow together, so it can't happen however many visits there are.

PRIMER shows it. Its 21 visits were scored in two folds against a bank of 7 visits each, 4,770 tiles in all with about 2,950 calibration tiles each, and nothing reaches $q \le 0.1$. Two tiles score above every one of their calibration tiles, where 1.6 are expected by chance. The first is blank sky. The second sits on the diffraction spikes of a saturated star, which also accounts for the cluster of low p-values in that visit. BH would need 17 tiles at the lowest possible p before it passed any of them, so PRIMER can say nothing in it stands out from chance, but not that any one tile does.

## The Split

COSMOS-Web (program 1727) has 148 visits with F150W, F277W and F444W at 1031 s. 60 of them, shuffled with seed 0, were split again with seed 0 into 4 search visits, 10 bank visits and 46 calibration visits, and the other 88 joined the calibration. Every visit was equally likely to be searched, so the search and calibration visits are exchangeable.

With $\alpha = 0.1$, a searched fraction $f$ of the scored tiles leaves $`n \approx (1 - f) N / f`$, and one tile can pass once $f \le 1/11$. The search is 4 of 138 scored visits, or 3%, which with the tile counts expected when this was written left room for a tile with two calibration tiles above it to pass, and for several tiles to pass together. The counts the rules left in the end allowed one calibration tile above it ([The Result](#the-result)).

## Which Tiles Count

Each rule depends only on where a tile is, never on its score, and applies to search and calibration tiles alike.

1. Every tile counts, and the visit has to be scored with `--score_region owned`. Tiles overlap by half, but each one only scores [the patches nearer its centre than any other tile's](method.md#which-patches-a-tile-scores), so every bit of sky is scored by exactly one tile. That also keeps whatever a tile scores at least a quarter of a tile (80 px, or 5 arcseconds) from the edge of the image the backbone sees, so an object isn't cut off by the edge unless it's more than about 10 arcseconds across. The version in 13a7b26 kept only tiles on a 320 px grid, and as those score only their middle quarter, it left three quarters of the sky unsearched. Neighbouring tiles still see each other's surroundings, so one object can raise two of them, and the rehearsal below is what checks that this doesn't make false detections more likely. A check on 2026-11-09, after the verdict, found that exactly one tile holds inside a visit only, as two neighbouring tiles both score the patches along its outside edge (5.4% of the scored patches in jw01727-o001_t001). It also found that the quarter-tile margin only holds for a tile with all eight neighbours. 7,676 of the 40,643 kept tiles (19%) lack one, at a visit's edge or beside a hole, and those score out to their own edge on that side. They were 488 of the 2,033 highest tiles, where their share of the tiles would give 384. Scoring every tile on its middle 14 by 14 patches afterwards (`survey --middle`, as [on MINERVA](minerva.md#a-rescore-after-the-result)) leaves them at 368 and changes little else, with no detection, the rehearsal at 2.8% again and a lowest p of 0.0017 on the same two tiles. That rescore wasn't planned and isn't a result. The rule and the scores are as they were.
2. A tile goes if it may share sky with a tile of any other visit, meaning their centres are less than a tile's diagonal apart. Neighbouring visits overlap at their edges, and an object seen twice would otherwise sit in the calibration as its own twin.
3. A tile goes if a Gaia DR3 star brighter than G = 16 is within half a tile's diagonal plus $`1.5'' \times 10^{0.2 (19 - G)}`$, which is 6 arcseconds at G = 16 and 4 arcminutes at G = 8 (this said 2.4 arcminutes until 2026-11-10, a slip in the arithmetic, and the code has always used the formula). Diffraction spikes are the commonest rare thing in this kind of sky. On PRIMER, which this search doesn't use, the rule drops 6.6% of the tiles and 32% of the top 2%. The rule only changes what the search can find, not whether its p-values are valid.

The kNN bank is the 10 bank visits, scored with `--reference_roots` and `--no-calibrate`, and the reference tiles that share sky with the tile being scored are left out.

## The Model

The search uses a LoRA fine-tune of DINOv3 ViT-L SAT that makes the same sky give the same patch features in two overlapping tiles, and [model.md](model.md) has how it was trained and every result behind it. The first version of this section said the default would be used unless a change beat it on the UDS bench in total and on 4 of 5 seeds, and its two changes both lost. A fine-tune on COSMOS-Web tiles (ExPLoRA, with step 2,000 picked on SMACS 0723) found 108 planted sources to the default's 111 and beat it on 2 of the 5 seeds, and subtracting each grid cell's mean patch feature found 95 and beat it on none. A third change, a fine-tune that makes the same sky give the same features in two overlapping tiles, failed the same rule on o008 (148 to 145), though it won on two other UDS visits, and that verdict stands. A separate and larger test on 10 UDS visits nobody had used was then written down before any of its benches ran, with alpha split between the tries it allowed, and [selection.md](selection.md) has it and every rule after it. All of that came before any COSMOS-Web tile was scored, but it is still a second try after a failure, which is worth keeping in mind when reading its p-value. The model that came out of it found more planted sources than the default on those 10 visits, with p = 0.0078 against the 0.0125 of alpha it had. The other condition here still holds as written. A model that saw search or calibration tiles in training would treat them differently, so it trained only on the 10 bank visits and four fields outside COSMOS-Web.

## The Rehearsal

The one approximation is that tiles in a visit aren't independent of each other, as the calibration pools them (Dunn, Wasserman and Ramdas 2023 cover this two-level case). So before the search is read, the same procedure runs 1,000 times with 4 calibration visits drawn at random as the search, where every detection is false. BH at $\alpha$ keeps the share of those draws with any detection at about $\alpha$. If more than 15% of draws have one, the search names nothing.

## What Counts as a Detection

A search tile with $q \le 0.1$ is a detection, as long as the rehearsal passed. Every other tile is a candidate for a look and nothing more. A detection gets looked at in each filter and matched against SIMBAD, NED and VizieR before anyone calls it anything, since rare is not the same as new. The p-values come from the same formula as in [the method](method.md#another-field-as-reference), and BH over them keeps the false discovery rate at $\alpha$ (Bates et al. 2023).

Every visit is prepped with:

```sh
jwst-peculiar prep --sensor nircam --rgb_filters f444w f277w f150w --tile_size 320 \
    --upscale 1 --stride 160 --wcs_in_name \
    --pattern "{tile}_nircam_clear-{filter}_i2d.fits" ...
```

and the search and the first 60 visits also with `--save_cutouts`, for looking at what's found. Only the four search visits and two calibration visits (o005 and o008) kept their cutouts in the end, as the rest were deleted for disk space.

## The Result

Every visit was scored on 2026-11-07 with the model in [model.md](model.md), and `results_cosmosweb/verdict.json` has the numbers. After the overlap and star rules, 1,345 search tiles were tested against 39,298 calibration tiles (3,048 tiles were dropped as overlaps and 6,355 as near a bright star). The rehearsal passed, with 2.8% of its 1,000 draws getting any detection, and nothing in the search reached $q \le 0.1$. The lowest p of a search tile was 0.0018, so 70 or so calibration tiles scored above it, where naming a tile on its own needed $7.4 \times 10^{-5}$ (at most one calibration tile above it). Every q came out at 0.70. As far as this detector can tell, the four search visits hold nothing stranger than the 134 calibration visits do.

How far it can tell is the catch. Planted into two calibration visits, none of the 1,442 sources the rules kept would have been named, not even rings at AB 22 that score above every other tile in their own visit, as 60 to 500 calibration tiles score higher than those (the numbers are in [model.md](model.md#found-by-the-search)). So this says the search tiles aren't stranger than the strangest few dozen calibration tiles, and not much more.

Looking at those tiles afterwards, 61 of the 70 highest come from one calibration visit, jw01727-o165_t131. A very bright star sits in its corner, and prep sets each filter's upper stretch level from a percentile of the whole mosaic, so the star's light pushed that level 4 to 12 times above the median visit's. Every tile in the visit came out darker, with the sky crushed to black, and its median score is 0.22 where the median visit's is 0.066. The star rule dropped the tiles near the star but couldn't fix the stretch of the rest. It's milder elsewhere, though not rare. 23 of the 148 visits have a filter's level over 1.5 times the median, and the visits with the highest levels also have the highest median scores.

None of this changes the verdict, which stands as the rules above give it, and a heavier tail in the calibration can only make detections harder to get, never easier. Leaving that visit out after the fact, which wasn't planned and can't count as a result, still gives no detection (9 calibration tiles above the top search tile, p = 0.00026). But it does change what the search could have seen, as [model.md](model.md#found-by-the-search) shows. A survey with one depth everywhere should have one stretch for every visit, measured across visits instead of per mosaic. Prep didn't do that when this search ran, and `--levels` ([usage.md](usage.md#one-stretch-for-a-survey)) does now, for the next survey. A split with `"standardise": true` also puts each visit's scores on its own median and MAD before the conformal p, which takes out what's left of a shift like this one. The MINERVA search ([minerva.md](minerva.md)) is planned with both.

The bank is also a random sample. Its 10 visits have about 2.85 million patches and the bank keeps 1 million, so the draw depends on the seed, and a different seed moves a few tiles' scores by up to 0.06 while most don't move at all. The search used seed 0 only.

## References

- Bates, Candès, Lei, Romano, Sesia. Testing for Outliers with Conformal p-values. Annals of Statistics 51, 149, 2023.
- Dunn, Wasserman, Ramdas. Distribution-Free Prediction Sets for Two-Layer Hierarchical Models. Journal of the American Statistical Association, 2023.
