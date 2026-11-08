# The MINERVA Search

This is the plan for searching MINERVA (JWST program 7814) the way [search.md](search.md) searched COSMOS-Web, written before any of its deep tiles are scored, with what COSMOS-Web taught built in. Where it doesn't say otherwise, the rules are COSMOS-Web's.

## Data

The four deep mosaics, o001 and o002 in UDS and o003 and o004 in COSMOS, as STScI's level-3 i2d files from MAST, with F460M, F250M and F182M as red, green and blue. They're 22,000 to 30,000 s deep in each filter. The AEGIS visits are shallower and are the model confirmation set ([selection.md](selection.md#the-model-confirmation-on-aegis)), so they're not in this search.

Prep is COSMOS-Web's (320 px tiles every 160 px on the F460M grid, 0.063 arcsec a pixel), except that each filter's stretch is the median over the four mosaics (`--levels`, written from a first cut of the same four with their own stretch).

## Blocks Instead of Visits

Four mosaics are far too few to calibrate visit against visit, so each one is split into square blocks 1,920 px (about 2 arcminutes) a side, and the blocks play the part COSMOS-Web's visits played (`cut_blocks` in [survey.py](../jwstdetector/survey.py)). A block takes a tile only if all of it lies inside, so no two blocks share a pixel, and the tiles that straddle a border belong to none, which leaves a 160 px strip along each border unsearched. A full block holds 121 tiles, and one with fewer than 30 is left out. The blocks are sorted by name (mosaic, row, column) and shuffled with seed 0. Then:

- the first 15 blocks are the kNN bank,
- the next blocks, as many as keep the search's tiles at 1/12 of the search's and calibration's together or fewer, are the search,
- and the rest calibrate.

The overlap rule drops any search or calibration tile that shares sky with a tile of another block, the bank's included. Two tiles of one mosaic overlap only if their boxes do, which blocks never do, so this only bites where two mosaics meet. It also drops any tile on sky the adopted model was trained on (`training_data` in the split), as the fine-tune has seen that sky as normal.

## Scoring and the Verdict

Every block is scored in bank mode against the bank with the model the AEGIS confirmation keeps. Each block's kept scores go on its own median and MAD (`"standardise": true`) before the conformal p, and the rest is COSMOS-Web's: BH at q <= 0.1 over the search's grid tiles, and a rehearsal on calibration blocks alone that has to find something in at most 15% of draws.

The star rule reaches fainter stars than COSMOS-Web's did. [minerva-stars.json](minerva-stars.json) holds every Gaia DR3 source brighter than G = 16 over the four mosaics, and the ones from G = 16 to 19 that Gaia sees moving (proper motion or parallax over 5 sigma), since compact galaxies and quasars don't move and nearly every Gaia source that bright is a star. The split sets `"star_g_max": 19`, and the reach is COSMOS-Web's. The cut-off came from COSMOS-Web's scores after its verdict. With the overlap rule alone, the old rule dropped 13.5% of its tiles and 44.5% of the top 1%, and this one drops 20.7% and 57.7% (and 61.7% of the top 0.1%, against 38.3%). Going fainter than G = 19 dropped tiles barely faster than chance. Of the 20 diffraction spikes among the trial list's 50 sources (below), it drops 14.

## The Vetting List

The test above can only name a tile that beats nearly every calibration tile, so most of what the search sees never reaches it. A trial on COSMOS-Web's 50 highest sources, ranked as below after its verdict, found 20 diffraction spikes, 9 detector artefacts (mostly in F150W alone), 4 trails, 6 edges of bright galaxies cut off by the tile, 9 ordinary galaxies and pairs, and 2 red compact galaxies that are already catalogued. Nothing in it was new, and two thirds of it was something a person would throw out in seconds, which is what the star rule above and the flag below are for.

[vet.py](../jwstdetector/vet.py) writes the list for MINERVA, once the verdict is in. It ranks every scored block, calibration as well as search, since the list makes no statistical claim, after the overlap and star rules and on each block's median and MAD. Each source is listed once, at its highest tile (hot spots within 1.5 arcsec are one source). Its hottest patch is checked in each filter of the tile's cutout, as MADs off the tile for both the patch mean and its smoothed peak, and it's flagged if it shows in no filter (all under 5) or in one filter only (that one at least 1.6 times the next and the next under 8). The flagged ones are listed after the 50 that aren't. On COSMOS-Web's trial list, where these numbers were picked, the flag caught 8 of the 13 artefacts and trails and 1 of the 17 real sources. Every source gets a sheet with the tile, each filter and its hot spot in colour, and the nearest SIMBAD and NED entries within 2 arcsec.

## The Colour Search

The model sees three filters as a picture, but MINERVA has eight medium bands, and what most often stands out in medium bands is a spectrum, such as an emission line bright enough to light up one band, which a picture of three bands can miss. So a second list ranks sources by how unlike every well-measured source their spectrum is. It's unsupervised like the first and makes no claim either.

- All eight bands of the four deep mosaics (F140M, F162M, F182M, F210M, F250M, F300M, F360M and F460M), each smoothed with a Gaussian to F460M's resolution, using the PSF widths in JDox, and resampled onto the F460M grid.
- Each band loses a local sky, a clipped median in boxes about 6 arcsec across, since the halo of a bright star reaches well past the star rule and turned whole regions into one source. Sources are found with photutils on the noise-weighted sum of the eight bands (5 connected pixels over 3 sigma), deblended, and placed at their segment's peak rather than its centroid, as a segment can hold a source and a stretch of halo or detector stripe beside it. Each is measured in a 0.32 arcsec aperture in each band. A band's error is the scatter of the same aperture on empty sky in its 1,920 px block, with 5% of the flux added in quadrature for PSF and calibration mismatch. Sources need every pixel within 0.24 arcsec covered in all eight bands, 10 sigma in the sum, and to be outside a bright star's halo (the star rule's reach without the tile's half-diagonal, as a source is a point, which takes 2.3% of o003 instead of 15.6%).
- The sources at 30 sigma or more in every band form the library. Each source's score is the fifth-smallest chi-squared over library sources (leaving out itself and anything within 1 arcsec), with the library spectrum scaled to fit, so a kind of spectrum needs five well-measured examples before it counts as ordinary, and a faint source with big errors can't score high on noise alone.
- A source more concentrated than a star is left out of the ranking and counted. The flux inside 0.08 arcsec over the flux inside 0.24 arcsec puts o003's point sources at 0.35 to 0.40, and above 0.45 sat 412 of its 526 sources scoring over 300, against 2 of 16,223 scoring under 30. Those are hot pixels and cosmic rays in one band, which score about 400 because one band off by its whole flux costs (1 / 0.05) squared.
- The 50 highest sources (once each) get a sheet with all eight bands and their spectrum against the five library fits, plus the catalogue match, and go after the first list.

There's one known answer in o003. Festin et al. (2026) list a brown dwarf candidate from COSMOS-Web at 150.06343 +2.35522, and it's the highest source on o003's list, 0.23 arcsec from where COSMOS-Web saw it, as it has moved since. With only medium bands, though, a hotter T dwarf and a galaxy at z of about 2.3 with [OIII] in F162M and H-alpha in F210M look alike, so this list doesn't call anything a brown dwarf by itself.

Only search tiles with q <= 0.1 count as detections. Anything else, both lists included, is a candidate for a person to look at and nothing more. A detection gets looked at in every filter and matched against catalogues, and nothing is said about it to anyone outside before that, or before the MINERVA team has seen it. The lists are looked at the same way, and only what survives a first look goes any further.
