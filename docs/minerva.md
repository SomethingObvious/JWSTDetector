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

Every block is scored in bank mode against the bank with the model the AEGIS confirmation keeps. Each block's kept scores go on its own median and MAD (`"standardise": true`) before the conformal p, and the rest is COSMOS-Web's: BH at q <= 0.1 over the search's grid tiles, a rehearsal on calibration blocks alone that has to find something in at most 15% of draws, and the star rule.

## What It Claims

Only search tiles with q <= 0.1 count as detections. Anything else is a ranked candidate at most. A detection gets looked at in every filter and matched against catalogues, and nothing is said about it to anyone outside before that, or before the MINERVA team has seen it.
