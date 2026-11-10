# The Model

The COSMOS-Web search scores tiles with a fine-tune of DINOv3 ViT-L SAT, called G in [selection.md](selection.md), which has every rule that led to it, each written before the benches it governs. This page has the results those rules were applied to, plus the checks that came after. The weights are 1.2 GB, so they aren't in git. [training/overlap_ft.py](../training/overlap_ft.py) is the script that made them, kept exactly as it ran, and `--model_name models/overlap-ft-noise` loads them once they're there.

## How It Was Trained

It's a LoRA of rank 16 on the query, value and both MLP projections of all 24 layers. Tiles are cut every 160 px, so two neighbouring 320 px tiles share half their sky, and the loss pulls the features of each shared patch together while a second term keeps every feature close to the frozen model's. Each view gets its own turn or flip, and tiles with a FITS cutout are rendered again from flux with Gaussian noise of up to half the sky's sigma, so what the training takes out is where a patch sits in its tile, its orientation and the noise pattern.

It trained for 2,000 steps with a batch of 12 and seed 0 on 4,686 tiles from CEERS, GLASS, Abell 370, MACS J0416 and the 10 COSMOS-Web bank visits:

```
python training/overlap_ft.py --roots datasets/ceers1_nircam datasets/glass_nircam \
    datasets/abell370_nircam datasets/macs0416_nircam datasets/cosmosweb/<the 10 bank visits> \
    --out <dir> --steps 2000 --batch 12 --noise 0.5 --export_at 500 1000 2000
```

None of the search or calibration visits and none of PRIMER-UDS went into it. Step 2,000 was picked on SMACS 0723, where it found 172 planted sources to the default's 143.

## The Confirmation

Both fine-tunes, F without the noise and G with it, were benched against the default on the same 10 UDS visits, each one in bank mode against the other 12 UDS visits, with 5 seeds, AB 22 to 28 and the four original shapes (arcs, points, rings and streaks). None of those visits had been used for anything before. G found 1,475 of the 4,200 planted sources to the default's 1,359, which is 11.6 more a visit (sd 10.5), better on 8 visits and worse on 2. The exact one-sided sign-flip test over the visits gives p = 0.0078 against its share of alpha, 0.0125. F passed too, with 1,463 at p = 0.0078 against 0.025, and the rule takes the higher total of the ones that pass.

The gain is not even across shapes or magnitudes. These are found in the top 5% over all seeds, with m50 where recall crosses 50%, and in brackets where the 95% Wilson interval's two ends cross it ("none" means it never does).

| Shape | Default | G | Default m50 | G m50 |
|---|---|---|---|---|
| Arc | 295/1050 | 385/1050 | 25.23 (none to 25.69) | 25.78 (25.50 to 26.05) |
| Point | 29/1050 | 36/1050 | none | none |
| Ring | 553/1050 | 530/1050 | 25.06 (24.88 to 25.22) | 24.89 (24.76 to 25.06) |
| Streak | 482/1050 | 524/1050 | 24.89 (24.69 to 25.09) | 24.78 (24.66 to 24.92) |
| All | 1359/4200 | 1475/4200 | 24.39 (24.18 to 24.60) | 24.53 (24.37 to 24.70) |

By magnitude, the default finds 316, 328 and 344 of 600 at AB 22, 23 and 24, and G finds 357, 366 and 377. From AB 25 on they're level, and both are at chance by AB 27 (the top 5% catches about 30 of 600 by luck). So the gain is bright arcs and streaks. At AB 25, G actually does a bit worse on rings and streaks (67 and 57 of 150, to the default's 79 and 69), which is why their m50 is lower. Point sources are at chance for both models at every magnitude, probably because one more compact source looks like the many already in every tile.

## Above Every Ordinary Tile

Being in the top 5% is a lot easier than what the search needs, which is to score above nearly every calibration tile. So the bench also records each source's rank among the tiles nothing was planted in, and beats_all counts the sources that come out above every one of them. The default's confirmation benches ran before that existed and only seed 0's scores were kept, so this comparison is seed 0 alone, worked out afterwards from the saved scores (every recorded rank came out the same). It's a report, not a test.

| Shape | Default | G |
|---|---|---|
| Arc | 7/210 | 16/210 |
| Point | 1/210 | 0/210 |
| Ring | 101/210 | 102/210 |
| Streak | 60/210 | 71/210 |
| All | 169/840 | 189/840 |

G is ahead on 7 visits, behind on 1 and level on 2, and the sign-flip test would give p = 0.016. Over all 5 seeds, G puts 982 of 4,200 above every ordinary tile, nearly all of them at AB 22 to 25. Its m50 on that measure is 24.74 for rings and 24.00 for streaks, and arcs never get to 50%.

## Shapes It Was Never Picked On

Every choice above was made on the same four shapes, so a gain could be tuning to them rather than seeing more. The check written for this before any of its benches ran uses three shapes that no model, step or setting was ever chosen with. Clumps are 3 to 6 knots on a line, diffuse is one wide Gaussian at low surface brightness, and a pair is two point sources 2 to 5 px apart. They go into the same tiles, places and colours as the confirmation's sources.

| Shape | Default found | G found | Default above all | G above all |
|---|---|---|---|---|
| Clumps | 65/1400 | 68/1400 | 0/1400 | 1/1400 |
| Diffuse | 114/1400 | 116/1400 | 12/1400 | 1/1400 |
| Pair | 69/1400 | 104/1400 | 8/1400 | 8/1400 |
| All | 248/4200 | 288/4200 | 20/4200 | 10/4200 |

The default didn't do better at 0.05 on either measure (p = 0.13 for beats_all, where G was 10 lower, and 0.86 for found), so by the rule G stays. G didn't do better on beats_all either, so this page can't say it sees more on new shapes. Neither model sees them much at all, though. With about 227 tiles a visit, chance alone puts roughly 200 of 4,200 in the top 5% and 20 above every ordinary tile, so the default sits at chance on both, and G is a little above it on found and below it on beats_all. Like single points, these look like ordinary sources. That means the check could only have caught a large harm, and the drop on diffuse sources (12 to 1) is worth keeping in mind even though it's small enough to be chance.

## Completeness on COSMOS-Web

COSMOS-Web isn't UDS, so both models were also benched on two COSMOS-Web calibration visits, jw01727-o005_t017 and jw01727-o008_t026, against the search's 10 bank visits with the search's run settings. Neither visit was in the training. Two visits can't carry a test, so this is only reported.

| Shape | Default found | G found | Default m50 | G m50 | Default above all | G above all |
|---|---|---|---|---|---|---|
| Arc | 84/210 | 99/210 | 25.38 (none to 25.77) | 25.33 (24.74 to 25.76) | 19/210 | 33/210 |
| Point | 0/210 | 8/210 | none | none | 0/210 | 0/210 |
| Ring | 94/210 | 91/210 | 24.53 (24.25 to 24.82) | 24.53 (24.25 to 24.82) | 88/210 | 87/210 |
| Streak | 86/210 | 98/210 | 24.36 (23.96 to 24.71) | 24.50 (24.13 to 24.87) | 53/210 | 64/210 |
| All | 264/840 | 296/840 | 24.13 (23.39 to 24.46) | 24.31 (24.01 to 24.59) | 160/840 | 184/840 |

On the three held-out shapes, the default found 48 of 840 and G 61, with 42 or so expected by chance, and 0 and 2 above every ordinary tile.

### Found by the Search

The other measure in selection.md asks whether the search itself would have named a planted source. Each planted tile from G's benches on the two visits gets a conformal p against the search's kept calibration tiles, leaving out its own visit's, and it counts as found if that p is at or below 0.1 / 1,345, which is what BH needs to name one tile alone. Only tiles the overlap and star rules keep are counted, 206 a magnitude over all seven shapes and 5 seeds.

None of them was found, at any magnitude, for any shape. The brightest planted sources still had plenty of calibration tiles above them. With seed 0, where the bench's bank is exactly the search's (every unplanted tile scores the same in both), the best of the 24 sources planted at AB 22 was a ring with 64 calibration tiles at or above it, and the other rings and streaks had 70 to 515. A tile needs at most one above it. So the search's completeness is zero for every shape here, and what limits it is the few dozen most unusual calibration tiles, not how the detector ranks tiles within a visit.

Most of those come from one visit whose stretch a bright star threw off ([search.md](search.md#the-result) has the details). Leaving it out of the calibration afterwards, which is a check and not a result, the search would have named 15 of 27 rings at AB 22, 16 at 23, 12 at 24 and 4 at 25, plus 6 streaks in all over AB 22 to 24, and still nothing of the other five shapes.

Dropping a visit by hand isn't something a search can do in advance, though. Putting each visit's kept scores on its own median and MAD before the conformal p is, since it treats every visit the same way. Done afterwards on the same scores, it gives about the same as leaving o165 out (16 of 27 rings at AB 22, 16 at 23, 10 at 24 and 4 at 25, and 3, 6 and 1 streaks at AB 22 to 24), without dropping anything. The 70 highest calibration tiles then come from many visits, at most 5 from any one, and the search tiles still hold nothing (p = 0.00048 at best). The other five shapes stay at zero even at AB 22, which looks like a limit of patch kNN at this scale, not of the calibration.

## What This Doesn't Show

Every source is synthetic, and each goes into one tile only, though in the sky the tiles that overlap it would see part of it too. Part of a ring or streak also falls on patches its tile doesn't score ([the benchmark](benchmark.md#what-is-planted) has the shares). The top 5% of a visit's 227 or so tiles is 11, so at most 11 of the 12 sources in a run can count as found. The benches' copies of a visit also had no mosaic record, so bank tiles of a neighbouring visit that share sky with it weren't left out the way the search leaves them out. Both models ran that way, and the bench keeps the record now. Nothing here is a real, labelled class of rare object. The gain is in shapes that already stood out (bright arcs and streaks), and at AB 25 G is a little worse on rings and streaks. The 10 confirmation visits were also used to choose between F and G (with alpha split between them) and then again for the held-out shapes, but never with those shapes. And the bank is a random million of its visits' 2.85 million patches, so a different seed moves a few tiles' scores by up to 0.06.
