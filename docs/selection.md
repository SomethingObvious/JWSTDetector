# Choosing the Model

These are the rules the search's model is chosen by, in the order they were written,
each one before any bench it governs had run. They lived in working notes until this
page was committed, so the times given before that commit rest on the notes alone, and
git has the times from it on. Two sentences were reworded on the way, one that named
who asked for more tries and one that pointed at a private note. The scripts and
benches they refer to are in [benchmark.md](benchmark.md) and [search.md](search.md).

## Sensitivity Round

This rule is fixed before any of its benches has run.

Every bench runs the way the search scores. Each PRIMER-UDS bench visit is scored against
a bank of the other 12 UDS visits with `--no-calibrate`, 5 seeds, AB 22 to 28, kNN, and a
planted source counts as found in the top 5% of tiles. None of this touches COSMOS-Web.

The baseline is DINOv3 ViT-L SAT with the default bank cap of 1,000,000 patches, which
is what the search does now. The candidates are:

- A, DINOv3 and raw pixels together. Each model's tile scores become z-scores against
  the median and MAD, `(score - median) / (1.4826 MAD)` over the run's unplanted tiles,
  and the tile's score is their plain mean. Equal weights, nothing tuned.
- B, DINOv3 with the bank cap raised to 2,200,000, which holds every patch of the 12
  bank visits.
- C, A built from B's DINOv3 run.

A candidate wins if, on jw01837-o008_t010, it finds more planted sources than the
baseline in total and on at least 4 of the 5 seeds, and then also finds more in total
on jw01837-o024_t028, which no bench has used. Among the winners the one with the
higher o008 total is adopted, and a tie goes to the simpler one (B, then A, then C).
With no winner the search stays as it is.

## The Fine-Tune

This part is fixed before training starts. A fourth candidate, F, is DINOv3 with
LoRA trained by overlap_ft.py, so that a patch of sky seen in two overlapping tiles
gives the same features whatever its place in the tile or its orientation, while staying close to the frozen model's features. It trains on the
COSMOS-Web bank visits on disk when it starts and on the CEERS, GLASS, Abell 370 and
MACS 0416 control fields. None of those is a search, calibration or bench visit.

The step (500, 1000 or 2000) is picked on the SMACS 0723 bench scored the old way,
against its own tiles, by total found with ties to the earlier step, where the
default model gets 143. The picked step then has to win like any other candidate: on
o008 in total and on 4 of 5 seeds, then in total on o024 and also on o031_t035, which
no bench has used either. If it wins, the search uses it, and it can be combined
with B only through a fresh round with its own rule.

In the search, A's z-scores would use the median and MAD of the bank tiles' own
scores, fixed before any search or calibration tile is scored.

## Confirmation Round for F, Fixed Before Any of Its Benches Ran

F failed the rule above on o008, 148 to 145 and better on 1 of 5 seeds, though it won
o024 (128 to 143, 5 of 5) and o031_t035 (155 to 170, 4 of 5). That verdict stands.
This is a separate test, written down on 2026-11-06 at about 04:20 before any bench
in it started, on the 10 UDS visits no bench has touched: o009_t011, o014_t016,
o021_t023, o022_t024, o023_t025, o025_t029, o032_t036, o033_t037, o034_t038 and
o035_t039. F never trained on UDS.

Each visit is benched in bank mode (the other 12 UDS visits as the bank,
--no-calibrate, 5 seeds, AB 22 to 28) with the default and with ft/step500 exactly as
exported, no retraining and no other step. The 50 visit and seed pairs are paired by
construction, since both models get the same planted sources.

F passes only if both hold. Its total over the 10 visits is higher than the
default's, and a one-sided sign test over the 50 pairs (ties dropped) gives p at or
below 0.05. o008, o024 and o031 are not counted, in either direction. If F passes,
the search uses it, and docs/search.md gets amended and pushed before any COSMOS-Web
tile is scored. If it fails, the default stays and F is done.

## G, and Splitting the 5% Between Tries

Written on 2026-11-06 at about 10:10, while F's confirmation was running and before
any of its results had been read (4 of its 20 benches had finished). The aim was to
keep trying until something can be adopted. Trying one candidate after another at 5%
each would sooner or later adopt one that is no better, so the 5% gets split. F's
confirmation now passes at p at or below 0.025 instead of 0.05, G gets 0.0125, and any
later candidate gets half of what is left (0.00625, then 0.003125 and so on). All the
tries together then adopt a model that is no better at most 5% of the time, and that
holds even though F and G are tested on the same visits.

G is F with one change. Lupton clips the sky at 0, so F's noise augmentation never
did anything. G draws a tile that has a FITS cutout again from its flux, after the
noise equalisation and before the stretch, with Gaussian noise of U(0, 1) times that
filter's sky sigma added, and each view of a pair gets its own draw. Only the four
control fields kept cutouts, so half of each batch comes from their pairs and half
from the COSMOS-Web bank visits as PNGs. It trains with overlap_ft.py `--noise 1.0`
on the same rule as F (the control fields and every bank visit on disk when it
starts, which is all 10 now), batch 12, 2000 steps. The step (500, 1000 or 2000) is
picked on the SMACS self bench by total found, ties to the earlier step.

G is then benched on the same 10 fresh UDS visits as F, against the same default
benches, and passes the same way at 0.0125. G hasn't seen UDS either. If F and G both
pass, the one with the higher total over the 10 visits is adopted, and a tie goes to
F. If neither passes the default stays, and any later candidate needs visits no bench
has used. The COSMOS-Web search doesn't score a tile until this verdict is in.

Scoring with the mean of the default's and F's z-scores isn't registered. On the dev
visits it found 149, 145 and 167 against F's 145, 143 and 170, which isn't worth
doubling the search's GPU time.

## The Test Counts Visits, Not Seeds

Changed on 2026-11-06 at about 10:40, with F's confirmation results still unread
(4 of 20 benches done, all run before 9 am). The 5 seeds of a visit plant into the
same tiles, so when the fine-tune's gain differs from visit to visit, all 5 seed
pairs of a visit move together and the sign test over 50 pairs counts one visit's
luck five times. In a simulation of this design with the gain varying between
visits as much as its size, that test passed a model that was no better 8.6% of the
time at a stated 0.025.

So for F, G and anything after them, the p value comes from the 10 visit differences
(the candidate's total minus the default's on each visit), by an exact one-sided
sign-flip test over all 1024 ways of flipping their signs. Everything else stays as
written above. The total still has to be higher, the alphas stay 0.025 for F and
0.0125 for G, and the judge prints the old 50-pair sign test only for reference.
The smallest p this test can give is 1/1024, so with 10 visits no candidate past the
sixth could pass, and a later round needs more fresh visits.

## G's Noise Lowered, and H

Changed on 2026-11-06 at about 10:50, with F's confirmation results still unread and
G not yet trained. A review of the literature points out that making features ignore noise is exactly
what would hide an AB 27 to 28 point source, which sits at the noise level. So G adds
U(0, 0.5) sky sigma instead of U(0, 1), and nothing else about G changes.

H needs no training. It is the default model with the tile drawn at 640 px instead of
448, so a patch covers 8 original pixels instead of 11, and the tile's score is the
mean of its 2 worst owned patches (`--resolution 640 --score_mode topk_mean
--score_top_frac 0.005`), which is the same 2 patches top1p takes at 448. Resolution
is the one lever the industrial literature shows helping small defects (PatchCore
on MVTec AD 2 went from 28.8 to 41.9 AU-PRO between 256 and 512 px), and nothing here
is tuned. H is first benched on o008, o024 and o031. If its total there isn't above
the default's 431, H stops and spends no alpha. Otherwise it is benched on the same
10 fresh visits and passes the same way at 0.00625. Among the candidates that pass,
the higher total is adopted, and ties go to F, then G, then H.

## Faster Benches, Same Ranks

From c_o025_t029_dinov3 on, every bench runs on cd62826, which keeps the bank and
the distances of tiles it has already scored between the runs of a bench. Nothing
about what is scored changes. Benched on o008 with the default model, it gave the
same rank for all 420 planted sources as the bench the old code made, in 7 to 8
minutes instead of 26.

## Checking the Adopted Model on Shapes It Was Never Picked On

Written on 2026-11-06 at about 19:35, after F passed its confirmation, while G's step
was still being picked on SMACS, and before any of G's or H's confirmation benches or
any bench below had run. The adopted model here is whichever of F, G, H or the
default the rules above end up adopting.

Every choice so far was made on four invented shapes, so a gain could just as well be
tuning to them as seeing more. So the bench gets three more shapes (`HELD_OUT` in
benchmark.py), and no model, step or setting has ever been chosen with them. Clumps
is 3 to 6 equal knots a pixel wide on a line twice the radius long, like a chain
galaxy or a clumpy arc. Diffuse is one round Gaussian with a sigma of half the
radius, so the same flux is spread out to a low surface brightness. Pair is two point
sources 2 to 5 pixels apart, like a close merger or a lensed pair. A seed plants them
in the same tiles, at the same places and in the same colours as the four shapes in
the confirmation benches, so only the shape differs.

The bench now also records each source's null rank, its rank among the tiles nothing
was planted in. A null rank of 1 means it scored above every ordinary tile in the
visit, which is a lot closer to what the search needs (a tile has to beat nearly all
of its calibration tiles) than being in the top 5%. That count, beats_all, is the
main measure from here on, with the top 5% reported next to it.

The default and the adopted model are each benched on the 10 confirmation visits in
bank mode as before, with 5 seeds, AB 22 to 28 and `--kinds clumps diffuse pair`. On
each visit the adopted model's total minus the default's is taken for beats_all and
for found, over every shape, magnitude and seed, and both go through the exact
one-sided sign-flip test over the 10 visits.

- If the default does better at p at or below 0.05 on either measure, the adopted
  model is dropped and the search uses the default. Two measures at 0.05 each make
  dropping it easier, which is the cautious way round.
- Only if the adopted model does better on beats_all at p at or below 0.05 is it
  described as seeing more on shapes it was never picked on. Otherwise the write-up
  says that it doesn't show.
- If the default is the adopted model, it's benched alone and only reported.

The confirmation visits were used to choose between passing candidates, but never with
these shapes, and the shapes are what this checks.

## Completeness on COSMOS-Web

Written at the same time. COSMOS-Web isn't UDS (F150W, F277W and F444W at 1031 s), so
the completeness the search quotes is measured on it, and only reported. Both models
are benched on jw01727-o005_t017 and jw01727-o008_t026, the two calibration visits
whose cutouts were kept, against the search's 10 bank visits with the search's run
settings, all seven shapes, 5 seeds and AB 22 to 28. There are two measures.

1. Found and beats_all as above, against the visit's own 363 tiles.
2. Found by the search. Each planted tile gets a conformal p against the search's
   kept calibration tiles, leaving out the planted visit's own, from the search's
   scoring with the adopted model. It counts as found if p is at or below alpha / N,
   with N the kept search tiles, which is what BH needs to name a tile on its own.
   Other detections only loosen that, so this is a floor. Only planted tiles that the
   overlap and star rules keep count.

Two visits can't carry a test, so there is none. The second measure needs the search's
scores, so it's worked out after the verdict and changes nothing about it.

For both fields the write-up gives, per shape and model, recall at each magnitude with
its 95% Wilson interval, and m50 with the range between where the interval's two ends
cross 50%. It also says what this still can't show. The shapes are all synthetic. A
source goes into one tile only, though in the sky the overlapping tiles would see
part of it too. And nothing here is a real, labelled class of rare object.

## After the COSMOS-Web Search

Written 2026-11-07, before any of the runs below. MINERVA's search has its own plan in
[minerva.md](minerva.md), and the AEGIS confirmation here picks the model it uses.

## S, G With Stretch Jitter

S is G's recipe exactly (same roots, 2,000 steps, batch 12, seed 0, flux noise 0.5) plus
`--stretch 4`, so each view drawn from flux gets each filter's range scaled by its own
log-uniform factor in [1/4, 4]. Steps 500, 1000 and 2000 are exported and the step is
picked the way G's was, the most found on SMACS 0723.

Dev check, reported and not tested: score jw01727-o165_t131 and six ordinary calibration
visits with S against the same bank, and compare the ratio of o165's median score to the
six visits' median with G's ratio from the search (0.2208 / 0.0644 = 3.43, the six being
the first six calibration visits in the split after leaving out o165, o005 and o008). If
S's ratio isn't 1.72 or less (half of G's), S is dropped and nothing else is run for it.
If it is, S joins the MINERVA confirmation as a candidate.

## The Model Confirmation on AEGIS

MINERVA's 10 AEGIS visits (o024 to o033, all 1,890 s in F182M and 3,779 s in F250M and
F460M) are one depth, so they're the confirmation set, and with 10 visits a search there
could never name a lone tile anyway. The deep mosaics (o001 to o004) are for the search.
Every tile is F460M, F250M and F182M as red, green and blue.

- The 10 AEGIS visits and the two small UDS tiles (o050, o051) are cut with their own
  stretch first, a levels file is written from all 12, and all 12 are cut again with
  `--levels` and `--save_cutouts`. That's the only thing anyone looks at before the
  benches, and it's the records, not the tiles.
- M is G's recipe (noise 0.5, 2,000 steps, batch 12, seed 0) trained on the four control
  fields plus o050 and o051, and it's step 2,000 with no step pick, since G's pick was
  2,000 and SMACS has none of these filters.
- Each AEGIS visit is benched in bank mode against the other 9, with 5 seeds, AB 22 to
  28 and arcs, points, rings and streaks, the same as the PRIMER confirmation. The
  default DINOv3 is benched too, for the record.
- The candidates are S (only if it passed its dev check) and M, each against G. A
  candidate passes if its total found beats G's with the exact one-sided sign-flip test
  over the 10 visit differences at p <= 0.025 / (number of candidates). If more than one
  passes, the higher total is adopted. If none does, G stays. The
  above-every-ordinary-tile counts are reported and not tested.
- AEGIS lies in the EGS, where CEERS is, and CEERS1 tiles are in every candidate's
  training set, G's included. Any of those visits that shares sky with CEERS1 is said so
  in the results.

S failed its dev check at a ratio of 2.29 against the 1.72 it needed, so M was the only
candidate. Out of 420 planted sources a visit, G found 1,253 over the 10 visits, M found
1,249 and the default DINOv3 found 1,246. M beat G on 3 visits and lost on 6 (p = 0.67),
so G stays. It's hardly ahead of the default, though. The default beat it on 4 of the 10
visits and they end up 7 sources apart, so on MINERVA's filters the fine-tune is about
level with DINOv3 rather than better. o024 lies wholly on CEERS1, and it isn't where G
gains (137 there against 139 for the default).
