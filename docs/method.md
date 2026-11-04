# Method

## Tiles and Features

Prep cuts each mosaic into square tiles, overlapping by half a tile by default, and renders them with one asinh or Lupton stretch per mosaic, so tiles from the same mosaic are comparable. A DINOv3 ViT-L/16 (the SAT-493M weights, trained on satellite imagery) turns each tile at 448 px into a 28 by 28 grid of patch features $f_p \in \mathbb{R}^{1024}$.

## Even Noise

A mosaic isn't equally deep everywhere. On SMACS 0723 some sky along the edges and in the gap between NIRCam's two modules was covered by one exposure where most of it had three, so its noise is about 1.7 times higher, and that alone put tiles reaching into those strips in the top four places, each one peaking on blank sky.

So when there's a weight map, prep scales each pixel's flux above the sky by

```math
\sqrt{\min\left(\frac{w}{\tilde w}, 1\right)}
```

in each filter, where $w$ is the pixel's weight and $\tilde w$ the mosaic's median weight. That is how a detection image is weighted. The sky noise comes out the same across the mosaic, and a source in shallow sky looks only as strong as its signal to noise. Pixels with no weight, like a saturated star's core, are left alone. The FITS cutouts keep the real flux, and the factors go in a `NOISEEQ` extension so the benchmark can render a planted source the same way.

The 57 tiles that reach into sky under half the median weight are 20% of the field, and now they hold 2 of kNN's top 20 places and 6% of its top 50. That is fewer than their share, which seems right, as a faint source in shallow sky is less certain.

## Patch Distance

The kNN detector, after AnomalyDINO (Damm et al. 2025), keeps every patch of the reference tiles in a memory bank $\mathcal{M}$ and scores a patch by the cosine distance to its nearest neighbour there:

```math
d(p) = \min_{m \in \mathcal{M} \setminus \mathcal{X}(t)} \left(1 - \frac{f_p \cdot f_m}{\lVert f_p \rVert \, \lVert f_m \rVert}\right)
```

$\mathcal{X}(t)$ is every reference patch from tile $t$ itself or from a tile that overlaps it on the sky, taken from the footprints in `manifest.csv`. Without that exclusion an overlapping neighbour holds the same object and the object matches itself, which hides exactly what the detector is for.

The subspace detector, after SubspaceAD (Lendering et al. 2026), fits a PCA to the reference patches and keeps the top $r$ directions $U_r$ that hold `--subspace_variance` (99%) of the variance. A patch scores its squared residual off that subspace:

```math
d(p) = \lVert (I - U_r U_r^\top)(f_p - \mu) \rVert^2
```

## Tile Score

A tile's score is the mean of the top 1% of the patch distances it scores (the next section says which), as in AnomalyDINO. That's its 2 worst patches for an interior tile and up to 8 for one scoring its whole grid, and the alternatives in `--score_mode` trade between catching one small oddity and catching a tile that is odd all over.

## Which Patches a Tile Scores

Tiles overlap by half, so most of the sky is in four of them. An object cut off by one tile's edge looks odd to the backbone however ordinary it is, and with every patch scored, the outermost ring of patches held the peak of 45% of kNN's top 20 tiles on SMACS 0723, against 14% by chance. Skipping a ring or two only moved the excess inward, since the patches next to a cut-off object are raised as well.

So each patch only counts towards the tile whose centre it is nearest, measured as the larger of the x and y offsets. That splits every overlap into square cells, hands each bit of sky to exactly one tile and drops none, and works the same for tiles flush with the far edge of a mosaic. With half-tile overlap an interior tile scores its central 14 by 14 patches, while its map still shows all 28 by 28, and the viewer draws the scored part as a dashed outline. `--score_region all` scores every patch the old way.

Scoring that way, 3 of kNN's top 20 tiles peak on the outer ring of the part they score, where chance would put 5, and over all 288 tiles it's 25% against 23%.

## The Bootstrap

There are no labels, so the reference has to come from the tiles themselves. Pass 1 draws a random 10% of them, clamped to between 500 and 5,000 and to at most half the set. Pass 2 rebuilds the reference from the lowest-scoring 20% of pass 1 and scores everything again. The overlap exclusion means an object in the random reference can't match itself, but a second object like it elsewhere in the field would match it and score low, which [the benchmark](benchmark.md#why-knn-finds-faint-arcs-more-often-than-bright-ones) measures. Pass 2's reference is the most ordinary tiles, which keeps rare things out of it. The viewer shows both passes and the change between them.

That pruning doesn't know a rare object from a rare kind of ordinary sky, though. Before the noise was evened out, two tiles in the gap between the modules ranked 216th and 175th in pass 1 and 2nd and 3rd in pass 2, because pass 2 had dropped every tile like them from its reference. So anything systematic and rare, like depth, has to be dealt with before scoring rather than left to the bootstrap.

## Twins in the Reference

`--places N` scores a patch by its distance to the nearest match in the N-th separate place in the bank instead of the nearest match anywhere, where two places are separate when their tiles share no sky:

```math
d_N(p) = \min_{m \in \mathcal{M}_N(p)} \left(1 - \cos(f_p, f_m)\right)
```

with $\mathcal{M}_N(p)$ the bank patches left once the closest matches from $N - 1$ separate places are set aside. Ordinary sky has close matches all over the field, so its score hardly moves. A rare object with a twin or two in the reference can only fill some of the places, so it keeps a high score. It's off by default, as [the benchmark](benchmark.md#why-knn-finds-faint-arcs-more-often-than-bright-ones) found no gain on the seeds kept back to check it.

## Another Field as Reference

`--reference_roots` takes the reference from other fields cut the same way, which keeps a rare object's twins in the target field out of it entirely and gives each tile a p-value. Each reference field is scored against the other reference fields, and those scores are the calibration set. A tile with score $s$ then gets the conformal p-value

```math
p = \frac{1 + \left|\{ j : c_j \ge s \}\right|}{n + 1}
```

over the $n$ calibration scores $c_j$ (Bates et al. 2023), and Benjamini-Hochberg turns those into q-values, so the tiles with $q \le 0.1$ are a set where about a tenth or fewer are expected to be ordinary. It is valid when the calibration tiles are exchangeable with the ordinary tiles of the target. The target is scored against every reference field and each calibration field against one fewer, which only lowers the target's scores, so the p-values err high.

On SMACS 0723 against Abell 370, MACS J0416, CEERS and GLASS, no tile reaches $q \le 0.1$. The lowest p is 0.0085, as the arcs and bright stars of the other three clusters score as high as SMACS's do, and with 1,056 calibration tiles p can't go below 0.00095 anyway. The top tile in that ranking is blank sky with faint horizontal striping, whose row medians scatter by about 1 sigma there against 0.2 in a typical tile. All five mosaics went through the same pipeline version, so it's something in the SMACS data itself that the other fields don't share. That is the catch with another field as reference, as anything that differs between the fields ranks ahead of what is rare within one, so the reference fields need to match the target in instrument, filters, depth and reduction as closely as they can.

## Precision and Speed

The backbone runs in fp16 when a probe tile's features come out finite and in bf16 when they don't, which is the case for the LVD-1689M checkpoints on every input. The memory bank is kept in fp16 with fp32 results, normalised first, and on SMACS 0723 that gives the same ranking as an fp32 bank (Spearman 1.00000, the same top 50) and the same recall, in two thirds of the time. The search is bound by memory rather than arithmetic, and against a 500,000 patch bank it scores 17.4 tiles a second on a laptop RTX 4070, so 50,000 tiles against a million patches take about an hour and three quarters. A backbone too big for the GPU, like the 7B, streams its layers from disk two at a time.

## Tile Descriptors

Each tile also gets a descriptor, the mean feature of its top 1% patches:

```math
g_t = \frac{1}{|P_t|} \sum_{p \in P_t} f_p
```

where $P_t$ is the top 1% by distance of the patches the tile scores, whatever `--score_mode` is. It describes what made the tile anomalous rather than the whole tile, so two tiles with the same kind of diffraction spike end up close even when the rest of each field differs. The viewer gets them reduced by PCA to 32 dimensions and scaled to unit length, which keeps a run of 100,000 tiles at about 13 MB.

## Cross-Matching

The anomaly is usually one small thing in a 20 arcsecond tile, so the match uses the sky position of the most anomalous patch, not the tile centre. For a $G \times G$ grid over a tile of side $s$ pixels at mosaic offset $(x_0, y_0)$, the patch in PNG row $r$ and column $c$ is centred at

```math
x = x_0 + \frac{(c + \tfrac{1}{2})\,s}{G} - \tfrac{1}{2}, \qquad y = y_0 + s - \frac{(r + \tfrac{1}{2})\,s}{G} - \tfrac{1}{2}
```

in zero-based FITS pixels, since PNG rows run down and FITS rows run up. The mosaic WCS turns that into RA and Dec, and astropy's `match_to_catalog_sky` finds the nearest SIMBAD, NED and Gaia DR3 source. A tile is uncatalogued when none of them lies within 1.5 arcseconds, about two patches.

## Learning From Verdicts

Every verdict says how much a tile was worth looking at, with interesting as 1, a known object as 0.5, and an artifact or not anomalous as 0. The viewer spreads those values over descriptor space by Nadaraya-Watson kernel regression with a prior of 0.5:

```math
\hat v(t) = \frac{\tfrac{1}{2} w_0 + \sum_j k_{tj} \, v_j}{w_0 + \sum_j k_{tj}}, \qquad k_{tj} = \exp\left(\frac{\cos(g_t, g_j) - 1}{\tau}\right)
```

with $w_0 = 0.5$ and $\tau = 0.15$. On SMACS 0723 a random pair of tiles sits near cosine 0 and the median nearest neighbour at 0.64, so a typical nearest neighbour counts about a tenth as much as an identical tile and an unrelated one counts for almost nothing. Sorting by Learned then ranks tiles by

```math
\text{priority}(t) = \left(1 - \frac{\text{rank}(t) - 1}{N}\right) \cdot 2\,\hat v(t)
```

which is the anomaly ranking unchanged until there are verdicts. It runs in the browser on every verdict, at a cost of the tile count times the labelled count.

On SMACS 0723, ranks 1 and 12 of kNN's ranking are neighbouring tiles that peak on the same horizontal diffraction spike, which comes out as a rainbow because the PSF widens with wavelength and the three filters separate. Ranks 45 and 123 peak on fainter spikes like it. Each of the four has the other three among its four nearest descriptors, at cosines from 0.52 to 0.71, and marking rank 12 alone as an artifact takes rank 1 out of the top 20 as well, while the other 18 stay in it.

This is the same idea as the active learning in Astronomaly (Lochner and Bassett 2021), which weighs a learned relevance score against the anomaly score. The differences here are the self-supervised patch features in place of hand-built ones, the descriptor built from the anomalous patches only, and the whole loop running in the viewer with no server.

## What Each Part Assumes

The score is relative to the reference, so an artifact shared by most tiles counts as normal. The bootstrap assumes anomalies are rare, which holds for real objects and doesn't for systematic ones like a detector stripe that crosses every tile. Uncatalogued means none of the three catalogues lists a source there, and on a JWST deep field that mostly means a galaxy too faint for them, which is why it's a filter to combine with verdicts rather than a claim of discovery. Even noise takes the weight map to be inverse variance or exposure time, which is what the JWST and HST pipelines write.

## References

- Damm, Laszkiewicz, Lederer, Fischer, Weiss. AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2. WACV 2025.
- Lendering et al. SubspaceAD. arXiv:2602.23013, 2026.
- Siméoni et al. DINOv3. arXiv:2508.10104, 2025.
- Lochner, Bassett. Astronomaly: Personalised Active Anomaly Detection in Astronomical Data. Astronomy and Computing 36, 100481, 2021.
- Lupton et al. Preparing Red-Green-Blue Images from CCD Data. PASP 116, 133, 2004.
- Bates, Candès, Lei, Romano, Sesia. Testing for Outliers with Conformal p-values. Annals of Statistics 51, 149, 2023.
- Benjamini, Hochberg. Controlling the False Discovery Rate. JRSS B 57, 289, 1995.
