# JWSTDetector

Unsupervised anomaly detection on JWST imagery. It cuts FITS mosaics into PNG tiles, embeds every
tile with a DINO backbone, and ranks the tiles by how far they sit from a reference set of ordinary
ones. The output is a sorted list for a person to look through: artifacts, rare morphologies,
whatever the reference set does not account for. The detector started as AnomalyDINO and was adapted
for the query-only case, where nothing is labelled.

## How it works

Both runners share one detector in `src/detection.py`. It builds a memory bank of patch embeddings
from the reference tiles, then scores each query tile by the distance from its patches to their
nearest neighbours in that bank.

`run_query_bootstrap.py` needs no labels and is the entry point for JWST data. Pass 1 picks a random
reference subset out of the query set and scores everything against it. Pass 2 throws that reference
away, rebuilds it from the lowest-scoring tiles, and scores again. The assumption is that most of
the sky is unremarkable, so the bottom of the first ranking is a decent stand-in for normal.

`run_anomalydino.py` is the classic setup: a `train/` folder you have already decided is clean, and
a `test/` folder to score.

## Install

```
pip install -e .
pip install -e '.[viewer]'   # only if you want the desktop viewer window
```

Python 3.11 or newer. A CUDA GPU is strongly recommended but no longer required: `--device` accepts
`cpu`, `cuda`, `cuda:1`, `mps`, or `auto`.

DINOv3 weights are gated on the Hugging Face Hub. Accept the licence on the model page, then run
`hf auth login` once. DINOv2 goes through `torch.hub` instead and needs network access on first use
to clone the upstream repo.

## Which backbone

The default is `dinov3-vitl16-pretrain-sat493m`. That checkpoint was pretrained on 493 million
satellite images rather than web photos, and a JWST mosaic tile has far more in common with an
overhead scene than with the object-centred pictures DINOv2 saw. DINOv3 also holds its dense
features together over long training runs through Gram anchoring, and dense patch quality is the
only thing this method actually consumes.

Other options that work:

| Name | Where it comes from | Notes |
|---|---|---|
| `dinov3-vitl16-pretrain-sat493m` | transformers | default, satellite pretraining |
| `dinov3-vitl16-pretrain-lvd1689m` | transformers | same size, web pretraining |
| `dinov3-vits16-pretrain-lvd1689m` | transformers | small and quick, good for a first pass |
| `dinov2_vitl14`, `dinov2_vits14` | torch.hub | the original AnomalyDINO backbones |

Any `facebook/...` repo id also works verbatim. `--resolution` is snapped down to a multiple of the
patch size, so 672 gives a 48x48 grid on DINOv2 and 42x42 on DINOv3.

## Usage

### 1. FITS to PNG tiles

```
python prep_data.py --sensor miri --indir rawfits --out datasets/miri \
  --pattern "*sci.fits" --sci_ext SCI --wht_ext WHT \
  --tile_size 672 --upscale 1 --stride 644 --min_wht_frac 0.5 \
  --p_lo 1 --p_hi 99.8 --asinh_q 10.0 --wcs_in_name
```

Tiles land in `datasets/miri/query/`. MIRI expects paired `*sci.fits` and `*wht.fits` files; the
weight map drops tiles with too little coverage. NIRCam mode (`--sensor nircam`) builds RGB tiles
from filter mosaics instead. `--wcs_in_name` puts the RA and Dec of each tile centre in its
filename, which is how you find a candidate again once it has been ranked.

Two knobs decide what gets thrown away:

- `--min_wht_frac` drops tiles whose exposure coverage is too thin.
- `--source_sigma` and `--min_source_frac` drop empty sky. A pixel counts as signal when it sits
  `source_sigma` above the mosaic's own noise, measured as a median absolute deviation, and a tile
  is kept when at least `min_source_frac` of its pixels qualify. The defaults, 5 sigma and 2e-4, keep
  a tile holding a source a few tens of pixels across and drop blank fields.

The stretch is measured once per mosaic by default (`--stretch_scope mosaic`), so two tiles that
look equally bright really are equally bright. This matters more than it sounds: the detector is
comparing tiles against each other, and a per-tile stretch rescales each one independently, which
throws away exactly the difference you were trying to find. Set `--stretch_scope tile` for the old
behaviour. One very bright object can dominate the mosaic percentiles, so check the `sky`, `sigma`
and `stretch` line each mosaic logs and lower `--p_hi` if the range looks wrong.

`--dry-run` does all the cutting and filtering and writes nothing, which is how you tune the
thresholds without touching disk. `--limit N` stops after N mosaics.

### 2. Score the tiles

```
python run_query_bootstrap.py --data_root datasets/miri --query_subdir query \
  --model_name dinov3-vitl16-pretrain-sat493m --resolution 672 \
  --batch_size 16 --num_workers 8 --k_neighbors 20 --rotation \
  --device cuda:0 --seed 0 \
  --init_ref_frac 0.2 --init_ref_min 100 --init_ref_max 425 \
  --bootstrap_keep_frac 0.1 --bootstrap_keep_min 100 --bootstrap_keep_max 425 \
  --out_dir results_query_only --tag miri_672
```

This writes `pass1/` and `pass2/` under the output directory, copies the final scores to
`measurements_final.csv`, and saves a patch-distance map per tile under `anomaly_maps/seed=<n>/` as
`.npy`, plus `.tiff` with `--save_tiffs`. Each `measurements.csv` has the columns `Sample`,
`AnomalyScore`, `MemoryBankTimeSec` and `InferenceTimeSec`.

Knobs worth knowing:

- `--batch_size` and `--num_workers`. Tiles go through the backbone in batches now, with the PNG
  decoding in worker processes. Raise both until the GPU is the bottleneck.
- `--max_bank_patches` caps the memory bank at a million patches by default, roughly 4 GB of float32
  at 1024 dimensions. Above the cap, patches are dropped at a fixed rate as they arrive, so the peak
  memory stays flat no matter how large the reference set is. Random sampling is used rather than
  greedy coreset selection, which is the accuracy-per-patch upgrade if you ever need it.
- `--score_mode` decides how one tile's patch distances collapse to one number. `top1p` is the
  AnomalyDINO default and a reasonable place to start. `peak_minus_median` measures how far the
  worst patch stands above the rest of its own tile, which suits a small bright oddity on otherwise
  ordinary sky. `max` and `quantile` are more sensitive and noisier.
- `--rotation` adds the eight square symmetries of each reference tile to the bank. They are exact
  rotations and reflections, so nothing gets resampled and no reflected border creeps into the
  corners.
- `--masking` scores only the patches that carry signal, using the first principal component of the
  patch embeddings. The cut is set in standard deviations of that component, so it transfers between
  backbones.
- `--seed` fixes the reference draw and seeds `random`, NumPy and torch. `--deterministic` also puts
  cuDNN in deterministic mode, which is slower and off by default.

### 3. Summarize and view

```
python summarize_results.py --results_root results_query_only_miri_672 --outdir results_summary

python viewer.py \
  --maps_dir results_query_only_miri_672/pass2/anomaly_maps/seed=0 \
  --png_dir datasets/miri/query \
  --csv_path results_summary/results_query_only_miri_672/samples_sorted_by_anomaly_score.csv
```

`summarize_results.py` gathers every run under `--results_root`, meaning any folder holding an
`args.yaml`, into summary CSVs including `samples_sorted_by_anomaly_score.csv`. `viewer.py` walks
the ranked tiles showing each PNG next to its patch-distance map. Enter for the next one, `q` to
quit.

`unzipper.py <dir>` expands gzipped FITS in a directory, which COSMOS-Web weight mosaics need.

## Layout

- `prep_data.py`, FITS mosaics to PNG tiles for MIRI and NIRCam.
- `run_query_bootstrap.py` and `run_anomalydino.py`, the two scoring entry points.
- `summarize_results.py`, run directories to summary CSVs.
- `viewer.py`, step through ranked tiles and their anomaly maps.
- `src/backbones.py`, model loading and background masking.
- `src/detection.py`, the memory bank, the nearest-neighbour search and the scoring loop.
- `src/scoring.py`, patch distances to one number per tile.
- `src/data.py`, tile listing, loading and result IO.
- `src/seeding.py`, reproducibility.

## Tests

```
python test_helpers.py     # numpy and astropy only, no GPU stack
python test_detection.py   # needs torch, runs on CPU with a stand-in backbone
```

`test_helpers.py` covers the tile maths, the emptiness cut and the scoring rules.
`test_detection.py` runs the memory bank, the search, masking, augmentation and map writing without
downloading any weights, by swapping in a small deterministic module in place of DINO.

## Limitations

- Scores are relative to the reference set you build. If the query set is mostly artifacts then
  artifacts are what "normal" means, so tune `--init_ref_*` and `--bootstrap_keep_*` to your data.
- There is no ground-truth evaluation. This fork drops the MVTec and VisA metric code because JWST
  tiles carry no labels. The ranking is for a person to read.
- `prep_data.py` assumes specific FITS layouts: MIRI `*sci.fits` and `*wht.fits` pairs, and a NIRCam
  filename pattern with filter and tile fields.
- The mosaic-wide stretch is more comparable across tiles but more exposed to a single very bright
  object skewing the percentiles. The per-mosaic log line tells you when that happened.

## What changed in 0.2

The empty-sky filter was measuring brightness on the stretched PNG, after a per-tile stretch had
already normalised blank sky up to full brightness. At the thresholds in `instructions.txt` it kept
empty tiles and discarded tiles containing a bright source, which is backwards. It now reads raw
flux against the mosaic's noise. `--mean_thresh` and `--var_thresh` are gone and the script says so
if you pass them.

Also in this release: DINOv3 backbones including the satellite checkpoints; batched inference
through a DataLoader instead of one tile at a time; a capped memory bank; nearest-neighbour search
in torch, which retires FAISS along with the CPU and GPU build mismatch it came with; `--score_mode`
wired up to the CLI, where it previously existed in the detector but could not be reached; a
background-mask threshold that transfers between backbones; and exact dihedral augmentation in place
of interpolated rotation. `scikit-learn`, `opencv-python` and `faiss-cpu` are no longer needed.
`torchvision` and `scipy` were imported but never declared, so a clean install used to fail on
import; dependencies now live in `pyproject.toml`.

## Credit

The detector and the DINO wrappers are adapted from
[AnomalyDINO](https://github.com/dammsi/AnomalyDINO) (Damm et al., WACV 2025). Backbones are
[DINOv2](https://github.com/facebookresearch/dinov2) and
[DINOv3](https://github.com/facebookresearch/dinov3) from Meta AI.
