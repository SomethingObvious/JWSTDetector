// Writes a synthetic run bundle into web/public/run/ that follows bundle spec version 1.
// Everything is drawn from a seeded generator, so two runs give identical files.
//
//   node scripts/make-fixture.mjs [--writable]

import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { fits, fitsCard, png, sampleBilinear, stretcher } from './encode.mjs';
import { gauss, makeMosaic, pix2world, rand, tanCd } from './sky.mjs';

const OUT = fileURLToPath(new URL('../public/run/', import.meta.url));
const WRITABLE = process.argv.includes('--writable');

// Mosaic px per tile side.
const CROP = 320;
const STRIDE = 216;
// PNG side, what prep_data writes.
const TILE_PX = 256;
// 448 px at patch 16.
const GRID = 28;
const CUTOUTS_PER_MOSAIC = 40;

const MOSAICS = [
  {
    name: 'jw02736-o001_t001_nircam_clear-f444w',
    filters: ['F444W', 'F356W', 'F277W'],
    instrument: 'NIRCAM',
    rows: 3072,
    cols: 3328,
    scale: 0.063,
    theta: 20,
    crval: [110.8375, -73.4547],
    sky: [0.21, 0.17, 0.12],
    noise: [0.011, 0.009, 0.008],
    // Galaxies redden with distance, stars are blue-white here.
    channels: 3,
  },
  {
    name: 'jw02736-o009_t001_miri_f770w',
    filters: ['F770W'],
    instrument: 'MIRI',
    rows: 2816,
    cols: 3072,
    scale: 0.11,
    theta: -35,
    crval: [110.8296, -73.4602],
    sky: [6.4],
    noise: [0.09],
    channels: 1,
  },
];

function header(m, wcs) {
  const filt = m.filters.join(', ');
  return {
    SIMPLE: 'T',
    TELESCOP: 'JWST',
    INSTRUME: m.instrument,
    DETECTOR: m.instrument === 'MIRI' ? 'MIRIMAGE' : 'MULTIPLE',
    FILTER: m.filters[0],
    PUPIL: 'CLEAR',
    TARGNAME: 'SMACS J0723.3-7327',
    TARGPROP: 'SMACS0723',
    PROGRAM: '02736',
    OBSERVTN: m.instrument === 'MIRI' ? '009' : '001',
    VISIT: '001',
    'DATE-OBS': '2022-06-07',
    'TIME-OBS': m.instrument === 'MIRI' ? '06:12:08.110' : '04:41:42.352',
    EFFEXPTM: m.instrument === 'MIRI' ? '2775.168' : '7537.191',
    BUNIT: 'MJy/sr',
    PIXAR_SR: String((m.scale / 206264.806) ** 2),
    PIXAR_A2: String(m.scale ** 2),
    RADESYS: 'ICRS',
    CTYPE1: wcs.CTYPE1,
    CTYPE2: wcs.CTYPE2,
    CRPIX1: String(wcs.CRPIX1),
    CRPIX2: String(wcs.CRPIX2),
    CRVAL1: String(wcs.CRVAL1),
    CRVAL2: String(wcs.CRVAL2),
    CD1_1: wcs.CD1_1.toExponential(10),
    CD1_2: wcs.CD1_2.toExponential(10),
    CD2_1: wcs.CD2_1.toExponential(10),
    CD2_2: wcs.CD2_2.toExponential(10),
    NAXIS1: String(m.cols),
    NAXIS2: String(m.rows),
    CAL_VER: '1.14.0',
    CRDS_CTX: 'jwst_1241.pmap',
    COLOURS: filt,
    SYNTHETIC: 'T',
    COMMENT:
      'Synthetic fixture for the viewer, drawn from a seeded generator, not JWST data',
  };
}

// Preview at half scale, each pixel the mean of a 2 by 2 block.
function writePreview(m, img, stretch) {
  const pw = m.cols / 2;
  const ph = m.rows / 2;
  const prev = Buffer.alloc(pw * ph * m.channels);
  for (let y = 0; y < ph; y++) {
    for (let x = 0; x < pw; x++) {
      const i = 2 * y * m.cols + 2 * x;
      for (let c = 0; c < m.channels; c++) {
        const p = img[c];
        prev[(y * pw + x) * m.channels + c] = stretch(
          c,
          (p[i] + p[i + 1] + p[i + m.cols] + p[i + m.cols + 1]) / 4,
        );
      }
    }
  }
  writeFileSync(`${OUT}mosaics/${m.name}.png`, png(pw, ph, m.channels, prev));
}

// prep_data drops tiles that are mostly empty.
function mostlyEmpty(m, valid, x0, y0) {
  let good = 0;
  for (let y = y0; y < y0 + CROP; y += 8) {
    for (let x = x0; x < x0 + CROP; x += 8) {
      good += valid[y * m.cols + x];
    }
  }
  return good < (CROP / 8) ** 2 * 0.35;
}

// Mean oddness and empty fraction over one patch cell of a crop.
function cellStats(m, sky, x0, y0, r, c) {
  const cell = CROP / GRID;
  let o = 0;
  let empty = 0;
  let count = 0;
  for (let y = Math.floor(y0 + r * cell); y < y0 + (r + 1) * cell; y++) {
    for (let x = Math.floor(x0 + c * cell); x < x0 + (c + 1) * cell; x++) {
      const i = y * m.cols + x;
      o += sky.odd[i];
      empty += sky.valid[i] ? 0 : 1;
      count++;
    }
  }
  return { o: o / count, edge: empty / count };
}

function patchMaps(m, sky, x0, y0) {
  const map2 = new Float32Array(GRID * GRID);
  const map1 = new Float32Array(GRID * GRID);
  for (let r = 0; r < GRID; r++) {
    for (let c = 0; c < GRID; c++) {
      const { o, edge } = cellStats(m, sky, x0, y0, r, c);
      const base = 0.105 + 0.012 * gauss();
      map2[r * GRID + c] = Math.max(
        0.02,
        base + 0.42 * Math.tanh(o * 2.2) + 0.03 * edge,
      );
      map1[r * GRID + c] = Math.max(
        0.02,
        base + 0.015 * gauss() + 0.36 * Math.tanh(o * 2.0) + 0.16 * edge,
      );
    }
  }
  return { map1, map2 };
}

const top = (a) => {
  const s = Array.from(a).sort((p, q) => q - p);
  const k = Math.max(1, Math.round(s.length * 0.01));
  return s.slice(0, k).reduce((p, q) => p + q, 0) / k;
};

function peakOf(map) {
  let peak = 0;
  for (let i = 1; i < map.length; i++) {
    if (map[i] > map[peak]) {
      peak = i;
    }
  }
  return [Math.floor(peak / GRID), peak % GRID];
}

function mosaicTiles(m, sky, wcs, stretch) {
  const tiles = [];
  for (let y0 = 0; y0 + CROP <= m.rows; y0 += STRIDE) {
    for (let x0 = 0; x0 + CROP <= m.cols; x0 += STRIDE) {
      if (mostlyEmpty(m, sky.valid, x0, y0)) {
        continue;
      }
      const { map1, map2 } = patchMaps(m, sky, x0, y0);
      const [ra, dec] = pix2world(wcs, x0 + CROP / 2 - 0.5, y0 + CROP / 2 - 0.5);
      tiles.push({
        m,
        img: sky.img,
        stretch,
        x0,
        y0,
        ra,
        dec,
        map1,
        map2,
        score: top(map2),
        score_pass1: top(map1),
        peak: peakOf(map2),
      });
    }
  }
  return tiles;
}

// PNG tile, resampled from the crop to TILE_PX.
function tilePixels(t) {
  const { m } = t;
  const pix = Buffer.alloc(TILE_PX * TILE_PX * m.channels);
  for (let v = 0; v < TILE_PX; v++) {
    for (let u = 0; u < TILE_PX; u++) {
      const x = t.x0 + ((u + 0.5) * CROP) / TILE_PX - 0.5;
      const y = t.y0 + ((v + 0.5) * CROP) / TILE_PX - 0.5;
      for (let c = 0; c < m.channels; c++) {
        pix[(v * TILE_PX + u) * m.channels + c] = t.stretch(
          c,
          sampleBilinear(t.img[c], m.cols, m.rows, x, y),
        );
      }
    }
  }
  return pix;
}

function writeCutout(t, id, w) {
  const { m } = t;
  const planes = t.img.map((p) => {
    const a = new Float32Array(CROP * CROP);
    for (let y = 0; y < CROP; y++) {
      a.set(
        p.subarray((t.y0 + y) * m.cols + t.x0, (t.y0 + y) * m.cols + t.x0 + CROP),
        y * CROP,
      );
    }
    return a;
  });
  // Once the rows are flipped, cutout FITS pixel `(i, j)` is mosaic pixel `(x0 + i, y0 + CROP - j)`.
  const cards = [
    fitsCard('SIMPLE', true),
    fitsCard('BITPIX', -32),
    fitsCard('NAXIS', m.channels === 3 ? 3 : 2),
    fitsCard('NAXIS1', CROP),
    fitsCard('NAXIS2', CROP),
    ...(m.channels === 3 ? [fitsCard('NAXIS3', 3, 'planes R, G, B')] : []),
    fitsCard('BUNIT', 'MJy/sr'),
    fitsCard('RADESYS', 'ICRS'),
    fitsCard('CTYPE1', 'RA---TAN'),
    fitsCard('CTYPE2', 'DEC--TAN'),
    fitsCard('CRPIX1', w.CRPIX1 - t.x0),
    fitsCard('CRPIX2', t.y0 + CROP + 1 - w.CRPIX2),
    fitsCard('CRVAL1', w.CRVAL1),
    fitsCard('CRVAL2', w.CRVAL2),
    fitsCard('CD1_1', w.CD1_1),
    fitsCard('CD1_2', -w.CD1_2),
    fitsCard('CD2_1', w.CD2_1),
    fitsCard('CD2_2', -w.CD2_2),
    fitsCard('FILTERS', m.filters.join(',')),
    fitsCard('MOSAIC', m.name.slice(0, 68)),
    fitsCard('X0', t.x0, 'crop origin in mosaic px'),
    fitsCard('Y0', t.y0, 'crop origin in mosaic px'),
    fitsCard('SYNTHET', true, 'not JWST data'),
  ];
  writeFileSync(`${OUT}cutouts/${id}.fits`, fits(cards, planes, CROP, CROP));
}

function tileKey(t, i) {
  const cx = t.x0 + CROP / 2;
  const cy = t.y0 + CROP / 2;
  const decTag = (t.dec < 0 ? 'm' : 'p') + Math.abs(t.dec).toFixed(2);
  return `query/${t.m.name}_cx${String(cx).padStart(6, '0')}_cy${String(cy).padStart(6, '0')}_ra${t.ra.toFixed(2)}_dec${decTag}_i${String(i).padStart(8, '0')}.png`;
}

function drawMosaics() {
  const mosaics = {};
  const tiles = [];
  for (const m of MOSAICS) {
    process.stdout.write(`drawing ${m.name}\n`);
    const sky = makeMosaic(m);
    const wcs = {
      CTYPE1: 'RA---TAN',
      CTYPE2: 'DEC--TAN',
      CRPIX1: m.cols / 2 + 0.5,
      CRPIX2: m.rows / 2 + 0.5,
      CRVAL1: m.crval[0],
      CRVAL2: m.crval[1],
      ...tanCd(m.scale, m.theta),
    };
    const stretch = stretcher(m);
    writePreview(m, sky.img, stretch);
    mosaics[m.name] = {
      shape: [m.rows, m.cols],
      preview: `mosaics/${m.name}.png`,
      preview_scale: 0.5,
      filters: m.filters,
      pixel_scale_arcsec: m.scale,
      wcs,
      header: header(m, wcs),
    };
    tiles.push(...mosaicTiles(m, sky, wcs, stretch));
  }
  return { mosaics, tiles };
}

function writeTiles(tiles, mosaics, refs) {
  const cutoutCount = Object.fromEntries(MOSAICS.map((m) => [m.name, 0]));
  return tiles.map((t, i) => {
    const id = String(i).padStart(6, '0');
    const { m } = t;
    writeFileSync(
      `${OUT}tiles/${id}.png`,
      png(TILE_PX, TILE_PX, m.channels, tilePixels(t)),
    );
    writeFileSync(`${OUT}maps/${id}.f32`, Buffer.from(t.map2.buffer));
    writeFileSync(`${OUT}maps1/${id}.f32`, Buffer.from(t.map1.buffer));

    const hasCutout = cutoutCount[m.name] < CUTOUTS_PER_MOSAIC;
    if (hasCutout) {
      cutoutCount[m.name]++;
      writeCutout(t, id, mosaics[m.name].wcs);
    }

    return {
      key: tileKey(t, i),
      rank: i + 1,
      score: Number(t.score.toFixed(5)),
      score_pass1: Number(t.score_pass1.toFixed(5)),
      in_reference: refs.has(i),
      mosaic: m.name,
      x0: t.x0,
      y0: t.y0,
      size: CROP,
      ra: Number(t.ra.toFixed(6)),
      dec: Number(t.dec.toFixed(6)),
      peak: t.peak,
      has_cutout: hasCutout,
    };
  });
}

function runInfo(refs) {
  return {
    name: 'synthetic_smacs0723_fixture',
    model: 'dinov3-vitl16-pretrain-sat493m',
    detector: 'knn',
    resolution: 448,
    patch_size: 16,
    grid: [GRID, GRID],
    score_mode: 'top1p',
    passes: 2,
    n_reference: refs.size,
    created: '2026-11-03T05:00:00Z',
    args: {
      backbone: 'dinov3-vitl16-pretrain-sat493m',
      resolution: 448,
      k: 1,
      score_mode: 'top1p',
      passes: 2,
      ref_fraction: 0.4,
      seed: 0,
      note: 'synthetic fixture, not a real run',
    },
  };
}

// A handful of saved verdicts so the index shows its glyphs.
function writeVerdicts(out) {
  const kinds = ['interesting', 'artifact', 'known', 'not_anomalous'];
  const verdicts = {};
  for (let i = 0; i < Math.min(14, out.length); i++) {
    if (i === 0 || i === 2) {
      continue;
    }
    verdicts[out[i].key] = {
      verdict: kinds[i % 4],
      star: i % 5 === 1,
      note: i === 1 ? 'Arc segment, check against the lens model.' : '',
      updated: '2026-11-03T05:10:00Z',
    };
  }
  writeFileSync(`${OUT}verdicts.json`, JSON.stringify(verdicts, null, 1));
}

function main() {
  rmSync(OUT, { recursive: true, force: true });
  for (const d of ['tiles', 'cutouts', 'maps', 'maps1', 'mosaics']) {
    mkdirSync(OUT + d, { recursive: true });
  }

  const { mosaics, tiles } = drawMosaics();
  tiles.sort((a, b) => b.score - a.score);
  // Pass 2 takes its references from the quiet half of pass 1.
  const quiet = tiles.map((t, i) => [t.score_pass1, i]).sort((a, b) => a[0] - b[0]);
  const refs = new Set(
    quiet
      .slice(0, Math.round(tiles.length * 0.4))
      .filter(() => rand() < 0.85)
      .map((q) => q[1]),
  );

  const out = writeTiles(tiles, mosaics, refs);
  const index = {
    version: 1,
    writable: WRITABLE,
    run: runInfo(refs),
    mosaics,
    tiles: out,
  };
  writeFileSync(`${OUT}index.json`, JSON.stringify(index));
  writeVerdicts(out);
  process.stdout.write(`wrote ${out.length} tiles to ${OUT}\n`);
}

main();
