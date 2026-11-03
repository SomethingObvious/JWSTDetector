// Writes a synthetic run bundle into web/public/run/ that follows bundle spec version 1.
// Everything is drawn from a seeded generator, so two runs give identical files.
//
//   node scripts/make-fixture.mjs [--writable]

import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const OUT = fileURLToPath(new URL('../public/run/', import.meta.url));
const WRITABLE = process.argv.includes('--writable');

const CROP = 320; // mosaic px per tile side
const STRIDE = 216;
const TILE_PX = 256; // PNG side, what prep_data writes
const GRID = 28; // 448 px at patch 16
const CUTOUTS_PER_MOSAIC = 40;

let seed = 863642;
function rand() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
function gauss() {
  return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}
const between = (a, b) => a + (b - a) * rand();

// PNG writer

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}
function png(width, height, channels, pixels) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = channels === 3 ? 2 : 0;
  const stride = width * channels;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) pixels.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

// FITS writer

function fitsCard(key, value, comment = '') {
  let v;
  if (typeof value === 'string') v = `'${value.replace(/'/g, "''").padEnd(8)}'`.padEnd(20);
  else if (typeof value === 'boolean') v = (value ? 'T' : 'F').padStart(20);
  else if (Number.isInteger(value)) v = String(value).padStart(20);
  else v = value.toExponential(12).toUpperCase().padStart(20);
  const card = `${key.padEnd(8)}= ${v}${comment ? ' / ' + comment : ''}`;
  return card.slice(0, 80).padEnd(80);
}
function fits(cards, planes, width, height) {
  const head = [...cards, 'END'.padEnd(80)].join('');
  const headLen = Math.ceil(head.length / 2880) * 2880;
  const dataBytes = planes.length * width * height * 4;
  const buf = Buffer.alloc(headLen + Math.ceil(dataBytes / 2880) * 2880);
  buf.fill(32, 0, headLen);
  buf.write(head, 0, 'ascii');
  let o = headLen;
  for (const plane of planes) {
    // FITS row 1 is the bottom, so the last displayed row goes first.
    for (let row = height - 1; row >= 0; row--) {
      for (let x = 0; x < width; x++) {
        buf.writeFloatBE(plane[row * width + x], o);
        o += 4;
      }
    }
  }
  return buf;
}

// The sky

const D2R = Math.PI / 180;

function tanCd(scaleArcsec, thetaDeg) {
  const s = scaleArcsec / 3600;
  const c = Math.cos(thetaDeg * D2R);
  const n = Math.sin(thetaDeg * D2R);
  // Screen rows run down and north is up, rotated by theta.
  return { CD1_1: -s * c, CD1_2: s * n, CD2_1: -s * n, CD2_2: -s * c };
}

function pix2world(w, x, y) {
  const dx = x + 1 - w.CRPIX1;
  const dy = y + 1 - w.CRPIX2;
  const xi = (w.CD1_1 * dx + w.CD1_2 * dy) * D2R;
  const eta = (w.CD2_1 * dx + w.CD2_2 * dy) * D2R;
  const d0 = w.CRVAL2 * D2R;
  const den = Math.cos(d0) - eta * Math.sin(d0);
  const ra = w.CRVAL1 + Math.atan2(xi, den) / D2R;
  const dec = Math.atan2(eta * Math.cos(d0) + Math.sin(d0), Math.hypot(xi, den)) / D2R;
  return [(ra + 360) % 360, dec];
}

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

function makeMosaic(m) {
  const { rows, cols, channels } = m;
  const n = rows * cols;
  const img = Array.from({ length: channels }, () => new Float32Array(n));
  const odd = new Float32Array(n);
  const valid = new Uint8Array(n);

  // The footprint is a rotated rectangle with ragged corners, like a real drizzled mosaic.
  const cx = cols / 2;
  const cy = rows / 2;
  const fa = (m.theta * 0.35 + 6) * D2R;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const u = (x - cx) * Math.cos(fa) + (y - cy) * Math.sin(fa);
      const v = -(x - cx) * Math.sin(fa) + (y - cy) * Math.cos(fa);
      const inside = Math.abs(u) < cols * 0.43 && Math.abs(v) < rows * 0.42 && Math.abs(u) + Math.abs(v) < (cols + rows) * 0.39;
      if (!inside) continue;
      const i = y * cols + x;
      valid[i] = 1;
      for (let c = 0; c < channels; c++) {
        const grad = 1 + 0.06 * ((x / cols) - 0.5) - 0.04 * ((y / rows) - 0.5);
        img[c][i] = m.sky[c] * grad + m.noise[c] * gauss();
      }
    }
  }

  const add = (c, x, y, v) => {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    img[c][y * cols + x] += v;
  };
  const addOdd = (x, y, v) => {
    if (x < 0 || y < 0 || x >= cols || y >= rows) return;
    odd[y * cols + x] += v;
  };

  // A source with an elliptical exponential or de Vaucouleurs profile.
  function galaxy(gx, gy, flux, re, q, pa, colour, oddWeight, sersic = 1) {
    const r = Math.min(320, re * (sersic > 2 ? 9 : 6));
    const cp = Math.cos(pa);
    const sp = Math.sin(pa);
    const k = sersic > 2 ? 7.67 : 1.68;
    // A de Vaucouleurs core is about 2000 times its value at re, so the amplitude is brought down to match.
    const amp = sersic > 2 ? flux * 0.06 : flux;
    for (let y = Math.floor(gy - r); y <= gy + r; y++) {
      for (let x = Math.floor(gx - r); x <= gx + r; x++) {
        const dx = x - gx;
        const dy = y - gy;
        const a = dx * cp + dy * sp;
        const b = (-dx * sp + dy * cp) / q;
        const rr = Math.hypot(a, b) / re;
        if (rr > r / re) continue;
        const prof = amp * Math.exp(-k * (rr ** (1 / sersic) - 1));
        for (let c = 0; c < channels; c++) add(c, x, y, prof * colour[c]);
        if (oddWeight) addOdd(x, y, oddWeight * Math.min(1, prof / amp));
      }
    }
  }

  function star(sx, sy, flux, colour, spikes, oddWeight) {
    const r = spikes ? 110 : 8;
    const pa = (m.theta + 30) * D2R;
    for (let y = Math.floor(sy - r); y <= sy + r; y++) {
      for (let x = Math.floor(sx - r); x <= sx + r; x++) {
        const dx = x - sx;
        const dy = y - sy;
        const d = Math.hypot(dx, dy);
        let v = flux * Math.exp(-(d * d) / (2 * 1.3 * 1.3)) + flux * 0.04 * Math.exp(-d / 4);
        if (spikes) {
          // JWST's hexagonal mirror gives six bright spikes and two faint ones.
          for (let s = 0; s < 4; s++) {
            const ang = pa + (s < 3 ? (s * Math.PI) / 3 : Math.PI / 2);
            const along = dx * Math.cos(ang) + dy * Math.sin(ang);
            const perp = -dx * Math.sin(ang) + dy * Math.cos(ang);
            const amp = s < 3 ? 1 : 0.35;
            v += amp * flux * 0.05 * Math.exp(-(perp * perp) / 0.7) * (Math.abs(along) + 3) ** -1.1;
          }
        }
        for (let c = 0; c < channels; c++) add(c, x, y, v * colour[c]);
        if (oddWeight && v > flux * 0.002) addOdd(x, y, oddWeight);
      }
    }
  }

  const col = (red) => (channels === 3 ? [1 + red, 0.85, 0.75 - red * 0.5] : [1]);
  const unit = channels === 3 ? 1 : 9; // MIRI surface brightness runs higher
  const ccx = cols * 0.52;
  const ccy = rows * 0.47;

  // Field galaxies, with a cluster's ellipticals piled up near the centre.
  const nField = Math.round((rows * cols) / 4200);
  for (let i = 0; i < nField; i++) {
    const re = 1.2 + rand() ** 3 * 14;
    galaxy(rand() * cols, rand() * rows, unit * between(0.02, 0.5) / Math.sqrt(re), re, between(0.3, 1), rand() * Math.PI, col(rand() * 0.6), 0.004 * re, 1);
  }
  for (let i = 0; i < 70; i++) {
    const rr = Math.abs(gauss()) * 380;
    const a = rand() * 2 * Math.PI;
    const re = between(4, 18);
    galaxy(ccx + rr * Math.cos(a), ccy + rr * Math.sin(a), unit * between(0.3, 1.2), re, between(0.6, 1), rand() * Math.PI, col(0.45), 0.003 * re, 4);
  }
  galaxy(ccx, ccy, unit * 3.5, 46, 0.78, 0.4, col(0.5), 0.05, 4); // brightest cluster galaxy

  // Lensed arcs around the cluster: strings of small blobs on a circle.
  for (const [rad, a0, len, f] of [
    [330, 0.4, 0.55, 0.9],
    [520, 2.6, 0.35, 0.6],
    [270, 4.1, 0.4, 0.5],
  ]) {
    for (let t = 0; t < 60; t++) {
      const a = a0 + (t / 60) * len;
      galaxy(ccx + rad * Math.cos(a), ccy + rad * Math.sin(a), unit * f * (0.6 + 0.4 * Math.sin((t / 60) * Math.PI)), 3.2, 0.35, a + Math.PI / 2, channels === 3 ? [0.55, 0.85, 1.35] : [1], 0.09);
    }
  }

  // Stars, a few bright enough for diffraction spikes.
  for (let i = 0; i < 160; i++) {
    const bright = i < 7;
    star(rand() * cols, rand() * rows, unit * (bright ? between(40, 140) : between(0.5, 8)), channels === 3 ? [0.8, 0.95, 1.2] : [1], bright, bright ? 0.12 : 0);
  }

  // The oddities the detector should find.
  // A collisional ring galaxy.
  {
    const gx = cols * 0.24;
    const gy = rows * 0.3;
    for (let t = 0; t < 90; t++) {
      const a = (t / 90) * 2 * Math.PI;
      galaxy(gx + 38 * Math.cos(a), gy + 24 * Math.sin(a), unit * 0.35, 3, 0.8, a, channels === 3 ? [0.6, 0.9, 1.5] : [1], 0.25);
    }
    galaxy(gx + 9, gy - 4, unit * 1.2, 5, 0.9, 0, col(0.3), 0.3, 4);
  }
  // A snowball: a round, flat-topped hit in one channel only.
  {
    const gx = cols * 0.7;
    const gy = rows * 0.68;
    for (let y = Math.floor(gy - 14); y <= gy + 14; y++) {
      for (let x = Math.floor(gx - 14); x <= gx + 14; x++) {
        const d = Math.hypot(x - gx, y - gy);
        const v = unit * 4 / (1 + Math.exp((d - 8) * 1.1));
        add(channels - 1, x, y, v);
        if (v > 0.01) addOdd(x, y, 0.8);
      }
    }
  }
  // A satellite trail across several tiles.
  {
    const x0 = cols * 0.1;
    const y0 = rows * 0.82;
    const ang = -0.32;
    for (let s = 0; s < 1400; s += 0.5) {
      const x = x0 + s * Math.cos(ang);
      const y = y0 + s * Math.sin(ang);
      for (let w = -3; w <= 3; w++) {
        const px = Math.round(x - w * Math.sin(ang));
        const py = Math.round(y + w * Math.cos(ang));
        for (let c = 0; c < channels; c++) add(c, px, py, (unit * 0.18 * Math.exp(-(w * w) / 2)) / 2);
        addOdd(px, py, 0.05);
      }
    }
  }
  // A patch of dead pixels.
  {
    const gx = Math.round(cols * 0.6);
    const gy = Math.round(rows * 0.22);
    for (let y = gy - 14; y < gy + 14; y++) {
      for (let x = gx - 30; x < gx + 30; x++) {
        if ((x * 7 + y * 13) % 5 === 0 || Math.abs(y - gy) < 3) {
          for (let c = 0; c < channels; c++) img[c][y * cols + x] = NaN;
          addOdd(x, y, 0.9);
        }
      }
    }
  }

  for (let i = 0; i < n; i++) {
    if (!valid[i]) for (let c = 0; c < channels; c++) img[c][i] = NaN;
  }
  return { img, odd, valid };
}

// Asinh stretch to 8 bits, with one fixed scaling per mosaic as prep_data does.
function stretcher(m) {
  const lo = m.sky.map((s, c) => s - 2 * m.noise[c]);
  const soft = m.sky.map((_, c) => m.noise[c] * 6);
  const top = m.channels === 3 ? 6 : 60;
  return (c, v) => {
    if (!Number.isFinite(v)) return 0;
    const t = Math.asinh((v - lo[c]) / soft[c]) / Math.asinh((top - lo[c]) / soft[c]);
    return Math.max(0, Math.min(255, Math.round(t * 255)));
  };
}

function sampleBilinear(plane, cols, rows, x, y) {
  const x0 = Math.max(0, Math.min(cols - 2, Math.floor(x)));
  const y0 = Math.max(0, Math.min(rows - 2, Math.floor(y)));
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * cols + x0;
  const a = plane[i], b = plane[i + 1], c = plane[i + cols], d = plane[i + cols + 1];
  if (!Number.isFinite(a + b + c + d)) return plane[Math.round(y) * cols + Math.round(x)];
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}

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
    COMMENT: 'Synthetic fixture for the viewer, drawn from a seeded generator, not JWST data',
  };
}

function main() {
  rmSync(OUT, { recursive: true, force: true });
  for (const d of ['tiles', 'cutouts', 'maps', 'maps1', 'mosaics']) mkdirSync(OUT + d, { recursive: true });

  const mosaics = {};
  const tiles = [];
  const cutoutJobs = [];

  for (const m of MOSAICS) {
    process.stdout.write(`drawing ${m.name}\n`);
    const { img, odd, valid } = makeMosaic(m);
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

    // Preview at half scale, each pixel the mean of a 2 by 2 block.
    const pw = m.cols / 2;
    const ph = m.rows / 2;
    const prev = Buffer.alloc(pw * ph * m.channels);
    for (let y = 0; y < ph; y++) {
      for (let x = 0; x < pw; x++) {
        for (let c = 0; c < m.channels; c++) {
          const p = img[c];
          const i = 2 * y * m.cols + 2 * x;
          prev[(y * pw + x) * m.channels + c] = stretch(c, (p[i] + p[i + 1] + p[i + m.cols] + p[i + m.cols + 1]) / 4);
        }
      }
    }
    writeFileSync(OUT + `mosaics/${m.name}.png`, png(pw, ph, m.channels, prev));

    mosaics[m.name] = {
      shape: [m.rows, m.cols],
      preview: `mosaics/${m.name}.png`,
      preview_scale: 0.5,
      filters: m.filters,
      pixel_scale_arcsec: m.scale,
      wcs,
      header: header(m, wcs),
    };

    for (let y0 = 0; y0 + CROP <= m.rows; y0 += STRIDE) {
      for (let x0 = 0; x0 + CROP <= m.cols; x0 += STRIDE) {
        let good = 0;
        for (let y = y0; y < y0 + CROP; y += 8) for (let x = x0; x < x0 + CROP; x += 8) good += valid[y * m.cols + x];
        if (good < (CROP / 8) ** 2 * 0.35) continue; // prep_data drops tiles that are mostly empty

        const map2 = new Float32Array(GRID * GRID);
        const map1 = new Float32Array(GRID * GRID);
        const cell = CROP / GRID;
        for (let r = 0; r < GRID; r++) {
          for (let c = 0; c < GRID; c++) {
            let o = 0;
            let empty = 0;
            let count = 0;
            for (let y = Math.floor(y0 + r * cell); y < y0 + (r + 1) * cell; y++) {
              for (let x = Math.floor(x0 + c * cell); x < x0 + (c + 1) * cell; x++) {
                const i = y * m.cols + x;
                o += odd[i];
                empty += valid[i] ? 0 : 1;
                count++;
              }
            }
            o /= count;
            const edge = empty / count;
            const base = 0.105 + 0.012 * gauss();
            map2[r * GRID + c] = Math.max(0.02, base + 0.42 * Math.tanh(o * 2.2) + 0.03 * edge);
            map1[r * GRID + c] = Math.max(0.02, base + 0.015 * gauss() + 0.36 * Math.tanh(o * 2.0) + 0.16 * edge);
          }
        }
        const top = (a) => {
          const s = Array.from(a).sort((p, q) => q - p);
          const k = Math.max(1, Math.round(s.length * 0.01));
          return s.slice(0, k).reduce((p, q) => p + q, 0) / k;
        };
        let peak = 0;
        for (let i = 1; i < map2.length; i++) if (map2[i] > map2[peak]) peak = i;
        const [ra, dec] = pix2world(wcs, x0 + CROP / 2 - 0.5, y0 + CROP / 2 - 0.5);
        tiles.push({
          m,
          img,
          stretch,
          x0,
          y0,
          ra,
          dec,
          map1,
          map2,
          score: top(map2),
          score_pass1: top(map1),
          peak: [Math.floor(peak / GRID), peak % GRID],
        });
      }
    }
    cutoutJobs.push(m.name);
  }

  tiles.sort((a, b) => b.score - a.score);
  // Pass 2 takes its references from the quiet half of pass 1.
  const quiet = tiles.map((t, i) => [t.score_pass1, i]).sort((a, b) => a[0] - b[0]);
  const refs = new Set(quiet.slice(0, Math.round(tiles.length * 0.4)).filter(() => rand() < 0.85).map((q) => q[1]));
  const cutoutCount = Object.fromEntries(MOSAICS.map((m) => [m.name, 0]));

  const out = tiles.map((t, i) => {
    const id = String(i).padStart(6, '0');
    const m = t.m;
    const cx = t.x0 + CROP / 2;
    const cy = t.y0 + CROP / 2;
    const decTag = (t.dec < 0 ? 'm' : 'p') + Math.abs(t.dec).toFixed(2);
    const key = `query/${m.name}_cx${String(cx).padStart(6, '0')}_cy${String(cy).padStart(6, '0')}_ra${t.ra.toFixed(2)}_dec${decTag}_i${String(i).padStart(8, '0')}.png`;

    // PNG tile, resampled from the crop to TILE_PX.
    const pix = Buffer.alloc(TILE_PX * TILE_PX * m.channels);
    for (let v = 0; v < TILE_PX; v++) {
      for (let u = 0; u < TILE_PX; u++) {
        const x = t.x0 + ((u + 0.5) * CROP) / TILE_PX - 0.5;
        const y = t.y0 + ((v + 0.5) * CROP) / TILE_PX - 0.5;
        for (let c = 0; c < m.channels; c++) pix[(v * TILE_PX + u) * m.channels + c] = t.stretch(c, sampleBilinear(t.img[c], m.cols, m.rows, x, y));
      }
    }
    writeFileSync(OUT + `tiles/${id}.png`, png(TILE_PX, TILE_PX, m.channels, pix));
    writeFileSync(OUT + `maps/${id}.f32`, Buffer.from(t.map2.buffer));
    writeFileSync(OUT + `maps1/${id}.f32`, Buffer.from(t.map1.buffer));

    const hasCutout = cutoutCount[m.name] < CUTOUTS_PER_MOSAIC;
    if (hasCutout) {
      cutoutCount[m.name]++;
      const planes = t.img.map((p) => {
        const a = new Float32Array(CROP * CROP);
        for (let y = 0; y < CROP; y++) a.set(p.subarray((t.y0 + y) * m.cols + t.x0, (t.y0 + y) * m.cols + t.x0 + CROP), y * CROP);
        return a;
      });
      const w = mosaics[m.name].wcs;
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
      writeFileSync(OUT + `cutouts/${id}.fits`, fits(cards, planes, CROP, CROP));
    }

    return {
      key,
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

  const index = {
    version: 1,
    writable: WRITABLE,
    run: {
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
    },
    mosaics,
    tiles: out,
  };
  writeFileSync(OUT + 'index.json', JSON.stringify(index));

  // A handful of saved verdicts so the index shows its glyphs.
  const kinds = ['interesting', 'artifact', 'known', 'not_anomalous'];
  const verdicts = {};
  for (let i = 0; i < Math.min(14, out.length); i++) {
    if (i === 0 || i === 2) continue;
    verdicts[out[i].key] = {
      verdict: kinds[i % 4],
      star: i % 5 === 1,
      note: i === 1 ? 'Arc segment, check against the lens model.' : '',
      updated: '2026-11-03T05:10:00Z',
    };
  }
  writeFileSync(OUT + 'verdicts.json', JSON.stringify(verdicts, null, 1));
  process.stdout.write(`wrote ${out.length} tiles to ${OUT}\n`);
}

main();
