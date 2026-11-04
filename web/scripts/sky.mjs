// The seeded generator and the synthetic sky drawn from it. The drawing steps
// call rand() in a fixed order, so changing that order changes every file.

let seed = 863642;
export function rand() {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
export function gauss() {
  return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}
const between = (a, b) => a + (b - a) * rand();

export const D2R = Math.PI / 180;

export function tanCd(scaleArcsec, thetaDeg) {
  const s = scaleArcsec / 3600;
  const c = Math.cos(thetaDeg * D2R);
  const n = Math.sin(thetaDeg * D2R);
  // Screen rows run down and north is up, rotated by theta.
  return { CD1_1: -s * c, CD1_2: s * n, CD2_1: -s * n, CD2_2: -s * c };
}

export function pix2world(w, x, y) {
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

function canvas(m) {
  const { rows, cols, channels } = m;
  const n = rows * cols;
  const img = Array.from({ length: channels }, () => new Float32Array(n));
  const odd = new Float32Array(n);
  const outside = (x, y) => x < 0 || y < 0 || x >= cols || y >= rows;
  return {
    m,
    rows,
    cols,
    channels,
    img,
    odd,
    valid: new Uint8Array(n),
    // MIRI surface brightness runs higher.
    unit: channels === 3 ? 1 : 9,
    tint: (red) => (channels === 3 ? [1 + red, 0.85, 0.75 - red * 0.5] : [1]),
    add(c, x, y, v) {
      if (!outside(x, y)) {
        img[c][y * cols + x] += v;
      }
    },
    addOdd(x, y, v) {
      if (!outside(x, y)) {
        odd[y * cols + x] += v;
      }
    },
  };
}

// The footprint is a rotated rectangle with ragged corners, like a real drizzled mosaic.
function footprint(s) {
  const { m, rows, cols, channels, img, valid } = s;
  const cx = cols / 2;
  const cy = rows / 2;
  const fa = (m.theta * 0.35 + 6) * D2R;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const u = (x - cx) * Math.cos(fa) + (y - cy) * Math.sin(fa);
      const v = -(x - cx) * Math.sin(fa) + (y - cy) * Math.cos(fa);
      const inside =
        Math.abs(u) < cols * 0.43 &&
        Math.abs(v) < rows * 0.42 &&
        Math.abs(u) + Math.abs(v) < (cols + rows) * 0.39;
      if (!inside) {
        continue;
      }
      const i = y * cols + x;
      valid[i] = 1;
      for (let c = 0; c < channels; c++) {
        const grad = 1 + 0.06 * (x / cols - 0.5) - 0.04 * (y / rows - 0.5);
        img[c][i] = m.sky[c] * grad + m.noise[c] * gauss();
      }
    }
  }
}

// A source with an elliptical exponential or de Vaucouleurs profile.
function galaxy(s, { x: gx, y: gy, flux, re, q, pa, colour, oddWeight, sersic = 1 }) {
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
      if (rr > r / re) {
        continue;
      }
      const prof = amp * Math.exp(-k * (rr ** (1 / sersic) - 1));
      for (let c = 0; c < s.channels; c++) {
        s.add(c, x, y, prof * colour[c]);
      }
      if (oddWeight) {
        s.addOdd(x, y, oddWeight * Math.min(1, prof / amp));
      }
    }
  }
}

// Adds onto v one spike at a time, since summing them first rounds differently.
function withSpikes(core, dx, dy, pa, flux) {
  let v = core;
  // JWST's hexagonal mirror gives six bright spikes and two faint ones.
  for (let s = 0; s < 4; s++) {
    const ang = pa + (s < 3 ? (s * Math.PI) / 3 : Math.PI / 2);
    const along = dx * Math.cos(ang) + dy * Math.sin(ang);
    const perp = -dx * Math.sin(ang) + dy * Math.cos(ang);
    const amp = s < 3 ? 1 : 0.35;
    v +=
      amp *
      flux *
      0.05 *
      Math.exp(-(perp * perp) / 0.7) *
      (Math.abs(along) + 3) ** -1.1;
  }
  return v;
}

function star(s, { x: sx, y: sy, flux, colour, spikes, oddWeight }) {
  const r = spikes ? 110 : 8;
  const pa = (s.m.theta + 30) * D2R;
  for (let y = Math.floor(sy - r); y <= sy + r; y++) {
    for (let x = Math.floor(sx - r); x <= sx + r; x++) {
      const dx = x - sx;
      const dy = y - sy;
      const d = Math.hypot(dx, dy);
      let v =
        flux * Math.exp(-(d * d) / (2 * 1.3 * 1.3)) + flux * 0.04 * Math.exp(-d / 4);
      if (spikes) {
        v = withSpikes(v, dx, dy, pa, flux);
      }
      for (let c = 0; c < s.channels; c++) {
        s.add(c, x, y, v * colour[c]);
      }
      if (oddWeight && v > flux * 0.002) {
        s.addOdd(x, y, oddWeight);
      }
    }
  }
}

// Field galaxies, with a cluster's ellipticals piled up near the centre.
function galaxies(s, ccx, ccy) {
  const { rows, cols, unit } = s;
  const nField = Math.round((rows * cols) / 4200);
  for (let i = 0; i < nField; i++) {
    const re = 1.2 + rand() ** 3 * 14;
    galaxy(s, {
      x: rand() * cols,
      y: rand() * rows,
      flux: (unit * between(0.02, 0.5)) / Math.sqrt(re),
      re,
      q: between(0.3, 1),
      pa: rand() * Math.PI,
      colour: s.tint(rand() * 0.6),
      oddWeight: 0.004 * re,
      sersic: 1,
    });
  }
  for (let i = 0; i < 70; i++) {
    const rr = Math.abs(gauss()) * 380;
    const a = rand() * 2 * Math.PI;
    const re = between(4, 18);
    galaxy(s, {
      x: ccx + rr * Math.cos(a),
      y: ccy + rr * Math.sin(a),
      flux: unit * between(0.3, 1.2),
      re,
      q: between(0.6, 1),
      pa: rand() * Math.PI,
      colour: s.tint(0.45),
      oddWeight: 0.003 * re,
      sersic: 4,
    });
  }
  // The brightest cluster galaxy.
  galaxy(s, {
    x: ccx,
    y: ccy,
    flux: unit * 3.5,
    re: 46,
    q: 0.78,
    pa: 0.4,
    colour: s.tint(0.5),
    oddWeight: 0.05,
    sersic: 4,
  });
}

// Lensed arcs around the cluster: strings of small blobs on a circle.
function arcs(s, ccx, ccy) {
  for (const [rad, a0, len, f] of [
    [330, 0.4, 0.55, 0.9],
    [520, 2.6, 0.35, 0.6],
    [270, 4.1, 0.4, 0.5],
  ]) {
    for (let t = 0; t < 60; t++) {
      const a = a0 + (t / 60) * len;
      galaxy(s, {
        x: ccx + rad * Math.cos(a),
        y: ccy + rad * Math.sin(a),
        flux: s.unit * f * (0.6 + 0.4 * Math.sin((t / 60) * Math.PI)),
        re: 3.2,
        q: 0.35,
        pa: a + Math.PI / 2,
        colour: s.channels === 3 ? [0.55, 0.85, 1.35] : [1],
        oddWeight: 0.09,
      });
    }
  }
}

// Stars, a few bright enough for diffraction spikes.
function stars(s) {
  for (let i = 0; i < 160; i++) {
    const bright = i < 7;
    star(s, {
      x: rand() * s.cols,
      y: rand() * s.rows,
      flux: s.unit * (bright ? between(40, 140) : between(0.5, 8)),
      colour: s.channels === 3 ? [0.8, 0.95, 1.2] : [1],
      spikes: bright,
      oddWeight: bright ? 0.12 : 0,
    });
  }
}

// A collisional ring galaxy.
function ring(s) {
  const gx = s.cols * 0.24;
  const gy = s.rows * 0.3;
  for (let t = 0; t < 90; t++) {
    const a = (t / 90) * 2 * Math.PI;
    galaxy(s, {
      x: gx + 38 * Math.cos(a),
      y: gy + 24 * Math.sin(a),
      flux: s.unit * 0.35,
      re: 3,
      q: 0.8,
      pa: a,
      colour: s.channels === 3 ? [0.6, 0.9, 1.5] : [1],
      oddWeight: 0.25,
    });
  }
  galaxy(s, {
    x: gx + 9,
    y: gy - 4,
    flux: s.unit * 1.2,
    re: 5,
    q: 0.9,
    pa: 0,
    colour: s.tint(0.3),
    oddWeight: 0.3,
    sersic: 4,
  });
}

// A snowball: a round, flat-topped hit in one channel only.
function snowball(s) {
  const gx = s.cols * 0.7;
  const gy = s.rows * 0.68;
  for (let y = Math.floor(gy - 14); y <= gy + 14; y++) {
    for (let x = Math.floor(gx - 14); x <= gx + 14; x++) {
      const d = Math.hypot(x - gx, y - gy);
      const v = (s.unit * 4) / (1 + Math.exp((d - 8) * 1.1));
      s.add(s.channels - 1, x, y, v);
      if (v > 0.01) {
        s.addOdd(x, y, 0.8);
      }
    }
  }
}

// A satellite trail across several tiles.
function trail(s) {
  const x0 = s.cols * 0.1;
  const y0 = s.rows * 0.82;
  const ang = -0.32;
  for (let t = 0; t < 1400; t += 0.5) {
    const x = x0 + t * Math.cos(ang);
    const y = y0 + t * Math.sin(ang);
    for (let w = -3; w <= 3; w++) {
      const px = Math.round(x - w * Math.sin(ang));
      const py = Math.round(y + w * Math.cos(ang));
      for (let c = 0; c < s.channels; c++) {
        s.add(c, px, py, (s.unit * 0.18 * Math.exp(-(w * w) / 2)) / 2);
      }
      s.addOdd(px, py, 0.05);
    }
  }
}

// A patch of dead pixels.
function deadPixels(s) {
  const gx = Math.round(s.cols * 0.6);
  const gy = Math.round(s.rows * 0.22);
  for (let y = gy - 14; y < gy + 14; y++) {
    for (let x = gx - 30; x < gx + 30; x++) {
      if ((x * 7 + y * 13) % 5 !== 0 && Math.abs(y - gy) >= 3) {
        continue;
      }
      for (let c = 0; c < s.channels; c++) {
        s.img[c][y * s.cols + x] = NaN;
      }
      s.addOdd(x, y, 0.9);
    }
  }
}

export function makeMosaic(m) {
  const s = canvas(m);
  footprint(s);
  const ccx = s.cols * 0.52;
  const ccy = s.rows * 0.47;
  galaxies(s, ccx, ccy);
  arcs(s, ccx, ccy);
  stars(s);
  // The oddities the detector should find.
  ring(s);
  snowball(s);
  trail(s);
  deadPixels(s);

  const { img, odd, valid } = s;
  for (let i = 0; i < valid.length; i++) {
    if (!valid[i]) {
      for (const plane of img) {
        plane[i] = NaN;
      }
    }
  }
  return { img, odd, valid };
}
