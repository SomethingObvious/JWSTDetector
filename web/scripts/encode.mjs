// Minimal PNG and FITS writers, just enough for the fixture bundle, and the
// stretch and resampling that turn float planes into PNG pixels.

import { deflateSync } from 'node:zlib';

const CRC = new Uint32Array(256).map((_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return c >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) {
    c = CRC[(c ^ b) & 0xff] ^ (c >>> 8);
  }
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
export function png(width, height, channels, pixels) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = channels === 3 ? 2 : 0;
  const stride = width * channels;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    pixels.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

export function fitsCard(key, value, comment = '') {
  let v;
  if (typeof value === 'string') {
    v = `'${value.replace(/'/gu, "''").padEnd(8)}'`.padEnd(20);
  } else if (typeof value === 'boolean') {
    v = (value ? 'T' : 'F').padStart(20);
  } else if (Number.isInteger(value)) {
    v = String(value).padStart(20);
  } else {
    v = value.toExponential(12).toUpperCase().padStart(20);
  }
  const card = `${key.padEnd(8)}= ${v}${comment ? ` / ${comment}` : ''}`;
  return card.slice(0, 80).padEnd(80);
}
export function fits(cards, planes, width, height) {
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

// Asinh stretch to 8 bits, with one fixed scaling per mosaic as prep_data does.
export function stretcher(m) {
  const lo = m.sky.map((s, c) => s - 2 * m.noise[c]);
  const soft = m.sky.map((_, c) => m.noise[c] * 6);
  const top = m.channels === 3 ? 6 : 60;
  return (c, v) => {
    if (!Number.isFinite(v)) {
      return 0;
    }
    const t = Math.asinh((v - lo[c]) / soft[c]) / Math.asinh((top - lo[c]) / soft[c]);
    return Math.max(0, Math.min(255, Math.round(t * 255)));
  };
}

export function sampleBilinear(plane, cols, rows, x, y) {
  const x0 = Math.max(0, Math.min(cols - 2, Math.floor(x)));
  const y0 = Math.max(0, Math.min(rows - 2, Math.floor(y)));
  const fx = x - x0;
  const fy = y - y0;
  const i = y0 * cols + x0;
  const a = plane[i];
  const b = plane[i + 1];
  const c = plane[i + cols];
  const d = plane[i + cols + 1];
  if (!Number.isFinite(a + b + c + d)) {
    return plane[Math.round(y) * cols + Math.round(x)];
  }
  return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy;
}
