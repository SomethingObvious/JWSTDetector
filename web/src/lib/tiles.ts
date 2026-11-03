import { readFits } from './fits';
import { tileId, type Bundle, type LimitMode } from './types';

export interface TileData {
  index: number;
  width: number;
  height: number;
  channels: 1 | 3;
  /** Float planes for a cutout, or null when the PNG is shown. */
  planes: Float32Array[] | null;
  /** RGBA bytes of the PNG fallback. */
  rgba: Uint8Array | null;
  bunit: string;
  source: 'cutout' | 'png';
  /** Why the cutout was not used, when it was asked for and failed. */
  fallback: string | null;
  map2: Float32Array;
  map1: Float32Array | null;
  /** Finite pixel values per plane, sorted, for limits and equalisation. */
  samples: Float32Array[];
  mins: number[];
  maxs: number[];
  /** Sorted map values, for the patch percentile. */
  map2Sorted: Float32Array;
}

const SAMPLE_MAX = 60000;

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  return res.arrayBuffer();
}

async function decodePng(url: string): Promise<{ width: number; height: number; rgba: Uint8Array }> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} returned ${res.status}`);
  const bmp = await createImageBitmap(await res.blob(), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  const canvas = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) throw new Error('This browser has no 2D canvas to decode the PNG');
  ctx.drawImage(bmp, 0, 0);
  const rgba = new Uint8Array(ctx.getImageData(0, 0, bmp.width, bmp.height).data.buffer);
  const out = { width: bmp.width, height: bmp.height, rgba };
  bmp.close();
  return out;
}

function sampleOf(get: (i: number) => number, n: number): { sample: Float32Array; min: number; max: number } {
  const step = Math.max(1, Math.floor(n / SAMPLE_MAX));
  const vals: number[] = [];
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = get(i);
    if (!Number.isFinite(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    if (i % step === 0) vals.push(v);
  }
  const sample = Float32Array.from(vals).sort();
  if (!sample.length) return { sample, min: 0, max: 1 };
  return { sample, min, max };
}

export async function loadTile(base: string, bundle: Bundle, index: number): Promise<TileData> {
  const t = bundle.tiles[index];
  const id = tileId(index);
  const mosaic = bundle.mosaics[t.mosaic];
  const wantsColour = (mosaic?.filters.length ?? 1) >= 3;

  const mapsP = fetchBytes(`${base}maps/${id}.f32`).then((b) => new Float32Array(b));
  const maps1P = bundle.run.passes >= 2 ? fetchBytes(`${base}maps1/${id}.f32`).then((b) => new Float32Array(b), () => null) : Promise.resolve(null);

  let planes: Float32Array[] | null = null;
  let rgba: Uint8Array | null = null;
  let width = 0;
  let height = 0;
  let channels: 1 | 3 = wantsColour ? 3 : 1;
  let bunit = String(mosaic?.header?.BUNIT ?? '');
  let fallback: string | null = null;

  if (t.has_cutout) {
    try {
      const f = readFits(await fetchBytes(`${base}cutouts/${id}.fits`));
      planes = f.planes.length >= 3 ? f.planes.slice(0, 3) : [f.planes[0]];
      channels = planes.length === 3 ? 3 : 1;
      width = f.width;
      height = f.height;
      if (typeof f.header.BUNIT === 'string') bunit = f.header.BUNIT;
    } catch (err) {
      fallback = err instanceof Error ? err.message : String(err);
    }
  }
  if (!planes) {
    const p = await decodePng(`${base}tiles/${id}.png`);
    rgba = p.rgba;
    width = p.width;
    height = p.height;
    bunit = '8-bit PNG';
  }

  const n = width * height;
  const stats = Array.from({ length: channels }, (_, c) => {
    if (planes) return sampleOf((i) => planes![c][i], n);
    const px = rgba!;
    return sampleOf((i) => px[i * 4 + c], n);
  });

  const map2 = await mapsP;
  const map1 = await maps1P;
  return {
    index,
    width,
    height,
    channels,
    planes,
    rgba,
    bunit,
    source: planes ? 'cutout' : 'png',
    fallback,
    map2,
    map1,
    samples: stats.map((x) => x.sample),
    mins: stats.map((x) => x.min),
    maxs: stats.map((x) => x.max),
    map2Sorted: Float32Array.from(map2).sort(),
  };
}

/** Value at sorted fraction q, interpolated. */
function quantile(s: Float32Array, q: number): number {
  if (!s.length) return 0;
  const x = q * (s.length - 1);
  const i = Math.floor(x);
  const f = x - i;
  return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i];
}

/** IRAF's zscale, run on the sorted sample. */
function zscale(sorted: Float32Array, contrast = 0.25): [number, number] {
  const step = Math.max(1, Math.floor(sorted.length / 1000));
  const s: number[] = [];
  for (let i = 0; i < sorted.length; i += step) s.push(sorted[i]);
  const n = s.length;
  if (n < 5) return [sorted[0] ?? 0, sorted[sorted.length - 1] ?? 1];
  const median = n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  const keep = s.map(() => true);
  let slope = 0;
  let icept = 0;
  let good = n;
  for (let iter = 0; iter < 5; iter++) {
    let sx = 0, sy = 0, sxx = 0, sxy = 0, m = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      sx += i;
      sy += s[i];
      sxx += i * i;
      sxy += i * s[i];
      m++;
    }
    const den = m * sxx - sx * sx;
    if (m < n * 0.5 || den === 0) break;
    slope = (m * sxy - sx * sy) / den;
    icept = (sy - slope * sx) / m;
    let ss = 0;
    for (let i = 0; i < n; i++) if (keep[i]) ss += (s[i] - (icept + slope * i)) ** 2;
    const sigma = Math.sqrt(ss / m);
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const k = Math.abs(s[i] - (icept + slope * i)) < 2.5 * sigma;
      if (k !== keep[i]) changed++;
      keep[i] = k;
    }
    good = keep.filter(Boolean).length;
    if (!changed) break;
  }
  if (good < n * 0.5) return [s[0], s[n - 1]];
  const z = slope / contrast;
  const centre = (n - 1) / 2;
  return [Math.max(s[0], median - centre * z), Math.min(s[n - 1], median + (n - 1 - centre) * z)];
}

function limitsOf(s: Float32Array, min: number, max: number, mode: LimitMode): [number, number] {
  switch (mode) {
    case 'p995':
      return [quantile(s, 0.0025), quantile(s, 0.9975)];
    case 'p99':
      return [quantile(s, 0.005), quantile(s, 0.995)];
    case 'p95':
      return [quantile(s, 0.025), quantile(s, 0.975)];
    case 'zscale':
      return zscale(s);
    default:
      return [min, max];
  }
}

/**
 * Low and high limits per plane. Each band of a float colour tile keeps its own low
 * limit, which takes out its sky, but all three share the widest span above it, so
 * the bands keep their real flux ratios and the colour survives.
 */
export function computeLimits(d: TileData, mode: LimitMode): Limits {
  const out = d.samples.map((s, c) => limitsOf(s, d.mins[c], d.maxs[c], mode));
  const lo = out.map((x) => x[0]);
  let hi = out.map((x) => x[1]);
  if (d.planes && d.channels === 3) {
    const span = Math.max(...hi.map((h, c) => h - lo[c]));
    hi = lo.map((l) => l + span);
  }
  return { lo, hi };
}

export interface Limits {
  lo: number[];
  hi: number[];
}

/** Fraction of sorted values at or below v. */
export function rankOf(sorted: Float32Array, v: number): number {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return sorted.length ? lo / sorted.length : 0;
}

/** 256 steps per plane mapping a linear position between lo and hi to its equalised position. */
export function equaliser(d: TileData, limits: Limits): Float32Array {
  const out = new Float32Array(256 * 3);
  d.samples.forEach((s, c) => {
    const lo = limits.lo[c];
    const hi = limits.hi[c];
    const r0 = rankOf(s, lo);
    const span = rankOf(s, hi) - r0 || 1;
    for (let i = 0; i < 256; i++) out[c * 256 + i] = Math.min(1, Math.max(0, (rankOf(s, lo + ((hi - lo) * i) / 255) - r0) / span));
  });
  return out;
}

/**
 * Keeps decoded tiles around the current one. A load that is already running is
 * shared, so stepping onto a tile being prefetched just waits for that fetch.
 */
export class TileCache {
  private done = new Map<number, TileData>();
  private running = new Map<number, Promise<TileData>>();
  private readonly base: string;
  private readonly bundle: Bundle;
  private readonly onReady: (d: TileData) => void;
  private readonly onEvict: (index: number) => void;
  private readonly capacity: number;

  constructor(base: string, bundle: Bundle, onReady: (d: TileData) => void, onEvict: (index: number) => void, capacity = 14) {
    this.base = base;
    this.bundle = bundle;
    this.onReady = onReady;
    this.onEvict = onEvict;
    this.capacity = capacity;
  }

  peek(index: number): TileData | undefined {
    const d = this.done.get(index);
    if (d) {
      // Map keeps insertion order, so re-inserting marks it as recently used.
      this.done.delete(index);
      this.done.set(index, d);
    }
    return d;
  }

  get(index: number): Promise<TileData> {
    const d = this.peek(index);
    if (d) return Promise.resolve(d);
    let p = this.running.get(index);
    if (!p) {
      p = loadTile(this.base, this.bundle, index).then(
        (data) => {
          this.running.delete(index);
          this.done.set(index, data);
          this.onReady(data);
          this.trim(index);
          return data;
        },
        (err) => {
          this.running.delete(index);
          throw err;
        },
      );
      this.running.set(index, p);
    }
    return p;
  }

  /** Loads these in order, one after another, so the next tile is never queued behind a far one. */
  async prefetch(indices: number[], keep: number[]): Promise<void> {
    this.keep = new Set(keep);
    for (const i of indices) {
      if (this.done.has(i)) continue;
      try {
        await this.get(i);
      } catch {
        // The plate shows the error if this tile is ever opened.
      }
    }
  }

  private keep = new Set<number>();

  private trim(justLoaded: number) {
    for (const k of this.done.keys()) {
      if (this.done.size <= this.capacity) break;
      if (k === justLoaded || this.keep.has(k)) continue;
      this.done.delete(k);
      this.onEvict(k);
    }
  }
}
