import type { Verdict } from './types';

/**
 * How much each verdict says a tile is worth following up. Known objects get half,
 * as a real source the detector was right about is still less useful than a new one.
 */
export const VERDICT_VALUE: Record<Verdict, number> = {
  interesting: 1,
  known: 0.5,
  artifact: 0,
  not_anomalous: 0,
};

/**
 * Kernel width in cosine distance. On SMACS 0723 a random pair of tiles sits near
 * cosine 0 and the median nearest neighbour at 0.64, which this weighs a tenth as
 * much as a twin while a random tile counts for almost nothing.
 */
export const TAU = 0.15;
/** The weight of the 0.5 prior, so a single verdict can't drive a far-off tile to 0 or 1. */
export const PRIOR_WEIGHT = 0.5;

function dot(e: Float32Array, dims: number, a: number, b: number): number {
  let s = 0;
  const oa = a * dims;
  const ob = b * dims;
  for (let d = 0; d < dims; d++) s += e[oa + d] * e[ob + d];
  return s;
}

/** The k tiles whose descriptors are closest to tile i by cosine, best first. Rows are unit length. */
export function similar(e: Float32Array, dims: number, i: number, k: number): { index: number; cos: number }[] {
  const n = e.length / dims;
  const out: { index: number; cos: number }[] = [];
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    const cos = dot(e, dims, i, j);
    if (out.length < k) {
      out.push({ index: j, cos });
      out.sort((a, b) => b.cos - a.cos);
    } else if (cos > out[k - 1].cos) {
      out[k - 1] = { index: j, cos };
      out.sort((a, b) => b.cos - a.cos);
    }
  }
  return out;
}

/**
 * Nadaraya-Watson regression of verdict value over descriptor space, with a 0.5
 * prior, so tiles that look like ones marked interesting rise and tiles that look
 * like artifacts sink. Tiles with no labelled neighbours stay near 0.5.
 */
export function learnedValue(e: Float32Array, dims: number, labels: Map<number, number>): Float32Array {
  const n = e.length / dims;
  const out = new Float32Array(n);
  // This is n times the labelled count, which is about half a second at 100,000 tiles
  // and 150 verdicts. A worker would keep that off the main thread if it ever matters.
  for (let i = 0; i < n; i++) out[i] = valueAt(e, dims, labels, i);
  return out;
}

/** learnedValue for one tile. */
export function valueAt(e: Float32Array, dims: number, labels: Map<number, number>, i: number): number {
  let w = PRIOR_WEIGHT;
  let s = 0.5 * PRIOR_WEIGHT;
  for (const [j, y] of labels) {
    const k = Math.exp((dot(e, dims, i, j) - 1) / TAU);
    w += k;
    s += k * y;
  }
  return s / w;
}

/**
 * The anomaly percentile times twice the learned value, so with no verdicts it's
 * the anomaly ranking unchanged, and a tile like the interesting ones can climb
 * past a stronger anomaly that looks like the artifacts.
 */
export function priority(ranks: ArrayLike<number>, scored: ArrayLike<boolean>, value: Float32Array): Float32Array {
  const n = value.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = scored[i] ? (1 - (ranks[i] - 1) / n) * 2 * value[i] : 0;
  return out;
}
