import { CMAPS } from './colormaps';
import type { Display } from './types';

const CMAP_IDS = CMAPS.map((c) => c.id);

// Saved prefs are shared by every run and may come from an older viewer, so each
// field has to be one this viewer can draw.
const CHOICES: Partial<Record<keyof Display, readonly unknown[]>> = {
  stretch: ['linear', 'sqrt', 'log', 'asinh', 'histeq'],
  limitMode: ['minmax', 'p995', 'p99', 'p95', 'zscale', 'manual'],
  cmap: CMAP_IDS,
  overlayCmap: CMAP_IDS,
  source: ['pass2', 'pass1', 'diff'],
};

function fits(k: keyof Display, v: unknown, d: unknown): boolean {
  const choices = CHOICES[k];
  if (choices) {
    return choices.includes(v);
  }
  if (Array.isArray(d)) {
    return Array.isArray(v) && v.length > 0 && v.every((x) => Number.isFinite(x));
  }
  if (typeof d === 'number') {
    // The only numbers are the overlay's opacity and threshold, both fractions.
    return typeof v === 'number' && v >= 0 && v <= 1;
  }
  return typeof v === typeof d;
}

/** Saved display prefs, with anything unknown or out of range back at its default. */
export function readDisplay(
  saved: unknown,
  defaults: Display,
  passes: number,
): Display {
  const s = (typeof saved === 'object' && saved !== null ? saved : {}) as Record<
    string,
    unknown
  >;
  const out: Record<string, unknown> = { ...defaults };
  for (const [k, d] of Object.entries(defaults)) {
    if (fits(k as keyof Display, s[k], d)) {
      out[k] = s[k];
    }
  }
  const display = out as unknown as Display;
  if (passes < 2) {
    display.source = defaults.source;
  }
  return display;
}

/** A rank typed into Go to Rank, or null when it isn't a whole number. */
export function parseRank(text: string): number | null {
  const t = text.trim();
  return /^\d+$/u.test(t) ? Number(t) : null;
}

/**
 * The run folder ?run= names, with a trailing slash, or null when it is on another
 * site. Runs come from this site or from a server on this computer, as a page on
 * another site could hand the viewer files made to mislead.
 */
export function runBase(raw: string | null, here: string): string | null {
  const b = raw ?? './run/';
  let u: URL;
  try {
    u = new URL(b, here);
  } catch {
    return null;
  }
  const local =
    (u.protocol === 'http:' || u.protocol === 'https:') &&
    (u.hostname === 'localhost' || u.hostname === '127.0.0.1');
  if (u.origin !== new URL(here).origin && !local) {
    return null;
  }
  return b.endsWith('/') ? b : `${b}/`;
}
