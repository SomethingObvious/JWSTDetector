import { Tan, separation, signOf } from './wcs';
import type { MosaicInfo, TileInfo } from './types';

/**
 * Maps between the pixels of the image on screen (cutout or PNG, origin at the
 * top-left corner of the first pixel), mosaic pixels and the sky. The image's top
 * row is the mosaic's highest row in the tile, as FITS counts rows from the bottom.
 */
export class TileFrame {
  readonly tile: TileInfo;
  readonly mosaic: MosaicInfo;
  /** Null for a mosaic without a WCS the viewer reads, which leaves it no sky positions. */
  readonly wcs: Tan | null;
  readonly width: number;
  readonly height: number;
  readonly scaleX: number;
  readonly scaleY: number;
  constructor(
    tile: TileInfo,
    mosaic: MosaicInfo,
    wcs: Tan | null,
    width: number,
    height: number,
  ) {
    this.tile = tile;
    this.mosaic = mosaic;
    this.wcs = wcs;
    this.width = width;
    this.height = height;
    this.scaleX = tile.size / width;
    this.scaleY = tile.size / height;
  }

  /** Image position (continuous, pixel centres at i + 0.5) to 0-based mosaic pixel. */
  toMosaic(ix: number, iy: number): [number, number] {
    return [
      this.tile.x0 + ix * this.scaleX - 0.5,
      this.tile.y0 + (this.height - iy) * this.scaleY - 0.5,
    ];
  }

  fromMosaic(x: number, y: number): [number, number] {
    return [
      (x + 0.5 - this.tile.x0) / this.scaleX,
      this.height - (y + 0.5 - this.tile.y0) / this.scaleY,
    ];
  }

  toSky(ix: number, iy: number): [number, number] | null {
    const [x, y] = this.toMosaic(ix, iy);
    return this.wcs?.pix2world(x + 1, y + 1) ?? null;
  }

  fromSky(ra: number, dec: number): [number, number] | null {
    const p = this.wcs?.world2pix(ra, dec);
    return p ? this.fromMosaic(p[0] - 1, p[1] - 1) : null;
  }

  get arcsecPerPx(): number {
    return this.mosaic.pixel_scale_arcsec * this.scaleX;
  }

  /** Unit vectors on screen toward north and east at the image centre. */
  compass(): { n: [number, number]; e: [number, number] } | null {
    const c = this.toSky(this.width / 2, this.height / 2);
    if (!c) {
      return null;
    }
    const [ra, dec] = c;
    const d = 5 / 3600;
    const at = (r: number, de: number) => this.fromSky(r, de) ?? [NaN, NaN];
    const [x0, y0] = at(ra, dec);
    const [xn, yn] = at(ra, dec + d);
    const [xe, ye] = at(ra + d / Math.cos((dec * Math.PI) / 180), dec);
    const unit = (x: number, y: number): [number, number] => {
      const l = Math.hypot(x, y) || 1;
      return [x / l, y / l];
    };
    return { n: unit(xn - x0, yn - y0), e: unit(xe - x0, ye - y0) };
  }
}

export interface Part {
  v: string;
  u: string;
}

export interface Tick {
  edge: 'top' | 'bottom' | 'left' | 'right';
  /** Distance along the edge in CSS px from its start (left or top). */
  at: number;
  parts: Part[];
}

const DEC_STEPS = [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600, 900, 1800, 3600];
const RA_STEPS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600];

function raParts(sec: number, step: number): Part[] {
  let places = 0;
  if (step < 0.1) {
    places = 2;
  } else if (step < 1) {
    places = 1;
  }
  const scale = 10 ** places;
  let total = Math.round(sec * scale);
  const day = 86400 * scale;
  total = ((total % day) + day) % day;
  const h = Math.floor(total / (3600 * scale));
  const m = Math.floor((total % (3600 * scale)) / (60 * scale));
  const s = (total % (60 * scale)) / scale;
  const parts = [
    { v: String(h).padStart(2, '0'), u: 'h' },
    { v: String(m).padStart(2, '0'), u: 'm' },
  ];
  if (step < 60) {
    parts.push({ v: s.toFixed(places).padStart(places ? places + 3 : 2, '0'), u: 's' });
  }
  return parts;
}

function decParts(arcsec: number, step: number): Part[] {
  const places = step < 1 ? 1 : 0;
  const scale = 10 ** places;
  const total = Math.round(Math.abs(arcsec) * scale);
  const d = Math.floor(total / (3600 * scale));
  const m = Math.floor((total % (3600 * scale)) / (60 * scale));
  const s = (total % (60 * scale)) / scale;
  const parts = [{ v: signOf(arcsec) + String(d).padStart(2, '0'), u: '°' }];
  if (step < 3600) {
    parts.push({ v: String(m).padStart(2, '0'), u: '′' });
  }
  if (step < 60) {
    parts.push({ v: s.toFixed(places).padStart(places ? places + 3 : 2, '0'), u: '″' });
  }
  return parts;
}

/** Like a printed atlas, repeat only the fields that changed since the last label. */
function changedParts(full: Part[], prev: Part[] | null): Part[] {
  if (!prev) {
    return full;
  }
  let same = 0;
  while (same < full.length - 1 && full[same]?.v === prev[same]?.v) {
    same++;
  }
  return full.slice(same);
}

/**
 * Finds where lines of constant RA and Dec cross the edges of a w by h view and
 * labels each edge with whichever coordinate runs along it.
 */
export function edgeTicks(
  w: number,
  h: number,
  sky: (x: number, y: number) => [number, number],
  target = 110,
): Tick[] {
  if (w < 40 || h < 40) {
    return [];
  }
  const [raC, decC] = sky(w / 2, h / 2);
  const [ra1, dec1] = sky(w / 2 + 100, h / 2);
  const perPx = (separation(raC, decC, ra1, dec1) * 3600) / 100;
  const decStep = DEC_STEPS.find((s) => s >= perPx * target) ?? 3600;
  const cosd = Math.cos((decC * Math.PI) / 180);
  const raStep = RA_STEPS.find((s) => s * 15 * cosd >= perPx * target) ?? 600;
  const unwrap = (ra: number) => {
    let d = ra - raC;
    if (d > 180) {
      d -= 360;
    }
    if (d < -180) {
      d += 360;
    }
    return raC + d;
  };

  const edges: [Tick['edge'], (t: number) => [number, number], number][] = [
    ['top', (t) => [t * w, 0], w],
    ['bottom', (t) => [t * w, h], w],
    ['left', (t) => [0, t * h], h],
    ['right', (t) => [w, t * h], h],
  ];
  const out: Tick[] = [];
  const N = 160;
  // Both arrays hold N + 1 samples.
  const span = (v: number[]) => Math.abs((v[N] ?? NaN) - (v[0] ?? NaN));
  for (const [edge, at, len] of edges) {
    const ra: number[] = [];
    const dec: number[] = [];
    for (let k = 0; k <= N; k++) {
      const [x, y] = at(k / N);
      const [r, d] = sky(x, y);
      ra.push((unwrap(r) / 15) * 3600);
      dec.push(d * 3600);
    }
    const useRa = span(ra) * 15 * cosd >= span(dec);
    const vals = useRa ? ra : dec;
    const step = useRa ? raStep : decStep;
    let prev: Part[] | null = null;
    for (const [k, v0] of vals.entries()) {
      const v1 = vals[k + 1];
      // The last sample has no segment after it.
      if (v1 === undefined) {
        continue;
      }
      const a = Math.floor(v0 / step);
      const b = Math.floor(v1 / step);
      if (a === b) {
        continue;
      }
      const v = Math.max(a, b) * step;
      const f = (v - v0) / (v1 - v0);
      const pos = ((k + f) / N) * len;
      if (pos < 18 || pos > len - 18) {
        continue;
      }
      const full = useRa ? raParts(v, step) : decParts(v, step);
      out.push({ edge, at: pos, parts: changedParts(full, prev) });
      prev = full;
    }
  }
  return out;
}

const SCALE_STEPS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 300] as const;

/** A round scale bar length in arcsec that comes out between about 50 and 140 px. */
export function scaleBar(arcsecPerCssPx: number): {
  arcsec: number;
  px: number;
  label: string;
} {
  const want = arcsecPerCssPx * 90;
  let arcsec: number = SCALE_STEPS[0];
  for (const s of SCALE_STEPS) {
    if (s <= want * 1.4) {
      arcsec = s;
    }
  }
  const label = arcsec >= 60 ? `${String(arcsec / 60)}′` : `${String(arcsec)}″`;
  return { arcsec, px: arcsec / arcsecPerCssPx, label };
}
