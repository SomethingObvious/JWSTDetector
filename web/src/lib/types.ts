import type { WcsCards } from './wcs';

export interface RunInfo {
  name: string;
  /** What verdicts are filed under when `--name` gave the run a name for show. */
  id?: string;
  model: string;
  detector: string;
  resolution: number;
  patch_size: number;
  grid: [number, number];
  /** Run-wide patch distances the overlay spans, from the median to the 99.9th percentile. */
  map_range?: [number, number];
  score_mode: string;
  passes: number;
  n_reference: number;
  /** Prep folders the tiles were scored against, when the reference was other fields. */
  reference_fields?: string[];
  /** Tiles from those fields that calibrate the p-values, or 0 without them. */
  n_calibration?: number;
  created: string;
  args: Record<string, unknown>;
  /** Width of each row of embedding.f32, or 0 when the run saved no descriptors. */
  embedding_dims?: number;
  xmatch?: { radius_arcsec: number; checked: string[]; failed: string[] } | null;
}

export interface MosaicInfo {
  shape: [number, number];
  preview: string;
  preview_scale: number;
  filters: string[];
  pixel_scale_arcsec: number;
  wcs: WcsCards;
  header: Record<string, string>;
}

export interface TileInfo {
  key: string;
  rank: number;
  /** Null when the tile had nothing left to compare against. */
  score: number | null;
  score_pass1?: number;
  /** Conformal p-value and Benjamini-Hochberg q-value against the reference fields. */
  p?: number;
  q?: number;
  in_reference: boolean;
  mosaic: string;
  x0: number;
  y0: number;
  size: number;
  ra: number;
  dec: number;
  peak: [number, number];
  /** RA and Dec of the peak patch's centre. */
  peak_sky?: [number, number];
  /** First and last patch row and column the tile was scored on, when it wasn't all of them. */
  owned?: [number, number, number, number];
  has_cutout: boolean;
  xmatch?: TileMatch;
}

export interface CatalogueMatch {
  name: string;
  type: string;
  note: string;
  /** Arcseconds from the peak patch. */
  sep: number;
}

export interface TileMatch {
  nearest: Record<string, CatalogueMatch>;
  uncatalogued: boolean;
}

/** A catalogued source over a tile, as [catalogue, name, type, note, u, v] from the top left. */
export type TileSource = [string, string, string, string, number, number];

export interface Bundle {
  version: number;
  writable: boolean;
  run: RunInfo;
  mosaics: Record<string, MosaicInfo>;
  tiles: TileInfo[];
}

export type Verdict = 'interesting' | 'artifact' | 'known' | 'not_anomalous';

export interface VerdictRecord {
  verdict: Verdict | null;
  star: boolean;
  note: string;
  updated: string;
}

export const VERDICTS: { id: Verdict; label: string; key: string }[] = [
  { id: 'interesting', label: 'Interesting', key: '1' },
  { id: 'artifact', label: 'Artifact', key: '2' },
  { id: 'known', label: 'Known Object', key: '3' },
  { id: 'not_anomalous', label: 'Not Anomalous', key: '4' },
];

export type Stretch = 'linear' | 'sqrt' | 'log' | 'asinh' | 'histeq';
export type LimitMode = 'minmax' | 'p995' | 'p99' | 'p95' | 'zscale' | 'manual';
export type Cmap = 'grey' | 'viridis' | 'cividis' | 'magma' | 'inferno' | 'heat';
export type MapSource = 'pass2' | 'pass1' | 'diff';

export interface Display {
  stretch: Stretch;
  limitMode: LimitMode;
  /** Manual limits, one per plane. */
  lo: number[];
  hi: number[];
  cmap: Cmap;
  invert: boolean;
  /** Shows a colour tile as the mean of its stretched bands, through the colour map. */
  mono: boolean;
  overlay: boolean;
  overlayOpacity: number;
  overlayCmap: Cmap;
  /** Fraction of the map's range below which patches are not drawn. */
  threshold: number;
  source: MapSource;
  contours: boolean;
  catalogues: boolean;
}

export type Theme = 'atlas' | 'plate' | 'classic';

export type SortKey = 'rank' | 'priority' | 'score' | 'delta' | 'ra' | 'dec';
export type MatchFilter = 'all' | 'uncatalogued' | 'catalogued';
export type VerdictFilter =
  | 'all'
  | 'unreviewed'
  | 'interesting'
  | 'artifact'
  | 'known'
  | 'not_anomalous'
  | 'starred';
export interface Filters {
  text: string;
  verdict: VerdictFilter;
  match: MatchFilter;
  mosaic: string;
  scoreMin: number | null;
  scoreMax: number | null;
  /** Highest q-value shown, or null for every tile. */
  qMax: number | null;
  sort: SortKey;
  desc: boolean;
}

export function tileId(index: number): string {
  return String(index).padStart(6, '0');
}

/** Drops the program and visit tokens and puts the observation last, so the instrument and filter read first. */
export function shortMosaic(name: string): string {
  const parts = name.split(/[-_]/u).filter((t) => !/^(?:jw\d+|t\d+|clear)$/iu.test(t));
  const obs = parts.filter((t) => /^o\d+$/iu.test(t));
  return [...parts.filter((t) => !obs.includes(t)), ...obs].join(' ');
}
