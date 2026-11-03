export type WcsCards = Record<string, string | number | boolean | undefined>;

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

/** A celestial TAN (gnomonic) WCS. Pixels are 1-based FITS pixels. */
export class Tan {
  readonly crpix: [number, number];
  readonly crval: [number, number];
  readonly cd: [number, number, number, number];
  private readonly inv: [number, number, number, number];

  constructor(cards: WcsCards) {
    const num = (k: string, fallback: number) => {
      const v = cards[k];
      return typeof v === 'number' ? v : fallback;
    };
    const t1 = String(cards.CTYPE1 ?? 'RA---TAN');
    const t2 = String(cards.CTYPE2 ?? 'DEC--TAN');
    if (!t1.endsWith('-TAN') || !t2.endsWith('-TAN')) throw new Error(`WCS is ${t1} ${t2}, the viewer reads TAN only`);
    this.crpix = [num('CRPIX1', 0), num('CRPIX2', 0)];
    this.crval = [num('CRVAL1', 0), num('CRVAL2', 0)];
    if (cards.CD1_1 !== undefined || cards.CD2_2 !== undefined) {
      this.cd = [num('CD1_1', 0), num('CD1_2', 0), num('CD2_1', 0), num('CD2_2', 0)];
    } else {
      const c1 = num('CDELT1', 1);
      const c2 = num('CDELT2', 1);
      this.cd = [c1 * num('PC1_1', 1), c1 * num('PC1_2', 0), c2 * num('PC2_1', 0), c2 * num('PC2_2', 1)];
    }
    const [a, b, c, d] = this.cd;
    const det = a * d - b * c;
    if (det === 0) throw new Error('WCS CD matrix is singular');
    this.inv = [d / det, -b / det, -c / det, a / det];
  }

  pix2world(p1: number, p2: number): [number, number] {
    const dx = p1 - this.crpix[0];
    const dy = p2 - this.crpix[1];
    const [a, b, c, d] = this.cd;
    const xi = (a * dx + b * dy) * D2R;
    const eta = (c * dx + d * dy) * D2R;
    const d0 = this.crval[1] * D2R;
    const den = Math.cos(d0) - eta * Math.sin(d0);
    const ra = this.crval[0] + Math.atan2(xi, den) * R2D;
    const dec = Math.atan2(eta * Math.cos(d0) + Math.sin(d0), Math.hypot(xi, den)) * R2D;
    return [((ra % 360) + 360) % 360, dec];
  }

  world2pix(ra: number, dec: number): [number, number] {
    const d0 = this.crval[1] * D2R;
    const da = (ra - this.crval[0]) * D2R;
    const dr = dec * D2R;
    const cosc = Math.sin(dr) * Math.sin(d0) + Math.cos(dr) * Math.cos(d0) * Math.cos(da);
    const xi = ((Math.cos(dr) * Math.sin(da)) / cosc) * R2D;
    const eta = ((Math.sin(dr) * Math.cos(d0) - Math.cos(dr) * Math.sin(d0) * Math.cos(da)) / cosc) * R2D;
    const [a, b, c, d] = this.inv;
    return [this.crpix[0] + a * xi + b * eta, this.crpix[1] + c * xi + d * eta];
  }
}

// ICRS to galactic rotation, as astropy uses it.
const GAL = [
  [-0.0548755604162154, -0.873437090234885, -0.4838350155487132],
  [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
  [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
];

export function icrsToGalactic(ra: number, dec: number): [number, number] {
  const a = ra * D2R;
  const d = dec * D2R;
  const v = [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)];
  const g = GAL.map((r) => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  const l = (((Math.atan2(g[1], g[0]) * R2D) % 360) + 360) % 360;
  const b = Math.asin(Math.max(-1, Math.min(1, g[2]))) * R2D;
  return [l, b];
}

function sexa(value: number, places: number, wrap = 0): string {
  const scale = 10 ** places;
  let total = Math.round(Math.abs(value) * 3600 * scale);
  if (wrap) total %= wrap * 3600 * scale;
  const d = Math.floor(total / (3600 * scale));
  const m = Math.floor((total % (3600 * scale)) / (60 * scale));
  const s = ((total % (60 * scale)) / scale).toFixed(places).padStart(places ? places + 3 : 2, '0');
  return `${String(d).padStart(2, '0')}:${String(m).padStart(2, '0')}:${s}`;
}

/** RA in degrees as hh:mm:ss.ss */
export function raHms(ra: number, places = 2): string {
  return sexa((((ra % 360) + 360) % 360) / 15, places, 24);
}

/** Dec in degrees as +dd:mm:ss.s, with a real minus sign. */
export function decDms(dec: number, places = 1): string {
  return signOf(dec) + sexa(dec, places);
}

/** The typographic minus for negative values, a plus otherwise. */
export function signOf(v: number): string {
  return v < 0 ? '−' : '+';
}

/** Formats a number with a typographic minus when it is negative. */
export function signed(v: number, digits: number): string {
  const s = Math.abs(v).toFixed(digits);
  return v < 0 && Number(s) !== 0 ? signOf(v) + s : s;
}

/** Angular separation in degrees. */
export function separation(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const a1 = ra1 * D2R;
  const d1 = dec1 * D2R;
  const a2 = ra2 * D2R;
  const d2 = dec2 * D2R;
  const s = Math.sin((d2 - d1) / 2) ** 2 + Math.cos(d1) * Math.cos(d2) * Math.sin((a2 - a1) / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(s))) * R2D;
}
