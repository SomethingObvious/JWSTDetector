export interface Fits {
  header: Record<string, string | number | boolean>;
  cards: string[];
  width: number;
  height: number;
  /** One Float32Array per plane (1 for one band, 3 for R, G, B), rows top to bottom as shown. */
  planes: Float32Array[];
}

const BLOCK = 2880;

function parseValue(raw: string): string | number | boolean {
  const v = raw.trim();
  if (v.startsWith("'")) {
    // FITS strings escape a quote by doubling it, and trailing spaces are not significant.
    const body = v.slice(1).match(/^((?:[^']|'')*)'/);
    return (body ? body[1] : v.slice(1)).replace(/''/g, "'").trimEnd();
  }
  if (v === 'T') return true;
  if (v === 'F') return false;
  const n = Number(v.replace(/[dD]/, 'E'));
  return Number.isFinite(n) && v !== '' ? n : v;
}

/** Reads the primary HDU of a FITS file with BITPIX -32 and NAXIS 2 or 3. */
export function readFits(buf: ArrayBuffer): Fits {
  const bytes = new Uint8Array(buf);
  const header: Fits['header'] = {};
  const cards: string[] = [];
  let offset = 0;
  let ended = false;
  while (!ended) {
    if (offset + BLOCK > bytes.length) throw new Error(`FITS header has no END card in ${bytes.length} bytes`);
    for (let i = 0; i < 36; i++) {
      const card = String.fromCharCode(...bytes.subarray(offset + i * 80, offset + i * 80 + 80));
      const key = card.slice(0, 8).trim();
      if (key === 'END') {
        ended = true;
        break;
      }
      if (!key) continue;
      cards.push(card.trimEnd());
      if (card.slice(8, 10) !== '= ') continue;
      const rest = card.slice(10);
      // A slash ends the value, except inside a quoted string.
      let value = rest;
      if (!rest.trimStart().startsWith("'")) value = rest.split('/')[0];
      header[key] = parseValue(value);
    }
    offset += BLOCK;
  }

  const bitpix = header.BITPIX;
  const naxis = header.NAXIS;
  if (bitpix !== -32) throw new Error(`FITS BITPIX is ${bitpix}, the viewer reads -32 only`);
  if (naxis !== 2 && naxis !== 3) throw new Error(`FITS NAXIS is ${naxis}, the viewer reads 2 or 3`);
  const width = Number(header.NAXIS1);
  const height = Number(header.NAXIS2);
  const nplanes = naxis === 3 ? Number(header.NAXIS3) : 1;
  const plane = width * height;
  if (offset + plane * nplanes * 4 > bytes.length) {
    throw new Error(`FITS data is short: needs ${plane * nplanes * 4} bytes after the header, has ${bytes.length - offset}`);
  }

  const view = new DataView(buf, offset);
  const planes: Float32Array[] = [];
  for (let p = 0; p < nplanes; p++) {
    const out = new Float32Array(plane);
    const base = p * plane;
    for (let row = 0; row < height; row++) {
      // FITS row 1 is the bottom of the image, so it becomes the last row here.
      const src = (base + row * width) * 4;
      const dst = (height - 1 - row) * width;
      for (let x = 0; x < width; x++) out[dst + x] = view.getFloat32(src + x * 4, false);
    }
    planes.push(out);
  }
  return { header, cards, width, height, planes };
}
