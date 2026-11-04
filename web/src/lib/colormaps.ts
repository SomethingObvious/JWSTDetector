import type { Cmap } from './types';

export const CMAPS: { id: Cmap; label: string }[] = [
  { id: 'grey', label: 'Grey' },
  { id: 'viridis', label: 'Viridis' },
  { id: 'cividis', label: 'Cividis' },
  { id: 'magma', label: 'Magma' },
  { id: 'inferno', label: 'Inferno' },
  { id: 'heat', label: 'Heat' },
];

type Vec3 = [number, number, number];

// Published sixth-order polynomial fits to matplotlib's maps, within a step or two of 255.
const POLY: Record<'viridis' | 'magma' | 'inferno', Vec3[]> = {
  viridis: [
    [0.2777273272234177, 0.005407344544966578, 0.3340998053353061],
    [0.1050930431085774, 1.404613529898575, 1.384590162594685],
    [-0.3308618287255563, 0.214847559468213, 0.09509516302823659],
    [-4.634230498983486, -5.799100973351585, -19.33244095627987],
    [6.228269936347081, 14.17993336680509, 56.69055260068105],
    [4.776384997670288, -13.74514537774601, -65.35303263337234],
    [-5.435455855934631, 4.645852612178535, 26.3124352495832],
  ],
  magma: [
    [-0.002136485053939582, -0.000749655052795221, -0.005386127855323933],
    [0.2516605407371642, 0.6775232436837668, 2.494026599312351],
    [8.353717279216625, -3.577719514958484, 0.3144679030132573],
    [-27.66873308576866, 14.26473078096533, -13.64921318813922],
    [52.17613981234068, -27.94360607168351, 12.94416944238394],
    [-50.76852536473588, 29.04658282127291, 4.23415299384598],
    [18.65570506591883, -11.48977351997711, -5.601961508734096],
  ],
  inferno: [
    [0.0002189403691192265, 0.001651004631001012, -0.01948089843709184],
    [0.1065134194856116, 0.5639564367884091, 3.932712388889277],
    [11.60249308247187, -3.972853965665698, -15.9423941062914],
    [-41.70399613139459, 17.43639888205313, 44.35414519872813],
    [77.162935699427, -33.40235894210092, -81.80730925738993],
    [-71.31942824499214, 32.62606426397723, 73.20951985803202],
    [25.13112622477341, -12.24266895238567, -23.07032500287172],
  ],
};

// Cividis has no polynomial fit, so it is nine even steps with straight lines between.
const CIVIDIS: Vec3[] = [
  [0, 0.1262, 0.3015],
  [0.0954, 0.2132, 0.4214],
  [0.2492, 0.3022, 0.4198],
  [0.3606, 0.3909, 0.4357],
  [0.4746, 0.4808, 0.4727],
  [0.5953, 0.5733, 0.4696],
  [0.7224, 0.6704, 0.4385],
  [0.8571, 0.7728, 0.3816],
  [0.9952, 0.9049, 0.231],
];

function evaluate(id: Cmap, t: number): Vec3 {
  switch (id) {
    case 'grey':
      return [t, t, t];
    case 'heat':
      // DS9's heat: red rises first, then green, then blue.
      return [
        Math.min(1, t * 3),
        Math.min(1, Math.max(0, t * 3 - 1)),
        Math.min(1, Math.max(0, t * 3 - 2)),
      ];
    case 'cividis': {
      const x = t * (CIVIDIS.length - 1);
      const i = Math.min(CIVIDIS.length - 2, Math.floor(x));
      const f = x - i;
      const a = CIVIDIS[i];
      const b = CIVIDIS[i + 1];
      if (!a || !b) {
        throw new RangeError(`Colour map position ${String(t)} is outside 0 to 1`);
      }
      return [
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f,
      ];
    }
    default: {
      const c = POLY[id];
      // Horner's rule from the highest power down.
      const channel = (k: 0 | 1 | 2) =>
        Math.min(
          1,
          Math.max(
            0,
            c.reduceRight((v, row) => v * t + row[k], 0),
          ),
        );
      return [channel(0), channel(1), channel(2)];
    }
  }
}

/** One row of 256 RGBA texels per map, in CMAPS order. */
export function lutTexels(): Uint8Array {
  const out = new Uint8Array(256 * CMAPS.length * 4);
  CMAPS.forEach(({ id }, row) => {
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = evaluate(id, i / 255);
      const o = (row * 256 + i) * 4;
      out[o] = Math.round(r * 255);
      out[o + 1] = Math.round(g * 255);
      out[o + 2] = Math.round(b * 255);
      out[o + 3] = 255;
    }
  });
  return out;
}

/** A CSS gradient for a legend swatch. */
export function cssGradient(id: Cmap, reverse = false): string {
  const stops: string[] = [];
  for (let i = 0; i <= 8; i++) {
    const t = i / 8;
    const [r, g, b] = evaluate(id, reverse ? 1 - t : t);
    stops.push(
      `rgb(${String(Math.round(r * 255))} ${String(Math.round(g * 255))} ${String(Math.round(b * 255))}) ${String(t * 100)}%`,
    );
  }
  return `linear-gradient(90deg, ${stops.join(', ')})`;
}

export const cmapRow = (id: Cmap) => CMAPS.findIndex((c) => c.id === id);
