import { describe, expect, it, vi } from 'vitest';
import { readFits } from '../src/lib/fits';
import { Tan, decDms, icrsToGalactic, raHms } from '../src/lib/wcs';
import { TileCache } from '../src/lib/cache';
import { computeLimits, loadTile, sampleOf, type TileData } from '../src/lib/tiles';
import { TileFrame } from '../src/lib/frame';
import type { Bundle, MosaicInfo, TileInfo } from '../src/lib/types';

function fitsBytes(cards: string[], data: number[]): ArrayBuffer {
  const head = [...cards, 'END'].map((c) => c.padEnd(80, ' ')).join('');
  const headLen = Math.ceil(head.length / 2880) * 2880;
  const dataLen = Math.ceil((data.length * 4) / 2880) * 2880;
  const buf = new ArrayBuffer(headLen + dataLen);
  const bytes = new Uint8Array(buf);
  bytes.fill(32, 0, headLen);
  for (let i = 0; i < head.length; i++) {
    bytes[i] = head.charCodeAt(i);
  }
  const view = new DataView(buf, headLen);
  data.forEach((v, i) => {
    view.setFloat32(i * 4, v, false);
  });
  return buf;
}

const card = (k: string, v: string) => `${k.padEnd(8)}= ${v.padStart(20)}`;

describe('readFits', () => {
  it('flips big-endian rows so row 0 is the top', () => {
    const buf = fitsBytes(
      [
        card('SIMPLE', 'T'),
        card('BITPIX', '-32'),
        card('NAXIS', '2'),
        card('NAXIS1', '3'),
        card('NAXIS2', '2'),
        "BUNIT   = 'MJy/sr  '           / surface brightness",
        "OBJECT  = 'it''s / here'",
        card('CRVAL1', '1.5D2'),
      ],
      [1, 2, 3, 4, NaN, -6.5],
    );
    const f = readFits(buf);
    expect(f.width).toBe(3);
    expect(f.height).toBe(2);
    expect(Array.from(f.planes[0]?.slice(0, 3) ?? [])).toEqual([4, NaN, -6.5]);
    expect(Array.from(f.planes[0]?.slice(3) ?? [])).toEqual([1, 2, 3]);
    expect(f.header['BUNIT']).toBe('MJy/sr');
    expect(f.header['OBJECT']).toBe("it's / here");
    expect(f.header['CRVAL1']).toBe(150);
    expect(f.header['SIMPLE']).toBe(true);
  });

  it('reads three planes for colour', () => {
    const buf = fitsBytes(
      [
        card('SIMPLE', 'T'),
        card('BITPIX', '-32'),
        card('NAXIS', '3'),
        card('NAXIS1', '1'),
        card('NAXIS2', '1'),
        card('NAXIS3', '3'),
      ],
      [0.25, 0.5, 0.75],
    );
    expect(readFits(buf).planes.map((p) => p[0])).toEqual([0.25, 0.5, 0.75]);
  });

  it('refuses integer data', () => {
    const buf = fitsBytes(
      [
        card('SIMPLE', 'T'),
        card('BITPIX', '16'),
        card('NAXIS', '2'),
        card('NAXIS1', '1'),
        card('NAXIS2', '1'),
      ],
      [0],
    );
    expect(() => readFits(buf)).toThrow('BITPIX is 16');
  });
});

describe('Tan', () => {
  // Reference values from astropy 8.0.2 with the same keywords.
  const wcs = new Tan({
    CTYPE1: 'RA---TAN',
    CTYPE2: 'DEC--TAN',
    CRPIX1: 2274.5,
    CRPIX2: 2306.5,
    CRVAL1: 110.8375,
    CRVAL2: -73.4547,
    CD1_1: -1.6444620863753398e-5,
    CD1_2: 5.985352508199202e-6,
    CD2_1: -5.985352508199202e-6,
    CD2_2: -1.6444620863753398e-5,
  });

  it('maps pixels to sky like astropy', () => {
    const [ra, dec] = wcs.pix2world(100, 4000);
    expect(ra).toBeCloseTo(110.9988034943329, 10);
    expect(dec).toBeCloseTo(-73.46947188260971, 10);
    const [ra0, dec0] = wcs.pix2world(2274.5, 2306.5);
    expect(ra0).toBeCloseTo(110.8375, 12);
    expect(dec0).toBeCloseTo(-73.4547, 12);
  });

  it('maps sky to pixels like astropy', () => {
    const [x, y] = wcs.world2pix(110.9, -73.5);
    expect(x).toBeCloseTo(2206.8567283593407, 7);
    expect(y).toBeCloseTo(5086.385386365631, 7);
  });

  it('formats sexagesimal and galactic coordinates', () => {
    expect(raHms(110.8375)).toBe('07:23:21.00');
    expect(decDms(-73.4547, 2)).toBe('−73:27:16.92');
    expect(raHms(359.99999999)).toBe('00:00:00.00');
    // astropy goes through FK5 for galactic, which moves it by about 0.02 arcsec.
    const [l, b] = icrsToGalactic(110.8375, -73.4547);
    expect(l).toBeCloseTo(284.9938224656199, 4);
    expect(b).toBeCloseTo(-23.69871978177526, 4);
  });
});

describe('sampleOf', () => {
  it('keeps the sample under 60,000 values on a 320 px tile', () => {
    const plane = Float32Array.from({ length: 320 * 320 }, (_, i) => i);
    plane[5] = NaN;
    const s = sampleOf(plane, plane.length);
    expect(s.sample.length).toBe(51200);
    expect(s.min).toBe(0);
    expect(s.max).toBe(102399);
    expect(s.sample[1]).toBe(2);
  });

  it('reads one channel out of RGBA', () => {
    const rgba = Uint8Array.from([1, 9, 9, 255, 3, 9, 9, 255, 2, 9, 9, 255]);
    const s = sampleOf(rgba, 3, 4, 0);
    expect([...s.sample]).toEqual([1, 2, 3]);
    expect([s.min, s.max]).toEqual([1, 3]);
  });
});

describe('computeLimits', () => {
  const tile = (planes: Float32Array[] | null, samples: number[][]) =>
    ({
      planes,
      channels: samples.length,
      samples: samples.map((s) => Float32Array.from(s).sort()),
      mins: samples.map((s) => Math.min(...s)),
      maxs: samples.map((s) => Math.max(...s)),
    }) as unknown as TileData;

  it('gives float colour bands one span above their own sky', () => {
    const p = [new Float32Array(1), new Float32Array(1), new Float32Array(1)];
    const l = computeLimits(
      tile(p, [
        [1, 3],
        [10, 11],
        [0, 4],
      ]),
      'minmax',
    );
    expect(l.lo).toEqual([1, 10, 0]);
    expect(l.hi).toEqual([5, 14, 4]);
  });

  it('leaves PNG bands on their own limits', () => {
    const l = computeLimits(
      tile(null, [
        [1, 3],
        [10, 11],
        [0, 4],
      ]),
      'minmax',
    );
    expect(l.hi).toEqual([3, 11, 4]);
  });
});

describe('TileFrame', () => {
  const tile = { x0: 100, y0: 200, size: 10 } as TileInfo;
  const f = new TileFrame(tile, {} as MosaicInfo, {} as Tan, 10, 10);

  it("maps the top image row to the tile's highest mosaic row", () => {
    expect(f.toMosaic(0.5, 0.5)).toEqual([100, 209]);
    expect(f.toMosaic(9.5, 9.5)).toEqual([109, 200]);
  });

  it('round-trips mosaic pixels', () => {
    expect(f.fromMosaic(...f.toMosaic(3.25, 7.75))).toEqual([3.25, 7.75]);
  });
});

describe('TileCache', () => {
  it('stops the loads a step leaves behind and keeps the rest', async () => {
    const asked: [string, AbortSignal][] = [];
    vi.stubGlobal('fetch', (url: string, init: { signal: AbortSignal }) => {
      asked.push([url, init.signal]);
      // Never answers, like a slow server, and fails the way fetch does when stopped.
      return new Promise((_resolve, reject) => {
        const stop = () => {
          reject(new DOMException('stopped', 'AbortError'));
        };
        if (init.signal.aborted) {
          stop();
        }
        init.signal.addEventListener('abort', stop);
      });
    });
    const tile = { mosaic: 'm', has_cutout: true };
    const bundle = {
      tiles: [tile, tile, tile],
      mosaics: {},
      run: { passes: 1 },
    } as unknown as Bundle;
    const cache = new TileCache(
      '',
      bundle,
      () => undefined,
      () => undefined,
    );
    const left = cache.get(0);
    const wanted = cache.get(1);
    void cache.prefetch([2], [1, 2]);
    await expect(left).rejects.toThrow('stopped');
    const stopped = (id: string) =>
      asked.filter(([url]) => url.includes(id)).map(([, signal]) => signal.aborted);
    expect(stopped('000000')).not.toContain(false);
    expect(stopped('000001')).toEqual([false, false]);
    expect(stopped('000002')).toEqual([false, false]);
    // Coming back asks again instead of handing over the stopped load.
    expect(cache.get(0)).not.toBe(left);
    expect(cache.get(1)).toBe(wanted);
    vi.unstubAllGlobals();
  });
});

describe('a tile whose map fails', () => {
  const cutout = fitsBytes(
    [
      card('SIMPLE', 'T'),
      card('BITPIX', '-32'),
      card('NAXIS', '2'),
      card('NAXIS1', '2'),
      card('NAXIS2', '1'),
    ],
    [1, 2],
  );
  const bundle = {
    tiles: [{ mosaic: 'm', has_cutout: true }],
    mosaics: {},
    run: { passes: 2 },
  } as unknown as Bundle;

  function serve(status: Record<string, number>) {
    const asked: string[] = [];
    vi.stubGlobal('fetch', (url: string) => {
      asked.push(url);
      const dir = url.split('/')[0] ?? '';
      const body = dir === 'cutouts' ? cutout : new Float32Array([0.5]).buffer;
      return Promise.resolve(new Response(body, { status: status[dir] ?? 200 }));
    });
    return asked;
  }

  it('shows the cutout without an overlay if the map is missing', async () => {
    serve({ maps: 404 });
    const d = await loadTile('', bundle, 0);
    expect(d.source).toBe('cutout');
    expect(d.map2).toBeNull();
    expect(Array.from(d.map1 ?? [])).toEqual([0.5]);
    expect(d.mapError).toBe(
      "Couldn't load the patch map (maps/000000.f32 returned 404), so there is no overlay. Step away and back to try again.",
    );
    vi.unstubAllGlobals();
  });

  it('asks again for a pass 1 map that failed', async () => {
    const asked = serve({ maps1: 500 });
    const cache = new TileCache(
      '',
      bundle,
      () => undefined,
      () => undefined,
    );
    const d = await cache.get(0);
    expect(d.map1).toBeNull();
    expect(Array.from(d.map2 ?? [])).toEqual([0.5]);
    expect(cache.peek(0)).toBeUndefined();
    await cache.get(0);
    expect(asked.filter((u) => u.startsWith('maps1/'))).toHaveLength(2);
    vi.unstubAllGlobals();
  });
});
