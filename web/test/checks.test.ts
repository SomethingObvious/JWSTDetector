import { describe, expect, it } from 'vitest';
import { parseRank, readDisplay, runBase } from '../src/lib/checks';
import { cmapRow } from '../src/lib/colormaps';
import { TileFrame } from '../src/lib/frame';
import type { Cmap, Display, MosaicInfo, TileInfo } from '../src/lib/types';
import { tanOf } from '../src/lib/wcs';

const defaults: Display = {
  stretch: 'asinh',
  limitMode: 'p995',
  lo: [0],
  hi: [1],
  cmap: 'grey',
  invert: false,
  mono: false,
  overlay: true,
  overlayOpacity: 0.45,
  overlayCmap: 'inferno',
  threshold: 0.6,
  source: 'pass2',
  contours: false,
  catalogues: true,
};

describe('readDisplay', () => {
  it('resets a colour map it has no row for', () => {
    const d = readDisplay({ cmap: 'plasma', overlayCmap: 'magma' }, defaults, 2);
    expect(d.cmap).toBe('grey');
    expect(d.overlayCmap).toBe('magma');
  });

  it('keeps a pass 1 map only in a two-pass run', () => {
    expect(readDisplay({ source: 'diff' }, defaults, 1).source).toBe('pass2');
    expect(readDisplay({ source: 'diff' }, defaults, 2).source).toBe('diff');
  });

  it('refuses values of the wrong type or out of range', () => {
    const d = readDisplay(
      {
        stretch: 'cube',
        lo: 'low',
        hi: [2, 'x'],
        invert: 'yes',
        overlayOpacity: 4,
        threshold: 0.3,
        mono: true,
      },
      defaults,
      2,
    );
    expect(d).toEqual({ ...defaults, threshold: 0.3, mono: true });
  });

  it('gives the defaults for prefs that are not an object', () => {
    expect(readDisplay(null, defaults, 2)).toEqual(defaults);
    expect(readDisplay('grey', defaults, 2)).toEqual(defaults);
  });
});

describe('cmapRow', () => {
  it('gives row 0 for a name it does not know, never -1', () => {
    expect(cmapRow('plasma' as Cmap)).toBe(0);
    expect(cmapRow('heat')).toBe(5);
  });
});

describe('parseRank', () => {
  it('takes a whole number with spaces around it', () => {
    expect(parseRank('12')).toBe(12);
    expect(parseRank(' 7 ')).toBe(7);
  });

  it('refuses anything else', () => {
    expect(parseRank('12abc')).toBeNull();
    expect(parseRank('1.5')).toBeNull();
    expect(parseRank('-3')).toBeNull();
    expect(parseRank('')).toBeNull();
  });
});

describe('runBase', () => {
  const here = 'https://example.org/viewer/index.html';

  it('takes a folder on this site and adds the slash', () => {
    expect(runBase(null, here)).toBe('./run/');
    expect(runBase('runs/a', here)).toBe('runs/a/');
    expect(runBase('https://example.org/other/', here)).toBe(
      'https://example.org/other/',
    );
  });

  it('takes a server on this computer', () => {
    expect(runBase('http://localhost:8765/run', here)).toBe(
      'http://localhost:8765/run/',
    );
    expect(runBase('http://127.0.0.1:8765/run/', here)).toBe(
      'http://127.0.0.1:8765/run/',
    );
  });

  it('refuses another site', () => {
    expect(runBase('https://evil.example/run/', here)).toBeNull();
    expect(runBase('//evil.example/run/', here)).toBeNull();
    expect(runBase('file:///C:/run/', here)).toBeNull();
  });
});

describe('a mosaic without a WCS', () => {
  const tile = { x0: 100, y0: 200, size: 10 } as TileInfo;

  it('has no Tan when the cards are missing or not TAN', () => {
    expect(tanOf(undefined)).toBeNull();
    expect(tanOf({ CTYPE1: 'RA---SIN', CTYPE2: 'DEC--SIN' })).toBeNull();
  });

  it('keeps its mosaic pixels but has no sky or compass', () => {
    const f = new TileFrame(tile, {} as MosaicInfo, null, 10, 10);
    expect(f.toMosaic(0.5, 0.5)).toEqual([100, 209]);
    expect(f.toSky(0.5, 0.5)).toBeNull();
    expect(f.compass()).toBeNull();
  });
});
