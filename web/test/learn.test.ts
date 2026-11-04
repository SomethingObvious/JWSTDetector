import { describe, expect, it } from 'vitest';
import { TAU, learnedValue, priority, similar } from '../src/lib/learn';

// Six unit vectors in 2D: three near the x axis and three near the y axis.
const angles = [0, 0.1, 0.2, Math.PI / 2, Math.PI / 2 - 0.1, Math.PI / 2 - 0.2];
const emb = Float32Array.from(angles.flatMap((a) => [Math.cos(a), Math.sin(a)]));

describe('similar', () => {
  it('ranks neighbours by cosine and skips the tile', () => {
    const near = similar(emb, 2, 0, 3);
    expect(near.map((x) => x.index)).toEqual([1, 2, 5]);
    expect(near[0]?.cos).toBeCloseTo(Math.cos(0.1), 6);
  });
});

describe('learnedValue', () => {
  it('stays at the prior with no verdicts', () => {
    expect([...learnedValue(emb, 2, new Map())]).toEqual([
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
    ]);
  });

  it('spreads a verdict to similar tiles only', () => {
    const v = learnedValue(
      emb,
      2,
      new Map([
        [0, 1],
        [3, 0],
      ]),
    );
    expect(v[1]).toBeGreaterThan(0.6);
    expect(v[2]).toBeGreaterThan(0.5);
    expect(v[1]).toBeGreaterThan(v[2] ?? NaN);
    expect(v[4]).toBeLessThan(0.4);
    // The labelled tile itself is pulled most of the way to its verdict.
    expect(v[0]).toBeCloseTo((0.25 + 1) / (1.5 + Math.exp(-1 / TAU)), 6);
  });
});

describe('priority', () => {
  it('keeps the anomaly order when nothing is learned', () => {
    const p = priority(
      [1, 2, 3, 4],
      [true, true, true, false],
      new Float32Array(4).fill(0.5),
    );
    expect([...p]).toEqual([1, 0.75, 0.5, 0]);
  });

  it('lifts a tile like the interesting ones', () => {
    const p = priority([1, 2], [true, true], Float32Array.from([0.2, 0.9]));
    expect(p[1]).toBeGreaterThan(p[0] ?? NaN);
  });
});
