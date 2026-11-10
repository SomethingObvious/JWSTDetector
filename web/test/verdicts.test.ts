import { describe, expect, it } from 'vitest';
import { readImport } from '../src/lib/verdicts.svelte';

const known = new Set(['a.png', 'b.png']);

describe('readImport', () => {
  it('only changes the columns a CSV has', () => {
    const text = 'key,star\na.png,TRUE\nb.png,false\nc.png,true\n';
    expect(readImport(text, known)).toEqual({
      changes: [
        ['a.png', { star: true }],
        ['b.png', { star: false }],
      ],
      skipped: 1,
    });
  });

  it('clears an empty verdict and leaves one it does not know', () => {
    const text = 'key,verdict,note\na.png,,"one, two"\nb.png,Maybe,\n';
    expect(readImport(text, known).changes).toEqual([
      ['a.png', { verdict: null, note: 'one, two' }],
      ['b.png', { note: '' }],
    ]);
  });

  it('reads a verdicts.json and drops a note that is not text', () => {
    const text = JSON.stringify({
      'a.png': { verdict: 'known', star: true, note: 5 },
      'b.png': null,
    });
    expect(readImport(text, known)).toEqual({
      changes: [['a.png', { verdict: 'known', star: true }]],
      skipped: 1,
    });
  });

  it('refuses a file with nothing to take', () => {
    expect(() => readImport('key,rank\na.png,1\n', known)).toThrow(
      /needs a key column/u,
    );
  });
});
