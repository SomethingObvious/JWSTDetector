import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { Verdicts } from '../src/lib/verdicts.svelte';
import type { Bundle } from '../src/lib/types';

const PENDING = 'jwst-peculiar:pending:r';
const LOCAL = 'jwst-peculiar:verdicts:r';

function bundle(writable: boolean): Bundle {
  return {
    writable,
    run: { name: 'r' },
    mosaics: {},
    tiles: [
      { key: 'a.png', rank: 1, mosaic: 'm', ra: 1, dec: 2, score: 0.5 },
      { key: 'b.png', rank: 2, mosaic: 'm', ra: 3, dec: 4, score: 0.25 },
    ],
  } as unknown as Bundle;
}

let stored: Map<string, string>;
let posted: string[];
let answer: (key: string) => Response;

beforeEach(() => {
  stored = new Map();
  posted = [];
  vi.stubGlobal('localStorage', {
    getItem: (k: string) => stored.get(k) ?? null,
    setItem: (k: string, v: string) => stored.set(k, v),
  });
  vi.stubGlobal('addEventListener', () => undefined);
  vi.stubGlobal('fetch', (url: string, init?: { body: string }) => {
    if (url.endsWith('verdicts.json')) {
      return Promise.resolve(new Response('', { status: 404 }));
    }
    const { key } = JSON.parse(init?.body ?? '{}') as { key: string };
    posted.push(key);
    return Promise.resolve(answer(key));
  });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const ok = () => new Response(JSON.stringify({ saved: '2026-11-13T00:00:00Z' }));
const pendingKeys = () =>
  (JSON.parse(stored.get(PENDING) ?? '[]') as { key: string }[]).map((p) => p.key);

describe('the save queue', () => {
  it('drops a refused save and sends the ones behind it', async () => {
    answer = (key) =>
      key === 'a.png'
        ? new Response(JSON.stringify({ error: 'a.png is not a tile here' }), {
            status: 400,
          })
        : ok();
    const v = new Verdicts(bundle(true), '');
    v.update('a.png', { verdict: 'artifact' });
    v.update('b.png', { verdict: 'known' });
    await vi.waitFor(() => {
      expect(posted).toEqual(['a.png', 'b.png']);
    });
    await vi.waitFor(() => {
      expect(v.state).toEqual({
        kind: 'rejected',
        key: 'a.png',
        error: 'a.png is not a tile here',
      });
    });
    expect(pendingKeys()).toEqual([]);
  });

  it('keeps a save the server cannot take yet and sends it again', async () => {
    vi.useFakeTimers();
    let first = true;
    answer = () => {
      const r = first
        ? new Response("Couldn't save it: locked", { status: 500 })
        : ok();
      first = false;
      return r;
    };
    const v = new Verdicts(bundle(true), '');
    v.update('a.png', { verdict: 'artifact' });
    await vi.waitFor(() => {
      expect(v.state).toEqual({
        kind: 'retrying',
        pending: 1,
        inSeconds: 1,
        error: "Couldn't save it: locked",
      });
    });
    expect(pendingKeys()).toEqual(['a.png']);
    await vi.advanceTimersByTimeAsync(1000);
    await vi.waitFor(() => {
      expect(v.state).toEqual({ kind: 'saved', at: '2026-11-13T00:00:00Z' });
    });
    expect(posted).toEqual(['a.png', 'a.png']);
    expect(pendingKeys()).toEqual([]);
  });

  it("replays only this run's saves and keeps the others stored", async () => {
    answer = ok;
    const item = (key: string) => ({ key, verdict: 'known', star: false, note: '' });
    stored.set(PENDING, JSON.stringify([item('x.png'), item('a.png')]));
    const v = new Verdicts(bundle(true), '');
    await v.load();
    await vi.waitFor(() => {
      expect(v.state.kind).toBe('saved');
    });
    expect(posted).toEqual(['a.png']);
    expect(pendingKeys()).toEqual(['x.png']);
    expect(v.get('x.png')).toBeUndefined();
  });
});

describe('verdicts kept in the browser', () => {
  const record = (updated: string) => ({
    verdict: 'interesting',
    star: false,
    note: '',
    updated,
  });

  it("counts and exports only this run's tiles", async () => {
    stored.set(
      LOCAL,
      JSON.stringify({ 'a.png': record('2026-01-01T00:00:00Z'), 'z.png': record('x') }),
    );
    const v = new Verdicts(bundle(false), '');
    await v.load();
    expect(v.reviewedCount()).toBe(1);
    expect(v.toCsv(bundle(false))).toBe(
      'key,rank,mosaic,ra,dec,score,verdict,star,note,updated\n' +
        'a.png,1,m,1,2,0.5,interesting,false,,2026-01-01T00:00:00Z\n',
    );
    v.update('b.png', { star: true });
    expect(Object.keys(JSON.parse(stored.get(LOCAL) ?? '{}') as object)).toEqual([
      'a.png',
      'z.png',
      'b.png',
    ]);
  });

  it('keeps what another tab saved before its event arrived', async () => {
    const v = new Verdicts(bundle(false), '');
    await v.load();
    stored.set(LOCAL, JSON.stringify({ 'a.png': record('2026-01-01T00:00:00Z') }));
    v.update('b.png', { verdict: 'artifact' });
    const saved = JSON.parse(stored.get(LOCAL) ?? '{}') as Record<
      string,
      { verdict: string }
    >;
    expect(saved['a.png']?.verdict).toBe('interesting');
    expect(saved['b.png']?.verdict).toBe('artifact');
  });
});
