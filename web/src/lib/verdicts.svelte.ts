import { SvelteMap } from 'svelte/reactivity';
import { VERDICTS, type Bundle, type Verdict, type VerdictRecord } from './types';

export type SaveState =
  | { kind: 'idle' }
  | { kind: 'saving'; pending: number }
  | { kind: 'saved'; at: string }
  | { kind: 'retrying'; pending: number; inSeconds: number; error: string }
  | { kind: 'local'; count: number };

interface Pending {
  key: string;
  verdict: Verdict | null;
  star: boolean;
  note: string;
}

function readJson(key: string): unknown {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeJson(key: string, value: unknown): boolean {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

const now = () => new Date().toISOString().replace(/\.\d+Z$/u, 'Z');

const isVerdict = (x: unknown): x is Verdict => VERDICTS.some((d) => d.id === x);

/** A record as it arrives from a file or the server, before it is checked. */
interface Incoming {
  verdict?: string | null;
  star?: boolean;
  note?: string;
  updated?: string;
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text.charAt(i);
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ',') {
      row.push(cell);
      cell = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') {
        i++;
      }
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += ch;
    }
  }
  row.push(cell);
  rows.push(row);
  return rows;
}

/** The rows of a CSV from Export, by key. */
function csvRecords(text: string): [string, Incoming][] {
  const rows = parseCsv(text);
  const head = rows.shift() ?? [];
  const col = (name: string) => head.indexOf(name);
  const [k, v, s, n] = [col('key'), col('verdict'), col('star'), col('note')];
  if (k < 0) {
    throw new Error(`The file has no key column, its header is: ${head.join(', ')}`);
  }
  return rows.flatMap((r): [string, Incoming][] => {
    const key = r[k];
    if (!key) {
      return [];
    }
    const note = n >= 0 ? (r[n] ?? '') : '';
    return [[key, { verdict: r[v] ?? null, star: r[s] === 'true', note }]];
  });
}

/**
 * Holds every verdict. With a writable run each change is posted to the server,
 * and anything not yet acknowledged is also kept in localStorage so a reload or a
 * crash resends it. Without one, localStorage is the store.
 */
export class Verdicts {
  readonly map = new SvelteMap<string, VerdictRecord>();
  state = $state<SaveState>({ kind: 'idle' });
  storageOk = $state(true);

  private readonly writable: boolean;
  private readonly base: string;
  private readonly localKey: string;
  private readonly pendingKey: string;
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- only read on unload, never by the UI
  private queue = new Map<string, Pending>();
  private flushing = false;
  private backoff = 1;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private tick: ReturnType<typeof setInterval> | null = null;

  constructor(bundle: Bundle, base: string) {
    this.writable = bundle.writable;
    this.base = base;
    this.localKey = `jwstdetector:verdicts:${bundle.run.name}`;
    this.pendingKey = `jwstdetector:pending:${bundle.run.name}`;
  }

  async load(): Promise<void> {
    try {
      const res = await fetch(`${this.base}verdicts.json`, { cache: 'no-store' });
      if (res.ok) {
        const all = (await res.json()) as Record<string, Incoming>;
        for (const [k, v] of Object.entries(all)) {
          this.map.set(k, {
            verdict: (v.verdict ?? null) as Verdict | null,
            star: Boolean(v.star),
            note: v.note ?? '',
            updated: v.updated ?? '',
          });
        }
      }
    } catch {
      // A run with no verdicts yet has no verdicts.json, which is fine.
    }
    if (this.writable) {
      const pending = (readJson(this.pendingKey) as Pending[] | null) ?? [];
      for (const p of pending) {
        this.map.set(p.key, {
          verdict: p.verdict,
          star: p.star,
          note: p.note,
          updated: now(),
        });
        this.queue.set(p.key, p);
      }
      if (this.queue.size) {
        void this.flush();
      }
    } else {
      const local =
        (readJson(this.localKey) as Record<string, VerdictRecord> | null) ?? {};
      for (const [k, v] of Object.entries(local)) {
        this.map.set(k, v);
      }
      this.state = { kind: 'local', count: this.reviewedCount() };
    }
  }

  get(key: string): VerdictRecord | undefined {
    return this.map.get(key);
  }

  reviewedCount(): number {
    let n = 0;
    for (const v of this.map.values()) {
      if (v.verdict) {
        n++;
      }
    }
    return n;
  }

  update(
    key: string,
    change: Partial<Pick<VerdictRecord, 'verdict' | 'star' | 'note'>>,
  ) {
    const prev = this.map.get(key) ?? {
      verdict: null,
      star: false,
      note: '',
      updated: '',
    };
    const next: VerdictRecord = { ...prev, ...change, updated: now() };
    if (
      next.verdict === prev.verdict &&
      next.star === prev.star &&
      next.note === prev.note
    ) {
      return;
    }
    this.map.set(key, next);
    if (this.writable) {
      this.queue.set(key, {
        key,
        verdict: next.verdict,
        star: next.star,
        note: next.note,
      });
      this.persistPending();
      void this.flush();
    } else {
      this.storageOk = writeJson(this.localKey, Object.fromEntries(this.map));
      this.state = { kind: 'local', count: this.reviewedCount() };
    }
  }

  hasUnsaved(): boolean {
    return this.queue.size > 0;
  }

  private persistPending() {
    this.storageOk = writeJson(this.pendingKey, [...this.queue.values()]);
  }

  private async flush(): Promise<void> {
    if (this.flushing) {
      return;
    }
    this.flushing = true;
    if (this.timer) {
      clearTimeout(this.timer);
    }
    if (this.tick) {
      clearInterval(this.tick);
    }
    try {
      let next = this.queue.entries().next();
      while (!next.done) {
        const [key, item] = next.value;
        this.state = { kind: 'saving', pending: this.queue.size };
        // eslint-disable-next-line no-await-in-loop -- saves go one at a time, in order
        const saved = await this.send(item);
        // Only drop it if nothing newer for this key arrived while the request was out.
        if (this.queue.get(key) === item) {
          this.queue.delete(key);
        }
        this.persistPending();
        this.backoff = 1;
        this.state = { kind: 'saved', at: saved ?? now() };
        next = this.queue.entries().next();
      }
    } catch (err) {
      const wait = this.backoff;
      this.backoff = Math.min(30, this.backoff * 2);
      const error = err instanceof Error ? err.message : String(err);
      let left = wait;
      this.state = {
        kind: 'retrying',
        pending: this.queue.size,
        inSeconds: left,
        error,
      };
      this.tick = setInterval(() => {
        left = Math.max(0, left - 1);
        this.state = {
          kind: 'retrying',
          pending: this.queue.size,
          inSeconds: left,
          error,
        };
      }, 1000);
      this.timer = setTimeout(() => {
        if (this.tick) {
          clearInterval(this.tick);
        }
        void this.flush();
      }, wait * 1000);
    } finally {
      this.flushing = false;
    }
  }

  private async send(item: Pending): Promise<string | undefined> {
    const res = await fetch(`${this.base}api/verdict`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(item),
    });
    if (!res.ok) {
      throw new Error(`the server answered ${String(res.status)}`);
    }
    const body = (await res.json().catch(() => ({}))) as { saved?: string };
    return body.saved;
  }

  retryNow() {
    this.backoff = 1;
    void this.flush();
  }

  toCsv(bundle: Bundle): string {
    const esc = (s: string) => (/[",\n]/u.test(s) ? `"${s.replace(/"/gu, '""')}"` : s);
    // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a throwaway lookup for one export
    const byKey = new Map(bundle.tiles.map((t) => [t.key, t]));
    const rows = ['key,rank,mosaic,ra,dec,score,verdict,star,note,updated'];
    for (const [key, v] of this.map) {
      const t = byKey.get(key);
      rows.push(
        [
          key,
          t?.rank ?? '',
          t?.mosaic ?? '',
          t?.ra ?? '',
          t?.dec ?? '',
          t?.score ?? '',
          v.verdict ?? '',
          v.star ? 'true' : 'false',
          v.note,
          v.updated,
        ]
          .map((x) => esc(String(x)))
          .join(','),
      );
    }
    return `${rows.join('\n')}\n`;
  }

  /** Reads a CSV from Export or a verdicts.json, and returns how many records it took. */
  importText(text: string): number {
    const trimmed = text.trim();
    const records = trimmed.startsWith('{')
      ? Object.entries(JSON.parse(trimmed) as Record<string, Incoming>)
      : csvRecords(trimmed);
    for (const [key, r] of records) {
      const verdict = isVerdict(r.verdict) ? r.verdict : null;
      this.update(key, { verdict, star: Boolean(r.star), note: r.note ?? '' });
    }
    return records.length;
  }
}
