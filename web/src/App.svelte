<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import IndexList from './components/IndexList.svelte';
  import Plate from './components/Plate.svelte';
  import Readout from './components/Readout.svelte';
  import { TileFrame } from './lib/frame';
  import type { Renderer } from './lib/gl';
  import { TileCache, computeLimits, type Limits, type TileData } from './lib/tiles';
  import { VERDICT_VALUE, learnedValue, priority, similar, valueAt } from './lib/learn';
  import {
    VERDICTS,
    tileId,
    type Bundle,
    type Display,
    type Filters,
    type SortKey,
    type Theme,
    type TileInfo,
    type TileSource,
    type Verdict,
    type VerdictRecord,
  } from './lib/types';
  import { Verdicts } from './lib/verdicts.svelte';
  import { Tan } from './lib/wcs';

  const params = new URLSearchParams(location.search);
  const base = (() => {
    const b = params.get('run') ?? './run/';
    return b.endsWith('/') ? b : `${b}/`;
  })();

  const PREFS = 'jwst-peculiar:prefs';
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
  const saved = (() => {
    try {
      return JSON.parse(localStorage.getItem(PREFS) ?? '{}') as {
        display?: Partial<Display>;
        theme?: Theme;
        advance?: boolean;
      };
    } catch {
      return {};
    }
  })();

  let bundle = $state.raw<Bundle | null>(null);
  let loadError = $state<string | null>(null);
  let verdicts = $state.raw<Verdicts | null>(null);
  let cache: TileCache | null = null;
  let renderer: Renderer | null = null;
  let wcs = new Map<string, Tan>();

  const initial: Display = { ...defaults, ...saved.display };
  if (!Array.isArray(initial.lo) || !Array.isArray(initial.hi)) {
    Object.assign(initial, { lo: [0], hi: [1], limitMode: 'p995' });
  }
  let display = $state<Display>(initial);
  let theme = $state<Theme>(
    saved.theme && ['atlas', 'plate', 'classic'].includes(saved.theme)
      ? saved.theme
      : 'atlas',
  );
  let advance = $state(saved.advance ?? true);
  let filters = $state<Filters>({
    text: '',
    verdict: 'all',
    match: 'all',
    mosaic: '',
    scoreMin: null,
    scoreMax: null,
    qMax: null,
    sort: 'rank',
    desc: false,
  });
  let embedding = $state.raw<Float32Array | null>(null);
  let sources = $state.raw<TileSource[]>([]);
  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a cache, sources is the state the UI reads
  const sourceCache = new Map<number, TileSource[]>();

  let current = $state(0);
  const tile = $derived(bundle?.tiles[current]);
  let data = $state.raw<TileData | null>(null);
  let tileError = $state<string | null>(null);
  let cursor = $state<{ ix: number; iy: number } | null>(null);
  let lastPos = 0;

  let plate: { fit: () => void; oneToOne: () => void } | undefined = $state();
  let filterInput: HTMLInputElement | undefined = $state();
  let noteEl: HTMLTextAreaElement | undefined = $state();
  let sheet: HTMLDialogElement | undefined = $state();
  let gotoEl: HTMLInputElement | undefined = $state();
  let goto = $state<string | null>(null);
  let gotoMsg = $state('');
  let narrow = $state(false);
  let tab = $state<'plate' | 'index' | 'readout'>('plate');

  // Before the other effects, as the plate and the locator read their colours from
  // the theme's CSS when it changes.
  $effect.pre(() => {
    document.documentElement.dataset['theme'] = theme;
  });

  $effect(() => {
    const snapshot = JSON.stringify({ display, theme, advance });
    try {
      localStorage.setItem(PREFS, snapshot);
    } catch {
      // Prefs are a convenience, so a blocked store just means defaults next time.
    }
  });

  /** The tile the address names with #rank=N, or -1. */
  function hashTile(b: Bundle): number {
    const rank = /rank=(?<rank>\d+)/u.exec(location.hash)?.groups?.['rank'];
    return rank === undefined ? -1 : b.tiles.findIndex((t) => t.rank === Number(rank));
  }

  async function load() {
    try {
      const res = await fetch(`${base}index.json`, { cache: 'no-store' });
      if (!res.ok) {
        throw new Error(`${base}index.json returned ${String(res.status)}`);
      }
      const b = (await res.json()) as Bundle;
      if (b.version !== 1) {
        throw new Error(
          `index.json is bundle version ${String(b.version)}, and this viewer reads version 1`,
        );
      }
      // The file comes from outside, so its shape is checked rather than trusted.
      if (!Array.isArray(b.tiles) || b.tiles.length === 0) {
        throw new Error('index.json lists no tiles');
      }
      wcs = new Map(Object.entries(b.mosaics).map(([k, m]) => [k, new Tan(m.wcs)]));
      const v = new Verdicts(b, base);
      await v.load();
      const width = b.run.embedding_dims ?? 0;
      if (width) {
        // Only similar tiles and the Learned sort use it, so nothing waits for it.
        fetch(`${base}embedding.f32`)
          .then((r) =>
            r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status))),
          )
          .then((buf) => {
            const e = new Float32Array(buf);
            if (e.length === b.tiles.length * width) {
              embedding = e;
            }
          })
          .catch(() => {
            // Without it the viewer just has no similar tiles.
          });
      }
      verdicts = v;
      cache = new TileCache(
        base,
        b,
        (d) => renderer?.upload(d, b.run.grid),
        (i) => renderer?.evict(i),
      );
      current = Math.max(0, hashTile(b));
      bundle = b;
      document.title = `${b.run.name} · jwst-peculiar`;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    }
  }

  onMount(() => {
    const mq = matchMedia('(max-width: 820px)');
    narrow = mq.matches;
    const onmq = () => (narrow = mq.matches);
    mq.addEventListener('change', onmq);
    void load();
    const unload = (e: BeforeUnloadEvent) => {
      if (verdicts?.hasUnsaved()) {
        e.preventDefault();
      }
    };
    addEventListener('beforeunload', unload);
    // A link to another rank pasted into this tab changes the hash without a reload.
    const onhash = () => {
      const at = bundle ? hashTile(bundle) : -1;
      if (at >= 0) {
        current = at;
      }
    };
    addEventListener('hashchange', onhash);
    return () => {
      mq.removeEventListener('change', onmq);
      removeEventListener('beforeunload', unload);
      removeEventListener('hashchange', onhash);
    };
  });

  const lowered = $derived(
    bundle ? bundle.tiles.map((t) => `${t.key} ${t.mosaic}`.toLowerCase()) : [],
  );

  const indexOfKey = $derived(new Map(bundle?.tiles.map((t, i) => [t.key, i]) ?? []));

  // Verdicts as values to learn from, by tile index.
  const labels = $derived(
    new Map(
      [...(verdicts?.map ?? [])].flatMap(([key, v]) => {
        const i = indexOfKey.get(key);
        return v.verdict && i !== undefined
          ? [[i, VERDICT_VALUE[v.verdict]] as const]
          : [];
      }),
    ),
  );
  const dims = $derived(bundle?.run.embedding_dims ?? 0);
  // Only worked out for every tile while the list is sorted by it.
  const priorities = $derived.by(() => {
    if (!bundle || !embedding || filters.sort !== 'priority') {
      return null;
    }
    const value = learnedValue(embedding, dims, labels);
    return priority(
      bundle.tiles.map((t) => t.rank),
      bundle.tiles.map((t) => t.score !== null),
      value,
    );
  });
  const learned = $derived(
    embedding
      ? { value: valueAt(embedding, dims, labels, current), from: labels.size }
      : null,
  );
  const alike = $derived(embedding ? similar(embedding, dims, current, 6) : []);

  function passesFields(t: TileInfo, f: Filters): boolean {
    if (f.mosaic && t.mosaic !== f.mosaic) {
      return false;
    }
    if (f.match !== 'all' && t.xmatch?.uncatalogued !== (f.match === 'uncatalogued')) {
      return false;
    }
    if (f.scoreMin !== null && (t.score === null || t.score < f.scoreMin)) {
      return false;
    }
    if (f.scoreMax !== null && (t.score === null || t.score > f.scoreMax)) {
      return false;
    }
    return !(f.qMax !== null && (t.q === undefined || t.q > f.qMax));
  }

  function passesReview(
    f: Filters,
    q: string,
    text: string,
    v: VerdictRecord | undefined,
  ): boolean {
    if (f.verdict === 'unreviewed' && v?.verdict) {
      return false;
    }
    if (f.verdict === 'starred' && !v?.star) {
      return false;
    }
    if (
      f.verdict !== 'all' &&
      f.verdict !== 'unreviewed' &&
      f.verdict !== 'starred' &&
      v?.verdict !== f.verdict
    ) {
      return false;
    }
    return !q || text.includes(q) || (v?.note ?? '').toLowerCase().includes(q);
  }

  function sortKey(
    sort: SortKey,
    tiles: TileInfo[],
    pr: Float32Array | null,
  ): (i: number) => number {
    if (pr) {
      return (i) => -(pr[i] ?? NaN);
    }
    switch (sort) {
      case 'score':
        return (i) => tiles[i]?.score ?? -Infinity;
      case 'delta':
        return (i) => {
          const t = tiles[i];
          return t ? (t.score ?? 0) - (t.score_pass1 ?? t.score ?? 0) : NaN;
        };
      case 'ra':
        return (i) => tiles[i]?.ra ?? NaN;
      case 'dec':
        return (i) => tiles[i]?.dec ?? NaN;
      default:
        return (i) => tiles[i]?.rank ?? NaN;
    }
  }

  const order = $derived.by(() => {
    if (!bundle || !verdicts) {
      return new Int32Array(0);
    }
    const f = filters;
    const q = f.text.trim().toLowerCase();
    const out: number[] = [];
    const tiles = bundle.tiles;
    const map = verdicts.map;
    tiles.forEach((t, i) => {
      if (!passesFields(t, f)) {
        return;
      }
      // The verdict is only looked up when a filter needs it.
      if (
        (f.verdict !== 'all' || q) &&
        !passesReview(f, q, lowered[i] ?? '', map.get(t.key))
      ) {
        return;
      }
      out.push(i);
    });
    if (f.sort !== 'rank' || f.desc) {
      const key = sortKey(f.sort, tiles, f.sort === 'priority' ? priorities : null);
      const rank = (i: number) => tiles[i]?.rank ?? NaN;
      const dir = f.desc ? -1 : 1;
      out.sort((a, b) => (key(a) - key(b)) * dir || rank(a) - rank(b));
    }
    return Int32Array.from(out);
  });

  // Catalogued sources over the current tile, for the plate markers and the readout.
  $effect(() => {
    const i = current;
    if (!bundle?.run.xmatch) {
      return;
    }
    const hit = sourceCache.get(i);
    sources = hit ?? [];
    if (hit) {
      return;
    }
    fetch(`${base}cats/${tileId(i)}.json`)
      .then((r) => (r.ok ? (r.json() as Promise<TileSource[]>) : []))
      .then((s) => {
        sourceCache.set(i, s);
        if (current === i) {
          sources = s;
        }
      })
      .catch(() => {
        // A tile without a catalogue file just has no markers.
      });
  });

  const posOf = $derived.by(() => {
    const p = new Int32Array(bundle?.tiles.length ?? 0).fill(-1);
    order.forEach((i, k) => (p[i] = k));
    return p;
  });
  const pos = $derived(posOf[current] ?? -1);
  $effect(() => {
    if (pos >= 0) {
      lastPos = pos;
    }
  });

  const reviewed = $derived.by(() => {
    if (!verdicts) {
      return 0;
    }
    let n = 0;
    for (const v of verdicts.map.values()) {
      if (v.verdict) {
        n++;
      }
    }
    return n;
  });

  $effect(() => {
    const i = current;
    // bundle comes first so the effect always tracks it. cache isn't reactive, and
    // bailing on it alone would leave the effect waiting for current to change,
    // which it never does when the first tile is the one asked for.
    if (!bundle || !cache) {
      return;
    }
    const c = cache;
    cursor = null;
    const hit = c.peek(i);
    tileError = null;
    data = hit ?? null;
    if (!hit) {
      c.get(i).then(
        (d) => {
          if (current === i) {
            data = d;
          }
        },
        (err: unknown) => {
          if (current === i) {
            tileError = `Couldn't load tile ${String(i)}. ${err instanceof Error ? err.message : String(err)}`;
          }
        },
      );
    }
    const p = untrack(() => (pos >= 0 ? pos : lastPos));
    const ord = untrack(() => order);
    const ahead = [1, 2, 3, -1].flatMap((d) => {
      const x = ord[p + d];
      return x !== undefined && x !== i ? [x] : [];
    });
    void c.prefetch(ahead, [i, ...ahead]);
    const t = bundle.tiles[i];
    if (t) {
      try {
        history.replaceState(
          null,
          '',
          `${location.pathname}${location.search}#rank=${String(t.rank)}`,
        );
      } catch {
        // Safari refuses more than 100 of these in 10 seconds, which holding J reaches.
      }
    }
  });

  const limits = $derived.by((): Limits => {
    const n = data?.channels ?? 1;
    // A PNG was already stretched when it was cut, so it's shown as it is.
    if (data && !data.planes) {
      return { lo: new Array<number>(n).fill(0), hi: new Array<number>(n).fill(255) };
    }
    if (
      display.limitMode === 'manual' &&
      Array.isArray(display.lo) &&
      display.lo.length >= n
    ) {
      return { lo: display.lo.slice(0, n), hi: display.hi.slice(0, n) };
    }
    if (!data) {
      return { lo: [0], hi: [1] };
    }
    return computeLimits(
      data,
      display.limitMode === 'manual' ? 'p995' : display.limitMode,
    );
  });

  const mapRange = $derived.by((): [number, number] => {
    if (!data) {
      return [0, 1];
    }
    let src = data.map2;
    if (display.source === 'pass1' && data.map1) {
      src = data.map1;
    }
    const run = bundle?.run.map_range;
    if (display.source !== 'diff' && run) {
      return run;
    }
    const map1 = data.map1;
    if (display.source === 'diff' && map1) {
      let m = 0;
      data.map2.forEach((v, k) => {
        m = Math.max(m, Math.abs(v - (map1[k] ?? NaN)));
      });
      return [-m || -1, m || 1];
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of src) {
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
    return [lo, hi > lo ? hi : lo + 1];
  });

  const frame = $derived.by(() => {
    if (!bundle || !data || !tile) {
      return null;
    }
    const m = bundle.mosaics[tile.mosaic];
    const w = wcs.get(tile.mosaic);
    return m && w ? new TileFrame(tile, m, w, data.width, data.height) : null;
  });

  function select(i: number) {
    if (bundle && i >= 0 && i < bundle.tiles.length) {
      current = i;
    }
  }

  function selectAt(k: number) {
    const i = order[k];
    if (i !== undefined) {
      select(i);
    }
  }

  function step(d: number) {
    if (!order.length) {
      return;
    }
    let p = pos;
    if (p < 0) {
      p = d > 0 ? lastPos - 1 : lastPos;
    }
    selectAt(Math.max(0, Math.min(order.length - 1, p + d)));
  }

  function neighbour(): number | null {
    const p = pos >= 0 ? pos + 1 : lastPos;
    return order[p] ?? null;
  }

  function setVerdict(v: Verdict | null) {
    if (!verdicts || !tile) {
      return;
    }
    const next = neighbour();
    verdicts.update(tile.key, { verdict: v });
    if (advance && v && next !== null) {
      select(next);
    }
  }

  function toggleStar() {
    if (!verdicts || !tile) {
      return;
    }
    const key = tile.key;
    verdicts.update(key, { star: !verdicts.get(key)?.star });
  }

  function setTheme(t: Theme) {
    if (t === theme) {
      return;
    }
    // The plate world reads tiles as a negative, the way plates were inspected on a light table.
    if (t === 'plate') {
      display.invert = true;
      display.mono = true;
      display.cmap = 'grey';
    } else if (theme === 'plate') {
      display.invert = false;
      display.mono = false;
    }
    theme = t;
  }

  function jump() {
    if (!bundle || goto === null) {
      return;
    }
    const r = parseInt(goto, 10);
    const i = bundle.tiles.findIndex((t) => t.rank === r);
    if (i >= 0) {
      select(i);
      goto = null;
    } else {
      gotoMsg = `No tile has rank ${goto}. Ranks run from 1 to ${String(bundle.tiles.length)}.`;
    }
  }

  // Keys that move through the list or open an input. Returns false for any other key.
  function moveKey(k: string): boolean {
    if (k === 'j' || k === 'ArrowDown' || k === 'ArrowRight') {
      step(1);
    } else if (k === 'k' || k === 'ArrowUp' || k === 'ArrowLeft') {
      step(-1);
    } else if (k === 'Home') {
      selectAt(0);
    } else if (k === 'End') {
      selectAt(order.length - 1);
    } else if (k === 'PageDown') {
      step(10);
    } else if (k === 'PageUp') {
      step(-10);
    } else if (k === 'g') {
      goto = '';
      gotoMsg = '';
      queueMicrotask(() => gotoEl?.focus());
    } else if (k === '/') {
      if (narrow) {
        tab = 'index';
      }
      queueMicrotask(() => filterInput?.focus());
    } else {
      return false;
    }
    return true;
  }

  // Keys that change the view or the verdict. Returns false for any other key.
  function actKey(k: string): boolean {
    const verdict = VERDICTS.find((d) => d.key === k);
    if (k === 'f') {
      plate?.fit();
    } else if (k === '0') {
      plate?.oneToOne();
    } else if (k === 'o') {
      display.overlay = !display.overlay;
    } else if (k === 'c') {
      display.catalogues = !display.catalogues;
    } else if (k === '?') {
      if (sheet?.open) {
        sheet.close();
      } else {
        sheet?.showModal();
      }
    } else if (verdict) {
      setVerdict(verdict.id);
    } else if (k === 'Delete' || k === 'Backspace') {
      setVerdict(null);
    } else if (k === 's') {
      toggleStar();
    } else if (k === 'n') {
      if (narrow) {
        tab = 'readout';
      }
      queueMicrotask(() => noteEl?.focus());
    } else {
      return false;
    }
    return true;
  }

  function focusKind(el: HTMLElement): { typing: boolean; widget: boolean } {
    const type = el instanceof HTMLInputElement ? el.type : '';
    // Sliders and checkboxes keep their own arrow keys but let every other key through.
    const widget = type === 'range' || type === 'checkbox';
    const typing =
      el.isContentEditable ||
      el instanceof HTMLTextAreaElement ||
      el instanceof HTMLSelectElement ||
      (el instanceof HTMLInputElement && !widget);
    return { typing, widget };
  }

  function onkeydown(e: KeyboardEvent) {
    const el = e.target as HTMLElement;
    const { typing, widget } = focusKind(el);
    if (e.key === 'Escape') {
      if (goto !== null) {
        goto = null;
      }
      if (typing || widget) {
        el.blur();
      }
      return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey || !bundle) {
      return;
    }
    if (widget && /^(?:Arrow|Home|End|Page)/u.test(e.key)) {
      return;
    }
    if (sheet?.open && e.key !== '?') {
      return;
    }
    // A held key steps through the list, but it mustn't mark every tile it passes.
    if (moveKey(e.key) || (!e.repeat && actKey(e.key))) {
      e.preventDefault();
    }
  }

  const save = $derived(verdicts?.state ?? { kind: 'idle' as const });
  const clock = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime())
      ? iso
      : d.toLocaleTimeString('en-CA', { hour12: false });
  };
</script>

<!-- A select keeps focus after a choice, which would swallow J and K, so it lets go once it changes. -->
<svelte:window
  {onkeydown}
  onchange={(e: Event) => {
    if (e.target instanceof HTMLSelectElement) {
      e.target.blur();
    }
  }}
/>

{#if loadError}
  <main class="fatal">
    <h1>Couldn't Open the Run</h1>
    <p>{loadError}</p>
    <p class="dim">
      The viewer reads <code>{base}index.json</code>. Point it at another run folder
      with <code>?run=path/to/run/</code> in the address.
    </p>
  </main>
{:else if !bundle || !verdicts || !tile}
  <main class="fatal">
    <p class="label">Opening Run</p>
  </main>
{:else}
  <div class="app" class:narrow data-tab={tab}>
    <h1 class="sr">jwst-peculiar, {bundle.run.name}</h1>
    {#if !bundle.writable}
      <div class="demo">
        <span class="label">Demo Run</span>
        <span class="name">{bundle.run.name}</span>
        <span class="dim"
          >Verdicts stay in this browser. Export them from the Verdict block.</span
        >
        <a
          href="https://github.com/SomethingObvious/jwst-peculiar"
          target="_blank"
          rel="noopener noreferrer">jwst-peculiar on GitHub</a
        >
      </div>
    {/if}

    {#if narrow}
      <nav class="tabs" aria-label="Panels">
        <button
          class="ctl"
          aria-pressed={tab === 'index'}
          onclick={() => (tab = 'index')}>Index</button
        >
        <button
          class="ctl"
          aria-pressed={tab === 'plate'}
          onclick={() => (tab = 'plate')}>Plate</button
        >
        <button
          class="ctl"
          aria-pressed={tab === 'readout'}
          onclick={() => (tab = 'readout')}>Readout</button
        >
      </nav>
    {/if}

    <div class="cols">
      <div class="col index-col">
        <IndexList
          {bundle}
          {order}
          {pos}
          {current}
          {verdicts}
          {reviewed}
          bind:filters
          bind:filterInput
          onselect={(i: number) => {
            select(i);
            if (narrow) {
              tab = 'plate';
            }
          }}
        />
      </div>
      <main class="col plate-col" aria-label="Plate">
        <Plate
          bind:this={plate}
          {tile}
          index={current}
          {data}
          error={tileError}
          {frame}
          grid={bundle.run.grid}
          hasPass1={bundle.run.passes >= 2}
          bind:display
          {limits}
          {mapRange}
          {theme}
          {cursor}
          sources={bundle.run.xmatch && display.catalogues ? sources : []}
          hasCatalogues={Boolean(bundle.run.xmatch)}
          onrenderer={(r: Renderer | null) => (renderer = r)}
          oncursor={(c: { ix: number; iy: number } | null) => (cursor = c)}
        />
      </main>
      <div class="col readout-col">
        <Readout
          {bundle}
          {base}
          {current}
          {data}
          {frame}
          {cursor}
          bind:display
          {limits}
          {verdicts}
          {order}
          {theme}
          {sources}
          {learned}
          {alike}
          bind:advance
          bind:noteEl
          onselect={select}
          onverdict={setVerdict}
          onstar={toggleStar}
        />
      </div>
    </div>

    <footer class="rail">
      {#if goto !== null}
        <form
          class="goto"
          onsubmit={(e) => {
            e.preventDefault();
            jump();
          }}
        >
          <label
            ><span class="label">Go to Rank</span><input
              class="ctl num"
              bind:this={gotoEl}
              bind:value={goto}
              inputmode="numeric"
            /></label
          >
          <button class="ctl" type="submit">Go</button>
          {#if gotoMsg}<span class="dim">{gotoMsg}</span>{/if}
        </form>
      {:else}
        <span class="hints">
          <span><kbd>j</kbd><kbd>k</kbd> Step</span>
          <span><kbd>1</kbd><kbd>2</kbd><kbd>3</kbd><kbd>4</kbd> Verdict</span>
          <span><kbd>s</kbd> Star</span>
          <span><kbd>n</kbd> Note</span>
          <span><kbd>o</kbd> Overlay</span>
          <span><kbd>g</kbd> Rank</span>
          <button class="keys" onclick={() => sheet?.showModal()}
            ><kbd>?</kbd> All Keys</button
          >
        </span>
      {/if}
      {#if narrow}
        <span class="step" role="group" aria-label="Step Through the List">
          <button
            class="ctl"
            disabled={pos === 0}
            onclick={() => {
              step(-1);
            }}>Prev</button
          >
          <button
            class="ctl"
            disabled={pos === order.length - 1}
            onclick={() => {
              step(1);
            }}>Next</button
          >
        </span>
      {/if}
      <span class="where" aria-live="polite" aria-atomic="true">
        <span>Rank <b class="num">{tile.rank}</b></span>
        <span class="dim"
          >{#if pos >= 0}<span class="num">{pos + 1}</span> of
            <span class="num">{order.length}</span> shown{:else}not in the filtered list{/if}</span
        >
      </span>
      <span class="save" role="status">
        {#if save.kind === 'saved'}
          <i class="led ok"></i>Saved {clock(save.at)}
        {:else if save.kind === 'saving'}
          <i class="led busy"></i>Saving {save.pending}
        {:else if save.kind === 'retrying'}
          <i class="led bad"></i>Not saved, {save.pending} waiting. Retrying in {save.inSeconds}
          s ({save.error})
          <button class="ctl mini" onclick={() => verdicts?.retryNow()}
            >Retry Now</button
          >
        {:else if save.kind === 'local'}
          <i class="led ok"></i><span
            ><span class="num">{save.count}</span>
            {save.count === 1 ? 'verdict' : 'verdicts'} in this browser</span
          >
        {:else}
          <i class="led ok"></i>Every verdict saved
        {/if}
      </span>
      <span class="themes" role="group" aria-label="Theme">
        <button
          class="ctl mini"
          aria-pressed={theme === 'atlas'}
          onclick={() => {
            setTheme('atlas');
          }}>Atlas</button
        >
        <button
          class="ctl mini"
          aria-pressed={theme === 'plate'}
          onclick={() => {
            setTheme('plate');
          }}>Plate</button
        >
        <button
          class="ctl mini"
          aria-pressed={theme === 'classic'}
          onclick={() => {
            setTheme('classic');
          }}>Classic</button
        >
      </span>
    </footer>

    <dialog
      bind:this={sheet}
      class="sheet"
      aria-label="Keyboard Keys"
      onclick={(e) => {
        if (e.target === sheet) {
          sheet.close();
        }
      }}
    >
      <div class="sheet-head">
        <h2 class="label">Keys</h2>
        <button class="ctl mini" onclick={() => sheet?.close()}>Close</button>
      </div>
      <dl>
        <dt><kbd>j</kbd> <kbd>Down</kbd> <kbd>Right</kbd></dt>
        <dd>Next candidate</dd>
        <dt><kbd>k</kbd> <kbd>Up</kbd> <kbd>Left</kbd></dt>
        <dd>Previous candidate</dd>
        <dt><kbd>Home</kbd> <kbd>End</kbd></dt>
        <dd>First or last in the list</dd>
        <dt><kbd>PgDn</kbd> <kbd>PgUp</kbd></dt>
        <dd>Ten forward or back</dd>
        <dt><kbd>g</kbd></dt>
        <dd>Go to a rank</dd>
        <dt><kbd>1</kbd></dt>
        <dd>Interesting</dd>
        <dt><kbd>2</kbd></dt>
        <dd>Artifact</dd>
        <dt><kbd>3</kbd></dt>
        <dd>Known object</dd>
        <dt><kbd>4</kbd></dt>
        <dd>Not anomalous</dd>
        <dt><kbd>Del</kbd></dt>
        <dd>Clear the verdict</dd>
        <dt><kbd>s</kbd></dt>
        <dd>Star or unstar</dd>
        <dt><kbd>n</kbd></dt>
        <dd>Write a note, <kbd>Esc</kbd> to leave it</dd>
        <dt><kbd>/</kbd></dt>
        <dd>Filter the index</dd>
        <dt><kbd>f</kbd></dt>
        <dd>Fit the tile</dd>
        <dt><kbd>0</kbd></dt>
        <dd>One image pixel per screen pixel</dd>
        <dt><kbd>o</kbd></dt>
        <dd>Overlay on or off</dd>
        <dt><kbd>c</kbd></dt>
        <dd>Catalogue markers on or off</dd>
        <dt><kbd>?</kbd></dt>
        <dd>This sheet</dd>
      </dl>
      <p class="dim">
        A verdict moves to the next tile unless you turn that off in the Verdict block.
        Drag the plate to pan, scroll to zoom and double-click to fit.
      </p>
    </dialog>
  </div>
{/if}

<style>
  .fatal {
    height: 100dvh;
    display: grid;
    place-content: center;
    padding: 16px;
    max-width: 60ch;
    margin: 0 auto;
  }
  .fatal h1 {
    font-size: 18px;
    font-weight: 500;
    color: var(--anno);
    margin: 0 0 8px;
  }
  .fatal p {
    margin: 0 0 8px;
  }
  .fatal .label {
    color: var(--anno);
  }
  code {
    font-family: var(--font-num);
    font-size: 11.5px;
    color: var(--anno);
  }
  .dim {
    color: var(--chalk-3);
  }
  .app {
    height: 100dvh;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .demo {
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding: 4px 12px;
    border-bottom: 1px solid var(--rule);
    background: var(--ground-sunk);
    font-size: 11.5px;
    white-space: nowrap;
    overflow: hidden;
  }
  .demo .label {
    color: var(--anno);
  }
  .demo .name {
    font-size: 13px;
  }
  .demo a {
    margin-left: auto;
    padding-block: 4px;
    margin-block: -4px;
  }
  .demo,
  .tabs {
    flex: none;
  }
  .cols {
    flex: 1 1 0;
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr) 340px;
    min-height: 0;
  }
  .col {
    min-height: 0;
    min-width: 0;
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    grid-template-rows: minmax(0, 1fr);
  }
  .rail {
    flex: none;
    height: 28px;
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 0 12px;
    border-top: 1px solid var(--rule);
    background: var(--ground-sunk);
    font-size: 11px;
    color: var(--chalk-2);
    white-space: nowrap;
    overflow: hidden;
  }
  .hints {
    display: flex;
    gap: 14px;
    align-items: center;
  }
  .hints > span {
    display: inline-flex;
    gap: 3px;
    align-items: center;
  }
  .hints > span > kbd:last-of-type {
    margin-right: 3px;
  }
  .keys {
    background: none;
    border: 0;
    padding: 0;
    color: var(--chalk-2);
    cursor: pointer;
    display: inline-flex;
    gap: 6px;
    align-items: center;
    min-height: 24px;
  }
  .keys:hover {
    color: var(--anno);
  }
  .where {
    margin-left: auto;
    display: inline-flex;
    gap: 8px;
    align-items: baseline;
  }
  .where b {
    color: var(--anno);
    font-weight: 500;
  }
  .save {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    color: var(--chalk-2);
  }
  .led {
    width: 6px;
    height: 6px;
    display: inline-block;
    transform: rotate(45deg);
  }
  .led.ok {
    background: var(--v-known);
  }
  .led.busy {
    background: var(--anno);
  }
  .led.bad {
    background: transparent;
    border: 1.5px solid var(--anno);
  }
  .themes {
    display: flex;
  }
  .themes .ctl + .ctl {
    margin-left: -1px;
  }
  .mini {
    height: 24px;
    padding: 0 7px;
    font-size: 11px;
  }
  .goto {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .goto label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--anno);
  }
  .goto input {
    width: 80px;
    height: 20px;
  }
  .sheet {
    background: var(--ground-2);
    color: var(--chalk);
    border: 1px solid var(--rule-strong);
    border-radius: 2px;
    padding: 14px 18px 16px;
    width: min(440px, calc(100vw - 32px));
  }
  .sheet::backdrop {
    background: rgb(4 8 16 / 0.6);
  }
  .sheet-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--rule);
  }
  .sheet h2 {
    margin: 0;
    color: var(--anno);
  }
  .sheet dl {
    display: grid;
    grid-template-columns: 150px 1fr;
    gap: 5px 12px;
    margin: 0 0 12px;
  }
  .sheet dt {
    display: flex;
    gap: 3px;
  }
  .sheet dd {
    margin: 0;
    color: var(--chalk-2);
  }
  .sheet p {
    margin: 0;
    font-size: 11.5px;
    line-height: 1.5;
  }

  /* Phones: the plate with tabs for the index and readout. */
  .narrow .rail {
    height: 52px;
    gap: 12px;
  }
  .step {
    display: inline-flex;
    gap: 6px;
  }
  .step .ctl {
    height: 40px;
    min-width: 64px;
  }
  .tabs {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
    padding: 6px 16px;
    border-bottom: 1px solid var(--rule);
  }
  .tabs .ctl {
    height: 28px;
  }
  .narrow .cols {
    grid-template-columns: minmax(0, 1fr);
  }
  .narrow .col {
    display: none;
  }
  .narrow[data-tab='index'] .index-col,
  .narrow[data-tab='plate'] .plate-col,
  .narrow[data-tab='readout'] .readout-col {
    display: grid;
  }
  .narrow .demo {
    padding: 4px 16px;
    flex-wrap: wrap;
    white-space: normal;
    gap: 2px 10px;
  }
  .narrow .demo .dim {
    display: none;
  }
  .narrow .rail {
    padding: 0 16px;
  }
  .narrow .hints,
  .narrow .themes {
    display: none;
  }
  .narrow .where {
    margin-left: 0;
  }
  .narrow .where .dim {
    display: none;
  }
  .narrow .save {
    margin-left: auto;
  }
  .narrow :global(.index),
  .narrow :global(.readout) {
    border: 0;
  }
  .narrow :global(.index .controls),
  .narrow :global(.readout .block) {
    padding-left: 16px;
    padding-right: 16px;
  }
  .narrow :global(.index .item) {
    padding-right: 16px;
  }
  .narrow :global(.plate .toolbar) {
    padding: 6px 16px;
  }
  @media (pointer: coarse) {
    .demo {
      align-items: center;
      padding-block: 0;
    }
    .demo a {
      display: inline-flex;
      align-items: center;
      min-height: 44px;
      margin-block: 0;
      padding-block: 0;
    }
  }
</style>
