<script lang="ts">
  import { onMount, untrack } from 'svelte';
  import IndexList, { type Filters } from './components/IndexList.svelte';
  import Plate from './components/Plate.svelte';
  import Readout from './components/Readout.svelte';
  import { TileFrame } from './lib/frame';
  import type { Renderer } from './lib/gl';
  import { TileCache, computeLimits, type Limits, type TileData } from './lib/tiles';
  import { VERDICT_VALUE, learnedValue, priority, similar, valueAt } from './lib/learn';
  import { tileId, type Bundle, type Display, type Theme, type TileSource, type Verdict } from './lib/types';
  import { Verdicts } from './lib/verdicts.svelte';
  import { Tan } from './lib/wcs';

  const params = new URLSearchParams(location.search);
  const base = (() => {
    const b = params.get('run') ?? './run/';
    return b.endsWith('/') ? b : b + '/';
  })();

  const PREFS = 'jwstdetector:prefs';
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
      return JSON.parse(localStorage.getItem(PREFS) ?? '{}') as { display?: Partial<Display>; theme?: Theme; advance?: boolean };
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
  if (!Array.isArray(initial.lo) || !Array.isArray(initial.hi)) Object.assign(initial, { lo: [0], hi: [1], limitMode: 'p995' });
  let display = $state<Display>(initial);
  let theme = $state<Theme>(saved.theme && ['atlas', 'plate', 'classic'].includes(saved.theme) ? saved.theme : 'atlas');
  let advance = $state(saved.advance ?? true);
  let filters = $state<Filters>({ text: '', verdict: 'all', match: 'all', mosaic: '', scoreMin: null, scoreMax: null, sort: 'rank', desc: false });
  let embedding = $state.raw<Float32Array | null>(null);
  let sources = $state.raw<TileSource[]>([]);
  const sourceCache = new Map<number, TileSource[]>();

  let current = $state(0);
  let data = $state.raw<TileData | null>(null);
  let tileError = $state<string | null>(null);
  let cursor = $state<{ ix: number; iy: number } | null>(null);
  let lastPos = 0;

  let plate: Plate | undefined = $state();
  let filterInput: HTMLInputElement | undefined = $state();
  let noteEl: HTMLTextAreaElement | undefined = $state();
  let sheet: HTMLDialogElement | undefined = $state();
  let gotoEl: HTMLInputElement | undefined = $state();
  let goto = $state<string | null>(null);
  let narrow = $state(false);
  let tab = $state<'plate' | 'index' | 'readout'>('plate');

  $effect(() => {
    document.documentElement.dataset.theme = theme;
  });

  $effect(() => {
    const snapshot = JSON.stringify({ display, theme, advance });
    try {
      localStorage.setItem(PREFS, snapshot);
    } catch {
      // Prefs are a convenience, so a blocked store just means defaults next time.
    }
  });

  onMount(() => {
    const mq = matchMedia('(max-width: 820px)');
    narrow = mq.matches;
    const onmq = () => (narrow = mq.matches);
    mq.addEventListener('change', onmq);
    void load();
    const unload = (e: BeforeUnloadEvent) => {
      if (verdicts?.hasUnsaved()) e.preventDefault();
    };
    addEventListener('beforeunload', unload);
    return () => {
      mq.removeEventListener('change', onmq);
      removeEventListener('beforeunload', unload);
    };
  });

  async function load() {
    try {
      const res = await fetch(`${base}index.json`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`${base}index.json returned ${res.status}`);
      const b = (await res.json()) as Bundle;
      if (b.version !== 1) throw new Error(`index.json is bundle version ${b.version}, and this viewer reads version 1`);
      if (!b.tiles?.length) throw new Error('index.json lists no tiles');
      wcs = new Map(Object.entries(b.mosaics).map(([k, m]) => [k, new Tan(m.wcs)]));
      const v = new Verdicts(b, base);
      await v.load();
      const width = b.run.embedding_dims ?? 0;
      if (width) {
        // Similar tiles and learning from verdicts need it, and the rest of the viewer doesn't.
        fetch(`${base}embedding.f32`)
          .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status)))))
          .then((buf) => {
            const e = new Float32Array(buf);
            if (e.length === b.tiles.length * width) embedding = e;
          })
          .catch(() => {});
      }
      verdicts = v;
      cache = new TileCache(
        base,
        b,
        (d) => renderer?.upload(d, b.run.grid),
        (i) => renderer?.evict(i),
      );
      const m = location.hash.match(/rank=(\d+)/);
      const fromHash = m ? b.tiles.findIndex((t) => t.rank === Number(m[1])) : -1;
      current = fromHash >= 0 ? fromHash : 0;
      bundle = b;
      document.title = `${b.run.name} · JWSTDetector`;
    } catch (err) {
      loadError = err instanceof Error ? err.message : String(err);
    }
  }

  const lowered = $derived(bundle ? bundle.tiles.map((t) => `${t.key} ${t.mosaic}`.toLowerCase()) : []);

  const order = $derived.by(() => {
    if (!bundle || !verdicts) return new Int32Array(0);
    const f = filters;
    const q = f.text.trim().toLowerCase();
    const out: number[] = [];
    const tiles = bundle.tiles;
    for (let i = 0; i < tiles.length; i++) {
      const t = tiles[i];
      if (f.mosaic && t.mosaic !== f.mosaic) continue;
      if (f.match !== 'all' && (!t.xmatch || t.xmatch.uncatalogued !== (f.match === 'uncatalogued'))) continue;
      if (f.scoreMin !== null && (t.score === null || t.score < f.scoreMin)) continue;
      if (f.scoreMax !== null && (t.score === null || t.score > f.scoreMax)) continue;
      if (f.verdict !== 'all' || q) {
        const v = verdicts.map.get(t.key);
        if (f.verdict === 'unreviewed' && v?.verdict) continue;
        if (f.verdict === 'starred' && !v?.star) continue;
        if (f.verdict !== 'all' && f.verdict !== 'unreviewed' && f.verdict !== 'starred' && v?.verdict !== f.verdict) continue;
        if (q && !lowered[i].includes(q) && !(v?.note ?? '').toLowerCase().includes(q)) continue;
      }
      out.push(i);
    }
    if (f.sort !== 'rank' || f.desc) {
      const pr = f.sort === 'priority' ? priorities : null;
      const key: (i: number) => number =
        pr
          ? (i) => -pr[i]
          : f.sort === 'score'
          ? (i) => tiles[i].score ?? -Infinity
          : f.sort === 'delta'
            ? (i) => (tiles[i].score ?? 0) - (tiles[i].score_pass1 ?? tiles[i].score ?? 0)
            : f.sort === 'ra'
              ? (i) => tiles[i].ra
              : f.sort === 'dec'
                ? (i) => tiles[i].dec
                : (i) => tiles[i].rank;
      const dir = f.desc ? -1 : 1;
      out.sort((a, b) => (key(a) - key(b)) * dir || tiles[a].rank - tiles[b].rank);
    }
    return Int32Array.from(out);
  });

  const indexOfKey = $derived(new Map(bundle?.tiles.map((t, i) => [t.key, i]) ?? []));

  // Verdicts as values to learn from, by tile index.
  const labels = $derived.by(() => {
    const m = new Map<number, number>();
    if (!verdicts) return m;
    for (const [key, v] of verdicts.map) {
      const i = indexOfKey.get(key);
      if (v.verdict && i !== undefined) m.set(i, VERDICT_VALUE[v.verdict]);
    }
    return m;
  });
  const dims = $derived(bundle?.run.embedding_dims ?? 0);
  // Only worked out for every tile while the list is sorted by it.
  const priorities = $derived.by(() => {
    if (!bundle || !embedding || filters.sort !== 'priority') return null;
    const value = learnedValue(embedding, dims, labels);
    return priority(
      bundle.tiles.map((t) => t.rank),
      bundle.tiles.map((t) => t.score !== null),
      value,
    );
  });
  const learned = $derived(embedding ? { value: valueAt(embedding, dims, labels, current), from: labels.size } : null);
  const alike = $derived(embedding ? similar(embedding, dims, current, 6) : []);

  // Catalogued sources over the current tile, for the plate markers and the readout.
  $effect(() => {
    const i = current;
    if (!bundle?.run.xmatch) return;
    const hit = sourceCache.get(i);
    sources = hit ?? [];
    if (hit) return;
    fetch(`${base}cats/${tileId(i)}.json`)
      .then((r) => (r.ok ? (r.json() as Promise<TileSource[]>) : []))
      .then((s) => {
        sourceCache.set(i, s);
        if (current === i) sources = s;
      })
      .catch(() => {});
  });

  const posOf = $derived.by(() => {
    const p = new Int32Array(bundle?.tiles.length ?? 0).fill(-1);
    order.forEach((i, k) => (p[i] = k));
    return p;
  });
  const pos = $derived(posOf[current] ?? -1);
  $effect(() => {
    if (pos >= 0) lastPos = pos;
  });

  const reviewed = $derived.by(() => {
    if (!verdicts) return 0;
    let n = 0;
    for (const v of verdicts.map.values()) if (v.verdict) n++;
    return n;
  });

  // Loads the current tile, from the cache when it is there, and prefetches its neighbours.
  $effect(() => {
    const i = current;
    // bundle comes first so the effect always tracks it. cache isn't reactive, and
    // bailing on it alone would leave the effect waiting for current to change,
    // which it never does when the first tile is the one asked for.
    if (!bundle || !cache) return;
    const c = cache;
    cursor = null;
    const hit = c.peek(i);
    tileError = null;
    data = hit ?? null;
    if (!hit) {
      c.get(i).then(
        (d) => {
          if (current === i) data = d;
        },
        (err) => {
          if (current === i) tileError = `Couldn't load tile ${i}. ${err instanceof Error ? err.message : String(err)}`;
        },
      );
    }
    const p = untrack(() => (pos >= 0 ? pos : lastPos));
    const ord = untrack(() => order);
    const ahead = [1, 2, 3, -1].map((d) => ord[p + d]).filter((x) => x !== undefined && x !== i);
    void c.prefetch(ahead, [i, ...ahead]);
    if (bundle) history.replaceState(null, '', `${location.pathname}${location.search}#rank=${bundle.tiles[i].rank}`);
  });

  const limits = $derived.by((): Limits => {
    const n = data?.channels ?? 1;
    // A PNG was already stretched when it was cut, so it's shown as it is.
    if (data && !data.planes) return { lo: Array(n).fill(0), hi: Array(n).fill(255) };
    if (display.limitMode === 'manual' && Array.isArray(display.lo) && display.lo.length >= n) {
      return { lo: display.lo.slice(0, n), hi: display.hi.slice(0, n) };
    }
    if (!data) return { lo: [0], hi: [1] };
    return computeLimits(data, display.limitMode === 'manual' ? 'p995' : display.limitMode);
  });

  const mapRange = $derived.by((): [number, number] => {
    if (!data) return [0, 1];
    let src: ArrayLike<number> = data.map2;
    if (display.source === 'pass1' && data.map1) src = data.map1;
    const run = bundle?.run.map_range;
    if (display.source !== 'diff' && run) return run;
    if (display.source === 'diff' && data.map1) {
      let m = 0;
      for (let k = 0; k < data.map2.length; k++) m = Math.max(m, Math.abs(data.map2[k] - data.map1[k]));
      return [-m || -1, m || 1];
    }
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < src.length; k++) {
      lo = Math.min(lo, src[k]);
      hi = Math.max(hi, src[k]);
    }
    return [lo, hi > lo ? hi : lo + 1];
  });

  const frame = $derived.by(() => {
    if (!bundle || !data) return null;
    const t = bundle.tiles[current];
    const m = bundle.mosaics[t.mosaic];
    const w = wcs.get(t.mosaic);
    return m && w ? new TileFrame(t, m, w, data.width, data.height) : null;
  });

  function select(i: number) {
    if (bundle && i >= 0 && i < bundle.tiles.length) current = i;
  }

  function step(d: number) {
    if (!order.length) return;
    const p = pos >= 0 ? pos : d > 0 ? lastPos - 1 : lastPos;
    const k = Math.max(0, Math.min(order.length - 1, p + d));
    select(order[k]);
  }

  function neighbour(): number | null {
    const p = pos >= 0 ? pos + 1 : lastPos;
    return p < order.length ? order[p] : null;
  }

  function setVerdict(v: Verdict | null) {
    if (!bundle || !verdicts) return;
    const next = neighbour();
    verdicts.update(bundle.tiles[current].key, { verdict: v });
    if (advance && v && next !== null) select(next);
  }

  function toggleStar() {
    if (!bundle || !verdicts) return;
    const key = bundle.tiles[current].key;
    verdicts.update(key, { star: !verdicts.get(key)?.star });
  }

  function setTheme(t: Theme) {
    if (t === theme) return;
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
    if (!bundle || goto === null) return;
    const r = parseInt(goto, 10);
    const i = bundle.tiles.findIndex((t) => t.rank === r);
    if (i >= 0) {
      select(i);
      goto = null;
    } else gotoMsg = `No tile has rank ${goto}. Ranks run from 1 to ${bundle.tiles.length}.`;
  }
  let gotoMsg = $state('');

  function onkeydown(e: KeyboardEvent) {
    const el = e.target as HTMLElement;
    const type = el instanceof HTMLInputElement ? el.type : '';
    // Sliders and checkboxes keep their own arrow keys but let every other key through.
    const widget = type === 'range' || type === 'checkbox';
    const typing = el.isContentEditable || el instanceof HTMLTextAreaElement || el instanceof HTMLSelectElement || (el instanceof HTMLInputElement && !widget);
    if (e.key === 'Escape') {
      if (goto !== null) goto = null;
      if (typing || widget) el.blur();
      return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey || !bundle) return;
    if (widget && /^(Arrow|Home|End|Page)/.test(e.key)) return;
    if (sheet?.open && e.key !== '?') return;
    const k = e.key;
    let handled = true;
    if (k === 'j' || k === 'ArrowDown' || k === 'ArrowRight') step(1);
    else if (k === 'k' || k === 'ArrowUp' || k === 'ArrowLeft') step(-1);
    else if (k === 'Home') order.length && select(order[0]);
    else if (k === 'End') order.length && select(order[order.length - 1]);
    else if (k === 'PageDown') step(10);
    else if (k === 'PageUp') step(-10);
    else if (k === 'g') {
      goto = '';
      gotoMsg = '';
      queueMicrotask(() => gotoEl?.focus());
    } else if (k === '/') {
      if (narrow) tab = 'index';
      queueMicrotask(() => filterInput?.focus());
    } else if (k === 'f') plate?.fit();
    else if (k === '0') plate?.oneToOne();
    else if (k === 'o') display.overlay = !display.overlay;
    else if (k === 'c') display.catalogues = !display.catalogues;
    else if (k === '?') sheet?.open ? sheet.close() : sheet?.showModal();
    else if (k >= '1' && k <= '4') setVerdict((['interesting', 'artifact', 'known', 'not_anomalous'] as const)[Number(k) - 1]);
    else if (k === 'Delete' || k === 'Backspace') setVerdict(null);
    else if (k === 's') toggleStar();
    else if (k === 'n') {
      if (narrow) tab = 'readout';
      queueMicrotask(() => noteEl?.focus());
    } else handled = false;
    if (handled) e.preventDefault();
  }

  const save = $derived(verdicts?.state ?? { kind: 'idle' as const });
  const clock = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString('en-CA', { hour12: false });
  };
</script>

<!-- A select keeps focus after a choice, which would swallow J and K, so it lets go once it changes. -->
<svelte:window {onkeydown} onchange={(e) => e.target instanceof HTMLSelectElement && e.target.blur()} />

{#if loadError}
  <main class="fatal">
    <h1>Couldn't Open the Run</h1>
    <p>{loadError}</p>
    <p class="dim">The viewer reads <code>{base}index.json</code>. Point it at another run folder with <code>?run=path/to/run/</code> in the address.</p>
  </main>
{:else if !bundle || !verdicts}
  <main class="fatal">
    <p class="label">Opening Run</p>
  </main>
{:else}
  {@const tile = bundle.tiles[current]}
  <div class="app" class:narrow data-tab={tab}>
    {#if !bundle.writable}
      <div class="demo">
        <span class="label">Demo Run</span>
        <span class="name">{bundle.run.name}</span>
        <span class="dim">Verdicts stay in this browser. Export them from the Verdict block.</span>
        <a href="https://github.com/SomethingObvious/JWSTDetector" target="_blank" rel="noopener noreferrer">JWSTDetector on GitHub</a>
      </div>
    {/if}

    {#if narrow}
      <nav class="tabs" aria-label="Panels">
        <button class="ctl" aria-pressed={tab === 'index'} onclick={() => (tab = 'index')}>Index</button>
        <button class="ctl" aria-pressed={tab === 'plate'} onclick={() => (tab = 'plate')}>Plate</button>
        <button class="ctl" aria-pressed={tab === 'readout'} onclick={() => (tab = 'readout')}>Readout</button>
      </nav>
    {/if}

    <div class="cols">
      <div class="col index-col">
        <IndexList {bundle} {order} {pos} {current} {verdicts} {reviewed} bind:filters bind:filterInput onselect={(i) => { select(i); if (narrow) tab = 'plate'; }} />
      </div>
      <div class="col plate-col">
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
          hasCatalogues={!!bundle.run.xmatch}
          onrenderer={(r) => (renderer = r)}
          oncursor={(c) => (cursor = c)}
        />
      </div>
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
        <form class="goto" onsubmit={(e) => { e.preventDefault(); jump(); }}>
          <label><span class="label">Go to Rank</span><input class="ctl num" bind:this={gotoEl} bind:value={goto} inputmode="numeric" /></label>
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
          <button class="keys" onclick={() => sheet?.showModal()}><kbd>?</kbd> All Keys</button>
        </span>
      {/if}
      {#if narrow}
        <span class="step" role="group" aria-label="Step Through the List">
          <button class="ctl" disabled={pos === 0} onclick={() => step(-1)}>Prev</button>
          <button class="ctl" disabled={pos === order.length - 1} onclick={() => step(1)}>Next</button>
        </span>
      {/if}
      <span class="where">
        <span>Rank <b class="num">{tile.rank}</b></span>
        <span class="dim num">{pos >= 0 ? `${pos + 1} of ${order.length} shown` : 'not in the filtered list'}</span>
      </span>
      <span class="save" role="status">
        {#if save.kind === 'saved'}
          <i class="led ok"></i>Saved {clock(save.at)}
        {:else if save.kind === 'saving'}
          <i class="led busy"></i>Saving {save.pending}
        {:else if save.kind === 'retrying'}
          <i class="led bad"></i>Not saved, {save.pending} waiting. Retrying in {save.inSeconds} s ({save.error})
          <button class="ctl mini" onclick={() => verdicts?.retryNow()}>Retry Now</button>
        {:else if save.kind === 'local'}
          <i class="led ok"></i>{save.count} in this browser
        {:else}
          <i class="led ok"></i>Every verdict saved
        {/if}
      </span>
      <span class="themes" role="group" aria-label="Theme">
        <button class="ctl mini" aria-pressed={theme === 'atlas'} onclick={() => setTheme('atlas')}>Atlas</button>
        <button class="ctl mini" aria-pressed={theme === 'plate'} onclick={() => setTheme('plate')}>Plate</button>
        <button class="ctl mini" aria-pressed={theme === 'classic'} onclick={() => setTheme('classic')}>Classic</button>
      </span>
    </footer>

    <dialog bind:this={sheet} class="sheet" aria-label="Keyboard Keys" onclick={(e) => e.target === sheet && sheet.close()}>
      <div class="sheet-head">
        <h2 class="label">Keys</h2>
        <button class="ctl mini" onclick={() => sheet?.close()}>Close</button>
      </div>
      <dl>
        <dt><kbd>j</kbd> <kbd>Down</kbd> <kbd>Right</kbd></dt><dd>Next candidate</dd>
        <dt><kbd>k</kbd> <kbd>Up</kbd> <kbd>Left</kbd></dt><dd>Previous candidate</dd>
        <dt><kbd>Home</kbd> <kbd>End</kbd></dt><dd>First or last in the list</dd>
        <dt><kbd>PgDn</kbd> <kbd>PgUp</kbd></dt><dd>Ten forward or back</dd>
        <dt><kbd>g</kbd></dt><dd>Go to a rank</dd>
        <dt><kbd>1</kbd></dt><dd>Interesting</dd>
        <dt><kbd>2</kbd></dt><dd>Artifact</dd>
        <dt><kbd>3</kbd></dt><dd>Known object</dd>
        <dt><kbd>4</kbd></dt><dd>Not anomalous</dd>
        <dt><kbd>Del</kbd></dt><dd>Clear the verdict</dd>
        <dt><kbd>s</kbd></dt><dd>Star or unstar</dd>
        <dt><kbd>n</kbd></dt><dd>Write a note, <kbd>Esc</kbd> to leave it</dd>
        <dt><kbd>/</kbd></dt><dd>Filter the index</dd>
        <dt><kbd>f</kbd></dt><dd>Fit the tile</dd>
        <dt><kbd>0</kbd></dt><dd>One image pixel per screen pixel</dd>
        <dt><kbd>o</kbd></dt><dd>Overlay on or off</dd>
        <dt><kbd>c</kbd></dt><dd>Catalogue markers on or off</dd>
        <dt><kbd>?</kbd></dt><dd>This sheet</dd>
      </dl>
      <p class="dim">A verdict moves to the next tile unless you turn that off in the Verdict block. Drag the plate to pan, scroll to zoom and double-click to fit.</p>
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
    height: 19px;
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
</style>
