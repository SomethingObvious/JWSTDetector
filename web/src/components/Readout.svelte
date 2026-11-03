<script lang="ts">
  import Glyph from './Glyph.svelte';
  import Locator from './Locator.svelte';
  import { CMAPS, cssGradient } from '../lib/colormaps';
  import { rankOf, type Limits, type TileData } from '../lib/tiles';
  import type { TileFrame } from '../lib/frame';
  import { VERDICTS, tileId, type Bundle, type Display, type LimitMode, type Stretch, type Theme, type TileSource, type Verdict } from '../lib/types';
  import { catShape, orList, sourceLink } from '../lib/catalogues';
  import { decDms, icrsToGalactic, raHms, signOf, signed } from '../lib/wcs';
  import type { Verdicts } from '../lib/verdicts.svelte';

  let {
    bundle,
    base,
    current,
    data,
    frame,
    cursor,
    display = $bindable(),
    limits,
    verdicts,
    order,
    theme,
    sources,
    learned,
    alike,
    advance = $bindable(),
    noteEl = $bindable(),
    onselect,
    onverdict,
    onstar,
  }: {
    bundle: Bundle;
    base: string;
    current: number;
    data: TileData | null;
    frame: TileFrame | null;
    cursor: { ix: number; iy: number } | null;
    display: Display;
    limits: Limits;
    verdicts: Verdicts;
    order: Int32Array;
    theme: Theme;
    sources: TileSource[];
    learned: { value: number; from: number } | null;
    alike: { index: number; cos: number }[];
    advance: boolean;
    noteEl?: HTMLTextAreaElement;
    onselect: (i: number) => void;
    onverdict: (v: Verdict | null) => void;
    onstar: () => void;
  } = $props();

  const tile = $derived(bundle.tiles[current]);
  const mosaic = $derived(bundle.mosaics[tile.mosaic]);
  const grid = $derived(bundle.run.grid);
  const record = $derived(verdicts.map.get(tile.key));

  // With no pointer on the plate the readout follows the peak patch, so it is never empty.
  const probe = $derived.by(() => {
    if (cursor) return { ...cursor, live: true };
    if (!data) return null;
    const [rows, cols] = grid;
    return { ix: ((tile.peak[1] + 0.5) * data.width) / cols, iy: ((tile.peak[0] + 0.5) * data.height) / rows, live: false };
  });

  const read = $derived.by(() => {
    if (!probe || !data || !frame) return null;
    const px = Math.floor(probe.ix);
    const py = Math.floor(probe.iy);
    const [mx, my] = frame.toMosaic(px + 0.5, py + 0.5);
    const [ra, dec] = frame.toSky(px + 0.5, py + 0.5);
    const [l, b] = icrsToGalactic(ra, dec);
    const i = py * data.width + px;
    let values: number[];
    if (data.planes) values = data.planes.map((p) => p[i]);
    else values = data.channels === 3 ? [data.rgba![i * 4], data.rgba![i * 4 + 1], data.rgba![i * 4 + 2]] : [data.rgba![i * 4]];
    const [rows, cols] = grid;
    const pr = Math.min(rows - 1, Math.floor((probe.iy / data.height) * rows));
    const pc = Math.min(cols - 1, Math.floor((probe.ix / data.width) * cols));
    const dist = data.map2[pr * cols + pc];
    const dist1 = data.map1 ? data.map1[pr * cols + pc] : null;
    return { px, py, mx, my, ra, dec, l, b, values, pr, pc, dist, dist1, pct: rankOf(data.map2Sorted, dist) * 100 };
  });

  // The catalogued source nearest the probe, if one is within an arcsecond.
  const near = $derived.by(() => {
    if (!probe || !data || !mosaic || !sources.length) return null;
    const scale = (mosaic.pixel_scale_arcsec * tile.size) / data.width;
    let best: { s: TileSource; sep: number } | null = null;
    for (const s of sources) {
      const sep = Math.hypot(s[4] * data.width - probe.ix, s[5] * data.height - probe.iy) * scale;
      if (sep <= 1 && (!best || sep < best.sep)) best = { s, sep };
    }
    return best;
  });
  const fmtSep = (sep: number) => (sep >= 60 ? `${(sep / 60).toFixed(1)}\u2032` : `${sep.toFixed(sep < 10 ? 2 : 1)}\u2033`);
  const xm = $derived(bundle.run.xmatch ?? null);

  const fmtVal = (v: number) => {
    if (!Number.isFinite(v)) return 'no data';
    if (!data?.planes) return String(v);
    const a = Math.abs(v);
    return a !== 0 && (a < 1e-3 || a >= 1e5) ? (v < 0 ? signOf(v) : '') + a.toExponential(3) : signed(v, a >= 100 ? 2 : 4);
  };
  const channelNames = $derived(data?.channels === 3 ? (mosaic?.filters.length === 3 ? mosaic.filters : ['R', 'G', 'B']) : [mosaic?.filters[0] ?? 'Value']);

  const delta = $derived(
    tile.score !== null && tile.score_pass1 !== undefined ? tile.score - tile.score_pass1 : null
  );
  const arcsec = $derived((mosaic?.pixel_scale_arcsec ?? 0) * tile.size);

  // Display block.
  const STRETCHES: { id: Stretch; label: string }[] = [
    { id: 'linear', label: 'Linear' },
    { id: 'sqrt', label: 'Sqrt' },
    { id: 'log', label: 'Log' },
    { id: 'asinh', label: 'Asinh' },
    { id: 'histeq', label: 'Hist Eq' },
  ];
  const LIMITS: { id: LimitMode; label: string }[] = [
    { id: 'minmax', label: 'Min Max' },
    { id: 'p995', label: '99.5%' },
    { id: 'p99', label: '99%' },
    { id: 'p95', label: '95%' },
    { id: 'zscale', label: 'ZScale' },
    { id: 'manual', label: 'Manual' },
  ];

  function setManual(plane: number, which: 'lo' | 'hi', value: string) {
    const v = parseFloat(value);
    if (!Number.isFinite(v)) return;
    const next = { lo: [...limits.lo], hi: [...limits.hi] };
    next[which][plane] = v;
    display.lo = next.lo;
    display.hi = next.hi;
    display.limitMode = 'manual';
  }

  // One histogram line per plane, on a shared axis, with the limits marked.
  const hist = $derived.by(() => {
    if (!data || !data.samples[0]?.length) return null;
    // The axis covers the display window with a margin either side, so the limits always show.
    const wlo = Math.min(...limits.lo);
    const whi = Math.max(...limits.hi);
    const pad = (whi - wlo || 1) * 0.15;
    const lo = wlo - pad;
    const hi = whi + pad;
    const span = hi - lo || 1;
    const at = (v: number) => Math.max(0, Math.min(100, ((v - lo) / span) * 100));
    const paths = data.samples.map((s) => {
      const bins = new Array(72).fill(0);
      for (const v of s) {
        const k = Math.floor(((v - lo) / span) * bins.length);
        if (k >= 0 && k < bins.length) bins[k]++;
      }
      const top = Math.log1p(Math.max(...bins));
      return bins.map((c, k) => `${k === 0 ? 'M' : 'L'}${(k / (bins.length - 1)) * 100} ${30 - (Math.log1p(c) / top) * 28}`).join(' ');
    });
    return { paths, lo: limits.lo.map(at), hi: limits.hi.map(at) };
  });

  const fmtLimit = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(4) : v.toPrecision(4));

  // Header block.
  let headerQuery = $state('');
  const cards = $derived.by(() => {
    const q = headerQuery.trim().toLowerCase();
    const all = Object.entries(mosaic?.header ?? {});
    return q ? all.filter(([k, v]) => k.toLowerCase().includes(q) || String(v).toLowerCase().includes(q)) : all;
  });

  // Note, saved after a pause in typing and whenever the tile changes.
  let draft = $state('');
  let draftKey = '';
  let timer: ReturnType<typeof setTimeout> | null = null;
  $effect(() => {
    const key = tile.key;
    if (key !== draftKey) {
      commitNote();
      draftKey = key;
      draft = verdicts.map.get(key)?.note ?? '';
    }
  });
  function commitNote() {
    if (timer) clearTimeout(timer);
    timer = null;
    if (draftKey && (verdicts.map.get(draftKey)?.note ?? '') !== draft) verdicts.update(draftKey, { note: draft });
  }
  function typed() {
    if (timer) clearTimeout(timer);
    timer = setTimeout(commitNote, 700);
  }

  const save = $derived(verdicts.state);
  let copied = $state('');
  async function copy(text: string, what: string) {
    try {
      await navigator.clipboard.writeText(text);
      copied = what;
    } catch {
      copied = 'blocked';
    }
    setTimeout(() => (copied = ''), 1600);
  }

  const radius = $derived(Math.max(5, (arcsec / 2) * Math.SQRT2));
  const links = $derived.by(() => {
    const ra = tile.ra.toFixed(6);
    const dec = tile.dec.toFixed(6);
    const fovDeg = Math.max(0.01, (arcsec * 2) / 3600).toFixed(4);
    const pos = encodeURIComponent(`${ra} ${dec}`);
    return [
      { label: 'ESASky', href: `https://sky.esa.int/esasky/?target=${pos}&fov=${fovDeg}&cooframe=J2000&sci=true` },
      { label: 'Aladin Lite', href: `https://aladin.cds.unistra.fr/AladinLite/?target=${pos}&fov=${fovDeg}&survey=CDS%2FP%2FDSS2%2Fcolor` },
      { label: 'SIMBAD Cone', href: `https://simbad.cds.unistra.fr/simbad/sim-coo?Coord=${pos}&CooFrame=ICRS&Radius=${radius.toFixed(1)}&Radius.unit=arcsec&submit=submit%20query` },
      { label: 'NED Cone', href: `https://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search&in_csys=Equatorial&in_equinox=J2000&coordinates=${encodeURIComponent(`${ra}d ${dec}d`)}&radius=${(radius / 60).toFixed(3)}` },
      { label: 'MAST Portal', href: `https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery=${pos}` },
    ];
  });

  // Import and export, for the hosted demo where verdicts stay in the browser.
  let importMsg = $state('');
  function exportCsv() {
    const blob = new Blob([verdicts.toCsv(bundle)], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `${bundle.run.name}-verdicts.csv`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  async function importFile(e: Event & { currentTarget: HTMLInputElement }) {
    const f = e.currentTarget.files?.[0];
    e.currentTarget.value = '';
    if (!f) return;
    try {
      const n = verdicts.importText(await f.text());
      importMsg = `Took ${n} verdicts from ${f.name}.`;
    } catch (err) {
      importMsg = `Couldn't read ${f.name}. ${err instanceof Error ? err.message : String(err)}`;
    }
  }
</script>

{#snippet head(title: string, aside?: string)}
  <h2 class="bh">
    <span class="label">{title}</span>
    <span class="rule"></span>
    {#if aside}<span class="aside num">{aside}</span>{:else}<span class="node"></span>{/if}
  </h2>
{/snippet}

<aside class="readout" aria-label="Readout">
  <section class="block">
    {@render head(probe?.live ? 'Cursor' : 'Peak Patch', probe?.live ? undefined : 'pointer off plate')}
    {#if read}
      <dl class="facts">
        <dt>Tile px</dt>
        <dd class="num">{read.px}, {read.py}</dd>
        <dt>Mosaic px</dt>
        <dd class="num">{read.mx.toFixed(1)}, {read.my.toFixed(1)}</dd>
        <dt>RA</dt>
        <dd class="num"><b>{raHms(read.ra, 3)}</b><span class="sub">{read.ra.toFixed(6)}&deg;</span></dd>
        <dt>Dec</dt>
        <dd class="num"><b>{decDms(read.dec, 2)}</b><span class="sub">{signed(read.dec, 6)}&deg;</span></dd>
        <dt>Galactic</dt>
        <dd class="num">l {read.l.toFixed(5)}&deg; <span class="gap"></span> b {signed(read.b, 5)}&deg;</dd>
        <dt>Value</dt>
        <dd class="num values">
          {#each read.values as v, k (k)}
            <span><i>{channelNames[k]}</i> {fmtVal(v)}</span>
          {/each}
          <span class="dim unit">{data?.bunit}</span>
        </dd>
        <dt>Patch</dt>
        <dd class="num">
          r{read.pr} c{read.pc} <span class="gap"></span> <b>{read.dist.toFixed(4)}</b>
          {#if read.dist1 !== null}<span class="sub">pass 1 {read.dist1.toFixed(4)}</span>{/if}
        </dd>
        <dt>Percentile</dt>
        <dd class="num"><span class="pbar"><i style:width="{read.pct}%"></i></span>{read.pct.toFixed(1)} within tile</dd>
        {#if near}
          <dt>Source</dt>
          <dd><span class="catname">{near.s[0]}</span> {near.s[1]} <span class="dim num">{fmtSep(near.sep)}</span></dd>
        {/if}
      </dl>
    {:else}
      <p class="quiet">Waiting for the tile to load.</p>
    {/if}
  </section>

  <section class="block">
    {@render head('Candidate')}
    <dl class="facts">
      <dt>Rank</dt>
      <dd class="num"><b>{tile.rank.toLocaleString('en-CA')}</b> <span class="dim">of {bundle.tiles.length.toLocaleString('en-CA')}</span></dd>
      <dt>Score</dt>
      <dd class="num">{#if tile.score === null}<b>unscored</b> <span class="dim">no reference tile was left to compare with</span>{:else}<b>{tile.score.toFixed(4)}</b> <span class="dim">{bundle.run.score_mode}</span>{/if}</dd>
      {#if tile.score_pass1 !== undefined && delta !== null}
        <dt>Pass 1</dt>
        <dd class="num">{tile.score_pass1.toFixed(4)} <span class="delta" class:up={delta > 0}>{delta > 0 ? '+' : ''}{signed(delta, 4)}</span></dd>
      {/if}
      {#if learned}
        <dt>Learned</dt>
        <dd class="num">
          {#if learned.from}<b>{learned.value.toFixed(2)}</b> <span class="dim">from {learned.from} {learned.from === 1 ? 'verdict' : 'verdicts'}, 0.5 is neutral</span>
          {:else}<span class="dim">No verdicts to learn from yet</span>{/if}
        </dd>
      {/if}
      <dt>Reference</dt>
      <dd>{tile.in_reference ? 'In the pass 2 reference set' : 'Not a reference tile'}</dd>
      <dt>Mosaic</dt>
      <dd class="name mosaic" title={tile.mosaic}>{tile.mosaic}</dd>
      <dt>Centre</dt>
      <dd class="num"><b>{raHms(tile.ra)}</b> <b>{decDms(tile.dec)}</b><span class="sub">{tile.ra.toFixed(5)}&deg; {signed(tile.dec, 5)}&deg;</span></dd>
      <dt>Size</dt>
      <dd class="num">{tile.size} px <span class="gap"></span> {arcsec.toFixed(2)}&Prime;</dd>
      <dt>Peak</dt>
      <dd class="num">r{tile.peak[0]} c{tile.peak[1]} <span class="dim">of {grid[0]} &times; {grid[1]}</span></dd>
      {#if tile.owned}
        {@const [r0, r1, c0, c1] = tile.owned}
        <dt>Scored</dt>
        <dd>
          <span class="num">{r1 - r0 + 1} &times; {c1 - c0 + 1}</span> patches inside the dashed outline
          <span class="dim small under">Overlapping tiles score the rest, each the part nearest its own centre.</span>
        </dd>
      {/if}
      <dt>Pixels</dt>
      <dd>
        {#if data}
          {data.source === 'cutout' ? 'FITS cutout' : 'PNG'} <span class="num">{data.width} &times; {data.height}</span>{data.channels === 3 ? ', RGB' : ''}
          {#if data.fallback}<span class="warn">The cutout failed ({data.fallback}), so this is the PNG.</span>{/if}
        {:else}
          <span class="dim">Loading</span>
        {/if}
      </dd>
    </dl>
  </section>

  {#if xm}
    <section class="block">
      {@render head('Catalogues', `${xm.radius_arcsec}\u2033 match radius`)}
      {#if tile.xmatch}
        {#if tile.xmatch.uncatalogued}
          <p class="lead new"><b>Uncatalogued.</b> Nothing in {orList(xm.checked)} lies within {xm.radius_arcsec}&Prime; of the peak.</p>
        {/if}
        <ul class="matches">
          {#each xm.checked as cat (cat)}
            {@const m = tile.xmatch.nearest[cat]}
            {@const href = m ? sourceLink(cat, m.name) : null}
            <li class:hit={!!m && m.sep <= xm.radius_arcsec}>
              <svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true"><path d={catShape(cat)} /></svg>
              <span class="catname">{cat}</span>
              {#if m}
                <span class="who">
                  {#if href}<a {href} target="_blank" rel="noopener noreferrer">{m.name}</a>{:else}{m.name}{/if}
                  {#if m.type || m.note}<span class="dim">{[m.type, m.note].filter(Boolean).join(', ')}</span>{/if}
                </span>
                <span class="num sep">{fmtSep(m.sep)}</span>
              {:else}
                <span class="who dim">Nothing over this mosaic</span>
              {/if}
            </li>
          {/each}
        </ul>
        <p class="quiet">
          The nearest source to the peak in each.
          {#if sources.length}{sources.length} catalogued {sources.length === 1 ? 'source lies' : 'sources lie'} on this tile, marked on the plate.{/if}
          {#if xm.failed.length}{orList(xm.failed)} couldn't be reached when this run was matched, so {xm.failed.length === 1 ? 'it was' : 'they were'} left out.{/if}
        </p>
      {:else}
        <p class="quiet">This tile's mosaic has no WCS, so it wasn't matched.</p>
      {/if}
    </section>
  {/if}

  <section class="block">
    {@render head('Verdict', record?.updated ? record.updated.replace('T', ' ').replace('Z', ' UTC') : undefined)}
    <div class="verdicts">
      {#each VERDICTS as v (v.id)}
        <button class="ctl vbtn" aria-pressed={record?.verdict === v.id} onclick={() => onverdict(v.id)}>
          <Glyph verdict={v.id} />
          <span>{v.label}</span>
          <kbd>{v.key}</kbd>
        </button>
      {/each}
      <button class="ctl vbtn" onclick={() => onverdict(null)} disabled={!record?.verdict}>
        <Glyph verdict={null} />
        <span>Clear</span>
        <kbd>Del</kbd>
      </button>
      <button class="ctl vbtn" aria-pressed={!!record?.star} onclick={onstar}>
        <Glyph verdict={record?.star ? 'star' : 'star-off'} size={11} />
        <span>{record?.star ? 'Starred' : 'Star'}</span>
        <kbd>s</kbd>
      </button>
    </div>
    <label class="note">
      <span class="fl label">Note <kbd>n</kbd></span>
      <textarea
        bind:this={noteEl}
        bind:value={draft}
        oninput={typed}
        onblur={commitNote}
        rows="3"
        placeholder="What you see here, for whoever follows it up"
      ></textarea>
    </label>
    <div class="saverow">
      <label class="check"><input type="checkbox" bind:checked={advance} /> Next tile after a verdict</label>
    </div>
    {#if !bundle.writable}
      <div class="io">
        <button class="ctl" onclick={exportCsv}>Export CSV</button>
        <label class="ctl file">Import<input type="file" accept=".csv,.json,text/csv,application/json" onchange={importFile} /></label>
        <span class="dim small">
          {#if save.kind === 'local'}{save.count} in this browser{/if}
          {#if !verdicts.storageOk}This browser is blocking storage, so export before you close the tab.{/if}
        </span>
      </div>
      {#if importMsg}<p class="quiet">{importMsg}</p>{/if}
    {/if}
  </section>

  {#if alike.length}
    <section class="block">
      {@render head('Similar Tiles', 'cosine')}
      <div class="alike" class:neg={theme === 'plate'}>
        {#each alike as a (a.index)}
          {@const t = bundle.tiles[a.index]}
          {@const v = verdicts.map.get(t.key)}
          <button class="thumb" onclick={() => onselect(a.index)} title="Rank {t.rank}, cosine {a.cos.toFixed(3)}">
            <img src="{base}tiles/{tileId(a.index)}.png" alt="Tile ranked {t.rank}" loading="lazy" decoding="async" />
            <span class="cap">
              <span class="num">{t.rank}</span>
              <Glyph verdict={v?.verdict ?? null} size={9} />
              <span class="num dim">{a.cos.toFixed(2)}</span>
            </span>
          </button>
        {/each}
      </div>
      <p class="quiet">Closest by the descriptor of each tile's most anomalous patches. A verdict here moves every tile like it once the index is sorted by Learned.</p>
    </section>
  {/if}

  <section class="block">
    {@render head('Mosaic Locator')}
    <Locator {bundle} {base} {current} {order} {theme} {onselect} />
  </section>

  <section class="block">
    {@render head('Display', data?.channels === 3 ? 'RGB' : undefined)}
    {#if data && !data.planes}
      <p class="quiet">This tile has no FITS cutout, so it shows the 8-bit PNG as it was cut. Stretch and limits work on cutouts.</p>
    {/if}
    <div class="field">
      <span class="fl label">Stretch</span>
      <div class="seg wrap" role="group" aria-label="Stretch">
        {#each STRETCHES as s (s.id)}
          <button class="ctl" aria-pressed={display.stretch === s.id} disabled={!!data && !data.planes} onclick={() => (display.stretch = s.id)}>{s.label}</button>
        {/each}
      </div>
    </div>
    <div class="field">
      <span class="fl label">Limits</span>
      <div class="seg wrap" role="group" aria-label="Limits">
        {#each LIMITS as m (m.id)}
          <button
            class="ctl"
            aria-pressed={display.limitMode === m.id}
            disabled={!!data && !data.planes}
            onclick={() => {
              if (m.id === 'manual') {
                display.lo = [...limits.lo];
                display.hi = [...limits.hi];
              }
              display.limitMode = m.id;
            }}>{m.label}</button
          >
        {/each}
      </div>
    </div>
    {#if hist}
      <svg class="hist" class:rgb={hist.paths.length === 3} viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
        {#each hist.paths as d, c (c)}
          <rect class="win p{c}" x={hist.lo[c]} width={Math.max(0.5, hist.hi[c] - hist.lo[c])} y={c * 2} height={32 - c * 4} />
          <path class="p{c}" {d} />
          <line class="p{c}" x1={hist.lo[c]} x2={hist.lo[c]} y1="0" y2="32" />
          <line class="p{c}" x1={hist.hi[c]} x2={hist.hi[c]} y1="0" y2="32" />
        {/each}
      </svg>
    {/if}
    <div class="limits" class:rgb={limits.lo.length === 3}>
      <span></span>
      <span class="label">Low</span>
      <span class="label">High</span>
      {#each limits.lo as lo, c (c)}
        <span class="band num">{channelNames[c]}</span>
        <input class="ctl num" type="number" step="any" aria-label="{channelNames[c]} low" value={fmtLimit(lo)} onchange={(e) => setManual(c, 'lo', e.currentTarget.value)} />
        <input class="ctl num" type="number" step="any" aria-label="{channelNames[c]} high" value={fmtLimit(limits.hi[c])} onchange={(e) => setManual(c, 'hi', e.currentTarget.value)} />
      {/each}
      <span></span>
      <span class="unit dim">{data?.bunit ?? ''}</span>
    </div>
    <div class="field">
      <span class="fl label">Colour</span>
      <div class="cmap">
        <select class="ctl" bind:value={display.cmap} disabled={data?.channels === 3 && !display.mono} aria-label="Colour map">
          {#each CMAPS as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
        <i class="swatch" style:background={data?.channels === 3 && !display.mono ? 'linear-gradient(90deg, #d33 0 33%, #3b3 33% 66%, #36d 66%)' : cssGradient(display.cmap, display.invert)}></i>
        <label class="check"><input type="checkbox" bind:checked={display.invert} /> Invert</label>
      </div>
    </div>
    {#if data?.channels === 3}
      <div class="field">
        <span class="fl label">Bands</span>
        <div class="seg" role="group" aria-label="Colour or one band">
          <button class="ctl" aria-pressed={!display.mono} onclick={() => (display.mono = false)}>RGB</button>
          <button class="ctl" aria-pressed={display.mono} onclick={() => (display.mono = true)}>Mean Through Map</button>
        </div>
      </div>
      <p class="quiet">{channelNames.join(', ')} share the stretch, and each band gets its own limits.</p>
    {/if}
  </section>

  <section class="block">
    {@render head('Links Out', 'tile centre')}
    <ul class="links">
      {#each links as l (l.label)}
        <li>
          <a href={l.href} target="_blank" rel="noopener noreferrer">
            {l.label}
            <svg width="9" height="9" viewBox="0 0 9 9" aria-hidden="true"><path d="M3 1.5 H7.5 V6 M7.5 1.5 L1.5 7.5" fill="none" stroke="currentColor" stroke-width="1.1" /></svg>
          </a>
        </li>
      {/each}
    </ul>
    <div class="io">
      <button class="ctl" onclick={() => copy(`${tile.ra.toFixed(6)} ${tile.dec.toFixed(6)}`, 'deg')}>{copied === 'deg' ? 'Copied' : 'Copy Degrees'}</button>
      <button class="ctl" onclick={() => copy(`${raHms(tile.ra)} ${tile.dec < 0 ? '-' : '+'}${decDms(Math.abs(tile.dec)).slice(1)}`, 'sexa')}>{copied === 'sexa' ? 'Copied' : 'Copy Sexagesimal'}</button>
      {#if copied === 'blocked'}<span class="dim small">The browser blocked the clipboard.</span>{/if}
    </div>
  </section>

  <section class="block">
    {@render head('Header', `${cards.length} cards`)}
    <input class="ctl hq" type="search" placeholder="Search keywords and values" bind:value={headerQuery} spellcheck="false" />
    <div class="cards">
      {#each cards as [k, v] (k)}
        <div class="card"><span class="k num">{k}</span><span class="v num">{v}</span></div>
      {:else}
        <p class="quiet">No header card matches {headerQuery}.</p>
      {/each}
    </div>
  </section>

  <section class="block">
    {@render head('Run')}
    <dl class="facts">
      <dt>Name</dt>
      <dd class="name">{bundle.run.name}</dd>
      <dt>Model</dt>
      <dd class="num small">{bundle.run.model}</dd>
      <dt>Detector</dt>
      <dd>{bundle.run.detector}, {bundle.run.passes} {bundle.run.passes === 1 ? 'pass' : 'passes'}, {bundle.run.n_reference} references</dd>
      <dt>Backbone</dt>
      <dd class="num">{bundle.run.resolution} px, patch {bundle.run.patch_size}, grid {grid[0]} &times; {grid[1]}</dd>
    </dl>
  </section>
</aside>

<style>
  .readout {
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-gutter: stable;
    min-height: 0;
    border-left: 1px solid var(--rule);
    background: var(--ground-2);
  }
  .block {
    padding: 10px 14px 12px;
    border-bottom: 1px solid var(--rule);
    display: grid;
    gap: 8px;
  }
  .bh {
    margin: 0;
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: inherit;
    font-weight: inherit;
  }
  .bh .label {
    color: var(--anno);
  }
  .bh .rule {
    flex: 1;
    height: 1px;
    background: var(--rule);
  }
  .bh .node {
    width: 4px;
    height: 4px;
    background: var(--anno);
    transform: rotate(45deg);
    opacity: 0.7;
  }
  .bh .aside {
    color: var(--chalk-3);
    font-size: 10px;
  }
  .facts {
    display: grid;
    grid-template-columns: 74px 1fr;
    gap: 3px 10px;
    margin: 0;
    align-items: baseline;
  }
  dt {
    color: var(--chalk-3);
    font-size: 11px;
  }
  dd {
    margin: 0;
    min-width: 0;
    color: var(--chalk);
  }
  dd b {
    font-weight: 500;
    color: var(--anno);
  }
  dd .sub {
    display: block;
    color: var(--chalk-3);
  }
  .dim {
    color: var(--chalk-3);
  }
  .gap {
    display: inline-block;
    width: 8px;
  }
  .values {
    display: flex;
    flex-wrap: wrap;
    gap: 2px 10px;
  }
  .values i {
    font-style: normal;
    color: var(--chalk-3);
  }
  .values .unit {
    display: inline;
  }
  .pbar {
    display: inline-block;
    width: 48px;
    height: 3px;
    background: var(--bar-track);
    vertical-align: middle;
    margin-right: 8px;
    position: relative;
  }
  .pbar i {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--bar);
  }
  .delta {
    color: var(--chalk-2);
    margin-left: 6px;
  }
  .delta.up {
    color: var(--anno);
  }
  .mosaic {
    font-size: 13.5px;
    overflow-wrap: anywhere;
  }
  .warn {
    display: block;
    color: var(--chalk-2);
    font-size: 11px;
  }
  .quiet {
    margin: 0;
    color: var(--chalk-2);
    font-size: 11.5px;
  }
  .small {
    font-size: 11px;
  }
  .under {
    display: block;
    margin-top: 2px;
  }
  .field {
    display: grid;
    grid-template-columns: 62px 1fr;
    gap: 8px;
    align-items: start;
  }
  .fl {
    color: var(--chalk-3);
    padding-top: 6px;
  }
  .seg {
    display: flex;
  }
  .seg.wrap {
    flex-wrap: wrap;
    gap: 4px;
  }
  .seg .ctl {
    padding: 0 7px;
    font-size: 11.5px;
  }
  .seg:not(.wrap) .ctl + .ctl {
    margin-left: -1px;
  }
  .hist {
    width: 100%;
    height: 34px;
    display: block;
    border-bottom: 1px solid var(--rule);
  }
  .hist path {
    fill: none;
    stroke: var(--chalk-2);
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }
  .hist line {
    stroke: var(--anno);
    stroke-width: 1;
    vector-effect: non-scaling-stroke;
  }
  .hist .win {
    fill: var(--anno-wash);
  }
  .limits {
    display: grid;
    grid-template-columns: 62px 1fr 1fr;
    gap: 4px 8px;
    align-items: center;
  }
  .limits > .label {
    color: var(--chalk-3);
    font-size: 9.5px;
  }
  .band {
    color: var(--chalk-3);
    font-size: 10.5px;
  }
  .limits input {
    width: 100%;
    min-width: 0;
    color: var(--anno);
    padding: 0 6px;
    height: 22px;
  }
  .limits .unit {
    font-size: 10.5px;
    grid-column: 2 / 4;
  }
  .hist.rgb .p0 {
    stroke: #e0605a;
  }
  .hist.rgb .p1 {
    stroke: #5fbf6a;
  }
  .hist.rgb .p2 {
    stroke: #6a8fe6;
  }
  .hist.rgb .win {
    fill: none;
  }
  .cmap {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cmap select {
    width: 96px;
  }
  .swatch {
    flex: 1;
    height: 8px;
    border: 1px solid var(--rule);
  }
  .check {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--chalk-2);
    cursor: pointer;
    white-space: nowrap;
  }
  .check input {
    accent-color: var(--anno);
    margin: 0;
  }
  .verdicts {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px;
  }
  .vbtn {
    display: flex;
    align-items: center;
    gap: 7px;
    height: 28px;
    color: var(--chalk);
  }
  .vbtn span {
    flex: 1;
    text-align: left;
  }
  .vbtn[aria-pressed='true'] span {
    color: var(--anno);
  }
  .note {
    display: grid;
    gap: 4px;
  }
  .note .fl {
    padding: 0;
    display: flex;
    gap: 6px;
    align-items: center;
  }
  textarea {
    resize: vertical;
    min-height: 54px;
    background: var(--ground-sunk);
    border: 1px solid var(--rule);
    border-radius: 2px;
    padding: 6px 8px;
    color: var(--chalk);
    line-height: 1.45;
  }
  textarea:focus {
    border-color: var(--anno);
    outline: none;
  }
  .io {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .file {
    display: inline-grid;
    place-items: center;
    position: relative;
  }
  .file input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }
  .file:focus-within {
    outline: 1px solid var(--anno);
  }
  .catname {
    color: var(--chalk-3);
    font-size: 11px;
  }
  .lead {
    margin: 0;
    line-height: 1.45;
  }
  .lead.new b {
    color: var(--cat);
    font-weight: 500;
  }
  .matches {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 4px;
  }
  .matches li {
    display: grid;
    grid-template-columns: 12px 62px minmax(0, 1fr) auto;
    gap: 8px;
    align-items: baseline;
  }
  .matches svg {
    align-self: center;
    overflow: visible;
  }
  .matches path {
    fill: none;
    stroke: var(--cat);
    stroke-width: 1.25;
  }
  .who {
    min-width: 0;
    overflow-wrap: anywhere;
  }
  .who .dim {
    display: block;
    font-size: 11px;
  }
  .sep {
    color: var(--chalk-2);
  }
  .matches li.hit .sep {
    color: var(--anno);
  }
  .alike {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 6px;
  }
  .thumb {
    display: grid;
    gap: 3px;
    padding: 0;
    border: 1px solid var(--rule);
    background: var(--ground-sunk);
    cursor: pointer;
    color: var(--chalk);
  }
  .thumb:hover,
  .thumb:focus-visible {
    border-color: var(--anno);
    outline: none;
  }
  .thumb img {
    width: 100%;
    aspect-ratio: 1;
    display: block;
    object-fit: cover;
  }
  .neg img {
    filter: grayscale(1) invert(1);
  }
  .cap {
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 0 5px 3px;
    font-size: 10.5px;
  }
  .cap .dim {
    margin-left: auto;
  }
  .links {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 12px;
  }
  .links a {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    text-decoration: none;
    padding: 2px 0;
  }
  .links a:hover {
    text-decoration: underline;
  }
  .hq {
    width: 100%;
  }
  .cards {
    max-height: 240px;
    overflow-y: auto;
    border-top: 1px solid var(--rule);
  }
  .card {
    display: grid;
    grid-template-columns: 82px 1fr;
    gap: 8px;
    padding: 2px 0;
    border-bottom: 1px solid var(--rule);
    font-size: 11px;
  }
  .card .k {
    color: var(--anno);
    font-size: 11px;
  }
  .card .v {
    color: var(--chalk);
    overflow-wrap: anywhere;
    font-size: 11px;
  }
</style>
