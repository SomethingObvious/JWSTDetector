<script lang="ts" module>
  export type SortKey = 'rank' | 'priority' | 'score' | 'delta' | 'ra' | 'dec';
  export type MatchFilter = 'all' | 'uncatalogued' | 'catalogued';
  export type VerdictFilter = 'all' | 'unreviewed' | 'interesting' | 'artifact' | 'known' | 'not_anomalous' | 'starred';
  export interface Filters {
    text: string;
    verdict: VerdictFilter;
    match: MatchFilter;
    mosaic: string;
    scoreMin: number | null;
    scoreMax: number | null;
    sort: SortKey;
    desc: boolean;
  }
</script>

<script lang="ts">
  import Glyph from './Glyph.svelte';
  import { shortMosaic, type Bundle } from '../lib/types';
  import { raHms, decDms } from '../lib/wcs';
  import { orList } from '../lib/catalogues';
  import type { Verdicts } from '../lib/verdicts.svelte';

  let {
    bundle,
    order,
    pos,
    current,
    verdicts,
    reviewed,
    filters = $bindable(),
    filterInput = $bindable(),
    onselect,
  }: {
    bundle: Bundle;
    order: Int32Array;
    pos: number;
    current: number;
    verdicts: Verdicts;
    reviewed: number;
    filters: Filters;
    filterInput?: HTMLInputElement;
    onselect: (index: number) => void;
  } = $props();

  const ROW = 40;
  let scroller: HTMLDivElement | undefined = $state();
  let scrollTop = $state(0);
  let height = $state(600);

  const scoreRange = $derived.by(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (const t of bundle.tiles) {
      if (t.score === null) continue;
      if (t.score < lo) lo = t.score;
      if (t.score > hi) hi = t.score;
    }
    if (lo > hi) return [0, 1] as const;
    return [lo, hi === lo ? lo + 1 : hi] as const;
  });

  const first = $derived(Math.max(0, Math.floor(scrollTop / ROW) - 4));
  const last = $derived(Math.min(order.length, Math.ceil((scrollTop + height) / ROW) + 4));
  const visible = $derived(Array.from({ length: Math.max(0, last - first) }, (_, k) => first + k));
  const mosaicNames = $derived(Object.keys(bundle.mosaics));
  const xmatch = $derived(bundle.run.xmatch);
  const unmatched = $derived(bundle.tiles.reduce((n, t) => n + (t.xmatch?.uncatalogued ? 1 : 0), 0));
  const matched = $derived(bundle.tiles.reduce((n, t) => n + (t.xmatch && !t.xmatch.uncatalogued ? 1 : 0), 0));

  // Keeps the current row in view without animating, so stepping never waits on a scroll.
  $effect(() => {
    if (!scroller || pos < 0) return;
    const top = pos * ROW;
    const st = scroller.scrollTop;
    if (top < st + ROW) scroller.scrollTop = Math.max(0, top - ROW);
    else if (top + ROW > st + height - ROW) scroller.scrollTop = top + 2 * ROW - height;
  });

  function num(v: string): number | null {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
  }
</script>

<section class="index" aria-label="Ranked Candidates">
  <div class="controls">
    <label class="search">
      <span class="sr">Filter</span>
      <input
        class="ctl"
        type="search"
        placeholder="Filter by key, mosaic or note"
        spellcheck="false"
        bind:value={filters.text}
        bind:this={filterInput}
      />
      <kbd>/</kbd>
    </label>
    <div class="row2">
      <select class="ctl" bind:value={filters.verdict} aria-label="Verdict filter">
        <option value="all">All Verdicts</option>
        <option value="unreviewed">Unreviewed</option>
        <option value="interesting">Interesting</option>
        <option value="artifact">Artifact</option>
        <option value="known">Known Object</option>
        <option value="not_anomalous">Not Anomalous</option>
        <option value="starred">Starred</option>
      </select>
      <select class="ctl" bind:value={filters.mosaic} aria-label="Mosaic filter">
        <option value="">All Mosaics</option>
        {#each mosaicNames as m (m)}
          <option value={m}>{shortMosaic(m)}</option>
        {/each}
      </select>
    </div>
    {#if xmatch}
      <div class="row2">
        <select class="ctl" bind:value={filters.match} aria-label="Catalogue filter" title="Uncatalogued means nothing in {orList(xmatch.checked)} within {xmatch.radius_arcsec}&Prime; of the peak">
          <option value="all">Catalogued or Not</option>
          <option value="uncatalogued">Uncatalogued ({unmatched.toLocaleString('en-CA')})</option>
          <option value="catalogued">Catalogued ({matched.toLocaleString('en-CA')})</option>
        </select>
      </div>
    {/if}
    <div class="row2">
      <span class="range">
        <span class="label">Score</span>
        <input
          class="ctl num"
          type="number"
          step="0.01"
          aria-label="Lowest score"
          placeholder={scoreRange[0].toFixed(2)}
          value={filters.scoreMin ?? ''}
          oninput={(e) => (filters.scoreMin = num(e.currentTarget.value))}
        />
        <input
          class="ctl num"
          type="number"
          step="0.01"
          aria-label="Highest score"
          placeholder={scoreRange[1].toFixed(2)}
          value={filters.scoreMax ?? ''}
          oninput={(e) => (filters.scoreMax = num(e.currentTarget.value))}
        />
      </span>
      <span class="sort">
        <select class="ctl" bind:value={filters.sort} aria-label="Sort by">
          <option value="rank">Rank</option>
          {#if bundle.run.embedding_dims}<option value="priority">Learned</option>{/if}
          <option value="score">Score</option>
          <option value="delta">Pass 1 Change</option>
          <option value="ra">RA</option>
          <option value="dec">Dec</option>
        </select>
        <button class="ctl dir" onclick={() => (filters.desc = !filters.desc)} aria-label={filters.desc ? 'Descending, switch to ascending' : 'Ascending, switch to descending'}>
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" class:flip={filters.desc}>
            <path d="M2 6.5 L5 3.5 L8 6.5" fill="none" stroke="currentColor" stroke-width="1.3" />
          </svg>
        </button>
      </span>
    </div>
    <div class="counts">
      <span><b class="num">{order.length.toLocaleString('en-CA')}</b> shown</span>
      <span><b class="num">{reviewed.toLocaleString('en-CA')}</b> reviewed</span>
      <span><b class="num">{bundle.tiles.length.toLocaleString('en-CA')}</b> total</span>
    </div>
  </div>

  <div class="list" bind:this={scroller} bind:clientHeight={height} onscroll={(e) => (scrollTop = e.currentTarget.scrollTop)} role="listbox" aria-label="Candidates" tabindex="-1">
    {#if order.length === 0}
      <p class="empty">Nothing matches these filters. Clear the text or set the verdict filter back to All Verdicts to see every tile.</p>
    {/if}
    <div class="spacer" style:height="{order.length * ROW}px">
      {#each visible as k (order[k])}
        {@const i = order[k]}
        {@const t = bundle.tiles[i]}
        {@const v = verdicts.map.get(t.key)}
        <button
          class="item"
          class:current={i === current}
          style:transform="translateY({k * ROW}px)"
          role="option"
          aria-selected={i === current}
          tabindex="-1"
          onclick={() => onselect(i)}
        >
          <span class="rank num">{t.rank}</span>
          <span class="mark"><Glyph verdict={v?.verdict ?? null} /></span>
          <span class="mark">{#if v?.star}<Glyph verdict="star" size={11} />{/if}</span>
          <span class="bar" aria-hidden="true"><i style:width="{t.score === null ? 0 : Math.max(2, ((t.score - scoreRange[0]) / (scoreRange[1] - scoreRange[0])) * 100)}%"></i></span>
          <span class="score num">{t.score === null ? 'unscored' : t.score.toFixed(4)}</span>
          <span class="line2">
            <span class="mname name">{shortMosaic(t.mosaic)}</span>
            {#if t.xmatch?.uncatalogued}<svg class="new" width="9" height="9" viewBox="-4.5 -4.5 9 9" role="img" aria-label="Uncatalogued"><title>Uncatalogued, nothing within {xmatch?.radius_arcsec}&Prime; of the peak</title><circle r="3.4" /></svg>{/if}
            <span class="coord num">{raHms(t.ra, 1)} {decDms(t.dec, 0)}</span>
          </span>
        </button>
      {/each}
    </div>
  </div>
</section>

<style>
  .index {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 0;
    border-right: 1px solid var(--rule);
    background: var(--ground-2);
  }
  .controls {
    padding: 10px 12px 8px;
    display: grid;
    gap: 6px;
    border-bottom: 1px solid var(--rule);
  }
  .sr {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  .search {
    position: relative;
    display: block;
  }
  .search input {
    width: 100%;
    padding-right: 28px;
  }
  .search kbd {
    position: absolute;
    right: 6px;
    top: 4px;
  }
  .row2 {
    display: flex;
    gap: 6px;
  }
  .row2 > select {
    flex: 1 1 0;
    min-width: 0;
  }
  .range {
    display: flex;
    align-items: center;
    gap: 4px;
    flex: 1 1 auto;
    min-width: 0;
  }
  .range .label {
    color: var(--chalk-3);
    margin-right: 2px;
  }
  .range input {
    width: 0;
    flex: 1 1 0;
    padding: 0 4px;
    appearance: textfield;
    -moz-appearance: textfield;
  }
  .range input::-webkit-inner-spin-button {
    display: none;
  }
  .sort {
    display: flex;
    gap: 4px;
  }
  .sort select {
    width: 96px;
  }
  .dir {
    width: 24px;
    padding: 0;
    display: grid;
    place-items: center;
  }
  .dir .flip {
    transform: rotate(180deg);
  }
  .counts {
    display: flex;
    gap: 12px;
    color: var(--chalk-3);
    font-size: 11px;
    padding-top: 2px;
  }
  .counts b {
    color: var(--anno);
    font-weight: 500;
  }
  .list {
    overflow-y: auto;
    overflow-x: hidden;
    position: relative;
    min-height: 0;
    outline: none;
  }
  .spacer {
    position: relative;
  }
  .empty {
    margin: 16px 12px;
    color: var(--chalk-2);
  }
  .item {
    position: absolute;
    left: 0;
    right: 0;
    top: 0;
    height: 40px;
    display: grid;
    grid-template-columns: 40px 14px 13px 1fr 50px;
    grid-template-rows: 18px 16px;
    column-gap: 6px;
    align-items: center;
    padding: 3px 12px 3px 0;
    border: 0;
    border-bottom: 1px solid var(--rule);
    background: transparent;
    text-align: left;
    cursor: pointer;
    color: var(--chalk);
  }
  .item:hover {
    background: var(--hover);
  }
  .item.current {
    background: var(--sel);
  }
  .rank {
    text-align: right;
    color: var(--chalk-2);
  }
  .current .rank {
    color: var(--anno);
  }
  .mark {
    display: grid;
    place-items: center;
    height: 14px;
  }
  .bar {
    height: 3px;
    background: var(--bar-track);
    position: relative;
  }
  .bar i {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--bar);
  }
  .score {
    text-align: right;
  }
  .line2 {
    grid-column: 2 / 6;
    grid-row: 2;
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
  }
  .mname {
    flex: 1 1 auto;
    min-width: 0;
    font-size: 12.5px;
    color: var(--chalk);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .new {
    flex: none;
    align-self: center;
    fill: none;
    stroke: var(--cat);
    stroke-width: 1.3;
    stroke-dasharray: 2.2 1.4;
  }
  .coord {
    flex: none;
    font-size: 10px;
    color: var(--chalk-2);
    white-space: nowrap;
  }
</style>
