<script lang="ts">
  import { onMount } from 'svelte';
  import { Renderer } from '../lib/gl';
  import { equaliser, type Limits, type TileData } from '../lib/tiles';
  import { edgeTicks, scaleBar, type TileFrame } from '../lib/frame';
  import { CMAPS, cssGradient } from '../lib/colormaps';
  import type { Display, Theme, TileInfo } from '../lib/types';

  let {
    tile,
    index,
    data,
    error,
    frame,
    grid,
    hasPass1,
    display = $bindable(),
    limits,
    mapRange,
    theme,
    cursor,
    onrenderer,
    oncursor,
  }: {
    tile: TileInfo;
    index: number;
    data: TileData | null;
    error: string | null;
    frame: TileFrame | null;
    grid: [number, number];
    hasPass1: boolean;
    display: Display;
    limits: Limits;
    mapRange: [number, number];
    theme: Theme;
    cursor: { ix: number; iy: number } | null;
    onrenderer: (r: Renderer | null, err?: string) => void;
    oncursor: (c: { ix: number; iy: number } | null) => void;
  } = $props();

  const M = 22;
  let canvas: HTMLCanvasElement | undefined = $state();
  let renderer: Renderer | null = $state(null);
  let glError = $state<string | null>(null);
  let cw = $state(0);
  let ch = $state(0);
  // Zoom relative to fit and the view centre as a fraction of the image, so both carry across tiles.
  let rel = $state(1);
  let cfx = $state(0.5);
  let cfy = $state(0.5);
  let nan = $state<[number, number, number]>([0.1, 0.14, 0.21]);

  const W = $derived(data?.width ?? 1);
  const H = $derived(data?.height ?? 1);
  const fitZoom = $derived(Math.max(1e-3, Math.min(cw / W, ch / H)));
  const zoom = $derived(fitZoom * rel);
  const cx = $derived(cfx * W);
  const cy = $derived(cfy * H);

  const toScreen = (ix: number, iy: number): [number, number] => [(ix - cx) * zoom + cw / 2, (iy - cy) * zoom + ch / 2];
  const toImage = (sx: number, sy: number): [number, number] => [(sx - cw / 2) / zoom + cx, (sy - ch / 2) / zoom + cy];

  export function fit() {
    rel = 1;
    cfx = 0.5;
    cfy = 0.5;
  }

  export function oneToOne() {
    rel = 1 / fitZoom;
  }

  onMount(() => {
    try {
      renderer = new Renderer(canvas!);
      onrenderer(renderer);
    } catch (err) {
      glError = err instanceof Error ? err.message : String(err);
      onrenderer(null, glError);
    }
    return () => onrenderer(null);
  });

  $effect(() => {
    void theme;
    const v = getComputedStyle(document.documentElement).getPropertyValue('--nan').trim();
    const m = v.match(/^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    if (m) nan = [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255];
  });

  $effect(() => {
    if (!renderer || !data?.planes || display.stretch !== 'histeq') return;
    renderer.setEqualiser(equaliser(data, limits));
  });

  $effect(() => {
    if (!renderer || !canvas) return;
    if (!data) {
      renderer.gl.clearColor(0, 0, 0, 0);
      renderer.gl.clear(renderer.gl.COLOR_BUFFER_BIT);
      return;
    }
    if (!renderer.has(data.index)) renderer.upload(data, grid);
    void cw;
    void ch;
    renderer.draw(
      {
        tile: data,
        display: data.planes ? { ...display } : { ...display, stretch: 'linear' },
        limits,
        mapRange,
        centre: [cx, cy],
        zoom,
        nan,
      },
      grid,
    );
  });

  // Furniture drawn over the canvas, kept to the part of the viewport the image covers.
  const vis = $derived.by(() => {
    const [x0, y0] = toScreen(0, 0);
    const [x1, y1] = toScreen(W, H);
    const l = Math.max(0, Math.round(x0));
    const t = Math.max(0, Math.round(y0));
    const r = Math.min(cw, Math.round(x1));
    const b = Math.min(ch, Math.round(y1));
    return data && r - l > 40 && b - t > 40 ? { l, t, w: r - l, h: b - t } : { l: 0, t: 0, w: cw, h: ch };
  });
  const ticks = $derived(
    frame && data && cw > 0 ? edgeTicks(vis.w, vis.h, (x, y) => frame.toSky(...toImage(vis.l + x, vis.t + y))) : [],
  );
  const compass = $derived(frame?.compass() ?? null);
  const bar = $derived(frame ? scaleBar(frame.arcsecPerPx / zoom) : null);
  const ring = $derived.by(() => {
    if (!data) return null;
    const [rows, cols] = grid;
    const [pr, pc] = tile.peak;
    const [x, y] = toScreen(((pc + 0.5) * W) / cols, ((pr + 0.5) * H) / rows);
    const r = Math.max(11, (0.85 * Math.max(W / cols, H / rows) * zoom));
    return { x, y, r };
  });
  // The same edges in the frame's coordinates, which sit M outside the viewport.
  const L = $derived(M + vis.l);
  const T = $derived(M + vis.t);
  const R = $derived(M + vis.l + vis.w);
  const B = $derived(M + vis.t + vis.h);
  // The scale bar takes the bottom corner the ring isn't near.
  const barRight = $derived(!!ring && ring.x < vis.l + vis.w / 2 && ring.y > vis.t + vis.h / 2);
  const patchBox = $derived.by(() => {
    if (!cursor || !data) return null;
    const [rows, cols] = grid;
    const c = Math.floor((cursor.ix / W) * cols);
    const r = Math.floor((cursor.iy / H) * rows);
    if (c < 0 || r < 0 || c >= cols || r >= rows) return null;
    const [x0, y0] = toScreen((c * W) / cols, (r * H) / rows);
    const [x1, y1] = toScreen(((c + 1) * W) / cols, ((r + 1) * H) / rows);
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  });

  // Pointer: drag pans, the wheel zooms about the pointer.
  let drag: { x: number; y: number; cfx: number; cfy: number } | null = null;

  function local(e: PointerEvent | WheelEvent): [number, number] {
    const r = canvas!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function onpointerdown(e: PointerEvent) {
    if (e.button !== 0) return;
    canvas!.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, cfx, cfy };
  }

  function onpointermove(e: PointerEvent) {
    if (drag) {
      cfx = drag.cfx - (e.clientX - drag.x) / zoom / W;
      cfy = drag.cfy - (e.clientY - drag.y) / zoom / H;
    }
    if (!data) return;
    const [ix, iy] = toImage(...local(e));
    oncursor(ix >= 0 && iy >= 0 && ix < W && iy < H ? { ix, iy } : null);
  }

  function onpointerup(e: PointerEvent) {
    drag = null;
    if (canvas?.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  }

  function onwheel(e: WheelEvent) {
    e.preventDefault();
    const [sx, sy] = local(e);
    const [ix, iy] = toImage(sx, sy);
    const f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018));
    const next = Math.min(64 / fitZoom, Math.max(0.25, rel * f));
    const z = fitZoom * next;
    rel = next;
    cfx = (ix - (sx - cw / 2) / z) / W;
    cfy = (iy - (sy - ch / 2) / z) / H;
  }

  const pct = (v: number) => `${Math.round(v * 100)}%`;
  const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : Math.abs(v) >= 1 ? v.toFixed(3) : v.toPrecision(3));
  const thresholdValue = $derived(mapRange[0] + display.threshold * (mapRange[1] - mapRange[0]));
</script>

{#snippet part(p: { v: string; u: string }, lastPart: boolean)}
  {@const sup = p.u === 'h' || p.u === 'm' || p.u === 's'}
  <tspan>{p.v}</tspan><tspan class="u" dy={sup ? '-0.45em' : undefined}>{p.u}</tspan>{#if sup || !lastPart}<tspan dy={sup ? '0.45em' : undefined}>{lastPart ? '' : ' '}</tspan>{/if}
{/snippet}

<section class="plate" aria-label="Plate">
  <div class="toolbar">
    <div class="group">
      <button class="ctl" onclick={fit} title="Fit the tile to the view (f)">Fit</button>
      <button class="ctl" onclick={oneToOne} title="One image pixel per screen pixel (0)">1:1</button>
      <span class="zoom num" title="Screen pixels per image pixel">{zoom.toFixed(2)}<span class="u">&times;</span></span>
    </div>
    <span class="sep"></span>
    <div class="group">
      <button class="ctl" aria-pressed={display.overlay} onclick={() => (display.overlay = !display.overlay)} title="Patch-distance overlay (o)">Overlay</button>
      <div class="seg" role="group" aria-label="Map">
        <button class="ctl" aria-pressed={display.source === 'pass2'} onclick={() => (display.source = 'pass2')}>Pass 2</button>
        <button class="ctl" aria-pressed={display.source === 'pass1'} disabled={!hasPass1} onclick={() => (display.source = 'pass1')}>Pass 1</button>
        <button class="ctl" aria-pressed={display.source === 'diff'} disabled={!hasPass1} onclick={() => (display.source = 'diff')} title="Pass 2 minus pass 1">Change</button>
      </div>
      <div class="seg" role="group" aria-label="Overlay style">
        <button class="ctl" aria-pressed={!display.contours} onclick={() => (display.contours = false)}>Patches</button>
        <button class="ctl" aria-pressed={display.contours} onclick={() => (display.contours = true)}>Contours</button>
      </div>
    </div>
    <span class="sep"></span>
    <div class="group sliders">
      <label class="slider">
        <span class="label">Opacity</span>
        <input type="range" min="0" max="1" step="0.01" bind:value={display.overlayOpacity} />
        <span class="num val">{pct(display.overlayOpacity)}</span>
      </label>
      <label class="slider">
        <span class="label">Threshold</span>
        <input type="range" min="0" max="0.95" step="0.01" bind:value={display.threshold} />
        <span class="num val">{fmt(thresholdValue)}</span>
      </label>
      <label class="legend">
        <span class="sr">Overlay colour map</span>
        <select class="ctl" bind:value={display.overlayCmap}>
          {#each CMAPS as c (c.id)}
            <option value={c.id}>{c.label}</option>
          {/each}
        </select>
        <span class="ramp">
          <span class="num">{fmt(mapRange[0])}</span>
          <i style:background={cssGradient(display.overlayCmap)}><b style:width="{display.threshold * 100}%"></b></i>
          <span class="num">{fmt(mapRange[1])}</span>
        </span>
      </label>
    </div>
  </div>

  <div class="stage">
    <div class="viewport" style:inset="{M}px" bind:clientWidth={cw} bind:clientHeight={ch}>
      <canvas
        bind:this={canvas}
        class:grab={!!data}
        {onpointerdown}
        {onpointermove}
        {onpointerup}
        onpointercancel={onpointerup}
        onpointerleave={() => oncursor(null)}
        {onwheel}
        ondblclick={fit}
        aria-label="Tile {tile.rank}, drag to pan and scroll to zoom"
      ></canvas>

      {#if !data && !error && !glError}
        <div class="skeleton" aria-live="polite">
          <span class="label">Loading Tile {String(index).padStart(6, '0')}</span>
        </div>
      {/if}
      {#if error || glError}
        <div class="failed" role="alert">
          <p>{glError ?? error}</p>
        </div>
      {/if}

      <svg class="over" width={cw} height={ch} aria-hidden="true">
        {#if patchBox}
          <rect class="patch" x={patchBox.x} y={patchBox.y} width={patchBox.w} height={patchBox.h} />
        {/if}
        {#if ring}
          <g class="ring" transform="translate({ring.x} {ring.y})">
            <circle r={ring.r} />
            <path d="M0 {-ring.r - 7} V{-ring.r - 2} M0 {ring.r + 2} V{ring.r + 7} M{-ring.r - 7} 0 H{-ring.r - 2} M{ring.r + 2} 0 H{ring.r + 7}" />
          </g>
        {/if}
        {#if compass}
          <g class="compass" transform="translate({vis.l + vis.w - 44} {vis.t + 44})">
            <circle r="2" />
            <line x1="0" y1="0" x2={compass.n[0] * 26} y2={compass.n[1] * 26} />
            <line x1="0" y1="0" x2={compass.e[0] * 18} y2={compass.e[1] * 18} />
            <text x={compass.n[0] * 34} y={compass.n[1] * 34} dy="0.35em">N</text>
            <text x={compass.e[0] * 26} y={compass.e[1] * 26} dy="0.35em">E</text>
          </g>
        {/if}
        {#if bar}
          <g class="scalebar" transform="translate({barRight ? vis.l + vis.w - 14 - bar.px : vis.l + 14} {vis.t + vis.h - 16})">
            <path d="M0 -4 V0 H{bar.px} V-4" />
            <text x={bar.px / 2} y="-8">{bar.label}</text>
          </g>
        {/if}
      </svg>
    </div>

    <svg class="frame" aria-hidden="true">
      <rect x={L - 0.5} y={T - 0.5} width={vis.w + 1} height={vis.h + 1} />
      {#each ticks as t, i (i)}
        {#if t.edge === 'top' || t.edge === 'bottom'}
          {@const x = L + t.at}
          {@const y = t.edge === 'top' ? T : B}
          <line x1={x} x2={x} y1={y} y2={t.edge === 'top' ? y + 6 : y - 6} />
          <text class="tl" x={x} y={t.edge === 'top' ? T - 7 : B + 15} text-anchor="middle">
            {#each t.parts as p, j (j)}{@render part(p, j === t.parts.length - 1)}{/each}
          </text>
        {:else}
          {@const y = T + t.at}
          {@const x = t.edge === 'left' ? L : R}
          <line x1={x} x2={t.edge === 'left' ? x + 6 : x - 6} y1={y} y2={y} />
          <text class="tl" transform="translate({t.edge === 'left' ? L - 8 : R + 8} {y}) rotate({t.edge === 'left' ? -90 : 90})" text-anchor="middle">
            {#each t.parts as p, j (j)}{@render part(p, j === t.parts.length - 1)}{/each}
          </text>
        {/if}
      {/each}
      {#each [[L - 0.5, T - 0.5, -1, -1], [R + 0.5, T - 0.5, 1, -1], [L - 0.5, B + 0.5, -1, 1], [R + 0.5, B + 0.5, 1, 1]] as [x, y, sx, sy], i (i)}
        <path class="corner" d="M{x + sx * 3} {y} H{x + sx * 12} M{x} {y + sy * 3} V{y + sy * 12}" />
      {/each}
    </svg>
  </div>
</section>

<style>
  .plate {
    display: grid;
    grid-template-rows: auto 1fr;
    min-width: 0;
    min-height: 0;
    background: var(--ground);
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px 10px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--rule);
    min-height: 37px;
  }
  .group {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
  }
  .sep {
    width: 1px;
    height: 16px;
    background: var(--rule);
  }
  .seg {
    display: flex;
  }
  @media (max-width: 820px) {
    .sep {
      display: none;
    }
    .toolbar {
      gap: 6px;
    }
  }
  .seg .ctl + .ctl {
    margin-left: -1px;
  }
  .seg .ctl[aria-pressed='true'] {
    position: relative;
    z-index: 1;
  }
  .zoom {
    color: var(--anno);
    min-width: 46px;
  }
  .zoom .u {
    color: var(--chalk-3);
    margin-left: 1px;
  }
  .slider {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .slider .label {
    color: var(--chalk-3);
  }
  .slider input {
    width: 72px;
  }
  .val {
    color: var(--anno);
    min-width: 40px;
  }
  .legend {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .legend select {
    width: 86px;
  }
  .ramp {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--chalk-2);
  }
  .ramp i {
    position: relative;
    width: 56px;
    height: 6px;
    border: 1px solid var(--rule);
  }
  .ramp b {
    position: absolute;
    inset: 0 auto 0 0;
    background: rgb(0 0 0 / 0.6);
    border-right: 1.5px solid var(--chalk);
  }
  .sr {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
    clip-path: inset(50%);
  }
  .stage {
    position: relative;
    min-height: 0;
    overflow: hidden;
  }
  .viewport {
    position: absolute;
    overflow: hidden;
    background: var(--ground-sunk);
  }
  canvas {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    display: block;
    touch-action: none;
    cursor: crosshair;
  }
  canvas.grab:active {
    cursor: grabbing;
  }
  .skeleton,
  .failed {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    pointer-events: none;
    background: var(--ground-sunk);
  }
  .skeleton .label {
    color: var(--anno);
    background: var(--ground-sunk);
    padding: 4px 8px;
  }
  .failed p {
    max-width: 46ch;
    background: var(--ground-sunk);
    padding: 10px 14px;
    border: 1px solid var(--rule-strong);
    color: var(--chalk);
  }
  .over {
    position: absolute;
    inset: 0;
    pointer-events: none;
    overflow: visible;
  }
  .patch {
    fill: none;
    stroke: var(--anno);
    stroke-width: 1;
    opacity: 0.85;
  }
  .ring circle,
  .ring path {
    fill: none;
    stroke: var(--ring);
    stroke-width: 1.5;
  }
  .compass line {
    stroke: var(--anno);
    stroke-width: 1.25;
  }
  .compass circle {
    fill: var(--anno);
  }
  .compass text,
  .scalebar text {
    fill: var(--anno);
    font-family: var(--font-num);
    font-variant-numeric: slashed-zero;
    font-size: 11px;
    text-anchor: middle;
    paint-order: stroke;
    stroke: var(--ground-sunk);
    stroke-width: 3px;
    stroke-linejoin: round;
  }
  .scalebar path {
    fill: none;
    stroke: var(--anno);
    stroke-width: 1.25;
  }
  .frame {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }
  .frame rect {
    fill: none;
    stroke: var(--rule-strong);
    stroke-width: 1;
  }
  .frame line,
  .corner {
    stroke: var(--anno);
    stroke-width: 1;
  }
  .tl {
    fill: var(--anno);
    font-family: var(--font-num);
    font-size: 11px;
    font-variant-numeric: tabular-nums slashed-zero;
  }
  .tl .u {
    font-size: 8px;
    fill: var(--anno-2);
  }
</style>
