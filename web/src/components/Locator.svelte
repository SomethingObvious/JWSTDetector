<script lang="ts">
  import { SvelteSet } from 'svelte/reactivity';
  import { track } from '../lib/track';
  import type { Bundle, Theme, TileInfo } from '../lib/types';

  let {
    bundle,
    base,
    current,
    order,
    theme,
    onselect,
  }: {
    bundle: Bundle;
    base: string;
    current: number;
    order: Int32Array;
    theme: Theme;
    onselect: (index: number) => void;
  } = $props();

  // eslint-disable-next-line svelte/prefer-svelte-reactivity -- a cache, the loaded counter drives redraws
  const images = new Map<string, HTMLImageElement>();
  let loaded = $state(0);
  // By mosaic, so moving to another mosaic doesn't carry the message along.
  const failed = new SvelteSet<string>();
  let canvas: HTMLCanvasElement | undefined = $state();
  let width = $state(316);

  const tile = $derived(bundle.tiles[current]);
  const mosaic = $derived(tile && bundle.mosaics[tile.mosaic]);
  const shape = $derived<[number, number]>(mosaic?.shape ?? [1, 1]);
  const height = $derived(Math.round((width * shape[0]) / shape[1]));
  const scale = $derived(width / shape[1]);
  // The preview is the right way up, so mosaic row 0 is its bottom edge.
  const px = (t: TileInfo) => (t.x0 + t.size / 2) * scale;
  const py = (t: TileInfo) => (shape[0] - t.y0 - t.size / 2) * scale;

  const dots = $derived.by(() => {
    const out: number[] = [];
    if (!tile) {
      return out;
    }
    for (const i of order) {
      if (bundle.tiles[i]?.mosaic === tile.mosaic && i !== current) {
        out.push(i);
      }
    }
    return out;
  });

  function image(name: string): HTMLImageElement | null {
    let img = images.get(name);
    if (!img) {
      const m = bundle.mosaics[name];
      if (!m) {
        return null;
      }
      img = new Image();
      img.decoding = 'async';
      img.onload = () => loaded++;
      img.onerror = () => failed.add(name);
      img.src = base + m.preview;
      images.set(name, img);
    }
    return img.complete && img.naturalWidth ? img : null;
  }

  $effect(() => {
    track(loaded, theme);
    if (!canvas || !mosaic || !tile) {
      return;
    }
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const v = (k: string) => css.getPropertyValue(k).trim();
    ctx.fillStyle = v('--ground-sunk');
    ctx.fillRect(0, 0, width, height);
    const img = image(tile.mosaic);
    if (img) {
      ctx.globalAlpha = theme === 'plate' ? 0.55 : 0.8;
      if (theme === 'plate') {
        ctx.filter = 'invert(1) grayscale(1)';
      }
      ctx.drawImage(img, 0, 0, width, height);
      ctx.filter = 'none';
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = v('--anno');
    ctx.globalAlpha = 0.85;
    for (const i of dots) {
      const t = bundle.tiles[i];
      if (t) {
        ctx.fillRect(px(t) - 1.5, py(t) - 1.5, 3, 3);
      }
    }
    ctx.globalAlpha = 1;
    const s = tile.size * scale;
    // Chalk, as red is kept for the ring on the plate.
    ctx.strokeStyle = v('--chalk');
    ctx.lineWidth = 1.5;
    ctx.strokeRect(px(tile) - s / 2 + 0.5, py(tile) - s / 2 + 0.5, s, s);
  });

  function click(e: MouseEvent) {
    if (!canvas) {
      return;
    }
    const r = canvas.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    let best = -1;
    let bestD = 9 * 9;
    for (const i of dots) {
      const t = bundle.tiles[i];
      if (!t) {
        continue;
      }
      const d = (px(t) - mx) ** 2 + (py(t) - my) ** 2;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best >= 0) {
      onselect(best);
    }
  }
</script>

<div class="locator" bind:clientWidth={width}>
  <!-- The keyboard reaches the same tiles through the index, so the map is a pointer shortcut. -->
  <canvas
    bind:this={canvas}
    style:height="{height}px"
    onclick={click}
    aria-label="Where this tile sits on the mosaic, with {dots.length} other candidates marked"
  ></canvas>
  <div class="legend">
    <span><i class="foot"></i>This tile</span>
    <span><i class="dot"></i>{dots.length} more in the list</span>
  </div>
  {#if mosaic && tile && failed.has(tile.mosaic)}<p class="err">
      Couldn't load {mosaic.preview}.
    </p>{/if}
</div>

<style>
  .locator {
    display: grid;
    gap: 6px;
  }
  canvas {
    width: 100%;
    display: block;
    cursor: pointer;
    border: 1px solid var(--rule);
  }
  .legend {
    display: flex;
    gap: 14px;
    font-size: 11px;
    color: var(--chalk-2);
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .foot {
    width: 9px;
    height: 9px;
    border: 1.5px solid var(--chalk);
  }
  .dot {
    width: 3px;
    height: 3px;
    background: var(--anno);
  }
  .err {
    margin: 0;
    color: var(--chalk-2);
  }
</style>
