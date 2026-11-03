<script lang="ts">
  import type { Verdict } from '../lib/types';

  let { verdict, size = 12 }: { verdict: Verdict | null | 'star' | 'star-off'; size?: number } = $props();
</script>

<!-- Shape carries the verdict, colour only repeats it. -->
<svg class="glyph {verdict ?? 'none'}" width={size} height={size} viewBox="-6 -6 12 12" aria-hidden="true">
  {#if verdict !== 'star' && verdict !== 'star-off' && verdict !== null}
    <circle class="ink" r="5.4" />
  {/if}
  {#if verdict === 'interesting'}
    <path d="M0 -3.6 L3.6 0 L0 3.6 L-3.6 0 Z" class="fill" />
  {:else if verdict === 'artifact'}
    <path d="M-2.9 -2.9 L2.9 2.9 M2.9 -2.9 L-2.9 2.9" class="stroke" />
  {:else if verdict === 'known'}
    <circle r="3" class="stroke" /><circle r="0.9" class="fill" />
  {:else if verdict === 'not_anomalous'}
    <path d="M-3.4 0 L3.4 0" class="stroke" />
  {:else if verdict === 'star' || verdict === 'star-off'}
    <path d="M0 -5.2 L1.2 -1.2 L5.2 0 L1.2 1.2 L0 5.2 L-1.2 1.2 L-5.2 0 L-1.2 -1.2 Z" class="fill" />
  {:else}
    <circle r="1.6" class="dot" />
  {/if}
</svg>

<style>
  .glyph {
    flex: none;
    overflow: visible;
  }
  .stroke {
    fill: none;
    stroke: currentColor;
    stroke-width: 1.5;
    stroke-linecap: square;
  }
  .fill {
    fill: currentColor;
  }
  .dot {
    fill: none;
    stroke: var(--chalk-3);
    stroke-width: 1;
  }
  .ink {
    fill: none;
    stroke: currentColor;
    stroke-width: 0.9;
    opacity: var(--v-ring);
  }
  .interesting {
    color: var(--v-interesting);
  }
  .artifact {
    color: var(--v-artifact);
  }
  .known {
    color: var(--v-known);
  }
  .not_anomalous {
    color: var(--v-not);
  }
  .star {
    color: var(--anno);
  }
  .star-off {
    color: var(--chalk-3);
    opacity: 0.5;
  }
</style>
