<script lang="ts">
  interface Props { lines?: number; }
  let { lines = 3 }: Props = $props();
</script>

<div
  class="rounded-lg p-4 animate-pulse"
  style="background: var(--color-card); border: 1px solid var(--color-border);"
>
  <div class="flex gap-3">
    <div class="rounded w-20 h-14 flex-shrink-0" style="background: var(--color-border);"></div>
    <div class="flex-1 space-y-2">
      {#each Array(lines) as _, i}
        <div
          class="rounded h-3"
          style="background: var(--color-border); width: {i === 0 ? '60%' : i === lines - 1 ? '40%' : '80%'};"
        ></div>
      {/each}
    </div>
  </div>
</div>
