<script lang="ts">
  interface Props {
    cameras: string[];
    camera: string;
    type: string;
    live: boolean;
    oncamerachange: (v: string) => void;
    ontypechange: (v: string) => void;
    onlivechange: (v: boolean) => void;
  }
  let { cameras, camera, type, live, oncamerachange, ontypechange, onlivechange }: Props = $props();
</script>

<div
  class="flex flex-wrap items-center gap-3 px-4 py-3 border-b"
  style="border-color: var(--color-border);"
>
  <select
    value={camera}
    onchange={(e) => oncamerachange((e.target as HTMLSelectElement).value)}
    class="text-sm rounded px-3 py-1.5"
    style="
      background: var(--color-card);
      color: var(--color-text-primary);
      border: 1px solid var(--color-border);
    "
  >
    <option value="">All Cameras</option>
    {#each cameras as cam}
      <option value={cam}>{cam}</option>
    {/each}
  </select>

  <select
    value={type}
    onchange={(e) => ontypechange((e.target as HTMLSelectElement).value)}
    class="text-sm rounded px-3 py-1.5"
    style="
      background: var(--color-card);
      color: var(--color-text-primary);
      border: 1px solid var(--color-border);
    "
  >
    <option value="">All Types</option>
    <option value="motion">Motion</option>
    <option value="arrival">Arrivals</option>
    <option value="departure">Departures</option>
    <option value="ding">Doorbell</option>
  </select>

  <label class="flex items-center gap-2 text-sm ml-auto cursor-pointer" style="color: var(--color-text-secondary);">
    <div
      class="relative w-9 h-5 rounded-full transition-colors cursor-pointer"
      style="background: {live ? 'var(--color-accent)' : 'var(--color-border)'};"
      role="switch"
      aria-checked={live}
      tabindex="0"
      onclick={() => onlivechange(!live)}
      onkeydown={(e) => e.key === 'Enter' && onlivechange(!live)}
    >
      <div
        class="absolute top-0.5 w-4 h-4 rounded-full transition-transform bg-white"
        style="left: {live ? '18px' : '2px'};"
      ></div>
    </div>
    Live
  </label>
</div>
