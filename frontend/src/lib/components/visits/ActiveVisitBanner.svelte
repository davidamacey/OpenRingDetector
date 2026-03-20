<script lang="ts">
  import type { VisitResponse } from '$lib/types';
  import StatusDot from '$lib/components/ui/StatusDot.svelte';
  import { onMount, onDestroy } from 'svelte';

  interface Props { visits: VisitResponse[]; }
  let { visits }: Props = $props();

  let now = $state(Date.now());
  let timer: ReturnType<typeof setInterval>;

  onMount(() => { timer = setInterval(() => (now = Date.now()), 30000); });
  onDestroy(() => clearInterval(timer));

  function elapsed(arrived_at: string): string {
    const ms = now - new Date(arrived_at).getTime();
    const min = Math.floor(ms / 60000);
    if (min < 60) return `${min}m`;
    return `${Math.floor(min / 60)}h ${min % 60}m`;
  }
</script>

{#if visits.length > 0}
  <div
    class="flex items-center gap-3 px-4 py-3 border-b flex-wrap"
    style="
      background: rgba(20,83,45,0.2);
      border-color: rgba(34,197,94,0.3);
    "
  >
    <span class="text-xs font-medium uppercase tracking-wide" style="color: var(--color-green);">
      Active Visits
    </span>
    {#each visits as v}
      <div
        class="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs"
        style="background: rgba(20,83,45,0.4); color: var(--color-green);"
      >
        <StatusDot status="ok" pulse size={6} />
        <span class="font-medium">{v.display_name}</span>
        <span style="color: rgba(74,222,128,0.7);">{v.camera_name} · {elapsed(v.arrived_at)}</span>
      </div>
    {/each}
  </div>
{/if}
