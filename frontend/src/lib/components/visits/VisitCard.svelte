<script lang="ts">
  import Badge from '$lib/components/ui/Badge.svelte';
  import StatusDot from '$lib/components/ui/StatusDot.svelte';
  import type { VisitResponse } from '$lib/types';

  interface Props { visit: VisitResponse; }
  let { visit }: Props = $props();

  const arrivedStr = $derived(new Date(visit.arrived_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  const departedStr = $derived(visit.departed_at
    ? new Date(visit.departed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : null);
</script>

<div
  class="flex gap-4 p-4 rounded-lg"
  style="background: var(--color-card); border: 1px solid var(--color-border);"
>
  {#if visit.snapshot_url}
    <div class="flex-shrink-0 w-20 h-14 rounded overflow-hidden" style="background: var(--color-border);">
      <img src={visit.snapshot_url} alt="visit snapshot" class="w-full h-full object-cover" loading="lazy" />
    </div>
  {/if}
  <div class="flex-1 min-w-0">
    <div class="flex items-start justify-between gap-2 mb-1">
      <div>
        <span class="text-sm font-semibold" style="color: var(--color-text-primary);">
          {visit.display_name}
        </span>
        <span class="text-xs ml-2" style="color: var(--color-text-secondary);">
          {visit.camera_name}
        </span>
      </div>
      {#if visit.is_active}
        <div class="flex items-center gap-1.5 flex-shrink-0">
          <StatusDot status="ok" pulse size={6} />
          <span class="text-xs font-medium" style="color: var(--color-green);">ACTIVE</span>
        </div>
      {:else}
        <Badge type="departed" size="sm" />
      {/if}
    </div>
    <p class="text-xs" style="color: var(--color-text-secondary);">
      Arrived {arrivedStr}
      {#if departedStr}
        · Departed {departedStr}
        {#if visit.duration_minutes !== null}
          · <span style="color: var(--color-text-primary);">{visit.duration_minutes}m</span>
        {/if}
      {:else}
        · <span style="color: var(--color-green);">ongoing</span>
      {/if}
    </p>
  </div>
</div>
