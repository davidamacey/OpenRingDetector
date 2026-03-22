<script lang="ts">
  import Badge from '$lib/components/ui/Badge.svelte';
  import type { EventResponse } from '$lib/types';
  import { formatDistanceToNow, parseISO } from 'date-fns';

  interface Props {
    event: EventResponse;
    onclick?: (e: EventResponse) => void;
  }
  let { event, onclick }: Props = $props();

  const borderColors: Record<string, string> = {
    arrival: 'var(--color-green)',
    departure: 'var(--color-red)',
    motion: 'var(--color-accent)',
    ding: 'var(--color-violet)'
  };
  const border = $derived(borderColors[event.event_type] ?? 'var(--color-border)');

  const relativeTime = $derived(
    formatDistanceToNow(parseISO(event.occurred_at), { addSuffix: true })
  );
  const absTime = $derived(
    new Date(event.occurred_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  );

  const uniqueClasses = $derived([
    ...new Set(event.detections.map((d) => d.class_name))
  ]);
</script>

<div
  class="flex gap-3 p-4 rounded-lg cursor-pointer transition-colors"
  style="
    background: var(--color-card);
    border: 1px solid var(--color-border);
    border-left: 3px solid {border};
  "
  role="button"
  tabindex="0"
  onclick={() => onclick?.(event)}
  onkeydown={(e) => e.key === 'Enter' && onclick?.(event)}
>
  <!-- Thumbnail -->
  {#if event.snapshot_url}
    <div class="flex-shrink-0 w-24 h-14 rounded overflow-hidden" style="background: var(--color-border);">
      <img
        src={event.snapshot_url}
        alt="snapshot"
        class="w-full h-full object-cover"
        loading="lazy"
      />
    </div>
  {:else}
    <div
      class="flex-shrink-0 w-24 h-14 rounded flex items-center justify-center"
      style="background: var(--color-border);"
    >
      <span class="text-xs" style="color: var(--color-text-muted);">No image</span>
    </div>
  {/if}

  <!-- Content -->
  <div class="flex-1 min-w-0">
    <div class="flex items-center gap-2 mb-1 flex-wrap">
      <Badge type={event.event_type} size="sm" />
      <span class="text-xs font-medium" style="color: var(--color-text-primary);">
        {event.camera_name}
      </span>
      <span
        class="text-xs font-mono ml-auto flex-shrink-0 text-right"
        style="color: var(--color-text-muted);"
        title={relativeTime}
      >
        {absTime}<br /><span class="text-xs" style="color: var(--color-text-muted); opacity: 0.7;">{relativeTime}</span>
      </span>
    </div>

    {#if event.display_name}
      <p class="text-sm font-medium mb-1" style="color: var(--color-text-primary);">
        {event.display_name}
        {#if event.event_type === 'arrival'}arrived{:else if event.event_type === 'departure'}departed{/if}
      </p>
    {/if}

    {#if event.detection_summary || event.caption}
      <p class="text-xs mb-1.5 line-clamp-2" style="color: var(--color-text-secondary);">
        {event.caption ?? event.detection_summary}
      </p>
    {/if}

    <div class="flex flex-wrap gap-1">
      {#each uniqueClasses as cls}
        <Badge type={cls} size="sm" />
      {/each}
    </div>
  </div>
</div>
