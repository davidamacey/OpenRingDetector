<script lang="ts">
  import Modal from '$lib/components/ui/Modal.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import type { EventResponse } from '$lib/types';

  interface Props {
    event: EventResponse | null;
    onclose: () => void;
  }
  let { event, onclose }: Props = $props();

  const open = $derived(event !== null);
</script>

<Modal {open} title={event ? `${event.camera_name} — ${new Date(event.occurred_at).toLocaleString()}` : ''} {onclose}>
  {#if event}
    <div class="space-y-4">
      <!-- Snapshot with detection overlay -->
      <div class="relative rounded overflow-hidden" style="background: var(--color-border);">
        {#if event.snapshot_url}
          <img src={event.snapshot_url} alt="snapshot" class="w-full" />
          <!-- Detection boxes as SVG overlay -->
          <svg
            class="absolute inset-0 w-full h-full"
            viewBox="0 0 1 1"
            preserveAspectRatio="none"
            style="pointer-events: none;"
          >
            {#each event.detections as d}
              <rect
                x={d.xcenter - d.width / 2}
                y={d.ycenter - d.height / 2}
                width={d.width}
                height={d.height}
                fill="none"
                stroke={d.class_name === 'person' ? '#a5b4fc' : '#2dd4bf'}
                stroke-width="0.003"
              />
            {/each}
          </svg>
        {:else}
          <div class="h-48 flex items-center justify-center">
            <span style="color: var(--color-text-muted);">No snapshot available</span>
          </div>
        {/if}
      </div>

      <!-- Details -->
      <div class="space-y-2">
        <div class="flex flex-wrap gap-1.5">
          <Badge type={event.event_type} />
          {#each [...new Set(event.detections.map(d => d.class_name))] as cls}
            <Badge type={cls} />
          {/each}
        </div>
        {#if event.caption || event.detection_summary}
          <p class="text-sm" style="color: var(--color-text-secondary);">
            {event.caption ?? event.detection_summary}
          </p>
        {/if}
        {#if event.display_name}
          <p class="text-sm font-medium" style="color: var(--color-text-primary);">
            {event.display_name}
          </p>
        {/if}
        <p class="text-xs font-mono" style="color: var(--color-text-muted);">
          {new Date(event.occurred_at).toLocaleString()}
        </p>
      </div>
    </div>
  {/if}
</Modal>
