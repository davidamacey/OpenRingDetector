<script lang="ts">
  import StatusDot from '$lib/components/ui/StatusDot.svelte';
  import type { SystemStatus } from '$lib/types';

  interface Props { status: SystemStatus | null; }
  let { status }: Props = $props();

  const rows = $derived(status ? [
    { label: 'Database', s: status.database },
    { label: 'GPU', s: status.gpu },
    { label: 'YOLO Model', s: status.yolo_model },
    { label: 'Ring Token', s: status.ring_token },
    { label: 'Face Models', s: status.face_models },
    { label: 'Ollama', s: status.ollama },
    { label: 'Archive', s: status.archive_dir },
  ] : []);
</script>

<div
  class="rounded-xl p-4 space-y-2.5"
  style="background: var(--color-card); border: 1px solid var(--color-border);"
>
  <p class="text-xs font-semibold uppercase tracking-wide mb-3" style="color: var(--color-text-muted);">
    System Status
  </p>
  {#if status}
    <!-- Watcher -->
    <div class="flex items-center justify-between text-sm">
      <span style="color: var(--color-text-secondary);">Watcher</span>
      <div class="flex items-center gap-1.5">
        <StatusDot status={status.watcher_running ? 'ok' : 'fail'} pulse={status.watcher_running} />
        <span style="color: var(--color-text-primary);">
          {status.watcher_running ? 'Running' : 'Stopped'}
        </span>
      </div>
    </div>
    {#each rows as row}
      <div class="flex items-center justify-between text-sm gap-2">
        <span class="flex-shrink-0" style="color: var(--color-text-secondary);">{row.label}</span>
        <div class="flex items-center gap-1.5 min-w-0">
          <StatusDot status={row.s.status} />
          <span class="text-xs truncate" style="color: var(--color-text-muted);" title={row.s.detail}>
            {row.s.detail.slice(0, 40)}
          </span>
        </div>
      </div>
    {/each}
    <div class="pt-1 text-xs" style="color: var(--color-text-muted);">
      Uptime: {Math.floor(status.uptime_seconds / 60)}m
    </div>
  {:else}
    <p class="text-sm" style="color: var(--color-text-muted);">Loading...</p>
  {/if}
</div>
