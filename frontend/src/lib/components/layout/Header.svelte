<script lang="ts">
  import { statusStore } from '$lib/stores/status';
  import WatcherPanel from '$lib/components/watcher/WatcherPanel.svelte';

  interface Props {
    title: string;
  }
  let { title }: Props = $props();

  const status = $derived($statusStore);
</script>

<header
  class="flex items-center justify-between px-6 py-4 border-b flex-shrink-0"
  style="
    background: var(--color-surface);
    border-color: var(--color-border);
    min-height: 60px;
  "
>
  <h1 class="text-lg font-semibold" style="color: var(--color-text-primary);">{title}</h1>

  <div class="flex items-center gap-4 text-sm" style="color: var(--color-text-secondary);">
    <WatcherPanel compact />
    {#if status?.gpu.status === 'ok'}
      <span class="hidden md:block" style="color: var(--color-text-muted);">|</span>
      <span class="hidden md:block text-xs" style="color: var(--color-text-muted);">
        GPU: {status.gpu.detail.split(',')[0]?.trim()}
      </span>
    {/if}
  </div>
</header>
