<script lang="ts">
  import { toast, toasts } from '$lib/stores/toast';
  import { CheckCircle, Info, X, XCircle } from 'lucide-svelte';

  const iconMap = { success: CheckCircle, error: XCircle, info: Info };
  const colorMap = {
    success: 'var(--color-green)',
    error: 'var(--color-red)',
    info: 'var(--color-accent)'
  };
</script>

<div class="fixed bottom-4 right-4 z-50 flex flex-col gap-2" style="pointer-events: none;">
  {#each $toasts as t (t.id)}
    <div
      class="flex items-center gap-3 px-4 py-3 rounded-lg shadow-lg text-sm"
      style="
        background: var(--color-card);
        border: 1px solid var(--color-border);
        color: var(--color-text-primary);
        pointer-events: all;
        min-width: 280px;
      "
    >
      <svelte:component
        this={iconMap[t.type]}
        size={16}
        style="color: {colorMap[t.type]}; flex-shrink: 0;"
      />
      <span class="flex-1">{t.message}</span>
      <button
        onclick={() => toast.remove(t.id)}
        style="color: var(--color-text-muted);"
      >
        <X size={14} />
      </button>
    </div>
  {/each}
</div>
