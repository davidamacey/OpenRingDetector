<script lang="ts">
  import { X } from 'lucide-svelte';
  interface Props {
    open: boolean;
    title?: string;
    onclose?: () => void;
  }
  let { open, title, onclose, children }: Props & { children?: any } = $props();
</script>

{#if open}
  <div
    class="fixed inset-0 z-50 flex items-center justify-center p-4"
    style="background: rgba(0,0,0,0.7);"
    role="dialog"
    aria-modal="true"
  >
    <div
      class="rounded-xl w-full max-w-lg shadow-2xl"
      style="background: var(--color-card); border: 1px solid var(--color-border);"
    >
      {#if title || onclose}
        <div
          class="flex items-center justify-between px-5 py-4 border-b"
          style="border-color: var(--color-border);"
        >
          {#if title}
            <h2 class="text-base font-semibold" style="color: var(--color-text-primary);">{title}</h2>
          {/if}
          {#if onclose}
            <button
              onclick={onclose}
              class="ml-auto p-1 rounded"
              style="color: var(--color-text-muted);"
            >
              <X size={18} />
            </button>
          {/if}
        </div>
      {/if}
      <div class="p-5">
        {@render children?.()}
      </div>
    </div>
  </div>
{/if}
