<script lang="ts">
  import { ArrowUp } from 'lucide-svelte';

  interface Props {
    onSend: (text: string) => void;
    disabled?: boolean;
  }
  let { onSend, disabled = false }: Props = $props();

  let text = $state('');

  function handleSend() {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    text = '';
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }
</script>

<div
  class="flex items-center gap-2 p-3 border-t"
  style="
    border-color: var(--color-border);
    background: var(--color-surface);
  "
>
  <input
    type="text"
    bind:value={text}
    onkeydown={handleKeydown}
    placeholder="Ask about your detection history..."
    {disabled}
    class="flex-1 rounded-lg px-3 py-2 text-sm outline-none transition-colors"
    style="
      background: var(--color-bg);
      border: 1px solid var(--color-border);
      color: var(--color-text-primary);
      opacity: {disabled ? '0.5' : '1'};
    "
  />
  <button
    onclick={handleSend}
    disabled={disabled || !text.trim()}
    class="flex items-center justify-center w-8 h-8 rounded-full transition-opacity flex-shrink-0"
    style="
      background: var(--color-accent);
      color: white;
      opacity: {disabled || !text.trim() ? '0.4' : '1'};
      pointer-events: {disabled || !text.trim() ? 'none' : 'auto'};
    "
    title="Send message"
  >
    <ArrowUp size={16} />
  </button>
</div>
