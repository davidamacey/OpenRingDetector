<script lang="ts">
  interface Props {
    status: 'ok' | 'warn' | 'fail' | 'missing' | 'off';
    pulse?: boolean;
    size?: number;
  }
  let { status, pulse = false, size = 8 }: Props = $props();

  const colorMap: Record<string, string> = {
    ok: 'var(--color-green)',
    warn: 'var(--color-amber)',
    fail: 'var(--color-red)',
    missing: 'var(--color-amber)',
    off: 'var(--color-text-muted)'
  };
  const color = $derived(colorMap[status] ?? 'var(--color-text-muted)');
</script>

<span
  class="inline-block rounded-full flex-shrink-0 {pulse && status === 'ok' ? 'animate-pulse' : ''}"
  style="width: {size}px; height: {size}px; background: {color};"
></span>
