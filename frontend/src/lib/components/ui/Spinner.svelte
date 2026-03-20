<script lang="ts">
  interface Props { size?: number; }
  let { size = 24 }: Props = $props();
</script>

<svg
  width={size}
  height={size}
  viewBox="0 0 24 24"
  fill="none"
  class="animate-spin"
  style="color: var(--color-accent);"
>
  <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" opacity="0.25" />
  <path
    d="M12 2a10 10 0 0 1 10 10"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
  />
</svg>
