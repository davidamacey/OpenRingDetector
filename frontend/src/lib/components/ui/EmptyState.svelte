<script lang="ts">
  import type { Component } from 'svelte';
  interface Props {
    icon?: Component;
    title: string;
    description?: string;
  }
  let { icon, title, description }: Props = $props();
</script>

<div class="flex flex-col items-center justify-center py-20 px-6 text-center">
  {#if icon}
    <div
      class="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
      style="background: var(--color-card);"
    >
      <svelte:component this={icon} size={32} style="color: var(--color-text-muted);" />
    </div>
  {/if}
  <p class="text-base font-medium mb-1" style="color: var(--color-text-primary);">{title}</p>
  {#if description}
    <p class="text-sm max-w-sm" style="color: var(--color-text-secondary);">{description}</p>
  {/if}
  <slot />
</div>
