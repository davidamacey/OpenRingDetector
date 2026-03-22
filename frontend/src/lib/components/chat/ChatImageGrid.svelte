<script lang="ts">
  import type { ChatImage } from '$lib/types';

  interface Props {
    images: ChatImage[];
  }
  let { images }: Props = $props();

  const displayImages = $derived(images.slice(0, 4));
  const columns = $derived(displayImages.length === 2 ? 2 : displayImages.length >= 3 ? 3 : 1);
</script>

<div
  class="grid gap-2"
  style="grid-template-columns: repeat({columns}, 1fr);"
>
  {#each displayImages as img}
    <a
      href={img.url}
      target="_blank"
      rel="noopener noreferrer"
      class="block rounded-lg overflow-hidden transition-opacity hover:opacity-80"
      style="border: 1px solid var(--color-border);"
    >
      <img
        src={img.url}
        alt={img.alt ?? 'image'}
        class="w-full h-auto object-cover"
        style="min-height: 80px; max-height: 200px;"
      />
      {#if img.alt}
        <p
          class="text-xs px-2 py-1 truncate"
          style="color: var(--color-text-muted); background: var(--color-surface);"
        >
          {img.alt}
        </p>
      {/if}
    </a>
  {/each}
</div>
