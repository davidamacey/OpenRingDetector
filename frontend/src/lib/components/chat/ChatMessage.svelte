<script lang="ts">
  import Badge from '$lib/components/ui/Badge.svelte';
  import type { ChatMessage } from '$lib/types';

  interface Props {
    message: ChatMessage;
  }
  let { message }: Props = $props();

  const isUser = $derived(message.role === 'user');
  const hasImages = $derived(message.images && message.images.length > 0);
  const hasReferences = $derived(message.referenceCards && message.referenceCards.length > 0);
  const hasFaces = $derived(message.faceCards && message.faceCards.length > 0);
  const hasEvents = $derived(message.eventDetails && message.eventDetails.length > 0);
  const hasError = $derived(!!message.error);
  const formattedTime = $derived(
    new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  );
</script>

<div class="flex {isUser ? 'justify-end' : 'justify-start'} mb-3">
  <div
    class="max-w-[85%] rounded-xl px-3.5 py-2.5"
    style="
      background: {hasError
        ? 'rgba(239,68,68,0.12)'
        : isUser
          ? 'rgba(59,130,246,0.18)'
          : 'var(--color-card)'};
      border: 1px solid {hasError
        ? 'rgba(239,68,68,0.25)'
        : isUser
          ? 'rgba(59,130,246,0.25)'
          : 'var(--color-border)'};
    "
  >
    <!-- Inline images -->
    {#if hasImages}
      <div class="flex flex-wrap gap-2 mb-2">
        {#each message.images! as img}
          <a href={img.url} target="_blank" rel="noopener noreferrer" class="block">
            <img
              src={img.url}
              alt={img.alt ?? 'snapshot'}
              class="rounded-lg object-cover cursor-pointer transition-opacity hover:opacity-80"
              style="max-height: 128px; max-width: 180px;"
            />
          </a>
        {/each}
      </div>
    {/if}

    <!-- Error text -->
    {#if hasError}
      <p class="text-sm" style="color: var(--color-red);">{message.error}</p>
    {/if}

    <!-- Message content -->
    {#if message.content}
      <p
        class="text-sm leading-relaxed whitespace-pre-wrap break-words"
        style="color: {isUser ? 'var(--color-text-primary)' : 'var(--color-text-secondary)'};"
      >{message.content}</p>
    {/if}

    <!-- Streaming indicator -->
    {#if message.isStreaming}
      <div class="flex items-center gap-1 mt-1">
        <span class="streaming-dot" style="animation-delay: 0ms;"></span>
        <span class="streaming-dot" style="animation-delay: 150ms;"></span>
        <span class="streaming-dot" style="animation-delay: 300ms;"></span>
      </div>
    {/if}

    <!-- Reference cards -->
    {#if hasReferences}
      <div class="flex flex-col gap-2 mt-2">
        {#each message.referenceCards! as ref}
          <div
            class="flex items-center gap-3 rounded-lg p-2.5"
            style="background: var(--color-surface); border: 1px solid var(--color-border);"
          >
            {#if ref.image_url}
              <img
                src={ref.image_url}
                alt={ref.display_name}
                class="w-10 h-10 rounded object-cover flex-shrink-0"
              />
            {:else}
              <div
                class="w-10 h-10 rounded flex items-center justify-center flex-shrink-0"
                style="background: var(--color-border);"
              >
                <span class="text-xs" style="color: var(--color-text-muted);">N/A</span>
              </div>
            {/if}
            <div class="min-w-0">
              <p class="text-sm font-medium truncate" style="color: var(--color-text-primary);">
                {ref.display_name}
              </p>
              <p class="text-xs" style="color: var(--color-text-muted);">
                {ref.visit_count} visit{ref.visit_count === 1 ? '' : 's'}
                {#if ref.last_seen}
                  - Last seen {ref.last_seen}
                {/if}
              </p>
            </div>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Face cards -->
    {#if hasFaces}
      <div class="flex flex-col gap-2 mt-2">
        {#each message.faceCards! as face}
          <div
            class="flex items-center gap-3 rounded-lg p-2.5"
            style="background: var(--color-surface); border: 1px solid var(--color-border);"
          >
            {#if face.image_url}
              <img
                src={face.image_url}
                alt={face.display_name}
                class="w-10 h-10 rounded-full object-cover flex-shrink-0"
              />
            {:else}
              <div
                class="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                style="background: var(--color-border);"
              >
                <span class="text-xs" style="color: var(--color-text-muted);">?</span>
              </div>
            {/if}
            <div class="min-w-0">
              <p class="text-sm font-medium truncate" style="color: var(--color-text-primary);">
                {face.display_name}
              </p>
              <p class="text-xs" style="color: var(--color-text-muted);">
                {face.visit_count} visit{face.visit_count === 1 ? '' : 's'}
                {#if face.last_seen}
                  - Last seen {face.last_seen}
                {/if}
              </p>
            </div>
          </div>
        {/each}
      </div>
    {/if}

    <!-- Event detail cards -->
    {#if hasEvents}
      <div class="flex flex-col gap-2 mt-2">
        {#each message.eventDetails! as evt}
          <div
            class="rounded-lg p-2.5"
            style="background: var(--color-surface); border: 1px solid var(--color-border);"
          >
            <div class="flex items-center gap-2 mb-1">
              <Badge type={evt.event_type} size="sm" />
              <span class="text-xs font-medium" style="color: var(--color-text-primary);">
                {evt.camera_name}
              </span>
              <span class="text-xs ml-auto" style="color: var(--color-text-muted);">
                {new Date(evt.occurred_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
            {#if evt.detection_summary}
              <p class="text-xs" style="color: var(--color-text-secondary);">
                {evt.detection_summary}
              </p>
            {/if}
            {#if evt.display_name}
              <p class="text-xs font-medium mt-0.5" style="color: var(--color-accent);">
                {evt.display_name}
              </p>
            {/if}
          </div>
        {/each}
      </div>
    {/if}

    <!-- Timestamp -->
    <p
      class="text-[10px] mt-1.5 {isUser ? 'text-right' : 'text-left'}"
      style="color: var(--color-text-muted);"
    >
      {formattedTime}
    </p>
  </div>
</div>

<style>
  .streaming-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--color-text-muted);
    animation: blink 1s infinite;
  }

  @keyframes blink {
    0%, 60%, 100% {
      opacity: 0.2;
    }
    30% {
      opacity: 1;
    }
  }
</style>
