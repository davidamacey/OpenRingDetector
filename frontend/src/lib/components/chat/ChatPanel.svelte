<script lang="ts">
  import { X, Trash2 } from 'lucide-svelte';
  import { onMount, tick } from 'svelte';
  import ChatMessage from '$lib/components/chat/ChatMessage.svelte';
  import ChatInput from '$lib/components/chat/ChatInput.svelte';
  import { chatStore } from '$lib/stores/chat';
  import { getChatStatus, warmupChat } from '$lib/api/chat';

  const messages = $derived($chatStore.messages);
  const isStreaming = $derived($chatStore.isStreaming);
  const isOpen = $derived($chatStore.isOpen);

  let scrollContainer: HTMLDivElement | undefined = undefined;
  let userScrolledUp = $state(false);
  let statusAvailable = $state<boolean | null>(null);
  let coldStartPhase = $state('');
  let coldStartTimer1: ReturnType<typeof setTimeout> | null = null;
  let coldStartTimer2: ReturnType<typeof setTimeout> | null = null;

  const suggestedChips = [
    'What happened today?',
    'Last visitor',
    'Events this week',
    'Anyone here now?'
  ];

  function handleScroll() {
    if (!scrollContainer) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
    // User is "scrolled up" if they're more than 60px from the bottom
    userScrolledUp = scrollHeight - scrollTop - clientHeight > 60;
  }

  async function scrollToBottom() {
    await tick();
    if (scrollContainer && !userScrolledUp) {
      scrollContainer.scrollTop = scrollContainer.scrollHeight;
    }
  }

  // Track last message content length for auto-scroll during streaming
  const lastMsgContent = $derived(
    messages.length > 0 ? messages[messages.length - 1].content.length : 0
  );

  // Auto-scroll when messages change or streaming content updates
  $effect(() => {
    // Track both message count and content length so effect re-runs on token append
    void lastMsgContent;
    if (messages.length > 0) {
      scrollToBottom();
    }
  });

  // Cold start phase tracking during streaming
  $effect(() => {
    if (isStreaming) {
      coldStartPhase = 'Sending...';
      coldStartTimer1 = setTimeout(() => {
        coldStartPhase = 'Loading AI model...';
      }, 3000);
      coldStartTimer2 = setTimeout(() => {
        coldStartPhase = 'Generating...';
      }, 8000);
    } else {
      coldStartPhase = '';
    }
    return () => {
      if (coldStartTimer1) clearTimeout(coldStartTimer1);
      if (coldStartTimer2) clearTimeout(coldStartTimer2);
      coldStartTimer1 = null;
      coldStartTimer2 = null;
    };
  });

  async function sendMessage(text: string) {
    chatStore.sendMessage(text);
  }

  async function handleClear() {
    chatStore.clearHistory();
  }

  function handleClose() {
    chatStore.close();
  }

  onMount(async () => {
    // Check chat availability
    try {
      const status = await getChatStatus();
      statusAvailable = status.available;
    } catch {
      statusAvailable = false;
    }

    // Preload AI model
    try {
      await warmupChat();
    } catch {
      // Warmup failure is non-critical
    }
  });
</script>

{#if isOpen}
  <!-- Backdrop for mobile -->
  <button
    class="fixed inset-0 z-30 md:hidden"
    style="background: rgba(0,0,0,0.5);"
    onclick={handleClose}
    aria-label="Close chat"
  ></button>

  <aside
    class="fixed top-0 right-0 bottom-0 z-40 flex flex-col w-full md:w-[400px]"
    style="
      background: var(--color-surface);
      border-left: 1px solid var(--color-border);
      transform: translateX(0);
      transition: transform 200ms ease-out;
    "
  >
    <!-- Header -->
    <div
      class="flex items-center justify-between px-4 py-3 border-b flex-shrink-0"
      style="border-color: var(--color-border);"
    >
      <h2 class="text-sm font-semibold" style="color: var(--color-text-primary);">Chat</h2>
      <div class="flex items-center gap-1">
        <button
          onclick={handleClear}
          class="p-1.5 rounded-lg transition-colors"
          style="color: var(--color-text-muted);"
          title="Clear history"
          disabled={messages.length === 0}
        >
          <Trash2 size={16} />
        </button>
        <button
          onclick={handleClose}
          class="p-1.5 rounded-lg transition-colors"
          style="color: var(--color-text-muted);"
          title="Close chat"
        >
          <X size={16} />
        </button>
      </div>
    </div>

    <!-- Status banner -->
    {#if statusAvailable === false}
      <div
        class="px-4 py-2 text-xs"
        style="background: rgba(245,158,11,0.12); color: var(--color-amber);"
      >
        Chat is not available. Check that the Ollama service is running.
      </div>
    {/if}

    <!-- Message area -->
    <div
      bind:this={scrollContainer}
      onscroll={handleScroll}
      class="flex-1 overflow-y-auto px-4 py-4"
    >
      {#if messages.length === 0}
        <!-- Empty state with suggested chips -->
        <div class="flex flex-col items-center justify-center h-full gap-4">
          <p class="text-sm" style="color: var(--color-text-muted);">
            Ask about events, visitors, or vehicles.
          </p>
          <div class="flex flex-wrap justify-center gap-2">
            {#each suggestedChips as chip}
              <button
                onclick={() => sendMessage(chip)}
                class="px-3 py-1.5 rounded-full text-xs font-medium transition-colors"
                style="
                  background: var(--color-card);
                  color: var(--color-text-secondary);
                  border: 1px solid var(--color-border);
                "
                disabled={isStreaming}
              >
                {chip}
              </button>
            {/each}
          </div>
        </div>
      {:else}
        {#each messages as msg (msg.id)}
          <ChatMessage message={msg} />
        {/each}
      {/if}
    </div>

    <!-- Cold start phase indicator -->
    {#if isStreaming && coldStartPhase}
      <div
        class="px-4 py-1.5 text-xs text-center"
        style="color: var(--color-text-muted); background: var(--color-bg);"
      >
        {coldStartPhase}
      </div>
    {/if}

    <!-- Input -->
    <ChatInput onSend={sendMessage} disabled={isStreaming || statusAvailable === false} />
  </aside>
{/if}
