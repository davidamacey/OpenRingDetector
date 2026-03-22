<script lang="ts">
  import '../app.css';
  import Sidebar from '$lib/components/layout/Sidebar.svelte';
  import ChatPanel from '$lib/components/chat/ChatPanel.svelte';
  import Toast from '$lib/components/ui/Toast.svelte';
  import { statusStore } from '$lib/stores/status';
  import { wsStore } from '$lib/stores/websocket';
  import { chatStore } from '$lib/stores/chat';
  import { onMount } from 'svelte';

  let { children } = $props();

  onMount(() => {
    statusStore.startPolling(30000);
    wsStore.connect();
  });

  function handleKeydown(e: KeyboardEvent) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      chatStore.togglePanel();
    }
  }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="flex h-screen overflow-hidden" style="background: var(--color-bg);">
  <Sidebar />
  <div class="flex-1 flex flex-col min-w-0 overflow-hidden">
    {@render children()}
  </div>
</div>
<ChatPanel />
<Toast />
