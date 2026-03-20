<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import ActiveVisitBanner from '$lib/components/visits/ActiveVisitBanner.svelte';
  import EventCard from '$lib/components/events/EventCard.svelte';
  import EventFilter from '$lib/components/events/EventFilter.svelte';
  import SnapshotModal from '$lib/components/events/SnapshotModal.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import SkeletonCard from '$lib/components/ui/SkeletonCard.svelte';
  import SystemStatusPanel from '$lib/components/status/SystemStatusPanel.svelte';
  import { getCameras } from '$lib/api/status';
  import { getActiveVisits } from '$lib/api/visits';
  import { eventsStore } from '$lib/stores/events';
  import { statusStore } from '$lib/stores/status';
  import { wsStore } from '$lib/stores/websocket';
  import type { EventResponse, VisitResponse } from '$lib/types';
  import { Activity } from 'lucide-svelte';
  import { onMount, onDestroy } from 'svelte';

  let loading = $state(true);
  let camera = $state('');
  let type = $state('');
  let live = $state(true);
  let cameras = $state<string[]>([]);
  let activeVisits = $state<VisitResponse[]>([]);
  let selectedEvent = $state<EventResponse | null>(null);
  let hasMore = $state(true);
  let loadingMore = $state(false);

  const events = $derived($eventsStore);
  const status = $derived($statusStore);

  async function loadEvents() {
    loading = true;
    try {
      const res = await eventsStore.load({ camera: camera || undefined, type: type || undefined });
      hasMore = res.items.length === 50;
    } finally {
      loading = false;
    }
  }

  async function loadMore() {
    if (loadingMore || !hasMore) return;
    loadingMore = true;
    try {
      const res = await eventsStore.loadMore(
        { camera: camera || undefined, type: type || undefined },
        events.length
      );
      hasMore = res.items.length === 50;
    } finally {
      loadingMore = false;
    }
  }

  function handleNewEvent(data: unknown) {
    if (live) eventsStore.prepend(data as EventResponse);
    getActiveVisits().then((v) => (activeVisits = v));
  }

  onMount(async () => {
    await loadEvents();
    const [camsData, visitsData] = await Promise.all([getCameras(), getActiveVisits()]);
    cameras = camsData.map((c) => c.name);
    activeVisits = visitsData;
    wsStore.on('event', handleNewEvent);
    wsStore.on('visit_started', () => getActiveVisits().then((v) => (activeVisits = v)));
    wsStore.on('visit_ended', () => getActiveVisits().then((v) => (activeVisits = v)));
  });

  onDestroy(() => {
    wsStore.off('event', handleNewEvent);
  });
</script>

<Header title="Live Feed" />
<ActiveVisitBanner visits={activeVisits} />
<EventFilter
  {cameras}
  {camera}
  {type}
  {live}
  oncamerachange={(v) => { camera = v; loadEvents(); }}
  ontypechange={(v) => { type = v; loadEvents(); }}
  onlivechange={(v) => (live = v)}
/>

<div class="flex flex-1 min-h-0 overflow-hidden">
  <!-- Event feed -->
  <div class="flex-1 overflow-y-auto p-4 space-y-2">
    {#if loading}
      {#each Array(5) as _}
        <SkeletonCard />
      {/each}
    {:else if events.length === 0}
      <EmptyState
        icon={Activity}
        title="No events yet"
        description="Waiting for Ring motion events. Make sure ring-watch is running."
      />
    {:else}
      {#each events as event (event.id)}
        <EventCard {event} onclick={(e) => (selectedEvent = e)} />
      {/each}
      {#if hasMore}
        <button
          onclick={loadMore}
          disabled={loadingMore}
          class="w-full py-2 text-sm rounded-lg transition-colors"
          style="
            background: var(--color-card);
            color: var(--color-text-secondary);
            border: 1px solid var(--color-border);
          "
        >
          {loadingMore ? 'Loading...' : 'Load more'}
        </button>
      {/if}
    {/if}
  </div>

  <!-- Status sidebar -->
  <div
    class="hidden lg:block w-72 p-4 overflow-y-auto border-l flex-shrink-0"
    style="border-color: var(--color-border);"
  >
    <SystemStatusPanel {status} />
  </div>
</div>

<SnapshotModal event={selectedEvent} onclose={() => (selectedEvent = null)} />
