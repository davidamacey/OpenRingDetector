<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import EventCard from '$lib/components/events/EventCard.svelte';
  import SnapshotModal from '$lib/components/events/SnapshotModal.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import SkeletonCard from '$lib/components/ui/SkeletonCard.svelte';
  import { getEvents } from '$lib/api/events';
  import { getCameras } from '$lib/api/status';
  import type { EventResponse } from '$lib/types';
  import { Clock } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let loading = $state(true);
  let events = $state<EventResponse[]>([]);
  let total = $state(0);
  let hasMore = $state(false);
  let loadingMore = $state(false);
  let selectedEvent = $state<EventResponse | null>(null);

  // Filters
  let camera = $state('');
  let type = $state('');
  let fromDate = $state('');
  let toDate = $state('');
  let cameras = $state<string[]>([]);

  async function load(reset = true) {
    if (reset) events = [];
    loading = reset;
    loadingMore = !reset;
    try {
      const res = await getEvents({
        limit: 50,
        offset: reset ? 0 : events.length,
        camera: camera || undefined,
        type: type || undefined,
        from: fromDate || undefined,
        to: toDate || undefined,
      });
      if (reset) {
        events = res.items;
      } else {
        events = [...events, ...res.items];
      }
      total = res.total;
      hasMore = events.length < total;
    } finally {
      loading = false;
      loadingMore = false;
    }
  }

  function resetFilters() {
    camera = '';
    type = '';
    fromDate = '';
    toDate = '';
    load();
  }

  // Group events by date
  const grouped = $derived(
    events.reduce(
      (acc, ev) => {
        const date = new Date(ev.occurred_at).toLocaleDateString('en-US', {
          weekday: 'long',
          month: 'long',
          day: 'numeric'
        });
        if (!acc[date]) acc[date] = [];
        acc[date].push(ev);
        return acc;
      },
      {} as Record<string, EventResponse[]>
    )
  );

  onMount(async () => {
    const [_, camsData] = await Promise.all([load(), getCameras()]);
    cameras = camsData.map((c) => c.name);
  });
</script>

<Header title="Event History" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Filters -->
  <div
    class="flex flex-wrap items-center gap-3 mb-4 p-3 rounded-xl"
    style="background: var(--color-card); border: 1px solid var(--color-border);"
  >
    <select
      bind:value={camera}
      onchange={() => load()}
      class="text-sm rounded-lg px-3 py-2"
      style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
    >
      <option value="">All Cameras</option>
      {#each cameras as cam}
        <option value={cam}>{cam}</option>
      {/each}
    </select>

    <select
      bind:value={type}
      onchange={() => load()}
      class="text-sm rounded-lg px-3 py-2"
      style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
    >
      <option value="">All Types</option>
      <option value="motion">Motion</option>
      <option value="arrival">Arrival</option>
      <option value="departure">Departure</option>
      <option value="ding">Doorbell</option>
    </select>

    <input
      type="date"
      bind:value={fromDate}
      onchange={() => load()}
      class="text-sm rounded-lg px-3 py-2"
      style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      placeholder="From"
    />

    <input
      type="date"
      bind:value={toDate}
      onchange={() => load()}
      class="text-sm rounded-lg px-3 py-2"
      style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      placeholder="To"
    />

    {#if camera || type || fromDate || toDate}
      <button
        onclick={resetFilters}
        class="text-xs px-3 py-2 rounded-lg"
        style="color: var(--color-accent); background: rgba(59,130,246,0.1);"
      >
        Clear filters
      </button>
    {/if}

    <span class="text-xs ml-auto" style="color: var(--color-text-muted);">
      {total} events
    </span>
  </div>

  <!-- Event list -->
  {#if loading}
    <div class="space-y-2">
      {#each Array(8) as _}
        <SkeletonCard />
      {/each}
    </div>
  {:else if events.length === 0}
    <EmptyState
      icon={Clock}
      title="No events found"
      description="No events match your filters, or ring-watch hasn't recorded any yet."
    />
  {:else}
    {#each Object.entries(grouped) as [date, dayEvents]}
      <div class="mb-6">
        <div
          class="text-xs font-semibold uppercase tracking-wider py-2 mb-2 border-b"
          style="color: var(--color-text-muted); border-color: var(--color-border);"
        >
          {date}
        </div>
        <div class="space-y-2">
          {#each dayEvents as event (event.id)}
            <EventCard {event} onclick={(e) => (selectedEvent = e)} />
          {/each}
        </div>
      </div>
    {/each}

    {#if hasMore}
      <button
        onclick={() => load(false)}
        disabled={loadingMore}
        class="w-full py-2.5 text-sm rounded-lg mt-2 transition-colors"
        style="background: var(--color-card); color: var(--color-text-secondary); border: 1px solid var(--color-border);"
      >
        {loadingMore ? 'Loading...' : 'Load more events'}
      </button>
    {/if}
  {/if}
</div>

<SnapshotModal event={selectedEvent} onclose={() => (selectedEvent = null)} />
