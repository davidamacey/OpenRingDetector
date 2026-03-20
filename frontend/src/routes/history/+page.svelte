<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import VisitCard from '$lib/components/visits/VisitCard.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import SkeletonCard from '$lib/components/ui/SkeletonCard.svelte';
  import { getVisits } from '$lib/api/visits';
  import type { VisitListResponse, VisitResponse } from '$lib/types';
  import { Clock } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let loading = $state(true);
  let visits = $state<VisitResponse[]>([]);
  let total = $state(0);
  let nameFilter = $state('');
  let activeOnly = $state(false);
  let offset = $state(0);
  let hasMore = $state(false);
  let debounceTimer: ReturnType<typeof setTimeout>;

  async function load(reset = true) {
    if (reset) {
      offset = 0;
      visits = [];
    }
    loading = true;
    try {
      const res = await getVisits({
        limit: 50,
        offset: reset ? 0 : offset,
        name: nameFilter || undefined,
        active_only: activeOnly || undefined,
      });
      if (reset) {
        visits = res.items;
      } else {
        visits = [...visits, ...res.items];
        offset += res.items.length;
      }
      total = res.total;
      hasMore = visits.length < res.total;
    } finally {
      loading = false;
    }
  }

  function onNameInput() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => load(), 300);
  }

  // Group by date
  const grouped = $derived(
    visits.reduce(
      (acc, v) => {
        const date = new Date(v.arrived_at).toLocaleDateString('en-US', {
          weekday: 'long',
          month: 'long',
          day: 'numeric'
        });
        if (!acc[date]) acc[date] = [];
        acc[date].push(v);
        return acc;
      },
      {} as Record<string, VisitResponse[]>
    )
  );

  onMount(() => load());
</script>

<Header title="Visit History" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Filters -->
  <div class="flex flex-wrap gap-3 mb-4">
    <input
      type="text"
      placeholder="Search by name..."
      bind:value={nameFilter}
      oninput={onNameInput}
      class="flex-1 min-w-48 text-sm px-3 py-2 rounded-lg"
      style="
        background: var(--color-card);
        color: var(--color-text-primary);
        border: 1px solid var(--color-border);
      "
    />
    <label class="flex items-center gap-2 text-sm cursor-pointer" style="color: var(--color-text-secondary);">
      <input
        type="checkbox"
        bind:checked={activeOnly}
        onchange={() => load()}
        class="rounded"
      />
      Active only
    </label>
    <span class="text-sm" style="color: var(--color-text-muted);">
      {total} visits total
    </span>
  </div>

  {#if loading && visits.length === 0}
    <div class="space-y-2">
      {#each Array(5) as _}
        <SkeletonCard />
      {/each}
    </div>
  {:else if visits.length === 0}
    <EmptyState
      icon={Clock}
      title="No visits found"
      description="No visits match your filters, or ring-watch hasn't recorded any yet."
    />
  {:else}
    {#each Object.entries(grouped) as [date, dayVisits]}
      <div class="mb-6">
        <div
          class="text-xs font-semibold uppercase tracking-wider py-2 mb-2 border-b"
          style="color: var(--color-text-muted); border-color: var(--color-border);"
        >
          {date}
        </div>
        <div class="space-y-2">
          {#each dayVisits as visit (visit.id)}
            <VisitCard {visit} />
          {/each}
        </div>
      </div>
    {/each}

    {#if hasMore}
      <button
        onclick={() => load(false)}
        class="w-full py-2 text-sm rounded-lg mt-2"
        style="
          background: var(--color-card);
          color: var(--color-text-secondary);
          border: 1px solid var(--color-border);
        "
      >
        Load more
      </button>
    {/if}
  {/if}
</div>
