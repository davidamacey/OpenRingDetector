<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import {
    getActivityHeatmap,
    getDetectionTypes,
    getEventsPerDay,
    getTopVisitors,
    getVisitDurations
  } from '$lib/api/analytics';
  import type {
    AnalyticsDetectionType,
    AnalyticsEventPerDay,
    AnalyticsHeatmap,
    AnalyticsTopVisitor,
    AnalyticsVisitDuration
  } from '$lib/types';
  import { BarChart3 } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let loading = $state(true);
  let days = $state(7);
  let eventsPerDay = $state<AnalyticsEventPerDay[]>([]);
  let heatmap = $state<AnalyticsHeatmap[]>([]);
  let topVisitors = $state<AnalyticsTopVisitor[]>([]);
  let detectionTypes = $state<AnalyticsDetectionType[]>([]);
  let visitDurations = $state<AnalyticsVisitDuration[]>([]);

  async function load() {
    loading = true;
    try {
      [eventsPerDay, heatmap, topVisitors, detectionTypes, visitDurations] = await Promise.all([
        getEventsPerDay(days),
        getActivityHeatmap(days),
        getTopVisitors(30),
        getDetectionTypes(days),
        getVisitDurations(30)
      ]);
    } finally {
      loading = false;
    }
  }

  const maxDay = $derived(Math.max(...eventsPerDay.map((d) => d.count), 1));
  const maxHour = $derived(Math.max(...heatmap.map((h) => h.count), 1));
  const heatmapByHour = $derived(
    Array.from({ length: 24 }, (_, i) => heatmap.find((h) => h.hour === i)?.count ?? 0)
  );

  onMount(load);
</script>

<Header title="Analytics" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Period selector -->
  <div class="flex items-center gap-3 mb-6">
    <span class="text-sm" style="color: var(--color-text-secondary);">Period:</span>
    {#each [[7, '7 days'], [14, '14 days'], [30, '30 days']] as [d, label]}
      <button
        onclick={() => { days = d as number; load(); }}
        class="px-3 py-1.5 rounded text-sm"
        style="
          background: {days === d ? 'var(--color-accent)' : 'var(--color-card)'};
          color: {days === d ? 'white' : 'var(--color-text-secondary)'};
          border: 1px solid {days === d ? 'var(--color-accent)' : 'var(--color-border)'};
        "
      >
        {label}
      </button>
    {/each}
  </div>

  {#if loading}
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {#each Array(4) as _}
        <div class="h-48 rounded-xl animate-pulse" style="background: var(--color-card);"></div>
      {/each}
    </div>
  {:else if eventsPerDay.length === 0 && topVisitors.length === 0}
    <EmptyState
      icon={BarChart3}
      title="Not enough data yet"
      description="Check back after a few days of monitoring."
    />
  {:else}
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <!-- Events per day -->
      <div class="rounded-xl p-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-4" style="color: var(--color-text-primary);">Events Per Day</h3>
        {#if eventsPerDay.length === 0}
          <p class="text-sm" style="color: var(--color-text-muted);">No data</p>
        {:else}
          <div class="flex items-end gap-1 h-32">
            {#each eventsPerDay as d}
              <div class="flex-1 flex flex-col items-center gap-1">
                <div
                  class="w-full rounded-t transition-all"
                  style="
                    height: {Math.max(4, (d.count / maxDay) * 112)}px;
                    background: var(--color-accent);
                    opacity: 0.8;
                  "
                  title="{d.date}: {d.count}"
                ></div>
                <span class="text-xs" style="color: var(--color-text-muted);">
                  {new Date(d.date).toLocaleDateString('en', { weekday: 'short' })}
                </span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Top visitors -->
      <div class="rounded-xl p-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-4" style="color: var(--color-text-primary);">Top Visitors (30 days)</h3>
        {#if topVisitors.length === 0}
          <p class="text-sm" style="color: var(--color-text-muted);">No visits recorded</p>
        {:else}
          <div class="space-y-2">
            {#each topVisitors.slice(0, 8) as v, i}
              <div class="flex items-center gap-3">
                <span class="text-xs w-4 text-right flex-shrink-0" style="color: var(--color-text-muted);">{i + 1}</span>
                <div class="flex-1 min-w-0">
                  <div class="flex items-center justify-between mb-0.5">
                    <span class="text-sm truncate" style="color: var(--color-text-primary);">{v.display_name}</span>
                    <span class="text-xs flex-shrink-0 ml-2" style="color: var(--color-text-secondary);">{v.visit_count}</span>
                  </div>
                  <div class="h-1.5 rounded-full" style="background: var(--color-border);">
                    <div
                      class="h-full rounded-full"
                      style="
                        width: {(v.visit_count / topVisitors[0].visit_count) * 100}%;
                        background: var(--color-accent);
                      "
                    ></div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Activity heatmap -->
      <div class="rounded-xl p-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-4" style="color: var(--color-text-primary);">Activity by Hour</h3>
        <div class="flex items-end gap-0.5 h-24">
          {#each heatmapByHour as count, hr}
            <div
              class="flex-1 rounded-sm"
              style="
                height: {count === 0 ? '4px' : Math.max(8, (count / maxHour) * 88) + 'px'};
                background: {count === 0 ? 'var(--color-border)' : 'var(--color-teal)'};
                opacity: {count === 0 ? 0.3 : 0.4 + (count / maxHour) * 0.6};
              "
              title="{hr}:00 — {count} events"
            ></div>
          {/each}
        </div>
        <div class="flex justify-between mt-1">
          <span class="text-xs" style="color: var(--color-text-muted);">12am</span>
          <span class="text-xs" style="color: var(--color-text-muted);">12pm</span>
          <span class="text-xs" style="color: var(--color-text-muted);">11pm</span>
        </div>
      </div>

      <!-- Detection types -->
      <div class="rounded-xl p-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-4" style="color: var(--color-text-primary);">Detection Types</h3>
        {#if detectionTypes.length === 0}
          <p class="text-sm" style="color: var(--color-text-muted);">No detections recorded</p>
        {:else}
          <div class="space-y-2">
            {#each detectionTypes as d}
              <div class="flex items-center gap-3">
                <span class="text-sm w-16 flex-shrink-0" style="color: var(--color-text-primary);">{d.class_name}</span>
                <div class="flex-1 h-2 rounded-full" style="background: var(--color-border);">
                  <div
                    class="h-full rounded-full"
                    style="width: {d.percentage}%; background: var(--color-teal);"
                  ></div>
                </div>
                <span class="text-xs w-12 text-right" style="color: var(--color-text-muted);">{d.percentage}%</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <!-- Visit durations -->
      <div class="rounded-xl p-4 lg:col-span-2" style="background: var(--color-card); border: 1px solid var(--color-border);">
        <h3 class="text-sm font-semibold mb-4" style="color: var(--color-text-primary);">Visit Duration Distribution (30 days)</h3>
        {#if visitDurations.every((d) => d.count === 0)}
          <p class="text-sm" style="color: var(--color-text-muted);">No completed visits yet</p>
        {:else}
          {@const maxDur = Math.max(...visitDurations.map((d) => d.count), 1)}
          <div class="flex items-end gap-3 h-28">
            {#each visitDurations as d}
              <div class="flex-1 flex flex-col items-center gap-2">
                <span class="text-xs" style="color: var(--color-text-secondary);">{d.count}</span>
                <div
                  class="w-full rounded-t"
                  style="
                    height: {Math.max(4, (d.count / maxDur) * 80)}px;
                    background: var(--color-violet);
                    opacity: 0.8;
                  "
                ></div>
                <span class="text-xs text-center leading-tight" style="color: var(--color-text-muted);">{d.bucket}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>
  {/if}
</div>
