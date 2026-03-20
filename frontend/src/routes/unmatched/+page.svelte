<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { api } from '$lib/api/client';
  import { toast } from '$lib/stores/toast';
  import type { UnmatchedFace, UnmatchedVehicle } from '$lib/types';
  import { CheckCircle, HelpCircle } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let vehicles = $state<UnmatchedVehicle[]>([]);
  let faces = $state<UnmatchedFace[]>([]);
  let loading = $state(true);
  let tab = $state<'vehicles' | 'faces'>('vehicles');

  async function loadData() {
    loading = true;
    try {
      const [v, f] = await Promise.all([
        api.get('/unmatched/vehicles').then((r) => r.data),
        api.get('/unmatched/faces').then((r) => r.data),
      ]);
      vehicles = v;
      faces = f;
    } catch {
      toast.error('Failed to load unmatched detections');
    } finally {
      loading = false;
    }
  }

  async function dismiss(type: 'vehicle' | 'face', id: string) {
    await api.post('/unmatched/dismiss', { type, id });
    await loadData();
    toast.success('Dismissed');
  }

  onMount(loadData);
</script>

<Header title="Unmatched Detections" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Tabs -->
  <div class="flex gap-1 mb-6 p-1 rounded-lg w-fit" style="background: var(--color-card);">
    {#each [['vehicles', 'Vehicles', vehicles.length], ['faces', 'Faces', faces.length]] as [key, label, count]}
      <button
        onclick={() => (tab = key as 'vehicles' | 'faces')}
        class="px-4 py-2 rounded-md text-sm font-medium transition-colors"
        style="
          background: {tab === key ? 'var(--color-accent)' : 'transparent'};
          color: {tab === key ? 'white' : 'var(--color-text-secondary)'};
        "
      >
        {label} ({count})
      </button>
    {/each}
  </div>

  {#if loading}
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
      {#each Array(6) as _}
        <div class="h-52 rounded-xl animate-pulse" style="background: var(--color-card);"></div>
      {/each}
    </div>
  {:else if tab === 'vehicles'}
    {#if vehicles.length === 0}
      <EmptyState
        icon={CheckCircle}
        title="No unmatched vehicles"
        description="All detected vehicles are identified."
      />
    {:else}
      <p class="text-sm mb-4" style="color: var(--color-text-secondary);">
        {vehicles.length} unmatched motion events in the last 30 days
      </p>
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {#each vehicles as v (v.cluster_id)}
          <div
            class="rounded-xl overflow-hidden"
            style="background: var(--color-card); border: 1px solid var(--color-border);"
          >
            <div class="h-32" style="background: var(--color-surface);">
              {#if v.representative_crop_url}
                <img src={v.representative_crop_url} alt="unmatched" class="w-full h-full object-cover" />
              {:else}
                <div class="h-full flex items-center justify-center">
                  <HelpCircle size={32} style="color: var(--color-text-muted);" />
                </div>
              {/if}
            </div>
            <div class="p-3">
              <p class="text-xs mb-1" style="color: var(--color-text-secondary);">
                {v.camera_name} · {new Date(v.last_seen).toLocaleDateString()}
              </p>
              <button
                onclick={() => dismiss('vehicle', v.cluster_id)}
                class="w-full mt-2 py-1.5 text-xs rounded"
                style="
                  background: var(--color-border);
                  color: var(--color-text-secondary);
                "
              >
                Dismiss
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {:else}
    {#if faces.length === 0}
      <EmptyState
        icon={CheckCircle}
        title="No unmatched faces"
        description="All detected faces are identified."
      />
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {#each faces as f (f.face_embedding_uuid)}
          <div
            class="rounded-xl overflow-hidden"
            style="background: var(--color-card); border: 1px solid var(--color-border);"
          >
            <div class="h-32" style="background: var(--color-surface);">
              {#if f.crop_url}
                <img src={f.crop_url} alt="face" class="w-full h-full object-cover" />
              {:else}
                <div class="h-full flex items-center justify-center">
                  <HelpCircle size={32} style="color: var(--color-text-muted);" />
                </div>
              {/if}
            </div>
            <div class="p-3">
              <p class="text-xs mb-1" style="color: var(--color-text-secondary);">
                {new Date(f.last_seen).toLocaleDateString()}
              </p>
              <button
                onclick={() => dismiss('face', f.face_embedding_uuid)}
                class="w-full mt-2 py-1.5 text-xs rounded"
                style="background: var(--color-border); color: var(--color-text-secondary);"
              >
                Dismiss
              </button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>
