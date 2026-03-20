<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import ProfileCard from '$lib/components/profiles/ProfileCard.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import { deleteReference, getReferences, updateReference } from '$lib/api/references';
  import { deleteFace, getFaces, updateFace } from '$lib/api/faces';
  import { toast } from '$lib/stores/toast';
  import type { FaceProfileResponse, ReferenceResponse } from '$lib/types';
  import { Car, Plus, Users } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let vehicles = $state<ReferenceResponse[]>([]);
  let faces = $state<FaceProfileResponse[]>([]);
  let loading = $state(true);
  let editingProfile = $state<ReferenceResponse | FaceProfileResponse | null>(null);
  let editType = $state<'vehicle' | 'face'>('vehicle');
  let editName = $state('');
  let saving = $state(false);
  let tab = $state<'vehicles' | 'people'>('vehicles');

  async function loadData() {
    loading = true;
    try {
      [vehicles, faces] = await Promise.all([getReferences(), getFaces()]);
    } catch {
      toast.error('Failed to load profiles');
    } finally {
      loading = false;
    }
  }

  function openEdit(profile: ReferenceResponse | FaceProfileResponse, type: 'vehicle' | 'face') {
    editingProfile = profile;
    editType = type;
    editName = profile.display_name;
  }

  async function saveEdit() {
    if (!editingProfile) return;
    saving = true;
    try {
      if (editType === 'vehicle') {
        await updateReference((editingProfile as ReferenceResponse).name, { display_name: editName });
        toast.success('Profile updated');
      } else {
        await updateFace((editingProfile as FaceProfileResponse).name, { display_name: editName });
        toast.success('Face profile updated');
      }
      editingProfile = null;
      await loadData();
    } catch {
      toast.error('Failed to save changes');
    } finally {
      saving = false;
    }
  }

  async function deleteProfile() {
    if (!editingProfile) return;
    if (!confirm(`Delete "${editingProfile.display_name}"? This cannot be undone.`)) return;
    try {
      if (editType === 'vehicle') {
        await deleteReference((editingProfile as ReferenceResponse).name);
      } else {
        await deleteFace((editingProfile as FaceProfileResponse).name);
      }
      toast.success('Profile deleted');
      editingProfile = null;
      await loadData();
    } catch {
      toast.error('Failed to delete profile');
    }
  }

  onMount(loadData);
</script>

<Header title="Profiles" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Tabs -->
  <div class="flex gap-1 mb-6 p-1 rounded-lg w-fit" style="background: var(--color-card);">
    {#each [['vehicles', 'Vehicles', vehicles.length], ['people', 'People', faces.length]] as [key, label, count]}
      <button
        onclick={() => (tab = key as 'vehicles' | 'people')}
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
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {#each Array(6) as _}
        <div class="rounded-xl h-48 animate-pulse" style="background: var(--color-card);"></div>
      {/each}
    </div>
  {:else if tab === 'vehicles'}
    {#if vehicles.length === 0}
      <EmptyState
        icon={Car}
        title="No vehicle profiles yet"
        description="Use ring-ref to create vehicle profiles from reference photos."
      />
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {#each vehicles as v (v.uuid)}
          <ProfileCard profile={v} type="vehicle" onedit={(p) => openEdit(p, 'vehicle')} />
        {/each}
      </div>
    {/if}
  {:else}
    {#if faces.length === 0}
      <EmptyState
        icon={Users}
        title="No face profiles yet"
        description="Use ring-face to add known people."
      />
    {:else}
      <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {#each faces as f (f.uuid)}
          <ProfileCard profile={f} type="face" onedit={(p) => openEdit(p, 'face')} />
        {/each}
      </div>
    {/if}
  {/if}
</div>

<!-- Edit Modal -->
<Modal open={!!editingProfile} title="Edit Profile" onclose={() => (editingProfile = null)}>
  {#if editingProfile}
    <div class="space-y-4">
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Display Name</label>
        <input
          type="text"
          bind:value={editName}
          class="w-full px-3 py-2 rounded-lg text-sm"
          style="
            background: var(--color-surface);
            color: var(--color-text-primary);
            border: 1px solid var(--color-border);
          "
        />
      </div>
      <div class="flex gap-2">
        <button
          onclick={saveEdit}
          disabled={saving}
          class="flex-1 py-2 rounded-lg text-sm font-medium"
          style="background: var(--color-accent); color: white;"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button
          onclick={deleteProfile}
          class="py-2 px-4 rounded-lg text-sm font-medium"
          style="background: rgba(239,68,68,0.15); color: var(--color-red);"
        >
          Delete
        </button>
      </div>
    </div>
  {/if}
</Modal>
