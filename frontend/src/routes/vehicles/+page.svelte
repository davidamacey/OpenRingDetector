<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import ProfileCard from '$lib/components/profiles/ProfileCard.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import { deleteReference, getReferences, updateReference } from '$lib/api/references';
  import { toast } from '$lib/stores/toast';
  import type { ReferenceResponse } from '$lib/types';
  import { Car, Plus, Upload } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let vehicles = $state<ReferenceResponse[]>([]);
  let loading = $state(true);
  let editingRef = $state<ReferenceResponse | null>(null);
  let editName = $state('');
  let editCategory = $state('');
  let saving = $state(false);

  // Add new reference
  let showAdd = $state(false);
  let newName = $state('');
  let newDisplayName = $state('');
  let newCategory = $state('vehicle');
  let newFiles = $state<FileList | null>(null);
  let adding = $state(false);

  async function loadData() {
    loading = true;
    try {
      vehicles = await getReferences();
    } catch {
      toast.error('Failed to load vehicle references');
    } finally {
      loading = false;
    }
  }

  function openEdit(profile: ReferenceResponse) {
    editingRef = profile;
    editName = profile.display_name;
    editCategory = profile.category;
  }

  async function saveEdit() {
    if (!editingRef) return;
    saving = true;
    try {
      await updateReference(editingRef.name, {
        display_name: editName,
        category: editCategory
      });
      toast.success('Reference updated');
      editingRef = null;
      await loadData();
    } catch {
      toast.error('Failed to save changes');
    } finally {
      saving = false;
    }
  }

  async function deleteRef() {
    if (!editingRef) return;
    if (!confirm(`Delete "${editingRef.display_name}"? This cannot be undone.`)) return;
    try {
      await deleteReference(editingRef.name);
      toast.success('Reference deleted');
      editingRef = null;
      await loadData();
    } catch {
      toast.error('Failed to delete reference');
    }
  }

  async function addReference() {
    if (!newName || !newDisplayName || !newFiles?.length) {
      toast.error('Name, display name, and at least one image are required');
      return;
    }
    adding = true;
    try {
      const formData = new FormData();
      formData.append('name', newName);
      formData.append('display_name', newDisplayName);
      formData.append('category', newCategory);
      for (const file of newFiles) {
        formData.append('images', file);
      }
      const { api } = await import('$lib/api/client');
      await api.post('/references', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000,
      });
      toast.success('Reference created — CLIP embeddings computed');
      showAdd = false;
      newName = '';
      newDisplayName = '';
      newCategory = 'vehicle';
      newFiles = null;
      await loadData();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to create reference');
    } finally {
      adding = false;
    }
  }

  onMount(loadData);
</script>

<Header title="Vehicles" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Top bar -->
  <div class="flex items-center justify-between mb-6">
    <p class="text-sm" style="color: var(--color-text-secondary);">
      {vehicles.length} vehicle reference{vehicles.length !== 1 ? 's' : ''}
    </p>
    <button
      onclick={() => (showAdd = true)}
      class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
      style="background: var(--color-accent); color: white;"
    >
      <Plus size={16} />
      Add Reference
    </button>
  </div>

  {#if loading}
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {#each Array(6) as _}
        <div class="rounded-xl h-48 animate-pulse" style="background: var(--color-card);"></div>
      {/each}
    </div>
  {:else if vehicles.length === 0}
    <EmptyState
      icon={Car}
      title="No vehicle references"
      description="Add vehicle references from photos to enable automatic identification."
    />
  {:else}
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {#each vehicles as v (v.uuid)}
        <ProfileCard profile={v} type="vehicle" onedit={() => openEdit(v)} />
      {/each}
    </div>
  {/if}
</div>

<!-- Edit Modal -->
<Modal open={!!editingRef} title="Edit Vehicle Reference" onclose={() => (editingRef = null)}>
  {#if editingRef}
    <div class="space-y-4">
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Key</label>
        <p class="text-sm font-mono px-3 py-2 rounded-lg"
          style="background: var(--color-surface); color: var(--color-text-muted);">
          {editingRef.name}
        </p>
      </div>
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Display Name</label>
        <input type="text" bind:value={editName}
          class="w-full px-3 py-2 rounded-lg text-sm"
          style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
        />
      </div>
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Category</label>
        <select bind:value={editCategory}
          class="w-full px-3 py-2 rounded-lg text-sm"
          style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
        >
          <option value="vehicle">Vehicle</option>
          <option value="other">Other</option>
        </select>
      </div>
      <div class="text-xs space-y-1" style="color: var(--color-text-muted);">
        <p>{editingRef.visit_count} visits recorded</p>
        {#if editingRef.last_seen}
          <p>Last seen: {new Date(editingRef.last_seen).toLocaleString()}</p>
        {/if}
      </div>
      <div class="flex gap-2">
        <button onclick={saveEdit} disabled={saving}
          class="flex-1 py-2 rounded-lg text-sm font-medium"
          style="background: var(--color-accent); color: white;"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button onclick={deleteRef}
          class="py-2 px-4 rounded-lg text-sm font-medium"
          style="background: rgba(239,68,68,0.15); color: var(--color-red);"
        >
          Delete
        </button>
      </div>
    </div>
  {/if}
</Modal>

<!-- Add Modal -->
<Modal open={showAdd} title="Add Vehicle Reference" onclose={() => (showAdd = false)}>
  <div class="space-y-4">
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Key (snake_case)</label>
      <input type="text" bind:value={newName} placeholder="e.g. cleaners_car"
        class="w-full px-3 py-2 rounded-lg text-sm"
        style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      />
    </div>
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Display Name</label>
      <input type="text" bind:value={newDisplayName} placeholder="e.g. Cleaner"
        class="w-full px-3 py-2 rounded-lg text-sm"
        style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      />
    </div>
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Category</label>
      <select bind:value={newCategory}
        class="w-full px-3 py-2 rounded-lg text-sm"
        style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      >
        <option value="vehicle">Vehicle</option>
        <option value="other">Other</option>
      </select>
    </div>
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">
        Reference Images
      </label>
      <label
        class="flex flex-col items-center justify-center w-full h-28 rounded-lg cursor-pointer transition-colors"
        style="border: 2px dashed var(--color-border); background: var(--color-surface);"
      >
        <Upload size={24} style="color: var(--color-text-muted);" />
        <span class="text-xs mt-2" style="color: var(--color-text-muted);">
          {newFiles?.length ? `${newFiles.length} file(s) selected` : 'Click or drag images here'}
        </span>
        <input type="file" accept="image/*" multiple class="hidden"
          onchange={(e) => (newFiles = (e.target as HTMLInputElement).files)}
        />
      </label>
    </div>
    <button onclick={addReference} disabled={adding}
      class="w-full py-2.5 rounded-lg text-sm font-medium"
      style="background: var(--color-accent); color: white;"
    >
      {adding ? 'Computing embeddings...' : 'Create Reference'}
    </button>
  </div>
</Modal>
