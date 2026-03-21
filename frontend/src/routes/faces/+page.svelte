<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import ProfileCard from '$lib/components/profiles/ProfileCard.svelte';
  import EmptyState from '$lib/components/ui/EmptyState.svelte';
  import Modal from '$lib/components/ui/Modal.svelte';
  import { deleteFace, getFaces, updateFace } from '$lib/api/faces';
  import { toast } from '$lib/stores/toast';
  import type { FaceProfileResponse } from '$lib/types';
  import { Plus, Upload, Users } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let faces = $state<FaceProfileResponse[]>([]);
  let loading = $state(true);
  let editingFace = $state<FaceProfileResponse | null>(null);
  let editName = $state('');
  let saving = $state(false);

  // Add new face
  let showAdd = $state(false);
  let newName = $state('');
  let newDisplayName = $state('');
  let newFile = $state<File | null>(null);
  let adding = $state(false);
  let previewUrl = $state<string | null>(null);

  async function loadData() {
    loading = true;
    try {
      faces = await getFaces();
    } catch {
      toast.error('Failed to load face profiles');
    } finally {
      loading = false;
    }
  }

  function openEdit(profile: FaceProfileResponse) {
    editingFace = profile;
    editName = profile.display_name;
  }

  async function saveEdit() {
    if (!editingFace) return;
    saving = true;
    try {
      await updateFace(editingFace.name, { display_name: editName });
      toast.success('Face profile updated');
      editingFace = null;
      await loadData();
    } catch {
      toast.error('Failed to save changes');
    } finally {
      saving = false;
    }
  }

  async function deleteProfile() {
    if (!editingFace) return;
    if (!confirm(`Delete "${editingFace.display_name}"? This cannot be undone.`)) return;
    try {
      await deleteFace(editingFace.name);
      toast.success('Face profile deleted');
      editingFace = null;
      await loadData();
    } catch {
      toast.error('Failed to delete profile');
    }
  }

  function onFileSelect(e: Event) {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    newFile = file;
    if (file) {
      previewUrl = URL.createObjectURL(file);
    } else {
      previewUrl = null;
    }
  }

  async function addFace() {
    if (!newName || !newDisplayName || !newFile) {
      toast.error('Name, display name, and a photo are required');
      return;
    }
    adding = true;
    try {
      const formData = new FormData();
      formData.append('name', newName);
      formData.append('display_name', newDisplayName);
      formData.append('image', newFile);
      const { api } = await import('$lib/api/client');
      await api.post('/faces', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 60000,
      });
      toast.success('Face profile created');
      showAdd = false;
      newName = '';
      newDisplayName = '';
      newFile = null;
      previewUrl = null;
      await loadData();
    } catch (e: any) {
      toast.error(e?.response?.data?.detail || 'Failed to create face profile');
    } finally {
      adding = false;
    }
  }

  onMount(loadData);
</script>

<Header title="Faces" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Top bar -->
  <div class="flex items-center justify-between mb-6">
    <p class="text-sm" style="color: var(--color-text-secondary);">
      {faces.length} face profile{faces.length !== 1 ? 's' : ''}
    </p>
    <button
      onclick={() => (showAdd = true)}
      class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
      style="background: var(--color-accent); color: white;"
    >
      <Plus size={16} />
      Add Face
    </button>
  </div>

  {#if loading}
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {#each Array(6) as _}
        <div class="rounded-xl h-48 animate-pulse" style="background: var(--color-card);"></div>
      {/each}
    </div>
  {:else if faces.length === 0}
    <EmptyState
      icon={Users}
      title="No face profiles"
      description="Add face profiles from photos to enable automatic person identification."
    />
  {:else}
    <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
      {#each faces as f (f.uuid)}
        <ProfileCard profile={f} type="face" onedit={() => openEdit(f)} />
      {/each}
    </div>
  {/if}
</div>

<!-- Edit Modal -->
<Modal open={!!editingFace} title="Edit Face Profile" onclose={() => (editingFace = null)}>
  {#if editingFace}
    <div class="space-y-4">
      {#if editingFace.sample_image_url}
        <div class="flex justify-center">
          <img
            src={editingFace.sample_image_url}
            alt={editingFace.display_name}
            class="w-24 h-24 rounded-full object-cover"
            style="border: 2px solid var(--color-border);"
          />
        </div>
      {/if}
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Key</label>
        <p class="text-sm font-mono px-3 py-2 rounded-lg"
          style="background: var(--color-surface); color: var(--color-text-muted);">
          {editingFace.name}
        </p>
      </div>
      <div>
        <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Display Name</label>
        <input type="text" bind:value={editName}
          class="w-full px-3 py-2 rounded-lg text-sm"
          style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
        />
      </div>
      <div class="text-xs space-y-1" style="color: var(--color-text-muted);">
        <p>{editingFace.visit_count} visits recorded</p>
        {#if editingFace.last_seen}
          <p>Last seen: {new Date(editingFace.last_seen).toLocaleString()}</p>
        {/if}
        <p>Created: {new Date(editingFace.created_at).toLocaleDateString()}</p>
      </div>
      <div class="flex gap-2">
        <button onclick={saveEdit} disabled={saving}
          class="flex-1 py-2 rounded-lg text-sm font-medium"
          style="background: var(--color-accent); color: white;"
        >
          {saving ? 'Saving...' : 'Save'}
        </button>
        <button onclick={deleteProfile}
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
<Modal open={showAdd} title="Add Face Profile" onclose={() => { showAdd = false; previewUrl = null; }}>
  <div class="space-y-4">
    {#if previewUrl}
      <div class="flex justify-center">
        <img src={previewUrl} alt="preview"
          class="w-28 h-28 rounded-full object-cover"
          style="border: 2px solid var(--color-accent);"
        />
      </div>
    {/if}
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Key (snake_case)</label>
      <input type="text" bind:value={newName} placeholder="e.g. david"
        class="w-full px-3 py-2 rounded-lg text-sm"
        style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      />
    </div>
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Display Name</label>
      <input type="text" bind:value={newDisplayName} placeholder="e.g. David"
        class="w-full px-3 py-2 rounded-lg text-sm"
        style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
      />
    </div>
    <div>
      <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Photo</label>
      <label
        class="flex flex-col items-center justify-center w-full h-28 rounded-lg cursor-pointer transition-colors"
        style="border: 2px dashed var(--color-border); background: var(--color-surface);"
      >
        <Upload size={24} style="color: var(--color-text-muted);" />
        <span class="text-xs mt-2" style="color: var(--color-text-muted);">
          {newFile ? newFile.name : 'Upload a clear face photo'}
        </span>
        <input type="file" accept="image/*" class="hidden" onchange={onFileSelect} />
      </label>
    </div>
    <p class="text-xs" style="color: var(--color-text-muted);">
      The photo will be processed to detect and extract the face automatically using SCRFD + ArcFace.
    </p>
    <button onclick={addFace} disabled={adding}
      class="w-full py-2.5 rounded-lg text-sm font-medium"
      style="background: var(--color-accent); color: white;"
    >
      {adding ? 'Detecting face...' : 'Create Profile'}
    </button>
  </div>
</Modal>
