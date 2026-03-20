<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import SystemStatusPanel from '$lib/components/status/SystemStatusPanel.svelte';
  import { api } from '$lib/api/client';
  import { statusStore } from '$lib/stores/status';
  import { toast } from '$lib/stores/toast';
  import { onMount } from 'svelte';

  let settings = $state<Record<string, Record<string, unknown>> | null>(null);
  let loading = $state(true);
  let saving = $state(false);
  let tab = $state<'detection' | 'notifications' | 'storage' | 'system'>('detection');

  const status = $derived($statusStore);

  async function loadSettings() {
    loading = true;
    try {
      const res = await api.get('/settings');
      settings = res.data;
    } catch {
      toast.error('Failed to load settings');
    } finally {
      loading = false;
    }
  }

  async function saveSettings() {
    if (!settings) return;
    saving = true;
    try {
      await api.patch('/settings', settings);
      toast.success('Settings saved — restart ring-watch to apply changes');
    } catch {
      toast.error('Failed to save settings');
    } finally {
      saving = false;
    }
  }

  onMount(loadSettings);
</script>

<Header title="Settings" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Tabs -->
  <div class="flex gap-1 mb-6 p-1 rounded-lg w-fit" style="background: var(--color-card);">
    {#each [['detection', 'Detection'], ['notifications', 'Notifications'], ['storage', 'Storage'], ['system', 'System']] as [key, label]}
      <button
        onclick={() => (tab = key as typeof tab)}
        class="px-4 py-2 rounded-md text-sm font-medium transition-colors"
        style="
          background: {tab === key ? 'var(--color-accent)' : 'transparent'};
          color: {tab === key ? 'white' : 'var(--color-text-secondary)'};
        "
      >
        {label}
      </button>
    {/each}
  </div>

  {#if loading || !settings}
    <div class="h-64 rounded-xl animate-pulse" style="background: var(--color-card);"></div>
  {:else}
    <div class="max-w-2xl space-y-4">
      {#if tab === 'detection'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Detection Settings</h3>

          <div class="grid gap-4">
            {#each [
              ['ring', 'cooldown_seconds', 'Motion Cooldown (seconds)', 'number'],
              ['ring', 'departure_timeout', 'Departure Timeout (seconds)', 'number'],
              ['model', 'device', 'CUDA Device', 'text'],
              ['model', 'yolo_model_path', 'YOLO Model Path', 'text'],
            ] as [section, key, label, inputType]}
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">{label}</label>
                <input
                  type={inputType}
                  bind:value={settings[section][key]}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="
                    background: var(--color-surface);
                    color: var(--color-text-primary);
                    border: 1px solid var(--color-border);
                  "
                />
              </div>
            {/each}
          </div>

          <div class="flex items-center gap-2">
            <input
              type="checkbox"
              id="captioner"
              bind:checked={settings.captioner.enabled as boolean}
            />
            <label for="captioner" class="text-sm" style="color: var(--color-text-secondary);">
              Enable Captioner (Gemma 3 via Ollama)
            </label>
          </div>
          {#if settings.captioner.enabled}
            <div>
              <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Ollama URL</label>
              <input
                type="text"
                bind:value={settings.captioner.ollama_url}
                class="w-full px-3 py-2 rounded-lg text-sm"
                style="
                  background: var(--color-surface);
                  color: var(--color-text-primary);
                  border: 1px solid var(--color-border);
                "
              />
            </div>
          {/if}
        </div>

      {:else if tab === 'notifications'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Notification Settings</h3>
          <div>
            <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">ntfy URL</label>
            <input
              type="text"
              bind:value={settings.notify.ntfy_url}
              class="w-full px-3 py-2 rounded-lg text-sm"
              style="
                background: var(--color-surface);
                color: var(--color-text-primary);
                border: 1px solid var(--color-border);
              "
            />
          </div>
          <div class="flex items-center gap-2">
            <input
              type="checkbox"
              id="attach_snap"
              bind:checked={settings.notify.attach_snapshot as boolean}
            />
            <label for="attach_snap" class="text-sm" style="color: var(--color-text-secondary);">
              Attach snapshot to notifications
            </label>
          </div>
        </div>

      {:else if tab === 'storage'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Storage Settings</h3>
          <div>
            <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Archive Directory</label>
            <input
              type="text"
              bind:value={settings.storage.archive_dir}
              class="w-full px-3 py-2 rounded-lg text-sm"
              style="
                background: var(--color-surface);
                color: var(--color-text-primary);
                border: 1px solid var(--color-border);
              "
            />
          </div>
        </div>

      {:else if tab === 'system'}
        <SystemStatusPanel {status} />
        <button
          onclick={() => statusStore.refresh()}
          class="mt-2 px-4 py-2 text-sm rounded-lg"
          style="
            background: var(--color-card);
            color: var(--color-text-secondary);
            border: 1px solid var(--color-border);
          "
        >
          Refresh Status
        </button>
      {/if}

      {#if tab !== 'system'}
        <button
          onclick={saveSettings}
          disabled={saving}
          class="px-5 py-2.5 rounded-lg text-sm font-medium"
          style="background: var(--color-accent); color: white;"
        >
          {saving ? 'Saving...' : 'Save Changes'}
        </button>
        <p class="text-xs" style="color: var(--color-text-muted);">
          Ring Watch must be restarted to apply most changes.
        </p>
      {/if}
    </div>
  {/if}
</div>
