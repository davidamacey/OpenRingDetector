<script lang="ts">
  import Header from '$lib/components/layout/Header.svelte';
  import SystemStatusPanel from '$lib/components/status/SystemStatusPanel.svelte';
  import { getSettings, patchSettings, testNotify } from '$lib/api/settings';
  import { statusStore } from '$lib/stores/status';
  import { toast } from '$lib/stores/toast';
  import { Bell } from 'lucide-svelte';
  import { onMount } from 'svelte';

  let settings = $state<Record<string, Record<string, unknown>> | null>(null);
  let loading = $state(true);
  let saving = $state(false);
  let testing = $state(false);
  let tab = $state<'detection' | 'video' | 'face' | 'notifications' | 'storage' | 'system'>('detection');

  const status = $derived($statusStore);

  async function loadSettings() {
    loading = true;
    try {
      settings = await getSettings();
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
      await patchSettings(settings);
      toast.success('Settings saved - restart ring-watch to apply changes');
    } catch {
      toast.error('Failed to save settings');
    } finally {
      saving = false;
    }
  }

  async function sendTestNotification() {
    testing = true;
    try {
      const res = await testNotify();
      if (res.success) {
        toast.success('Test notification sent');
      } else {
        toast.error(`Failed: ${res.detail}`);
      }
    } catch {
      toast.error('Failed to send test notification');
    } finally {
      testing = false;
    }
  }

  onMount(loadSettings);
</script>

<Header title="Settings" />

<div class="flex-1 overflow-y-auto p-4">
  <!-- Tabs -->
  <div class="flex flex-wrap gap-1 mb-6 p-1 rounded-lg w-fit" style="background: var(--color-card);">
    {#each [
      ['detection', 'Detection'],
      ['video', 'Video'],
      ['face', 'Face'],
      ['notifications', 'Notifications'],
      ['storage', 'Storage'],
      ['system', 'System']
    ] as [key, label]}
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
              ['model', 'batch_size', 'Batch Size', 'number'],
            ] as [section, key, label, inputType]}
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">{label}</label>
                <input
                  type={inputType}
                  bind:value={settings[section][key]}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                />
              </div>
            {/each}
          </div>
          <div class="flex items-center gap-2">
            <input type="checkbox" id="captioner" bind:checked={settings.captioner.enabled as boolean} />
            <label for="captioner" class="text-sm" style="color: var(--color-text-secondary);">
              Enable Captioner (Gemma 3 via Ollama)
            </label>
          </div>
          {#if settings.captioner.enabled}
            <div class="grid gap-4">
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Ollama URL</label>
                <input type="text" bind:value={settings.captioner.ollama_url}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                />
              </div>
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Model</label>
                <input type="text" bind:value={settings.captioner.model}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                />
              </div>
            </div>
          {/if}
        </div>

      {:else if tab === 'video'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Video Analysis</h3>
          <p class="text-xs" style="color: var(--color-text-muted);">
            When enabled, the watcher downloads and analyzes the full Ring video recording
            instead of relying solely on a single snapshot.
          </p>
          <div class="flex items-center gap-2">
            <input type="checkbox" id="video_enabled" bind:checked={settings.video.enabled as boolean} />
            <label for="video_enabled" class="text-sm" style="color: var(--color-text-secondary);">
              Enable Video Analysis
            </label>
          </div>
          {#if settings.video.enabled}
            <div class="grid gap-4">
              {#each [
                ['video', 'frame_interval', 'Frame Interval (extract every Nth frame)', 'number'],
                ['video', 'max_frames', 'Max Frames per Video', 'number'],
                ['video', 'wait_timeout', 'Wait Timeout (seconds)', 'number'],
              ] as [section, key, label, inputType]}
                <div>
                  <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">{label}</label>
                  <input
                    type={inputType}
                    bind:value={settings[section][key]}
                    class="w-full px-3 py-2 rounded-lg text-sm"
                    style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                  />
                </div>
              {/each}
            </div>
          {/if}
        </div>

      {:else if tab === 'face'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Face Detection & Recognition</h3>
          <p class="text-xs" style="color: var(--color-text-muted);">
            Uses SCRFD-10G for face detection and ArcFace w600k_r50 for 512-dim face embeddings.
          </p>
          <div class="flex items-center gap-2">
            <input type="checkbox" id="face_enabled" bind:checked={settings.face.enabled as boolean} />
            <label for="face_enabled" class="text-sm" style="color: var(--color-text-secondary);">
              Enable Face Detection
            </label>
          </div>
          {#if settings.face.enabled}
            <div class="grid gap-4">
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">
                  Match Threshold (0.0 - 1.0, higher = stricter)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  bind:value={settings.face.match_threshold}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                />
              </div>
              <div>
                <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">
                  Min Face Size (pixels)
                </label>
                <input
                  type="number"
                  min="10"
                  bind:value={settings.face.min_face_size}
                  class="w-full px-3 py-2 rounded-lg text-sm"
                  style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
                />
              </div>
            </div>
          {/if}
        </div>

      {:else if tab === 'notifications'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Notification Settings</h3>
          <div>
            <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">ntfy URL</label>
            <input type="text" bind:value={settings.notify.ntfy_url}
              class="w-full px-3 py-2 rounded-lg text-sm"
              style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
            />
          </div>
          <div class="flex items-center gap-2">
            <input type="checkbox" id="attach_snap" bind:checked={settings.notify.attach_snapshot as boolean} />
            <label for="attach_snap" class="text-sm" style="color: var(--color-text-secondary);">
              Attach snapshot to notifications
            </label>
          </div>
          <div class="pt-2 border-t" style="border-color: var(--color-border);">
            <button
              onclick={sendTestNotification}
              disabled={testing}
              class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
              style="background: var(--color-teal); color: white;"
            >
              <Bell size={14} />
              {testing ? 'Sending...' : 'Send Test Notification'}
            </button>
          </div>
        </div>

      {:else if tab === 'storage'}
        <div class="rounded-xl p-5 space-y-4" style="background: var(--color-card); border: 1px solid var(--color-border);">
          <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Storage Settings</h3>
          <div>
            <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Archive Directory</label>
            <input type="text" bind:value={settings.storage.archive_dir}
              class="w-full px-3 py-2 rounded-lg text-sm"
              style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
            />
          </div>
          <div>
            <label class="block text-sm mb-1" style="color: var(--color-text-secondary);">Camera Name</label>
            <input type="text" bind:value={settings.ring.camera_name}
              class="w-full px-3 py-2 rounded-lg text-sm"
              style="background: var(--color-surface); color: var(--color-text-primary); border: 1px solid var(--color-border);"
            />
          </div>
        </div>

      {:else if tab === 'system'}
        <SystemStatusPanel {status} />
        <div class="flex gap-3 mt-3">
          <button
            onclick={() => statusStore.refresh()}
            class="px-4 py-2 text-sm rounded-lg"
            style="background: var(--color-card); color: var(--color-text-secondary); border: 1px solid var(--color-border);"
          >
            Refresh Status
          </button>
        </div>
      {/if}

      {#if tab !== 'system'}
        <div class="flex items-center gap-3 pt-2">
          <button
            onclick={saveSettings}
            disabled={saving}
            class="px-5 py-2.5 rounded-lg text-sm font-medium"
            style="background: var(--color-accent); color: white;"
          >
            {saving ? 'Saving...' : 'Save Changes'}
          </button>
          <p class="text-xs" style="color: var(--color-text-muted);">
            Restart ring-watch to apply most changes.
          </p>
        </div>
      {/if}
    </div>
  {/if}
</div>
