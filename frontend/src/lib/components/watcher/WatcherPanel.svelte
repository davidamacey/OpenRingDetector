<script lang="ts">
  import StatusDot from '$lib/components/ui/StatusDot.svelte';
  import { getWatcherStatus, getWatcherLogs, startWatcher, stopWatcher } from '$lib/api/watcher';
  import type { WatcherStatus } from '$lib/api/watcher';
  import { toast } from '$lib/stores/toast';
  import { Play, Square, RefreshCw, Terminal } from 'lucide-svelte';
  import { onMount, onDestroy } from 'svelte';

  interface Props {
    compact?: boolean;
  }
  let { compact = false }: Props = $props();

  let status = $state<WatcherStatus | null>(null);
  let logs = $state<string[]>([]);
  let showLogs = $state(false);
  let starting = $state(false);
  let stopping = $state(false);
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let logPollTimer: ReturnType<typeof setInterval> | null = null;
  let logContainer: HTMLDivElement | undefined = undefined;

  const isRunning = $derived(status?.state === 'running');
  const uptime = $derived(() => {
    if (!status?.uptime_seconds) return '';
    const s = status.uptime_seconds;
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  });

  async function refresh() {
    try {
      status = await getWatcherStatus();
    } catch {
      // ignore
    }
  }

  async function refreshLogs() {
    if (!showLogs) return;
    try {
      const res = await getWatcherLogs(500);
      logs = res.lines;
      // Auto-scroll to bottom
      if (logContainer) {
        requestAnimationFrame(() => {
          if (logContainer) logContainer.scrollTop = logContainer.scrollHeight;
        });
      }
    } catch {
      // ignore
    }
  }

  async function handleStart() {
    starting = true;
    try {
      const res = await startWatcher();
      if (res.success) {
        toast.success('Watcher started');
      } else {
        toast.error(res.detail);
      }
      await refresh();
    } catch {
      toast.error('Failed to start watcher');
    } finally {
      starting = false;
    }
  }

  async function handleStop() {
    if (!confirm('Stop the Ring watcher? You will not receive motion alerts until it is restarted.')) return;
    stopping = true;
    try {
      const res = await stopWatcher();
      if (res.success) {
        toast.success('Watcher stopped');
      } else {
        toast.error(res.detail);
      }
      await refresh();
    } catch {
      toast.error('Failed to stop watcher');
    } finally {
      stopping = false;
    }
  }

  function toggleLogs() {
    showLogs = !showLogs;
    if (showLogs) {
      refreshLogs();
      logPollTimer = setInterval(refreshLogs, 3000);
    } else if (logPollTimer) {
      clearInterval(logPollTimer);
      logPollTimer = null;
    }
  }

  onMount(() => {
    refresh();
    pollTimer = setInterval(refresh, 5000);
  });

  onDestroy(() => {
    if (pollTimer) clearInterval(pollTimer);
    if (logPollTimer) clearInterval(logPollTimer);
  });
</script>

{#if compact}
  <!-- Compact mode for Live page header -->
  <div class="flex items-center gap-2">
    <StatusDot status={isRunning ? 'ok' : 'fail'} pulse={isRunning} />
    <span class="text-sm" style="color: var(--color-text-secondary);">
      {isRunning ? 'Watching' : 'Stopped'}
    </span>
    {#if isRunning}
      <span class="text-xs" style="color: var(--color-text-muted);">{uptime()}</span>
      <button
        onclick={handleStop}
        disabled={stopping}
        class="ml-1 p-1 rounded transition-colors"
        style="color: var(--color-red);"
        title="Stop watcher"
      >
        <Square size={14} />
      </button>
    {:else}
      <button
        onclick={handleStart}
        disabled={starting}
        class="ml-1 p-1 rounded transition-colors"
        style="color: var(--color-green);"
        title="Start watcher"
      >
        <Play size={14} />
      </button>
    {/if}
  </div>
{:else}
  <!-- Full panel for Settings page -->
  <div class="rounded-xl overflow-hidden" style="background: var(--color-card); border: 1px solid var(--color-border);">
    <div class="p-5">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-sm font-semibold" style="color: var(--color-text-primary);">Ring Watcher</h3>
        <div class="flex items-center gap-2">
          <StatusDot status={isRunning ? 'ok' : 'fail'} pulse={isRunning} size={10} />
          <span class="text-sm font-medium" style="color: {isRunning ? 'var(--color-green)' : 'var(--color-text-muted)'};">
            {isRunning ? 'Running' : 'Stopped'}
          </span>
        </div>
      </div>

      {#if isRunning}
        <div class="grid grid-cols-3 gap-3 mb-4">
          <div class="rounded-lg p-3" style="background: var(--color-surface);">
            <p class="text-xs" style="color: var(--color-text-muted);">PID</p>
            <p class="text-sm font-mono" style="color: var(--color-text-primary);">{status?.pid}</p>
          </div>
          <div class="rounded-lg p-3" style="background: var(--color-surface);">
            <p class="text-xs" style="color: var(--color-text-muted);">Uptime</p>
            <p class="text-sm font-mono" style="color: var(--color-text-primary);">{uptime()}</p>
          </div>
          <div class="rounded-lg p-3" style="background: var(--color-surface);">
            <p class="text-xs" style="color: var(--color-text-muted);">Status</p>
            <p class="text-sm" style="color: var(--color-green);">Listening</p>
          </div>
        </div>
      {:else if status?.exit_code !== null && status?.exit_code !== undefined}
        <div class="rounded-lg p-3 mb-4" style="background: var(--color-surface);">
          <p class="text-xs" style="color: var(--color-text-muted);">
            Last exit code: <span class="font-mono" style="color: {status.exit_code === 0 ? 'var(--color-green)' : 'var(--color-red)'};">{status.exit_code}</span>
          </p>
        </div>
      {/if}

      <p class="text-xs mb-4" style="color: var(--color-text-muted);">
        {#if isRunning}
          Listening for Ring motion events via Firebase push. Detected objects are identified, visitors tracked, and notifications sent automatically.
        {:else}
          Start the watcher to begin monitoring your Ring cameras. It will load AI models, connect to Ring, and listen for motion events.
        {/if}
      </p>

      <div class="flex gap-2">
        {#if isRunning}
          <button
            onclick={handleStop}
            disabled={stopping}
            class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            style="background: rgba(239,68,68,0.15); color: var(--color-red);"
          >
            <Square size={14} />
            {stopping ? 'Stopping...' : 'Stop Watcher'}
          </button>
        {:else}
          <button
            onclick={handleStart}
            disabled={starting}
            class="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
            style="background: var(--color-green); color: white;"
          >
            <Play size={14} />
            {starting ? 'Starting...' : 'Start Watcher'}
          </button>
        {/if}
        <button
          onclick={() => refresh()}
          class="p-2 rounded-lg transition-colors"
          style="background: var(--color-surface); color: var(--color-text-muted);"
          title="Refresh status"
        >
          <RefreshCw size={14} />
        </button>
        <button
          onclick={toggleLogs}
          class="flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition-colors"
          style="
            background: {showLogs ? 'var(--color-accent)' : 'var(--color-surface)'};
            color: {showLogs ? 'white' : 'var(--color-text-muted)'};
          "
        >
          <Terminal size={14} />
          Logs
        </button>
      </div>
    </div>

    <!-- Log viewer -->
    {#if showLogs}
      <div
        bind:this={logContainer}
        class="border-t overflow-y-auto font-mono text-xs p-4 space-y-0.5"
        style="
          border-color: var(--color-border);
          background: var(--color-bg);
          max-height: 400px;
          color: var(--color-text-muted);
        "
      >
        {#if logs.length === 0}
          <p style="color: var(--color-text-muted);">No logs yet. Start the watcher to see output.</p>
        {:else}
          {#each logs as line}
            <div
              class="whitespace-pre-wrap break-all leading-relaxed"
              style="color: {line.includes('ERROR') ? 'var(--color-red)' : line.includes('WARNING') ? 'var(--color-amber)' : line.includes('[manager]') ? 'var(--color-accent)' : 'var(--color-text-secondary)'};"
            >{line}</div>
          {/each}
        {/if}
      </div>
    {/if}
  </div>
{/if}
