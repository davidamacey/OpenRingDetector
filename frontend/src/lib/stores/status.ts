import { getStatus } from '$lib/api/status';
import type { SystemStatus } from '$lib/types';
import { writable } from 'svelte/store';

const { subscribe, set } = writable<SystemStatus | null>(null);

let pollTimer: ReturnType<typeof setInterval> | null = null;

async function refresh() {
  try {
    const s = await getStatus();
    set(s);
  } catch {
    // ignore fetch failures
  }
}

function startPolling(intervalMs = 30000) {
  refresh();
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(refresh, intervalMs);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
}

export const statusStore = { subscribe, refresh, startPolling, stopPolling };
