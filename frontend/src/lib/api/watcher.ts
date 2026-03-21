import { api } from './client';

export interface WatcherStatus {
  state: 'running' | 'stopped';
  pid: number | null;
  uptime_seconds: number;
  exit_code: number | null;
}

export interface WatcherLogs {
  lines: string[];
  total: number;
}

export async function getWatcherStatus(): Promise<WatcherStatus> {
  const res = await api.get('/watcher/status');
  return res.data;
}

export async function startWatcher(): Promise<{ success: boolean; detail: string }> {
  const res = await api.post('/watcher/start');
  return res.data;
}

export async function stopWatcher(): Promise<{ success: boolean; detail: string }> {
  const res = await api.post('/watcher/stop');
  return res.data;
}

export async function getWatcherLogs(tail = 200): Promise<WatcherLogs> {
  const res = await api.get('/watcher/logs', { params: { tail } });
  return res.data;
}
