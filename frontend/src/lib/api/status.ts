import type { SystemStatus } from '$lib/types';
import { api } from './client';

export async function getStatus(): Promise<SystemStatus> {
  const res = await api.get('/status');
  return res.data;
}

export async function getCameras(): Promise<{ name: string; type: string }[]> {
  const res = await api.get('/cameras');
  return res.data;
}
