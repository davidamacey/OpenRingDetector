import { api } from './client';

export async function getSettings(): Promise<Record<string, Record<string, unknown>>> {
  const res = await api.get('/settings');
  return res.data;
}

export async function patchSettings(body: Record<string, Record<string, unknown>>): Promise<void> {
  await api.patch('/settings', body);
}

export async function testNotify(): Promise<{ success: boolean; detail: string }> {
  const res = await api.post('/test/notify');
  return res.data;
}
