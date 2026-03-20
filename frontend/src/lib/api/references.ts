import type { ReferenceResponse } from '$lib/types';
import { api } from './client';

export async function getReferences(): Promise<ReferenceResponse[]> {
  const res = await api.get('/references');
  return res.data;
}

export async function updateReference(name: string, body: { display_name?: string; category?: string }): Promise<ReferenceResponse> {
  const res = await api.put(`/references/${name}`, body);
  return res.data;
}

export async function deleteReference(name: string): Promise<void> {
  await api.delete(`/references/${name}`);
}
