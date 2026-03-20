import type { FaceProfileResponse } from '$lib/types';
import { api } from './client';

export async function getFaces(): Promise<FaceProfileResponse[]> {
  const res = await api.get('/faces');
  return res.data;
}

export async function updateFace(name: string, body: { display_name?: string }): Promise<FaceProfileResponse> {
  const res = await api.put(`/faces/${name}`, body);
  return res.data;
}

export async function deleteFace(name: string): Promise<void> {
  await api.delete(`/faces/${name}`);
}
