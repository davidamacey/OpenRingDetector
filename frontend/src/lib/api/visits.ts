import type { VisitListResponse, VisitResponse } from '$lib/types';
import { api } from './client';

export async function getVisits(params: {
  limit?: number;
  offset?: number;
  from?: string;
  to?: string;
  name?: string;
  active_only?: boolean;
} = {}): Promise<VisitListResponse> {
  const res = await api.get('/visits', { params });
  return res.data;
}

export async function getActiveVisits(): Promise<VisitResponse[]> {
  const res = await api.get('/visits/active');
  return res.data;
}

export async function getVisit(id: number): Promise<VisitResponse> {
  const res = await api.get(`/visits/${id}`);
  return res.data;
}
