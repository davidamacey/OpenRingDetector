import type { EventListResponse, EventResponse } from '$lib/types';
import { api } from './client';

export async function getEvents(params: {
  camera?: string;
  type?: string;
  limit?: number;
  offset?: number;
  from?: string;
  to?: string;
} = {}): Promise<EventListResponse> {
  const res = await api.get('/events', { params });
  return res.data;
}

export async function getEvent(id: number): Promise<EventResponse> {
  const res = await api.get(`/events/${id}`);
  return res.data;
}
