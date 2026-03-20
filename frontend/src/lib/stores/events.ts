import { getEvents } from '$lib/api/events';
import type { EventResponse } from '$lib/types';
import { writable } from 'svelte/store';

const MAX_LIVE = 200;

const { subscribe, update, set } = writable<EventResponse[]>([]);

async function load(params: Parameters<typeof getEvents>[0] = {}) {
  const res = await getEvents({ limit: 50, ...params });
  set(res.items);
  return res;
}

function prepend(event: EventResponse) {
  update((items) => {
    const next = [event, ...items];
    return next.length > MAX_LIVE ? next.slice(0, MAX_LIVE) : next;
  });
}

async function loadMore(params: Parameters<typeof getEvents>[0] = {}, currentLength = 0) {
  const res = await getEvents({ limit: 50, offset: currentLength, ...params });
  update((items) => [...items, ...res.items]);
  return res;
}

export const eventsStore = { subscribe, load, prepend, loadMore };
