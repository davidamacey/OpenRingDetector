import { writable } from 'svelte/store';

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
}

const { subscribe, update } = writable<Toast[]>([]);

function addToast(type: Toast['type'], message: string) {
  const id = crypto.randomUUID();
  update((toasts) => [...toasts, { id, type, message }]);
  setTimeout(() => removeToast(id), 4000);
}

function removeToast(id: string) {
  update((toasts) => toasts.filter((t) => t.id !== id));
}

export const toasts = { subscribe };
export const toast = {
  success: (msg: string) => addToast('success', msg),
  error: (msg: string) => addToast('error', msg),
  info: (msg: string) => addToast('info', msg),
  remove: removeToast
};
