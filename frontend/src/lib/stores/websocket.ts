import { writable } from 'svelte/store';

type WsStatus = 'connecting' | 'connected' | 'disconnected';

interface WsStore {
  status: WsStatus;
}

const { subscribe, set } = writable<WsStore>({ status: 'disconnected' });

type MessageHandler = (data: unknown) => void;
const handlers = new Map<string, MessageHandler[]>();
let ws: WebSocket | null = null;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let reconnectDelay = 1000;

function connect() {
  if (typeof window === 'undefined') return;
  const url = `ws://${window.location.hostname}:${window.location.port}/api/ws`;
  set({ status: 'connecting' });

  ws = new WebSocket(url);

  ws.onopen = () => {
    set({ status: 'connected' });
    reconnectDelay = 1000;
  };

  ws.onclose = () => {
    set({ status: 'disconnected' });
    scheduleReconnect();
  };

  ws.onerror = () => {
    ws?.close();
  };

  ws.onmessage = (e) => {
    try {
      const msg = JSON.parse(e.data) as { type: string; data?: unknown };
      const cbs = handlers.get(msg.type) ?? [];
      cbs.forEach((cb) => cb(msg.data));
    } catch {
      // ignore malformed messages
    }
  };
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectTimer = setTimeout(() => {
    reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    connect();
  }, reconnectDelay);
}

function on(type: string, handler: MessageHandler) {
  if (!handlers.has(type)) handlers.set(type, []);
  handlers.get(type)!.push(handler);
}

function off(type: string, handler: MessageHandler) {
  const list = handlers.get(type) ?? [];
  handlers.set(type, list.filter((h) => h !== handler));
}

export const wsStore = { subscribe, connect, on, off };
