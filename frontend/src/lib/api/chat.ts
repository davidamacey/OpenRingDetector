import type {
  ChatEventDetail,
  ChatFaceCard,
  ChatImage,
  ChatMessage,
  ChatReferenceCard,
  ChatStatusResponse
} from '$lib/types';

export interface ChatCallbacks {
  onToken: (text: string) => void;
  onImage: (image: ChatImage) => void;
  onImages: (images: ChatImage[]) => void;
  onEventDetail: (detail: ChatEventDetail) => void;
  onReferenceCard: (card: ChatReferenceCard) => void;
  onFaceCard: (card: ChatFaceCard) => void;
  onDone: () => void;
  onError: (message: string, fallback?: boolean, structuredData?: unknown) => void;
}

interface HistoryEntry {
  role: 'user' | 'assistant';
  content: string;
}

function buildHistory(messages: ChatMessage[]): HistoryEntry[] {
  return messages
    .filter((m) => m.content.length > 0)
    .map((m) => ({ role: m.role, content: m.content }));
}

/**
 * Stream a chat response via SSE (POST with ReadableStream).
 * Returns an AbortController so the caller can cancel.
 */
export function streamChat(
  message: string,
  history: ChatMessage[],
  callbacks: ChatCallbacks
): AbortController {
  const controller = new AbortController();

  (async () => {
    let response: Response;
    try {
      response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          history: buildHistory(history)
        }),
        signal: controller.signal
      });
    } catch (err) {
      if (!controller.signal.aborted) {
        callbacks.onError(err instanceof Error ? err.message : 'Network error');
      }
      return;
    }

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        detail = body.detail ?? detail;
      } catch {
        // ignore parse errors
      }
      callbacks.onError(detail);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      callbacks.onError('Response body is not readable');
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE blocks (terminated by double newline)
        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          parseSSEBlock(block, callbacks);
          boundary = buffer.indexOf('\n\n');
        }
      }

      // Process any remaining partial block
      if (buffer.trim().length > 0) {
        parseSSEBlock(buffer.trim(), callbacks);
      }
    } catch (err) {
      if (!controller.signal.aborted) {
        callbacks.onError(err instanceof Error ? err.message : 'Stream read error');
      }
    }
  })();

  return controller;
}

function parseSSEBlock(block: string, callbacks: ChatCallbacks): void {
  let eventType = '';
  let dataLines: string[] = [];

  for (const line of block.split('\n')) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice(6));
    } else if (line.startsWith('data:')) {
      // "data:" with no space — value is rest of line
      dataLines.push(line.slice(5));
    }
  }

  if (!eventType && dataLines.length === 0) return;

  const rawData = dataLines.join('\n');

  switch (eventType) {
    case 'token':
      try {
        const parsed = JSON.parse(rawData);
        callbacks.onToken(parsed.text ?? '');
      } catch {
        callbacks.onToken(rawData);
      }
      break;

    case 'image':
      try {
        callbacks.onImage(JSON.parse(rawData) as ChatImage);
      } catch {
        // ignore malformed image data
      }
      break;

    case 'images':
      try {
        const parsed = JSON.parse(rawData);
        const items = parsed.items ?? parsed;
        callbacks.onImages(Array.isArray(items) ? items : []);
      } catch {
        // ignore malformed images data
      }
      break;

    case 'event_detail':
      try {
        callbacks.onEventDetail(JSON.parse(rawData) as ChatEventDetail);
      } catch {
        // ignore malformed event detail
      }
      break;

    case 'reference_card':
      try {
        callbacks.onReferenceCard(JSON.parse(rawData) as ChatReferenceCard);
      } catch {
        // ignore malformed reference card
      }
      break;

    case 'face_card':
      try {
        callbacks.onFaceCard(JSON.parse(rawData) as ChatFaceCard);
      } catch {
        // ignore malformed face card
      }
      break;

    case 'done':
      callbacks.onDone();
      break;

    case 'error':
      try {
        const parsed = JSON.parse(rawData);
        callbacks.onError(
          parsed.message ?? rawData,
          parsed.fallback ?? false,
          parsed.structured_data
        );
      } catch {
        callbacks.onError(rawData);
      }
      break;

    default:
      // Unknown event type — ignore
      break;
  }
}

/**
 * Check whether the chat backend (Ollama model) is available.
 */
export async function getChatStatus(): Promise<ChatStatusResponse> {
  const res = await fetch('/api/chat/status');
  if (!res.ok) {
    throw new Error(`Chat status check failed: HTTP ${res.status}`);
  }
  return res.json();
}

/**
 * Warm up the chat model so first response is faster.
 * Fire-and-forget — errors are silently ignored.
 */
export async function warmupChat(): Promise<void> {
  try {
    await fetch('/api/chat/warmup', { method: 'POST' });
  } catch {
    // fire and forget
  }
}
