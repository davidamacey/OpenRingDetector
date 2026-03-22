import { streamChat } from '$lib/api/chat';
import type {
  ChatEventDetail,
  ChatFaceCard,
  ChatImage,
  ChatMessage,
  ChatReferenceCard
} from '$lib/types';
import { writable } from 'svelte/store';

interface ChatState {
  messages: ChatMessage[];
  isStreaming: boolean;
  isOpen: boolean;
}

const initial: ChatState = {
  messages: [],
  isStreaming: false,
  isOpen: false
};

const { subscribe, update, set } = writable<ChatState>(initial);

let abortController: AbortController | null = null;

function togglePanel() {
  update((s) => ({ ...s, isOpen: !s.isOpen }));
}

function open() {
  update((s) => ({ ...s, isOpen: true }));
}

function close() {
  update((s) => ({ ...s, isOpen: false }));
}

function clearHistory() {
  cancelStream();
  update((s) => ({ ...s, messages: [], isStreaming: false }));
}

function setStreaming(v: boolean) {
  update((s) => ({ ...s, isStreaming: v }));
}

function cancelStream() {
  if (abortController) {
    abortController.abort();
    abortController = null;
  }
}

function updateAssistantMessage(
  state: ChatState,
  assistantId: string,
  updater: (msg: ChatMessage) => ChatMessage
): ChatState {
  return {
    ...state,
    messages: state.messages.map((m) => (m.id === assistantId ? updater(m) : m))
  };
}

function sendMessage(text: string) {
  const trimmed = text.trim();
  if (!trimmed) return;

  cancelStream();

  const userMsg: ChatMessage = {
    id: crypto.randomUUID(),
    role: 'user',
    content: trimmed,
    timestamp: new Date()
  };

  const assistantId = crypto.randomUUID();
  const assistantMsg: ChatMessage = {
    id: assistantId,
    role: 'assistant',
    content: '',
    timestamp: new Date(),
    images: [],
    eventDetails: [],
    referenceCards: [],
    faceCards: [],
    isStreaming: true
  };

  update((s) => ({
    ...s,
    messages: [...s.messages, userMsg, assistantMsg],
    isStreaming: true
  }));

  // Build history from messages before this exchange
  let currentMessages: ChatMessage[] = [];
  update((s) => {
    currentMessages = s.messages.slice(0, -2); // exclude the two we just added
    return s;
  });

  abortController = streamChat(trimmed, currentMessages, {
    onToken(token: string) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          content: m.content + token
        }))
      );
    },

    onImage(image: ChatImage) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          images: [...(m.images ?? []), image]
        }))
      );
    },

    onImages(images: ChatImage[]) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          images: [...(m.images ?? []), ...images]
        }))
      );
    },

    onEventDetail(detail: ChatEventDetail) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          eventDetails: [...(m.eventDetails ?? []), detail]
        }))
      );
    },

    onReferenceCard(card: ChatReferenceCard) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          referenceCards: [...(m.referenceCards ?? []), card]
        }))
      );
    },

    onFaceCard(card: ChatFaceCard) {
      update((s) =>
        updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          faceCards: [...(m.faceCards ?? []), card]
        }))
      );
    },

    onDone() {
      update((s) => ({
        ...updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          isStreaming: false
        })),
        isStreaming: false
      }));
      abortController = null;
    },

    onError(message: string) {
      update((s) => ({
        ...updateAssistantMessage(s, assistantId, (m) => ({
          ...m,
          isStreaming: false,
          error: message
        })),
        isStreaming: false
      }));
      abortController = null;
    }
  });
}

export const chatStore = {
  subscribe,
  togglePanel,
  open,
  close,
  sendMessage,
  clearHistory,
  setStreaming,
  cancelStream
};
