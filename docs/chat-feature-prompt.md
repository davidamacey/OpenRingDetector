# Implementation Prompt: Dashboard Chat Feature

Copy everything below this line and paste it as the first message to a fresh Claude Code session in the `/mnt/nvm/repos/ring_detector` project directory.

---

Implement the chat feature described in `docs/chat-feature-plan.md`. Read that file first — it contains the complete design including architecture, intent classification, SSE format, context management, fallback behavior, and file-by-file implementation plan.

## Important Constraints

- **DO NOT start any Docker containers or services.** The system is running GPU benchmarks. Only write code and files.
- **DO NOT modify the database schema.** No Alembic migrations. Query existing tables only.
- Read the existing codebase before writing anything. Key files to understand first:
  - `ring_detector/captioner.py` — Ollama API pattern (multi-modal, error handling)
  - `ring_detector/database.py` — All table schemas and query functions
  - `backend/app/routers/events.py` — Event query patterns, `_snapshot_url()`, `_get_db()`
  - `backend/app/routers/analytics.py` — Aggregation query patterns
  - `backend/app/main.py` — Router registration pattern
  - `backend/app/schemas.py` — Pydantic model conventions
  - `frontend/src/routes/+layout.svelte` — Layout structure
  - `frontend/src/lib/components/layout/Header.svelte` — Header buttons
  - `frontend/src/lib/components/layout/Sidebar.svelte` — Navigation items
  - `frontend/src/lib/stores/` — Store patterns (events.ts, toast.ts, status.ts)
  - `frontend/src/lib/api/client.ts` — API client pattern
  - `frontend/src/lib/components/events/SnapshotModal.svelte` — Image display pattern
  - `frontend/src/lib/components/watcher/WatcherPanel.svelte` — Complex component with loading/error states
  - `frontend/nginx.conf` — Proxy config (needs `proxy_buffering off` for SSE)

## Implementation Order

### Phase 1: Backend Core (no Ollama needed to test)
1. Create `backend/app/chat_entities.py`:
   - `parse_time_range(question: str) -> tuple[datetime | None, datetime | None]` — regex-based
   - `classify_intent(question: str, known_entities: list[str]) -> ChatIntent` — keyword matching
   - `resolve_entity(query: str, session) -> Reference | FaceProfile | None` — fuzzy match via difflib
   - `is_in_scope(question: str) -> bool` — domain keyword check
   - `ChatIntent` dataclass with category, time_from, time_to, entity_filter, camera_filter, question_type

2. Create `backend/app/chat_context.py`:
   - `build_chat_context(session, intent: ChatIntent) -> str` — orchestrator
   - `get_recent_events_context(session, from_dt, to_dt, ...) -> str` — event rows as compact text
   - `get_visit_history_context(session, from_dt, to_dt, ...) -> str` — visit summaries
   - `get_statistics_context(session, from_dt, to_dt) -> str` — aggregated counts
   - `get_entity_info_context(session) -> str` — configured references + faces + cameras
   - `get_active_visits_context(session) -> str` — currently active visits
   - `build_system_prompt(context: str) -> str` — wraps context in system instructions
   - Tiered summarization: last 24h detailed, 2-7 days aggregated, 8-30 days counts only
   - Token estimation: `len(text) / 3.5`, budget ~1800-2000 tokens for context

### Phase 2: Backend API
3. Add to `backend/app/schemas.py`:
   - `ChatRequest(message: str, history: list = [], conversation_id: str | None = None)`
   - `ChatStatusResponse(available: bool, model_loaded: bool, model_name: str, ollama_reachable: bool, detail: str)`

4. Create `backend/app/routers/chat.py`:
   - `POST /api/chat` — SSE streaming endpoint:
     - Parse intent via `chat_entities.classify_intent()`
     - Build context via `chat_context.build_chat_context()`
     - If image intent: look up snapshot_path, emit `image` SSE event, optionally send to Gemma multi-modal
     - If out-of-scope: return error SSE event immediately
     - Stream Gemma response via `httpx.AsyncClient` with `stream=True` to Ollama `/api/chat`
     - Emit SSE events: `token`, `image`, `images`, `event_detail`, `reference_card`, `face_card`, `done`, `error`
     - Rate limiting: in-memory, 10 req/60s
     - Concurrency: `asyncio.Semaphore(1)`, fail fast with 503
     - Fallback: if Ollama down, return structured DB data
   - `GET /api/chat/status` — health check (hit Ollama `/api/tags` + `/api/ps`)
   - `POST /api/chat/warmup` — fire-and-forget model load
   - Use `StreamingResponse(media_type="text/event-stream")` with `X-Accel-Buffering: no` header
   - Ollama URL from `settings.captioner.ollama_url`, model from `settings.captioner.model`

5. Register in `backend/app/main.py`: `from app.routers import chat` + `app.include_router(chat.router, prefix="/api")`

6. Add `httpx>=0.27` to `backend/requirements.txt`

7. Add `chat_timeout: int = int(os.getenv("CHAT_TIMEOUT", "90"))` to `CaptionerConfig` in `ring_detector/config.py`

### Phase 3: Frontend Infrastructure
8. Add types to `frontend/src/lib/types/index.ts`:
   - `ChatMessage { id, role, content, timestamp, images?, eventDetails?, referenceCards?, faceCards?, isStreaming?, error? }`
   - `ChatImage { url, alt, eventId? }`
   - `ChatEventDetail { id, event_type, camera_name, occurred_at, detections, ... }`
   - `ChatReferenceCard { name, display_name, image_url, visit_count, last_seen }`
   - `ChatFaceCard { name, display_name, image_url, visit_count, last_seen }`

9. Create `frontend/src/lib/api/chat.ts`:
   - `streamChat(message, history, callbacks)` — native `fetch` + `ReadableStream`, NOT axios
   - Parse multi-event-type SSE: `token`, `image`, `images`, `event_detail`, `reference_card`, `face_card`, `done`, `error`
   - AbortController support for cancellation
   - `getChatStatus()` — GET `/api/chat/status`
   - `warmupChat()` — POST `/api/chat/warmup`

10. Create `frontend/src/lib/stores/chat.ts`:
    - Svelte writable store following `toast.ts` / `events.ts` pattern
    - State: `messages: ChatMessage[]`, `isStreaming: boolean`, `isOpen: boolean`
    - Actions: `addUserMessage`, `appendToAssistant`, `setError`, `clearHistory`, `togglePanel`

### Phase 4: Frontend UI
11. Create `frontend/src/lib/components/chat/ChatMessage.svelte`:
    - Svelte 5 runes (`$props`)
    - User messages: right-aligned, accent background
    - Assistant messages: left-aligned, card background
    - Streaming: blinking cursor animation
    - Error: red-tinted with retry button
    - Inline images: clickable thumbnails (reuse SnapshotModal on click)
    - Reference/face cards: small card with thumbnail + metadata
    - Event detail cards: badge + camera + time + detection details
    - All colors via CSS custom properties (`var(--color-*)`) not Tailwind color classes

12. Create `frontend/src/lib/components/chat/ChatInput.svelte`:
    - Text input with placeholder "Ask about your detection history..."
    - Send on Enter, disabled during streaming
    - Send button with ArrowUp icon from lucide-svelte

13. Create `frontend/src/lib/components/chat/ChatPanel.svelte`:
    - Slide-out from right edge, fixed position
    - Width: `w-full md:w-[400px]`, z-index 40
    - Header bar: "Chat" title + Clear + Close buttons
    - Scrollable message area with auto-scroll (pause on manual scroll-up)
    - ChatInput at bottom
    - Status banner from `/api/chat/status` on mount
    - Suggested query chips when empty: "Vehicles today", "Last visitor", "Events this week", "Anyone here now?"
    - Cold start phases: "Sending..." -> "Loading AI model..." (3s) -> "Generating..." (8s)

14. Create `frontend/src/lib/components/chat/ChatImageGrid.svelte`:
    - Side-by-side image layout for compare intents
    - 2-column grid with labels

15. Modify `frontend/src/lib/components/layout/Header.svelte`:
    - Add MessageCircle icon button (from lucide-svelte)
    - Toggle chat panel open/close via store

16. Modify `frontend/src/routes/+layout.svelte`:
    - Import and render `<ChatPanel />` at layout level (persists across pages)

17. Modify `frontend/nginx.conf`:
    - Add `proxy_buffering off;` and `proxy_cache off;` inside the existing `/api/` location block

### Phase 5: Polish
18. Out-of-scope detection: return helpful message without calling Ollama
19. Structured fallback when Ollama is down
20. Post-generation hallucination check (extract counts, compare to context)
21. Keyboard shortcut Cmd/Ctrl+K to toggle panel

## Style Rules
- Backend: Python 3.12, type hints, 100 char line length, ruff-compatible
- Frontend: Svelte 5 runes ($state, $derived, $props), TypeScript, 2-space indent
- Colors: ALWAYS use CSS custom properties via `style="color: var(--color-text-primary);"` — NEVER Tailwind color classes
- Layout utilities: Tailwind classes (flex, gap, rounded, p-4, etc.) are fine
- Follow existing patterns in each file — read before writing
