# Chat Feature Implementation Plan

## Overview

Natural language chat interface for the OpenRingDetector dashboard. Users ask questions about detection events, vehicles, visitors, and activity. Gemma 3 4B (via Ollama on GPU 1) answers using DB context and can analyze snapshots on demand.

**UI**: Slide-out panel from right edge, accessible from any page via Header button.

## Example Queries

- "What happened today?"
- "When did the cleaner last come?"
- "Any visitors after 10pm this week?"
- "Show me the last motion event" (returns inline snapshot)
- "Describe what's in the photo from 3pm" (sends snapshot to Gemma)
- "How many motion events this week?"
- "Is anyone here right now?"
- "Compare the last two visitors" (side-by-side snapshots)
- "Show me the cleaner's car" (reference image + recent sightings)

## Architecture

```
User question (frontend)
  -> POST /api/chat (FastAPI SSE)
  -> Intent classification (keyword-based, no LLM)
  -> Time range parsing (regex)
  -> Entity resolution (fuzzy match against DB)
  -> Query relevant DB tables
  -> If image intent: look up snapshot, optionally send to Gemma multi-modal
  -> Assemble system prompt + context (token-budgeted)
  -> Stream Gemma response via Ollama (SSE)
  -> Frontend renders text + inline images + event/reference cards
```

## Intent Classification (Keyword-Based, No LLM)

| Intent | Examples | Data Source |
|--------|----------|-------------|
| `recent_events` | "What happened today?" | events |
| `vehicle_query` | "What vehicles were seen?" | events + visit_events + references |
| `person_query` | "Was David home today?" | events + visit_events + face_profiles |
| `visit_history` | "How long did the cleaner stay?" | visit_events |
| `active_visits` | "Is anyone here now?" | visit_events (departed_at IS NULL) |
| `statistics` | "How many events this week?" | events + detections (aggregates) |
| `doorbell` | "Did anyone ring the doorbell?" | events (type=ding) |
| `system_info` | "What cameras are set up?" | references, face_profiles |
| `describe_event` | "What's in the photo from 3pm?" | events + snapshot -> Gemma multi-modal |
| `show_event` | "Show me the last motion event" | events + snapshot URL |
| `show_reference` | "Show me the cleaner's car" | references + thumbnail |
| `show_visitors` | "Who visited today?" (with images) | events + snapshots |
| `compare_events` | "Compare the last two visitors" | events + multiple snapshots -> Gemma |
| `visual_search` | "Any red cars today?" | events + snapshots -> Gemma |

Image intents checked first, then fall back to text-only intents.

## Time Range Parsing (Regex, No LLM)

| Input | Resolution |
|-------|-----------|
| "today" / "this morning" / "tonight" | Start of day to now |
| "yesterday" / "yesterday evening" | Yesterday (with optional time-of-day) |
| "this week" | Monday 00:00 to now |
| "last week" | Previous Mon-Sun |
| "last N days/hours" | now - N to now |
| "a few days ago" | 3-5 days ago (widened range) |
| "March 15" / "on Tuesday" | Specific date, full day |
| No time keyword | Default: last 24 hours |

## Entity Resolution (Fuzzy Matching)

- Load all reference `display_name` + face profile `display_name` from DB (small set, <20)
- Case-insensitive substring match: "cleaners" -> "Cleaner"
- `difflib.get_close_matches()` for typos: "Dave" -> "David"
- Strip possessives: "cleaner's car" -> "cleaner"

## Context Window Management (Gemma 3 4B = 8K tokens)

**Token budget:**
- System prompt: ~300 tokens (fixed)
- System summary: ~80 tokens (always included)
- DB context: ~1800-2000 tokens
- User question: ~50 tokens
- Response: ~500-1000 tokens (`num_predict: 300-500`)

**Tiered summarization by time distance:**

| Tier | Range | Detail | Tokens |
|------|-------|--------|--------|
| Recent | Last 24h | Full event rows with timestamps | ~50/event |
| Medium | 2-7 days | Daily aggregates | ~30/day |
| Historical | 8-30 days | Period counts | ~20/period |

When user asks about a specific past day, promote it to "Recent" tier.

**Overflow handling:** If context exceeds budget:
1. Reduce event limit from 20 to 10
2. Remove captions (verbose)
3. Truncate detection summaries to 30 chars

## SSE Response Format

```
event: token
data: {"text": "The cleaner arrived at..."}

event: image
data: {"url": "/api/images/...", "alt": "Front Door - 3:15 PM", "event_id": 42}

event: images
data: {"items": [{"url": "...", "alt": "..."}, ...]}

event: event_detail
data: {"id": 42, "event_type": "arrival", "camera_name": "Front Door", ...}

event: reference_card
data: {"name": "cleaners_car", "display_name": "Cleaner", "image_url": "...", ...}

event: face_card
data: {"name": "david", "display_name": "David", "image_url": "...", ...}

event: done
data: {}

event: error
data: {"message": "No events found for that time range."}
```

Image/card events emitted BEFORE text streaming so user sees visual context while AI description loads.

## System Prompt

```
You are the OpenRingDetector assistant. You answer questions about security
camera activity ONLY using the data provided below. If the data doesn't
contain enough information, say so. Never invent events, times, or counts.

Current time: {datetime}

RULES:
- When referencing events, include timestamps
- Durations: "X minutes" for <60min, "X hours Y minutes" for longer
- Times: 12-hour format with am/pm
- Dates: "Today", "Yesterday", or "Weekday, Month Day"
- Use exact numbers, not approximations

{system_summary}
{category_context}
```

## Ollama Availability & Fallback

### Health Check: `GET /api/chat/status`
```json
{
  "available": true,
  "model_loaded": true,
  "model_name": "gemma3:4b",
  "ollama_reachable": true,
  "detail": "Ready"
}
```
Uses Ollama `GET /api/ps` to check if model is in VRAM vs just downloaded.

### Warmup: `POST /api/chat/warmup`
Fire-and-forget call when user opens chat panel. Sends trivial prompt to force model loading.

### Direct-Query Fallback
When Ollama is down, return structured DB results without LLM summarization:
```json
{
  "answer": null,
  "fallback": true,
  "fallback_reason": "ollama_unavailable",
  "structured_data": {"events": [...], "visit_summary": {...}}
}
```
Frontend renders structured data as formatted summary instead of natural language.

### Cold Start Handling
- Check `/api/ps` before request
- First request: 90s timeout (vs 60s normal)
- Frontend phases: "Sending..." -> "Loading AI model..." (3s) -> "Generating..." (8s)

## Out-of-Scope Detection

Before calling Ollama, check if question contains domain keywords (motion, arrival, vehicle, camera, visit, etc.). If none found, return immediately:
```json
{
  "out_of_scope": true,
  "answer": "I can only answer questions about your Ring camera events..."
}
```

## Rate Limiting

- In-memory token bucket: 10 requests per 60 seconds (single-user LAN system)
- Max message length: 500 characters
- `asyncio.Semaphore(1)` for concurrent requests — fail fast with 503 rather than queue
- Frontend disables send button during in-flight requests

## Hallucination Prevention

1. System prompt: "ONLY use data provided below. Never invent events."
2. Post-generation validation: extract counts from response, compare against context data
3. Source citations: event IDs in context, LLM instructed to reference them
4. `cited_events: int[]` extracted from response for frontend linking

## Conversation History

**V1: Single-turn, client-side only**
- Each request independent — no server-side history
- Frontend displays message history in component state
- API accepts optional `conversation_id` (ignored in V1, ready for V2)

**V2 (future): Multi-turn**
- Send last 3 turns to Ollama (6 messages max)
- Fresh DB queries per request (not from conversation history)
- Optional `chat_messages` DB table via Alembic migration

## Files to Create (9)

### Backend
| File | Purpose |
|------|---------|
| `backend/app/routers/chat.py` | POST `/api/chat` (SSE), GET `/api/chat/status`, POST `/api/chat/warmup` |
| `backend/app/chat_context.py` | Context builder: tiered summarization, token estimation, prompt assembly |
| `backend/app/chat_entities.py` | Intent classifier, time parser, entity resolver, scope checker |

### Frontend
| File | Purpose |
|------|---------|
| `frontend/src/lib/api/chat.ts` | SSE streaming fetch client (native fetch + ReadableStream) |
| `frontend/src/lib/stores/chat.ts` | Chat state store (messages, isStreaming, isOpen) |
| `frontend/src/lib/components/chat/ChatPanel.svelte` | Slide-out panel container |
| `frontend/src/lib/components/chat/ChatMessage.svelte` | Rich message bubble (text + inline images + cards) |
| `frontend/src/lib/components/chat/ChatInput.svelte` | Text input + send button |
| `frontend/src/lib/components/chat/ChatImageGrid.svelte` | Multi-image comparison layout |

## Files to Modify (7)

| File | Change |
|------|--------|
| `backend/app/main.py` | Register chat router |
| `backend/app/schemas.py` | Add ChatRequest, ChatResponse, ChatStatusResponse, ChatImagePayload, ChatEventDetail |
| `backend/requirements.txt` | Add `httpx>=0.27` for async streaming |
| `ring_detector/config.py` | Add `chat_timeout` to CaptionerConfig |
| `frontend/nginx.conf` | Add `proxy_buffering off;` to `/api/` block for SSE |
| `frontend/src/routes/+layout.svelte` | Add ChatPanel overlay + floating toggle button |
| `frontend/src/lib/components/layout/Header.svelte` | Add MessageCircle chat toggle button |

## Implementation Phases

### Phase 1: Backend Core (testable via curl, no frontend needed)
1. `chat_entities.py` — intent classifier, time parser, entity resolver
2. `chat_context.py` — DB queries, tiered context builder, token budgeting
3. Test independently: parse questions, verify context output

### Phase 2: Backend API
4. `chat.py` router — `/api/chat` SSE endpoint, `/api/chat/status`, `/api/chat/warmup`
5. Ollama streaming integration (follows captioner.py pattern)
6. Multi-modal image support for describe/compare intents
7. Register in `main.py`, add schemas
8. Add `httpx>=0.27` to requirements
9. Test: `curl -N -X POST http://localhost:9553/api/chat -H 'Content-Type: application/json' -d '{"message":"Who visited today?"}'`

### Phase 3: Frontend Infrastructure
10. Chat types in `types/index.ts`
11. SSE client `chat.ts` (multi-event-type parser)
12. Chat store `chat.ts`
13. `nginx.conf` — `proxy_buffering off;`

### Phase 4: Frontend UI
14. `ChatMessage.svelte` — text + inline images + reference/face/event cards
15. `ChatInput.svelte` — text input with send
16. `ChatPanel.svelte` — slide-out container, suggested queries, status banner
17. `ChatImageGrid.svelte` — side-by-side comparison layout
18. Header toggle button + layout integration

### Phase 5: Polish
19. Error UX: inline error messages with retry, cold start phases, Ollama-down banner
20. Hallucination checks: post-generation count validation
21. Rate limiting + semaphore
22. Click-to-expand snapshots (reuse SnapshotModal)
23. Keyboard shortcut (Cmd/Ctrl+K) to toggle panel
24. Suggested query chips ("Vehicles today", "Last visitor", etc.)

## Key Design Decisions

- **Keyword intent classifier** (not LLM) — zero latency, deterministic, reliable
- **Slide-out panel** (not separate page) — accessible from any page without navigation
- **SSE streaming** — tokens appear progressively, images/cards emitted first
- **Native fetch** for SSE (not EventSource which doesn't support POST, not axios which doesn't stream)
- **Multi-modal only on demand** — only send images to Gemma for "describe"/"compare" intents
- **Independent of CAPTIONER_ENABLED** — chat works even if captioning is disabled
- **No schema changes** — query existing tables, vehicle match similarity computed live via pgvector
- **Single-turn V1** — stateless backend, client-side message history, API ready for V2
- **Structured fallback** — returns raw data when Ollama is down (degraded, not broken)
- **Ollama concurrency handled by semaphore** — fail fast with 503, don't queue silently
