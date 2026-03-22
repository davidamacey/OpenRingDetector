"""Chat router -- SSE streaming endpoint using Ollama Gemma 3 4B.

Keyword-based intent classification + structured DB context + Ollama for
natural language responses about Ring camera events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.chat_context import (
    build_chat_context,
    build_system_prompt,
    lookup_event_snapshots,
    lookup_face_info,
    lookup_reference_info,
)
from app.chat_entities import ChatIntent, classify_intent, is_in_scope
from app.schemas import ChatRequest, ChatStatusResponse
from ring_detector.config import settings
from ring_detector.database import get_session

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# ---------------------------------------------------------------------------
# Rate limiting: in-memory token bucket (10 req / 60s)
# ---------------------------------------------------------------------------
_RATE_LIMIT = 10
_RATE_WINDOW = 60.0
_request_timestamps: list[float] = []

# Concurrency: single request at a time (Ollama GPU contention)
_chat_semaphore = asyncio.Semaphore(1)

# Max user message length
_MAX_MESSAGE_LEN = 500

# Ollama cold-start timeout (first request may need to load model into VRAM)
_COLD_START_TIMEOUT = 90


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _check_rate_limit() -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.monotonic()
    # Evict stale timestamps
    while _request_timestamps and _request_timestamps[0] < now - _RATE_WINDOW:
        _request_timestamps.pop(0)
    if len(_request_timestamps) >= _RATE_LIMIT:
        return False
    _request_timestamps.append(now)
    return True


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _build_ollama_messages(
    question: str,
    system_prompt: str,
    history: list,
) -> list[dict]:
    """Assemble the messages list for Ollama /api/chat."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    # Include conversation history (last 6 turns max)
    for turn in history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


async def _stream_ollama(
    messages: list[dict],
    ollama_url: str,
    model: str,
    timeout: float,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama /api/chat as SSE events.

    Yields SSE-formatted strings. On connection/timeout errors, yields an
    error event with fallback=true.
    """
    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=10.0)) as client,
            client.stream(
                "POST",
                f"{ollama_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500,
                    },
                },
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield _sse_event("token", {"text": token})
                if chunk.get("done"):
                    break
    except (httpx.ConnectError, httpx.ConnectTimeout):
        log.warning("Ollama not reachable at %s", ollama_url)
        yield _sse_event(
            "error",
            {
                "message": "Ollama is not reachable. Showing structured data instead.",
                "fallback": True,
            },
        )
    except httpx.TimeoutException:
        log.warning("Ollama request timed out after %.0fs", timeout)
        yield _sse_event(
            "error",
            {
                "message": "Response timed out. Showing structured data instead.",
                "fallback": True,
            },
        )
    except httpx.HTTPStatusError as exc:
        log.warning("Ollama returned %d: %s", exc.response.status_code, exc.response.text[:200])
        yield _sse_event(
            "error",
            {
                "message": f"Ollama error ({exc.response.status_code}).",
                "fallback": True,
            },
        )
    except Exception:
        log.exception("Unexpected Ollama streaming error")
        yield _sse_event(
            "error",
            {
                "message": "Unexpected error. Showing structured data instead.",
                "fallback": True,
            },
        )


async def _generate_chat_stream(
    request: ChatRequest,
    session: Session,
) -> AsyncGenerator[str, None]:
    """Full chat pipeline: scope check -> classify -> images/cards -> Ollama stream."""
    question = request.message.strip()

    # 1. Scope check
    if not is_in_scope(question):
        yield _sse_event(
            "error",
            {
                "message": (
                    "I can only answer questions about your Ring camera activity, "
                    "detected vehicles, recognized faces, and visit history. "
                    "Try asking something like 'What happened today?' or "
                    "'When did the cleaner arrive?'"
                ),
                "fallback": False,
            },
        )
        yield _sse_event("done", {})
        return

    # 2. Classify intent
    intent: ChatIntent = classify_intent(question, session)

    # 3. Image intents: emit images before text
    if intent.is_image_intent:
        snapshots = lookup_event_snapshots(session, intent)
        if snapshots:
            if len(snapshots) == 1:
                yield _sse_event("image", snapshots[0])
            else:
                yield _sse_event("images", {"items": snapshots})

    # 4. Reference card
    if intent.category == "show_reference" and intent.entity_name:
        ref_info = lookup_reference_info(session, intent.entity_name)
        if ref_info:
            yield _sse_event("reference_card", ref_info)

    # 5. Face card
    if intent.entity_type == "face_profile" and intent.entity_name:
        face_info = lookup_face_info(session, intent.entity_name)
        if face_info:
            yield _sse_event("face_card", face_info)

    # 6. Build context and system prompt
    context = build_chat_context(session, intent)
    system_prompt = build_system_prompt(session, context)

    # 7. Build messages and stream from Ollama
    messages = _build_ollama_messages(question, system_prompt, request.history)
    ollama_url = settings.captioner.ollama_url
    model = settings.captioner.model
    timeout = float(settings.captioner.chat_timeout)

    async for event in _stream_ollama(messages, ollama_url, model, timeout):
        if '"fallback": true' in event or '"fallback":true' in event:
            # Replace the generic fallback with one that includes structured data
            yield _sse_event(
                "error",
                {
                    "message": "AI chat is unavailable. Here is the raw data from your cameras.",
                    "fallback": True,
                    "structured_data": context,
                },
            )
        else:
            yield event

    # 8. Done
    yield _sse_event("done", {})


# ---------------------------------------------------------------------------
# POST /api/chat -- SSE streaming endpoint
# ---------------------------------------------------------------------------


@router.post("/chat")
async def chat(request: ChatRequest, session: Session = Depends(_get_db)):
    """Stream a chat response about Ring camera activity via SSE."""
    # Validate message length
    if len(request.message) > _MAX_MESSAGE_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long (max {_MAX_MESSAGE_LEN} characters).",
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Rate limit
    if not _check_rate_limit():
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a moment before trying again.",
        )

    # Concurrency guard
    if _chat_semaphore.locked():
        raise HTTPException(
            status_code=503,
            detail="Another chat request is in progress. Please try again shortly.",
        )

    async def stream_wrapper() -> AsyncGenerator[str, None]:
        async with _chat_semaphore:
            async for event in _generate_chat_stream(request, session):
                yield event

    return StreamingResponse(
        stream_wrapper(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/chat/status -- Ollama health check
# ---------------------------------------------------------------------------


@router.get("/chat/status", response_model=ChatStatusResponse)
async def chat_status():
    """Check if Ollama is reachable and the chat model is loaded in VRAM."""
    ollama_url = settings.captioner.ollama_url
    model = settings.captioner.model
    ollama_reachable = False
    model_found = False
    model_loaded = False
    detail = ""

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            tags_resp = await client.get(f"{ollama_url}/api/tags")
            tags_resp.raise_for_status()
            ollama_reachable = True

            models = [m["name"] for m in tags_resp.json().get("models", [])]
            model_found = any(model in m for m in models)

            if not model_found:
                detail = f"Model '{model}' not found. Available: {', '.join(models) or 'none'}"
            else:
                ps_resp = await client.get(f"{ollama_url}/api/ps")
                ps_resp.raise_for_status()
                running = [m.get("name", "") for m in ps_resp.json().get("models", [])]
                model_loaded = any(model in m for m in running)
                detail = (
                    "Ready"
                    if model_loaded
                    else ("Available (first request may be slow while model loads)")
                )

    except httpx.ConnectError:
        detail = f"Ollama not reachable at {ollama_url}"
    except httpx.TimeoutException:
        detail = f"Ollama timed out at {ollama_url}"
    except Exception as exc:
        detail = f"Error checking Ollama: {exc}"

    return ChatStatusResponse(
        available=ollama_reachable and model_found,
        model_loaded=model_loaded,
        model_name=model,
        ollama_reachable=ollama_reachable,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# POST /api/chat/warmup -- fire-and-forget model load
# ---------------------------------------------------------------------------


@router.post("/chat/warmup", status_code=202)
async def chat_warmup():
    """Send a trivial prompt to Ollama to force model loading into VRAM.

    Returns 202 immediately; model loading happens in the background.
    """
    ollama_url = settings.captioner.ollama_url
    model = settings.captioner.model

    async def _warmup():
        try:
            async with httpx.AsyncClient(timeout=_COLD_START_TIMEOUT) as client:
                await client.post(
                    f"{ollama_url}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "options": {"num_predict": 1},
                    },
                )
                log.info("Chat model '%s' warmed up successfully.", model)
        except Exception:
            log.warning("Chat model warmup failed (model may load on first request).")

    asyncio.create_task(_warmup())
    return {"status": "warmup_started", "model": model}
