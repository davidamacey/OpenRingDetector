"""WebSocket hub — broadcasts Ring events to all connected clients."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import WebSocket

log = logging.getLogger(__name__)


class WebSocketHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        log.info("WebSocket client connected (%d total)", len(self._clients))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("WebSocket client disconnected (%d remaining)", len(self._clients))

    async def broadcast(self, message: dict) -> None:
        payload = json.dumps(message)
        dead: set[WebSocket] = set()
        async with self._lock:
            clients = set(self._clients)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._clients -= dead

    async def broadcast_from_pg(self, connection, pid, channel, payload: str) -> None:
        """Called by asyncpg listener when pg_notify fires."""
        try:
            data = json.loads(payload)
            await self.broadcast({"type": "event", "data": data})
        except Exception:
            log.exception("Error broadcasting PG event")


ws_hub = WebSocketHub()
