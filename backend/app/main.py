"""FastAPI application — OpenRingDetector Dashboard API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.routers import (
    analytics,
    events,
    faces,
    images,
    references,
    settings_router,
    status,
    test,
    unmatched,
    visits,
    watcher,
)
from app.websocket import ws_hub

log = logging.getLogger(__name__)

_start_time = time.time()


def get_db_url_asyncpg() -> str:
    """Convert SQLAlchemy URL to asyncpg format."""
    from ring_detector.config import settings

    cfg = settings.db
    return f"postgresql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.name}"


async def _pg_listener(db_url: str) -> None:
    """Subscribe to ring_events channel and broadcast to WebSocket clients."""
    while True:
        try:
            conn = await asyncpg.connect(db_url)
            await conn.add_listener("ring_events", ws_hub.broadcast_from_pg)
            log.info("PostgreSQL LISTEN/NOTIFY active on 'ring_events'")
            # Keep alive until disconnect
            while True:
                await asyncio.sleep(30)
                await conn.execute("SELECT 1")
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("PG listener error — reconnecting in 5s")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    db_url = get_db_url_asyncpg()
    task = asyncio.create_task(_pg_listener(db_url), name="pg_listener")
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="OpenRingDetector API",
    description="Dashboard API for Ring motion detection, profiles, and visit history",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(events.router, prefix="/api")
app.include_router(visits.router, prefix="/api")
app.include_router(references.router, prefix="/api")
app.include_router(faces.router, prefix="/api")
app.include_router(unmatched.router, prefix="/api")
app.include_router(analytics.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(settings_router.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(test.router, prefix="/api")
app.include_router(watcher.router, prefix="/api")


@app.get("/api/health")
async def health():
    """Quick health check with component status summary."""
    from app.routers.status import (
        _check_archive,
        _check_db,
        _check_gpu,
        _check_ollama,
        _check_ring_token,
        _check_watcher,
        _check_yolo,
    )

    db = _check_db()
    return {
        "status": "ok" if db.status == "ok" else "degraded",
        "uptime_seconds": int(time.time() - _start_time),
        "components": {
            "database": db.status,
            "gpu": _check_gpu().status,
            "yolo_model": _check_yolo().status,
            "ring_token": _check_ring_token().status,
            "ollama": _check_ollama().status,
            "archive": _check_archive().status,
            "watcher": "ok" if _check_watcher() else "fail",
        },
    }


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_hub.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            import json

            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text('{"type":"pong"}')
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await ws_hub.disconnect(ws)


# Export start time for status endpoint
app.state.start_time = _start_time
