"""Watcher control router — start/stop/status/logs."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.watcher_manager import watcher_mgr

router = APIRouter(tags=["watcher"])


@router.get("/watcher/status")
async def watcher_status():
    return watcher_mgr.status


@router.post("/watcher/start")
async def watcher_start():
    started = await watcher_mgr.start()
    if not started:
        return {"success": False, "detail": "Watcher is already running"}
    return {"success": True, "detail": "Watcher started", **watcher_mgr.status}


@router.post("/watcher/stop")
async def watcher_stop():
    stopped = await watcher_mgr.stop()
    if not stopped:
        return {"success": False, "detail": "Watcher is not running"}
    return {"success": True, "detail": "Watcher stopped", **watcher_mgr.status}


@router.get("/watcher/logs")
async def watcher_logs(tail: int = Query(200, ge=1, le=2000)):
    lines = watcher_mgr.logs
    return {"lines": lines[-tail:], "total": len(lines)}
