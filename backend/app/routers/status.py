"""System status router."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Request

from app.schemas import ComponentStatus, SystemStatus

router = APIRouter(tags=["status"])


def _check_db() -> ComponentStatus:
    try:
        from ring_detector.database import get_session

        session = get_session()
        from sqlalchemy import text

        session.execute(text("SELECT 1"))
        session.close()
        return ComponentStatus(status="ok", detail="Connected")
    except Exception as e:
        return ComponentStatus(status="fail", detail=str(e))


def _check_gpu() -> ComponentStatus:
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info = result.stdout.strip().split("\n")[0]
            return ComponentStatus(status="ok", detail=info)
        return ComponentStatus(status="fail", detail="nvidia-smi failed")
    except Exception as e:
        return ComponentStatus(status="missing", detail=str(e))


def _check_yolo() -> ComponentStatus:
    try:
        from ring_detector.config import settings

        model_path = Path(settings.model.yolo_model_path)
        if model_path.exists():
            size_mb = model_path.stat().st_size // (1024 * 1024)
            return ComponentStatus(status="ok", detail=f"{model_path.name} ({size_mb} MB)")
        return ComponentStatus(status="missing", detail=f"{model_path} not found")
    except Exception as e:
        return ComponentStatus(status="fail", detail=str(e))


def _check_ring_token() -> ComponentStatus:
    try:
        from ring_detector.config import settings

        token_path = Path(settings.ring.token_path)
        if token_path.exists():
            return ComponentStatus(status="ok", detail="Token cache present")
        return ComponentStatus(status="missing", detail="No token cache — run ring auth")
    except Exception as e:
        return ComponentStatus(status="fail", detail=str(e))


def _check_face_models() -> ComponentStatus:
    # Face models not in this branch — report as off
    return ComponentStatus(status="off", detail="Face detection not configured in this build")


def _check_ollama() -> ComponentStatus:
    try:
        from ring_detector.config import settings

        if not settings.captioner.enabled:
            return ComponentStatus(status="off", detail="Captioner disabled")
        import urllib.request

        url = f"{settings.captioner.ollama_url}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            if resp.status == 200:
                url_str = settings.captioner.ollama_url
                return ComponentStatus(status="ok", detail=f"Ollama at {url_str}")
        return ComponentStatus(status="warn", detail="Ollama returned non-200")
    except Exception as e:
        return ComponentStatus(status="fail", detail=str(e))


def _check_archive() -> ComponentStatus:
    try:
        from ring_detector.config import settings

        archive = Path(settings.storage.archive_dir)
        if archive.exists():
            return ComponentStatus(status="ok", detail=str(archive))
        return ComponentStatus(status="warn", detail=f"{archive} not mounted")
    except Exception as e:
        return ComponentStatus(status="fail", detail=str(e))


def _check_watcher() -> bool:
    """Check if ring-watch is running via PID file or process name."""
    try:
        import subprocess

        result = subprocess.run(
            ["pgrep", "-f", "ring-watch"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


@router.get("/status", response_model=SystemStatus)
def system_status(request: Request):
    start_time = getattr(request.app.state, "start_time", time.time())
    return SystemStatus(
        database=_check_db(),
        gpu=_check_gpu(),
        yolo_model=_check_yolo(),
        ring_token=_check_ring_token(),
        face_models=_check_face_models(),
        ollama=_check_ollama(),
        archive_dir=_check_archive(),
        watcher_running=_check_watcher(),
        uptime_seconds=int(time.time() - start_time),
    )


@router.get("/cameras")
def list_cameras():
    """Return known camera names from the events log."""
    try:
        from sqlalchemy import text

        from ring_detector.database import get_session

        session = get_session()
        rows = session.execute(
            text("SELECT DISTINCT camera_name FROM events ORDER BY camera_name")
        ).fetchall()
        session.close()
        return [{"name": r[0], "type": "camera"} for r in rows]
    except Exception:
        return []
