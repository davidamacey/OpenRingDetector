"""Image serving router — serves snapshot files from the archive directory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from ring_detector.config import settings

router = APIRouter(tags=["images"])

_ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", str(settings.storage.archive_dir)))
_MODELS_DIR = Path("./models")


def _resolve_safe(path: str) -> Path:
    """Resolve path relative to allowed roots. Raises 403 on traversal attempt."""
    # Try absolute path first
    target = Path(path)
    if not target.is_absolute():
        target = _ARCHIVE_DIR / path

    # Ensure path is inside an allowed directory
    target = target.resolve()
    allowed = [_ARCHIVE_DIR.resolve(), _MODELS_DIR.resolve()]
    if not any(str(target).startswith(str(a)) for a in allowed):
        raise HTTPException(status_code=403, detail="Path not allowed")
    return target


@router.get("/images/{path:path}")
async def serve_image(path: str):
    try:
        target = _resolve_safe(path)
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=400, detail="Invalid path") from err

    if not target.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    suffix = target.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")

    return FileResponse(
        str(target),
        media_type=media_type,
        headers={"Cache-Control": "max-age=86400"},
    )
