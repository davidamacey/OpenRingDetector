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

# Collect all allowed root directories for image serving
_ALLOWED_ROOTS: list[Path] = []
for _d in [_ARCHIVE_DIR, _MODELS_DIR]:
    if _d.exists():
        _ALLOWED_ROOTS.append(_d.resolve())


def _resolve_safe(path: str) -> Path:
    """Resolve path relative to allowed roots. Raises 403 on traversal attempt."""
    # Restore leading / if it was stripped (URL routing eats the first /)
    if not path.startswith("/") and "/" in path:
        abs_candidate = Path("/" + path)
        if abs_candidate.exists():
            path = "/" + path

    target = Path(path)
    if not target.is_absolute():
        target = _ARCHIVE_DIR / path

    target = target.resolve()

    # Block path traversal
    if any(part == ".." for part in target.parts):
        raise HTTPException(status_code=403, detail="Path not allowed")

    # Check static allowed roots
    for root in _ALLOWED_ROOTS:
        if str(target).startswith(str(root)):
            return target

    # Allow any real file on mounted volumes (absolute snapshot paths from DB)
    if target.is_file():
        return target

    raise HTTPException(status_code=403, detail="Path not allowed")


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
