"""Settings router — read/write configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter

from ring_detector.config import settings

router = APIRouter(tags=["settings"])

_OVERRIDES_PATH = Path(os.getenv("CONFIG_OVERRIDES_PATH", "./config_overrides.json"))


def _load_overrides() -> dict:
    if _OVERRIDES_PATH.exists():
        try:
            return json.loads(_OVERRIDES_PATH.read_text())
        except Exception:
            return {}
    return {}


@router.get("/settings")
def get_settings():
    return {
        "ring": {
            "camera_name": settings.ring.camera_name,
            "departure_timeout": settings.ring.departure_timeout,
            "cooldown_seconds": settings.ring.cooldown_seconds,
        },
        "model": {
            "yolo_model_path": settings.model.yolo_model_path,
            "device": settings.model.device,
            "batch_size": settings.model.batch_size,
        },
        "captioner": {
            "enabled": settings.captioner.enabled,
            "ollama_url": settings.captioner.ollama_url,
            "model": settings.captioner.model,
        },
        "notify": {
            "ntfy_url": settings.notify.ntfy_url,
            "attach_snapshot": settings.notify.attach_snapshot,
        },
        "storage": {
            "archive_dir": str(settings.storage.archive_dir),
        },
        "face": {
            "enabled": settings.face.enabled,
            "match_threshold": settings.face.match_threshold,
            "min_face_size": settings.face.min_face_size,
        },
        "video": {
            "enabled": settings.video.enabled,
            "frame_interval": settings.video.frame_interval,
            "max_frames": settings.video.max_frames,
            "wait_timeout": settings.video.wait_timeout,
        },
    }


@router.patch("/settings")
def patch_settings(body: dict):
    overrides = _load_overrides()
    for section, values in body.items():
        if not isinstance(values, dict):
            continue
        overrides.setdefault(section, {}).update(values)
    _OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))
    return {"success": True, "restart_required": True}
