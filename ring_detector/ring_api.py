"""Async Ring API integration for auth, video download, and event monitoring.

Uses ring_doorbell >= 0.9.x which is fully async.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ring_doorbell import Auth, Ring
from ring_doorbell.const import USER_AGENT

from ring_detector.config import settings

log = logging.getLogger(__name__)


def _token_updated_callback(token: dict) -> None:
    """Persist updated OAuth token to disk."""
    settings.ring.token_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ring.token_path.write_text(json.dumps(token))
    log.info("Ring token refreshed and saved")


async def authenticate() -> Ring:
    """Authenticate with Ring using cached token. Returns Ring instance."""
    token_path = settings.ring.token_path
    if not token_path.is_file():
        raise FileNotFoundError(
            f"No cached token at {token_path}. "
            "Run `python -m ring_doorbell.cli --auth` to create one."
        )

    token_data = json.loads(token_path.read_text())
    auth = Auth(USER_AGENT, token_data, _token_updated_callback)
    ring = Ring(auth)
    await ring.async_update_data()
    log.info("Ring authenticated successfully")
    return ring


def get_camera(ring: Ring, camera_name: str | None = None):
    """Get a specific camera by name, or the first available one."""
    devices = ring.devices()
    camera_name = camera_name or settings.ring.camera_name

    # Search across all device types
    for device_type in ("doorbells", "stickup_cams", "other"):
        for device in devices.get(device_type, []):
            if not camera_name or device.name == camera_name:
                return device

    available = []
    for device_type in ("doorbells", "stickup_cams", "other"):
        available.extend(d.name for d in devices.get(device_type, []))
    raise ValueError(f"Camera '{camera_name}' not found. Available: {available}")


async def download_latest_video(
    ring: Ring,
    camera_name: str | None = None,
) -> Path | None:
    """Download the most recent motion video and archive to NAS."""
    cam = get_camera(ring, camera_name)
    history = cam.history(limit=1)
    if not history:
        log.warning("No video history found for %s", cam.name)
        return None

    event = history[0]
    vid_id = event["id"]
    created_at = event["created_at"]
    timestamp_str = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{cam.name}_{vid_id}_{timestamp_str}.mp4"

    output_dir = settings.storage.video_dir()
    # Organize by date
    date_dir = output_dir / created_at.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    output_path = date_dir / filename

    if output_path.exists():
        log.info("Video already downloaded: %s", output_path)
        return output_path

    try:
        cam.recording_download(vid_id, str(output_path))
        log.info("Downloaded video: %s", output_path)
        return output_path
    except Exception:
        log.exception("Failed to download video %s", vid_id)
        return None


async def download_snapshot(
    ring: Ring,
    camera_name: str | None = None,
) -> Path | None:
    """Take and download a snapshot from the camera."""
    cam = get_camera(ring, camera_name)

    try:
        snapshot = cam.get_snapshot()
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{cam.name}_{timestamp}.jpg"

        output_dir = settings.storage.snapshot_dir()
        date_dir = output_dir / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        output_path = date_dir / filename

        output_path.write_bytes(snapshot)
        log.info("Snapshot saved: %s", output_path)
        return output_path
    except Exception:
        log.exception("Failed to get snapshot from %s", cam.name)
        return None


async def get_recent_motion_events(
    ring: Ring,
    camera_name: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Get recent motion events from the camera."""
    cam = get_camera(ring, camera_name)
    history = cam.history(limit=limit, kind="motion")
    return history


async def get_new_motion_events(
    ring: Ring,
    camera_name: str | None = None,
    since_id: int | None = None,
) -> list[dict]:
    """Get motion events newer than the given event ID."""
    events = await get_recent_motion_events(ring, camera_name, limit=20)
    if since_id is None:
        return events

    new_events = []
    for event in events:
        if event["id"] == since_id:
            break
        new_events.append(event)
    return new_events
