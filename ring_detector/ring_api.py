"""Async Ring API integration using push-based event listener.

Uses ring_doorbell >= 0.9.x with RingEventListener for near-instant
motion/ding alerts via Firebase Cloud Messaging (no polling needed).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from ring_doorbell import Auth, Ring, RingEventListener
from ring_doorbell.const import USER_AGENT

from ring_detector.config import settings

log = logging.getLogger(__name__)


# --- Token / Credential Persistence ---


def _token_updated_callback(token: dict) -> None:
    """Persist updated OAuth token to disk."""
    settings.ring.token_path.parent.mkdir(parents=True, exist_ok=True)
    settings.ring.token_path.write_text(json.dumps(token))
    log.debug("Ring token refreshed and saved")


def _fcm_credentials_updated(credentials: dict) -> None:
    """Persist FCM credentials for push event listener."""
    path = settings.ring.fcm_credentials_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(credentials))
    log.debug("FCM credentials updated and saved")


def _load_fcm_credentials() -> dict | None:
    """Load persisted FCM credentials, or None if not yet created."""
    path = settings.ring.fcm_credentials_path
    if path.is_file():
        return json.loads(path.read_text())
    return None


# --- Authentication ---


async def authenticate() -> Ring:
    """Authenticate with Ring using cached token. Returns Ring instance."""
    token_path = settings.ring.token_path
    if not token_path.is_file():
        raise FileNotFoundError(
            f"No cached token at {token_path}. Run `ring-auth` to authenticate."
        )

    token_data = json.loads(token_path.read_text())
    auth = Auth(USER_AGENT, token_data, _token_updated_callback)
    ring = Ring(auth)
    await ring.async_update_data()
    log.info("Ring authenticated successfully")
    return ring


# --- Push Event Listener ---


def create_event_listener(
    ring: Ring,
    on_motion: Callable | None = None,
    on_ding: Callable | None = None,
) -> RingEventListener:
    """Create a push-based event listener using Firebase Cloud Messaging.

    This replaces polling — events arrive within seconds of motion/ding.
    """
    fcm_creds = _load_fcm_credentials()
    listener = RingEventListener(
        ring=ring,
        credentials=fcm_creds,
        credentials_updated_callback=_fcm_credentials_updated,
    )

    def _on_event(event):
        """Route events to the appropriate handler."""
        # Skip duplicate "update" events (battery doorbells send two)
        if hasattr(event, "is_update") and event.is_update:
            return

        log.info(
            "Ring event: %s on %s (%s)",
            event.kind,
            event.device_name,
            event.id,
        )

        if event.kind == "motion" and on_motion:
            on_motion(event)
        elif event.kind == "ding" and on_ding:
            on_ding(event)

    listener.add_notification_callback(_on_event)
    return listener


# --- Device Helpers ---


def get_camera(ring: Ring, camera_name: str | None = None):
    """Get a specific camera by name, or the first available one."""
    camera_name = camera_name or settings.ring.camera_name

    if camera_name:
        device = ring.get_video_device_by_name(camera_name)
        if device:
            return device
        available = [d.name for d in ring.video_devices()]
        raise ValueError(f"Camera '{camera_name}' not found. Available: {available}")

    video_devs = ring.video_devices()
    if video_devs:
        return video_devs[0]
    raise ValueError("No cameras found in Ring account")


# --- Video / Snapshot Download ---


async def download_latest_video(ring: Ring, camera_name: str | None = None) -> Path | None:
    """Download the most recent motion video and archive to NAS."""
    cam = get_camera(ring, camera_name)
    history = await cam.async_history(limit=1)
    if not history:
        log.warning("No video history found for %s", cam.name)
        return None

    event = history[0]
    vid_id = event["id"]
    created_at = event["created_at"]
    timestamp_str = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{cam.name}_{vid_id}_{timestamp_str}.mp4"

    output_dir = settings.storage.video_dir()
    date_dir = output_dir / created_at.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    output_path = date_dir / filename

    if output_path.exists():
        log.debug("Video already downloaded: %s", output_path)
        return output_path

    try:
        await cam.async_recording_download(vid_id, str(output_path))
        log.info("Downloaded video: %s", output_path)
        return output_path
    except Exception:
        log.exception("Failed to download video %s", vid_id)
        return None


async def download_video_with_retry(
    ring: Ring,
    camera_name: str | None = None,
    timeout: int = 60,
    retry_delay: int = 5,
    shutdown_event: asyncio.Event | None = None,
) -> Path | None:
    """Wait for video to become available on Ring servers, then download it.

    Ring processes videos server-side after motion events. This retries
    up to `timeout` seconds with `retry_delay` between attempts.
    Respects an optional shutdown_event for clean exit.
    """
    cam = get_camera(ring, camera_name)
    deadline = asyncio.get_event_loop().time() + timeout

    while asyncio.get_event_loop().time() < deadline:
        if shutdown_event and shutdown_event.is_set():
            return None

        try:
            history = await cam.async_history(limit=1)
            if not history:
                log.debug("No video history yet for %s, retrying...", cam.name)
            else:
                event = history[0]
                vid_id = event["id"]
                created_at = event["created_at"]
                age = (datetime.now() - created_at).total_seconds()

                # Only download videos from the last 5 minutes
                if age > 300:
                    log.debug("Most recent video is %ds old, waiting for new one...", int(age))
                else:
                    timestamp_str = created_at.strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{cam.name}_{vid_id}_{timestamp_str}.mp4"

                    output_dir = settings.storage.video_dir()
                    date_dir = output_dir / created_at.strftime("%Y-%m-%d")
                    date_dir.mkdir(parents=True, exist_ok=True)
                    output_path = date_dir / filename

                    if output_path.exists():
                        log.info("Video already downloaded: %s", output_path)
                        return output_path

                    try:
                        await cam.async_recording_download(vid_id, str(output_path))
                        log.info("Downloaded video: %s", output_path)
                        return output_path
                    except Exception:
                        log.debug("Video not ready yet, retrying...", exc_info=True)
        except Exception:
            log.debug("Error checking video history, retrying...", exc_info=True)

        # Wait retry_delay or until shutdown
        if shutdown_event:
            try:
                await asyncio.wait_for(asyncio.shield(shutdown_event.wait()), timeout=retry_delay)
                return None  # Shutdown requested
            except TimeoutError:
                pass
        else:
            await asyncio.sleep(retry_delay)

    log.warning("Timed out waiting for video from %s after %ds", cam.name, timeout)
    return None


async def download_snapshot(ring: Ring, camera_name: str | None = None) -> Path | None:
    """Take and download a snapshot from the camera."""
    cam = get_camera(ring, camera_name)

    try:
        snapshot = await cam.async_get_snapshot()
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
