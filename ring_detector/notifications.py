"""Push notifications via ntfy with optional snapshot attachments."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from ring_detector.config import settings

log = logging.getLogger(__name__)


def send_notification(
    message: str,
    title: str = "Ring Detector",
    tags: str = "camera",
    priority: str = "default",
    snapshot_path: str | Path | None = None,
) -> None:
    """Send a push notification via ntfy server, optionally with a snapshot image."""
    headers = {
        "Title": title,
        "Tags": tags,
        "Priority": priority,
    }

    try:
        if snapshot_path and settings.notify.attach_snapshot:
            snapshot = Path(snapshot_path)
            if snapshot.is_file():
                # Upload image to a private topic to get a hosted URL without notifying
                upload_url = settings.notify.ntfy_url.rsplit("/", 1)[0] + "/_uploads"
                with open(snapshot, "rb") as f:
                    resp = requests.put(
                        url=upload_url,
                        data=f,
                        headers={"Filename": snapshot.name},
                        timeout=15,
                    )
                resp.raise_for_status()
                image_url = (
                    resp_data.get("attachment", {}).get("url")
                    if (resp_data := resp.json())
                    else None
                )

                # Single notification with attachment + action button for iOS
                headers["Filename"] = snapshot.name
                headers["Message"] = message
                if image_url:
                    headers["Actions"] = f"view, View Snapshot, {image_url}, clear=true"
                with open(snapshot, "rb") as f:
                    requests.put(
                        url=settings.notify.ntfy_url,
                        data=f,
                        headers=headers,
                        timeout=15,
                    )
                log.info("Notification sent with snapshot: %s", title)
                return

        # Text-only notification (no snapshot or snapshot not found)
        requests.post(
            url=settings.notify.ntfy_url,
            data=message.encode("utf-8"),
            headers=headers,
            timeout=10,
        )
        log.info("Notification sent: %s - %s", title, message)
    except requests.RequestException:
        log.exception("Failed to send notification")


def notify_motion(camera_name: str, detail: str, snapshot_path: str | None = None) -> None:
    send_notification(
        message=f"Motion at {camera_name} ({detail})",
        title=f"Motion - {camera_name}",
        tags="camera,motion_detector",
        snapshot_path=snapshot_path,
    )


def notify_arrival(
    display_name: str,
    camera_name: str,
    detections_summary: str = "",
    snapshot_path: str | None = None,
) -> None:
    """Known reference arrived."""
    detail = f" ({detections_summary})" if detections_summary else ""
    send_notification(
        message=f"{display_name} arrived at {camera_name}{detail}",
        title=f"{display_name} - Arrived",
        tags="white_check_mark,car",
        priority="high",
        snapshot_path=snapshot_path,
    )


def notify_departure(display_name: str, camera_name: str, duration_mins: int) -> None:
    """Known reference left - time to pay."""
    send_notification(
        message=f"{display_name} left {camera_name} after ~{duration_mins} min. Time to pay!",
        title=f"{display_name} - Done! Pay Now",
        tags="money_with_wings,wave",
        priority="high",
    )


def notify_known_person(
    person_name: str,
    camera_name: str,
    detections_summary: str = "",
    snapshot_path: str | None = None,
) -> None:
    """Known person (face-matched) arrived at camera."""
    detail = f" ({detections_summary})" if detections_summary else ""
    send_notification(
        message=f"{person_name} arrived at {camera_name}{detail}",
        title=f"{person_name} - Arrived",
        tags="white_check_mark,bust_in_silhouette",
        priority="high",
        snapshot_path=snapshot_path,
    )


def notify_unknown_visitor(
    camera_name: str,
    detections_summary: str = "",
    snapshot_path: str | None = None,
) -> None:
    detail = f" ({detections_summary})" if detections_summary else ""
    send_notification(
        message=f"Unknown visitor at {camera_name}{detail}",
        title=f"Unknown Visitor - {camera_name}",
        tags="warning,camera",
        snapshot_path=snapshot_path,
    )


def notify_ding(camera_name: str, snapshot_path: str | None = None) -> None:
    send_notification(
        message=f"Someone rang the doorbell at {camera_name}",
        title=f"Doorbell - {camera_name}",
        tags="bell,door",
        priority="high",
        snapshot_path=snapshot_path,
    )
