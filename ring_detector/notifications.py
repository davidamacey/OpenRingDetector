"""Push notifications via ntfy."""

from __future__ import annotations

import logging

import requests

from ring_detector.config import settings

log = logging.getLogger(__name__)


def send_notification(
    message: str,
    title: str = "Ring Detector",
    tags: str = "camera",
    priority: str = "default",
) -> None:
    """Send a push notification via ntfy server."""
    try:
        requests.post(
            url=settings.notify.ntfy_url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Tags": tags,
                "Priority": priority,
            },
            timeout=10,
        )
        log.info("Notification sent: %s — %s", title, message)
    except requests.RequestException:
        log.exception("Failed to send notification")


def notify_motion(camera_name: str, timestamp: str) -> None:
    send_notification(
        message=f"Motion detected at {camera_name} ({timestamp})",
        title="Motion Detected",
        tags="camera,motion_detector",
    )


def notify_arrival(display_name: str, camera_name: str) -> None:
    """Someone known has arrived."""
    send_notification(
        message=f"{display_name} arrived at {camera_name}",
        title=f"{display_name} — Arrived",
        tags="white_check_mark,car",
        priority="high",
    )


def notify_departure(display_name: str, camera_name: str, duration_mins: int) -> None:
    """Someone known has left — time to pay!"""
    send_notification(
        message=(f"{display_name} left {camera_name} after ~{duration_mins} min. Time to pay!"),
        title=f"{display_name} — Done! Pay Now",
        tags="money_with_wings,wave",
        priority="high",
    )


def notify_unknown_visitor(camera_name: str) -> None:
    send_notification(
        message=f"Unknown person/vehicle at {camera_name}",
        title="Unknown Visitor",
        tags="warning,camera",
    )
