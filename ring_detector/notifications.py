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
        log.info("Notification sent: %s", title)
    except requests.RequestException:
        log.exception("Failed to send notification")


def notify_motion(camera_name: str, timestamp: str) -> None:
    send_notification(
        message=f"Motion detected at {camera_name} ({timestamp})",
        title="Motion Detected",
        tags="camera,motion_detector",
    )


def notify_person_detected(person_name: str, camera_name: str) -> None:
    send_notification(
        message=f"{person_name} detected at {camera_name}",
        title=f"{person_name} Arrived",
        tags="person,camera",
        priority="high",
    )


def notify_vehicle_detected(vehicle_label: str, camera_name: str) -> None:
    send_notification(
        message=f"{vehicle_label} detected at {camera_name}",
        title=f"{vehicle_label} Arrived",
        tags="car,camera",
        priority="high",
    )


def notify_unknown_visitor(camera_name: str) -> None:
    send_notification(
        message=f"Unknown person/vehicle at {camera_name}",
        title="Unknown Visitor",
        tags="warning,camera",
    )
