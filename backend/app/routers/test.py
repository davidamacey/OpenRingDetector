"""Test router — POST /api/test/notify."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["test"])


@router.post("/test/notify")
def test_notify():
    """Send a test notification via ntfy."""
    from ring_detector.notifications import send_notification

    try:
        send_notification(
            message="Test notification from OpenRingDetector dashboard",
            title="Dashboard Test",
            tags="white_check_mark,test_tube",
        )
        return {"success": True, "detail": "Notification sent"}
    except Exception as e:
        return {"success": False, "detail": str(e)}
