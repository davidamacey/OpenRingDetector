"""Tests for notification message formatting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ring_detector import notifications


def _capture_send(mock_post, mock_put):
    """Helper: return what was passed to send_notification."""
    calls = []
    mock_post.side_effect = lambda **kw: calls.append(("post", kw))
    mock_put.side_effect = lambda **kw: calls.append(("put", kw))
    return calls


@patch("ring_detector.notifications.requests.post")
@patch("ring_detector.notifications.requests.put")
def test_notify_motion_message(mock_put, mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifications.notify_motion("Front Door", "2025-01-01 12:00:00")
    assert mock_post.called
    args = mock_post.call_args
    assert "Front Door" in args.kwargs.get("data", b"").decode("utf-8")


@patch("ring_detector.notifications.requests.post")
def test_notify_arrival_high_priority(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifications.notify_arrival("Cleaner", "Front Door", "white van")
    assert mock_post.called
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Priority"] == "high"
    assert "Cleaner" in headers["Title"]


@patch("ring_detector.notifications.requests.post")
def test_notify_departure_message(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifications.notify_departure("Cleaner", "Front Door", 45)
    data = mock_post.call_args.kwargs["data"].decode("utf-8")
    assert "45 min" in data
    assert "Time to pay" in data


@patch("ring_detector.notifications.requests.post")
def test_notify_ding_high_priority(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifications.notify_ding("Front Door")
    headers = mock_post.call_args.kwargs["headers"]
    assert headers["Priority"] == "high"


@patch("ring_detector.notifications.requests.post")
def test_notify_unknown_visitor(mock_post):
    mock_post.return_value = MagicMock(status_code=200)
    notifications.notify_unknown_visitor("Front Door", "car, person")
    data = mock_post.call_args.kwargs["data"].decode("utf-8")
    assert "Unknown" in data
    assert "car, person" in data


@patch("ring_detector.notifications.requests.post")
def test_send_notification_swallows_request_error(mock_post):
    import requests as req

    mock_post.side_effect = req.ConnectionError("connection refused")
    # Should not raise — RequestException is caught internally
    notifications.send_notification("test message")


@patch("ring_detector.notifications.requests.put")
def test_send_notification_with_snapshot(mock_put, tmp_path):
    snap = tmp_path / "snap.jpg"
    snap.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # minimal JPEG header

    mock_put.return_value = MagicMock(status_code=200)
    notifications.send_notification("motion", snapshot_path=str(snap))
    assert mock_put.called
    headers = mock_put.call_args.kwargs["headers"]
    assert headers["Filename"] == "snap.jpg"
