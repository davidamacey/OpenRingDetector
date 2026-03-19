"""Tests for configuration module."""

from ring_detector.config import Settings


def test_settings_defaults():
    s = Settings()
    assert s.db.port == 5433
    assert s.model.image_size == 640
    assert s.ring.poll_interval_seconds == 60


def test_db_url_format():
    s = Settings()
    url = s.db.url
    assert url.startswith("postgresql://")
    assert str(s.db.port) in url
