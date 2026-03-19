"""Tests for captioner module."""

from ring_detector.captioner import caption_image, is_available
from ring_detector.config import Settings


def test_captioner_disabled_by_default():
    s = Settings()
    assert s.captioner.enabled is False
    assert s.captioner.model == "gemma3:4b"


def test_caption_returns_none_when_disabled():
    assert caption_image("/nonexistent/image.jpg") is None


def test_is_available_returns_false_when_disabled():
    assert is_available() is False
