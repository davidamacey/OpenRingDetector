"""Tests for watcher logic — deduplication, cooldown, detection summary."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ring_detector.watcher import RingWatcher


@pytest.fixture
def watcher():
    return RingWatcher()


# --- Deduplication ---


def test_first_event_not_duplicate(watcher):
    event = MagicMock()
    event.id = 12345
    assert watcher._is_duplicate(event) is False


def test_second_event_with_same_id_is_duplicate(watcher):
    event = MagicMock()
    event.id = 12345
    watcher._is_duplicate(event)  # first call registers it
    assert watcher._is_duplicate(event) is True


def test_different_ids_not_duplicate(watcher):
    e1 = MagicMock()
    e1.id = 1
    e2 = MagicMock()
    e2.id = 2
    watcher._is_duplicate(e1)
    assert watcher._is_duplicate(e2) is False


def test_event_without_id_never_duplicate(watcher):
    event = MagicMock(spec=[])  # no .id attribute
    assert watcher._is_duplicate(event) is False
    assert watcher._is_duplicate(event) is False


def test_seen_ids_pruned_at_max(watcher):
    watcher._max_seen_ids = 10
    for i in range(15):
        e = MagicMock()
        e.id = i
        watcher._is_duplicate(e)
    # After pruning, set should be at max_seen_ids // 2
    assert len(watcher._seen_event_ids) <= watcher._max_seen_ids


# --- Cooldown ---


def test_no_cooldown_on_first_event(watcher):
    assert watcher._is_on_cooldown("Front Door") is False


def test_cooldown_active_immediately_after_processing(watcher):
    watcher._last_processed["Front Door"] = datetime.now()
    assert watcher._is_on_cooldown("Front Door") is True


def test_cooldown_expired(watcher):
    from ring_detector.config import settings

    past = datetime.now() - timedelta(seconds=settings.ring.cooldown_seconds + 5)
    watcher._last_processed["Front Door"] = past
    assert watcher._is_on_cooldown("Front Door") is False


def test_cooldown_per_camera(watcher):
    watcher._last_processed["Front Door"] = datetime.now()
    assert watcher._is_on_cooldown("Back Door") is False


# --- Detection Summary ---


def test_detection_summary_empty(watcher):
    df = pd.DataFrame()
    assert watcher._detection_summary(df) == ""


def test_detection_summary_single(watcher):
    df = pd.DataFrame({"class_name": ["car"]})
    result = watcher._detection_summary(df)
    assert result == "car"


def test_detection_summary_multiple_same_class(watcher):
    df = pd.DataFrame({"class_name": ["car", "car", "car"]})
    result = watcher._detection_summary(df)
    assert "car x3" in result


def test_detection_summary_mixed(watcher):
    df = pd.DataFrame({"class_name": ["car", "person", "car"]})
    result = watcher._detection_summary(df)
    assert "car x2" in result
    assert "person" in result


# --- Shutdown helper ---


@pytest.mark.asyncio
async def test_wait_or_shutdown_returns_false_on_timeout():
    watcher = RingWatcher()
    result = await watcher._wait_or_shutdown(0.05)
    assert result is False


@pytest.mark.asyncio
async def test_wait_or_shutdown_returns_true_when_shutdown_set():
    watcher = RingWatcher()

    async def trigger():
        await asyncio.sleep(0.02)
        watcher._shutdown.set()

    asyncio.create_task(trigger())
    result = await watcher._wait_or_shutdown(5.0)
    assert result is True
