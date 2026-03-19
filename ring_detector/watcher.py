"""Main watcher: push-based Ring events → detect → identify → track → notify.

Uses RingEventListener (Firebase push) for near-instant motion alerts.
Tracks arrival/departure of known references (cleaner, yard guy, etc.)
and sends "arrived" and "time to pay" notifications with snapshot images.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ring_detector import ring_api
from ring_detector.config import settings
from ring_detector.database import (
    create_tables,
    get_active_visit_by_reference,
    get_active_visits,
    get_all_references,
    get_session,
    match_against_references,
    record_arrival,
    record_departure,
)
from ring_detector.detector import (
    clear_gpu_memory,
    compute_yolo_embeddings,
    run_detection,
)
from ring_detector.image_utils import pad_to_square, prepare_batch
from ring_detector.models import load_models
from ring_detector.notifications import (
    notify_arrival,
    notify_departure,
    notify_ding,
    notify_motion,
    notify_unknown_visitor,
)

log = logging.getLogger(__name__)


class RingWatcher:
    """Watches Ring cameras for motion, identifies known visitors, tracks visits."""

    def __init__(self):
        self.ring = None
        self.models = None
        self.session = None
        self.listener = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._shutdown = asyncio.Event()
        # Cooldown: track last processed time per camera to avoid spam
        self._last_processed: dict[str, datetime] = {}
        # Dedup: track processed Ring event IDs
        self._seen_event_ids: set[int] = set()
        self._max_seen_ids = 500  # rolling cap

    async def startup(self) -> None:
        """Initialize all components."""
        log.info("Starting Ring Watcher...")

        create_tables()
        self.session = get_session()
        self.models = load_models()
        self.ring = await ring_api.authenticate()

        # List cameras
        devices = self.ring.devices()
        for dtype in ("doorbells", "stickup_cams", "other"):
            for dev in devices.get(dtype, []):
                log.info("  Camera: %s (%s)", dev.name, dtype)

        # References
        refs = get_all_references(self.session)
        log.info("Loaded %d reference(s):", len(refs))
        for ref in refs:
            log.info("  - %s (%s)", ref.display_name, ref.category)

        # Push event listener
        self.listener = ring_api.create_event_listener(
            self.ring,
            on_motion=self._on_motion_event,
            on_ding=self._on_ding_event,
        )

        log.info("Ring Watcher ready")

    def _on_motion_event(self, event) -> None:
        self._event_queue.put_nowait(("motion", event))

    def _on_ding_event(self, event) -> None:
        self._event_queue.put_nowait(("ding", event))

    def _is_duplicate(self, event) -> bool:
        """Check if we've already processed this Ring event ID."""
        eid = getattr(event, "id", None)
        if eid is None:
            return False
        if eid in self._seen_event_ids:
            return True
        self._seen_event_ids.add(eid)
        # Keep set bounded
        if len(self._seen_event_ids) > self._max_seen_ids:
            to_remove = list(self._seen_event_ids)[: self._max_seen_ids // 2]
            self._seen_event_ids -= set(to_remove)
        return False

    def _is_on_cooldown(self, camera_name: str) -> bool:
        """Check if this camera is in cooldown period."""
        last = self._last_processed.get(camera_name)
        if last is None:
            return False
        elapsed = (datetime.now() - last).total_seconds()
        return elapsed < settings.ring.cooldown_seconds

    def _build_detection_summary(self, df_dets) -> str:
        """Build a short summary string like 'car, person x2'."""
        if len(df_dets) == 0:
            return ""
        counts = df_dets["class_name"].value_counts()
        parts = []
        for cls, count in counts.items():
            parts.append(f"{cls} x{count}" if count > 1 else cls)
        return ", ".join(parts)

    # --- Event Processing ---

    async def _process_events(self) -> None:
        while not self._shutdown.is_set():
            try:
                kind, event = await asyncio.wait_for(self._event_queue.get(), timeout=10.0)

                if self._is_duplicate(event):
                    continue

                cam_name = getattr(event, "device_name", None) or settings.ring.camera_name

                if kind == "ding":
                    snapshot = await ring_api.download_snapshot(self.ring, cam_name)
                    notify_ding(cam_name, str(snapshot) if snapshot else None)
                elif kind == "motion":
                    await self._handle_motion(event, cam_name)

            except TimeoutError:
                continue
            except Exception:
                log.exception("Error processing event")

    async def _handle_motion(self, event, cam_name: str) -> None:
        """Motion event pipeline: cooldown → visit check → snapshot → detect → match → notify."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        # --- Extend active visits ---
        active_visits = get_active_visits(self.session)
        if active_visits:
            for visit in active_visits:
                visit.arrived_at = now
            self.session.commit()
            log.info(
                "Motion extends %d active visit(s): %s",
                len(active_visits),
                ", ".join(v.display_name for v in active_visits),
            )
            return

        # --- Cooldown check (only if no active visits) ---
        if self._is_on_cooldown(cam_name):
            log.debug("Cooldown active for %s, skipping", cam_name)
            return
        self._last_processed[cam_name] = now

        log.info("Processing motion at %s (%s)", cam_name, timestamp_str)

        # --- Download snapshot ---
        snapshot_path = await ring_api.download_snapshot(self.ring, cam_name)
        if snapshot_path is None:
            notify_motion(cam_name, timestamp_str)
            return

        # Archive video in background
        asyncio.create_task(self._archive_video(cam_name))

        # --- YOLO detection ---
        paths, resized, padded = prepare_batch([str(snapshot_path)])
        if not paths:
            notify_motion(cam_name, timestamp_str, str(snapshot_path))
            return

        df_meta, df_dets, img_crops = run_detection(self.models, resized, paths)
        summary = self._build_detection_summary(df_dets)

        if len(df_dets) == 0:
            notify_motion(cam_name, timestamp_str, str(snapshot_path))
            return

        # --- Vehicle matching ---
        vehicle_classes = {2, 7}  # car, truck
        vehicle_dets = df_dets[df_dets["class_id"].isin(vehicle_classes)]
        person_dets = df_dets[df_dets["class_name"] == "person"]
        matched_refs = []

        if len(vehicle_dets) > 0 and img_crops:
            vehicle_indices = vehicle_dets.index.tolist()
            vehicle_crops = [img_crops[i] for i in vehicle_indices if i < len(img_crops)]

            if vehicle_crops:
                padded_crops = [pad_to_square(c) for c in vehicle_crops]
                embeddings = compute_yolo_embeddings(self.models, padded_crops)
                matches = match_against_references(self.session, embeddings)

                seen = set()
                for match in matches:
                    if match["reference_name"] not in seen:
                        matched_refs.append(match)
                        seen.add(match["reference_name"])

                clear_gpu_memory()

        # --- Record arrivals & notify ---
        for match in matched_refs:
            ref_name = match["reference_name"]
            display_name = match["display_name"]

            existing = get_active_visit_by_reference(self.session, ref_name)
            if existing:
                existing.arrived_at = now
                self.session.commit()
            else:
                record_arrival(self.session, ref_name, display_name, cam_name, str(snapshot_path))
                notify_arrival(display_name, cam_name, summary, str(snapshot_path))

        # --- Unmatched detections ---
        if not matched_refs:
            if len(person_dets) > 0:
                notify_unknown_visitor(cam_name, summary, str(snapshot_path))
            else:
                notify_motion(cam_name, f"{timestamp_str} — {summary}", str(snapshot_path))

    # --- Background Tasks ---

    async def _check_departures(self) -> None:
        """Check if anyone who arrived has left (no motion for timeout)."""
        timeout_secs = settings.ring.departure_timeout
        while not self._shutdown.is_set():
            await asyncio.sleep(60)
            try:
                now = datetime.now()
                for visit in get_active_visits(self.session):
                    if (now - visit.arrived_at).total_seconds() > timeout_secs:
                        duration_mins = record_departure(self.session, visit)
                        notify_departure(visit.display_name, visit.camera_name, duration_mins)
            except Exception:
                log.exception("Error checking departures")

    async def _refresh_ring_session(self) -> None:
        while not self._shutdown.is_set():
            await asyncio.sleep(3600)
            try:
                await self.ring.async_update_data()
                log.debug("Ring session refreshed")
            except Exception:
                log.exception("Failed to refresh Ring session")

    async def _archive_video(self, camera_name: str | None = None) -> None:
        try:
            await ring_api.download_latest_video(self.ring, camera_name)
        except Exception:
            log.exception("Failed to archive video")

    # --- Main Loop ---

    async def run(self) -> None:
        await self.startup()

        await self.listener.start(timeout=10)
        log.info("Event listener started (Firebase push)")

        tasks = [
            asyncio.create_task(self._process_events(), name="event-processor"),
            asyncio.create_task(self._check_departures(), name="departure-checker"),
            asyncio.create_task(self._refresh_ring_session(), name="session-refresh"),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            self._shutdown.set()
            await self.listener.stop()
            log.info("Ring Watcher stopped")


def main():
    """Entry point for ring-watch command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    watcher = RingWatcher()

    def _signal_handler(sig, frame):
        log.info("Received %s, shutting down...", signal.Signals(sig).name)
        watcher._shutdown.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        log.info("Shutting down Ring Watcher")


if __name__ == "__main__":
    main()
