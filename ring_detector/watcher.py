"""Main watcher: push-based Ring events → detect → identify → track → notify.

Uses RingEventListener (Firebase push) for near-instant motion alerts.
Tracks arrival/departure of known references (cleaner, yard guy, etc.)
and sends "arrived" and "time to pay" notifications.
"""

from __future__ import annotations

import asyncio
import logging
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

    async def startup(self) -> None:
        """Initialize all components."""
        log.info("Starting Ring Watcher...")

        # Database
        create_tables()
        self.session = get_session()

        # ML models
        self.models = load_models()

        # Ring API
        self.ring = await ring_api.authenticate()

        # Show loaded references
        refs = get_all_references(self.session)
        log.info("Loaded %d reference(s):", len(refs))
        for ref in refs:
            log.info("  - %s (%s)", ref.display_name, ref.category)

        # Push event listener (Firebase)
        self.listener = ring_api.create_event_listener(
            self.ring,
            on_motion=self._on_motion_event,
            on_ding=self._on_ding_event,
        )

        log.info("Ring Watcher ready — listening for push events")

    def _on_motion_event(self, event) -> None:
        """Callback from RingEventListener — puts event on the async queue."""
        self._event_queue.put_nowait(("motion", event))

    def _on_ding_event(self, event) -> None:
        """Callback from RingEventListener — puts event on the async queue."""
        self._event_queue.put_nowait(("ding", event))

    async def _process_events(self) -> None:
        """Process events from the push listener queue."""
        while not self._shutdown.is_set():
            try:
                kind, event = await asyncio.wait_for(self._event_queue.get(), timeout=10.0)
                if kind == "motion":
                    await self._handle_motion(event)
                elif kind == "ding":
                    cam_name = getattr(event, "device_name", settings.ring.camera_name)
                    notify_motion(cam_name, "Doorbell ring!")
            except TimeoutError:
                continue
            except Exception:
                log.exception("Error processing event")

    async def _check_departures(self) -> None:
        """Periodically check if anyone who arrived has left."""
        timeout_secs = settings.ring.departure_timeout
        while not self._shutdown.is_set():
            await asyncio.sleep(60)  # Check every minute
            try:
                active_visits = get_active_visits(self.session)
                now = datetime.now()
                for visit in active_visits:
                    elapsed = (now - visit.arrived_at).total_seconds()
                    if elapsed > timeout_secs:
                        duration_mins = record_departure(self.session, visit)
                        notify_departure(
                            visit.display_name,
                            visit.camera_name,
                            duration_mins,
                        )
                        log.info(
                            "%s departed after %d min — departure notification sent",
                            visit.display_name,
                            duration_mins,
                        )
            except Exception:
                log.exception("Error checking departures")

    async def _refresh_ring_session(self) -> None:
        """Periodically refresh Ring data to keep session alive."""
        while not self._shutdown.is_set():
            await asyncio.sleep(3600)  # Every hour
            try:
                await self.ring.async_update_data()
                log.debug("Ring session refreshed")
            except Exception:
                log.exception("Failed to refresh Ring session")

    async def _handle_motion(self, event) -> None:
        """Handle a motion event: snapshot → detect → identify → track → notify."""
        cam_name = getattr(event, "device_name", settings.ring.camera_name) or "Camera"
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        log.info("Motion at %s (%s)", cam_name, timestamp_str)

        # First check: is someone we're tracking still here?
        # If so, this motion extends their visit (reset departure timer)
        active_visits = get_active_visits(self.session)
        if active_visits:
            # Update arrival time to push back departure timeout
            for visit in active_visits:
                visit.arrived_at = timestamp
                self.session.commit()
                log.info("Extended visit for %s (still present)", visit.display_name)
            # Don't re-process if we already know who's there
            # unless it's been a while since the last detection
            return

        # Download snapshot for detection
        snapshot_path = await ring_api.download_snapshot(self.ring)
        if snapshot_path is None:
            log.warning("No snapshot available")
            notify_motion(cam_name, timestamp_str)
            return

        # Archive video in background (don't block detection)
        asyncio.create_task(self._archive_video())

        # Run YOLO detection
        paths, resized, padded = prepare_batch([str(snapshot_path)])
        if not paths:
            notify_motion(cam_name, timestamp_str)
            return

        df_meta, df_dets, img_crops = run_detection(self.models, resized, paths)

        if len(df_dets) == 0:
            log.info("No objects detected in snapshot")
            notify_motion(cam_name, timestamp_str)
            return

        # Check vehicles against references
        vehicle_classes = {2, 7}  # car, truck in COCO
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
                # Deduplicate by reference name
                seen = set()
                for match in matches:
                    if match["reference_name"] not in seen:
                        matched_refs.append(match)
                        seen.add(match["reference_name"])

                clear_gpu_memory()

        # Handle matches — record arrivals
        for match in matched_refs:
            ref_name = match["reference_name"]
            display_name = match["display_name"]

            # Check if already tracked as active visit
            existing = get_active_visit_by_reference(self.session, ref_name)
            if existing:
                # Already here, just extend
                existing.arrived_at = timestamp
                self.session.commit()
                log.info("%s still present (visit extended)", display_name)
            else:
                # New arrival!
                record_arrival(
                    self.session,
                    ref_name,
                    display_name,
                    cam_name,
                    str(snapshot_path),
                )
                notify_arrival(display_name, cam_name)

        # Handle unmatched detections
        if not matched_refs:
            if len(person_dets) > 0:
                notify_unknown_visitor(cam_name)
            else:
                classes_found = df_dets["class_name"].unique().tolist()
                notify_motion(
                    cam_name,
                    f"{timestamp_str} — detected: {', '.join(classes_found)}",
                )

    async def _archive_video(self) -> None:
        """Download and archive the latest video to NAS."""
        try:
            await ring_api.download_latest_video(self.ring)
        except Exception:
            log.exception("Failed to archive video")

    async def run(self) -> None:
        """Start the event listener and all background tasks."""
        await self.startup()

        # Start the push event listener
        await self.listener.start(timeout=10)
        log.info("Event listener started (Firebase push)")

        # Run concurrent tasks
        tasks = [
            asyncio.create_task(self._process_events()),
            asyncio.create_task(self._check_departures()),
            asyncio.create_task(self._refresh_ring_session()),
        ]

        try:
            await asyncio.gather(*tasks)
        finally:
            self._shutdown.set()
            await self.listener.stop()
            log.info("Event listener stopped")


def main():
    """Entry point for ring-watch command."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    watcher = RingWatcher()
    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        log.info("Shutting down Ring Watcher")


if __name__ == "__main__":
    main()
