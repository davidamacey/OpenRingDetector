"""Main watcher loop: poll Ring for motion → detect → identify → notify.

This is the primary entry point for the ring-detector service.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from ring_detector import ring_api
from ring_detector.config import settings
from ring_detector.database import (
    create_tables,
    get_all_references,
    get_session,
    match_against_references,
)
from ring_detector.detector import (
    clear_gpu_memory,
    compute_yolo_embeddings,
    run_detection,
)
from ring_detector.image_utils import pad_to_square, prepare_batch
from ring_detector.models import load_models
from ring_detector.notifications import (
    notify_motion,
    notify_unknown_visitor,
    notify_vehicle_detected,
)

log = logging.getLogger(__name__)


class RingWatcher:
    """Watches Ring cameras for motion and processes events."""

    def __init__(self):
        self.ring = None
        self.models = None
        self.session = None
        self.last_event_id: int | None = None

    async def startup(self) -> None:
        """Initialize all components."""
        log.info("Starting Ring Watcher...")

        # Database (PostgreSQL + pgvector)
        create_tables()
        self.session = get_session()

        # Models
        self.models = load_models()

        # Ring API
        self.ring = await ring_api.authenticate()

        # Verify we have references loaded
        refs = get_all_references(self.session)
        log.info("Loaded %d reference vectors from database", len(refs))
        for ref in refs:
            log.info("  - %s (%s)", ref.display_name, ref.category)

        log.info("Ring Watcher initialized — monitoring for motion events")

    async def check_for_motion(self) -> None:
        """Poll Ring for new motion events and process them."""
        try:
            await self.ring.async_update_data()
            new_events = await ring_api.get_new_motion_events(
                self.ring, since_id=self.last_event_id
            )

            if not new_events:
                return

            log.info("Found %d new motion event(s)", len(new_events))
            self.last_event_id = new_events[0]["id"]

            for event in reversed(new_events):  # Process oldest first
                await self._process_motion_event(event)

        except Exception:
            log.exception("Error checking for motion events")

    async def _process_motion_event(self, event: dict) -> None:
        """Process a single motion event: download → detect → identify → notify."""
        created_at = event["created_at"]
        timestamp_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        cam_name = settings.ring.camera_name or "Camera"
        log.info("Processing motion event from %s", timestamp_str)

        # Download the video (archived to NAS)
        await ring_api.download_latest_video(self.ring)

        # Grab a snapshot for quick detection
        snapshot_path = await ring_api.download_snapshot(self.ring)

        if snapshot_path is None:
            log.warning("No snapshot available for motion event")
            notify_motion(cam_name, timestamp_str)
            return

        # Run detection on the snapshot
        paths, resized, padded = prepare_batch([str(snapshot_path)])
        if not paths:
            notify_motion(cam_name, timestamp_str)
            return

        df_meta, df_dets, img_crops = run_detection(self.models, resized, paths)

        if len(df_dets) == 0:
            log.info("No objects detected in snapshot")
            notify_motion(cam_name, timestamp_str)
            return

        # Check detected vehicles against references in pgvector
        vehicle_classes = {2, 7}  # car, truck
        vehicle_dets = df_dets[df_dets["class_id"].isin(vehicle_classes)]
        person_dets = df_dets[df_dets["class_name"] == "person"]

        matched_anyone = False

        if len(vehicle_dets) > 0 and img_crops:
            vehicle_indices = vehicle_dets.index.tolist()
            vehicle_crops = [img_crops[i] for i in vehicle_indices if i < len(img_crops)]

            if vehicle_crops:
                padded_crops = [pad_to_square(c) for c in vehicle_crops]
                embeddings = compute_yolo_embeddings(self.models, padded_crops)

                # Match against all references in PostgreSQL
                matches = match_against_references(self.session, embeddings)
                for match in matches:
                    log.info(
                        "Matched %s (similarity: %.3f)",
                        match["display_name"],
                        match["similarity"],
                    )
                    notify_vehicle_detected(match["display_name"], cam_name)
                    matched_anyone = True

                clear_gpu_memory()

        if len(person_dets) > 0:
            log.info("Person(s) detected: %d", len(person_dets))
            if not matched_anyone:
                notify_unknown_visitor(cam_name)
                matched_anyone = True

        if not matched_anyone:
            classes_found = df_dets["class_name"].unique().tolist()
            notify_motion(cam_name, f"{timestamp_str} — detected: {', '.join(classes_found)}")

    async def run(self) -> None:
        """Main polling loop."""
        await self.startup()
        poll_interval = settings.ring.poll_interval_seconds

        log.info("Polling every %d seconds", poll_interval)
        while True:
            await self.check_for_motion()
            await asyncio.sleep(poll_interval)


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
