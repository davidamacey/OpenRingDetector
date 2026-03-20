"""Main watcher: push-based Ring events → detect → identify → track → notify.

Uses RingEventListener (Firebase push) for near-instant motion alerts.
Tracks arrival/departure of known references (vehicles) and face-matched persons,
sending notifications with snapshot images via ntfy.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ring_detector import ring_api
from ring_detector.captioner import caption_image
from ring_detector.config import settings
from ring_detector.database import (
    create_tables,
    extend_visit,
    get_active_visit_by_reference,
    get_active_visits,
    get_all_references,
    get_session,
    match_against_face_profiles,
    match_against_references,
    record_arrival,
    record_departure,
    store_watcher_face_embedding,
)
from ring_detector.detector import (
    clear_gpu_memory,
    compute_clip_embeddings,
    detect_faces_simple,
    run_detection,
)
from ring_detector.image_utils import pad_to_square, prepare_batch
from ring_detector.models import load_models
from ring_detector.notifications import (
    notify_arrival,
    notify_departure,
    notify_ding,
    notify_known_person,
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
        self._last_processed: dict[str, datetime] = {}
        self._seen_event_ids: set[int] = set()
        self._max_seen_ids = 500

    async def startup(self) -> None:
        log.info("Starting Ring Watcher...")

        create_tables()
        self.session = get_session()
        self.models = load_models()
        self.ring = await ring_api.authenticate()

        for dev in self.ring.video_devices():
            log.info("  Camera: %s", dev.name)

        refs = get_all_references(self.session)
        log.info("References: %d loaded", len(refs))
        for ref in refs:
            log.info("  - %s (%s)", ref.display_name, ref.category)

        if settings.captioner.enabled:
            from ring_detector.captioner import is_available

            if is_available():
                log.info("Captioner: %s via Ollama", settings.captioner.model)
            else:
                log.warning("Captioner enabled but not available")

        face_enabled = settings.face.enabled and self.models.face_detector is not None
        log.info(
            "Face detection: %s (threshold=%.2f, min_size=%dpx)",
            "enabled" if face_enabled else "disabled",
            settings.face.match_threshold,
            settings.face.min_face_size,
        )

        self.listener = ring_api.create_event_listener(
            self.ring,
            on_motion=lambda e: self._event_queue.put_nowait(("motion", e)),
            on_ding=lambda e: self._event_queue.put_nowait(("ding", e)),
        )
        log.info("Ring Watcher ready")

    def _is_duplicate(self, event) -> bool:
        eid = getattr(event, "id", None)
        if eid is None:
            return False
        if eid in self._seen_event_ids:
            return True
        self._seen_event_ids.add(eid)
        if len(self._seen_event_ids) > self._max_seen_ids:
            self._seen_event_ids = set(list(self._seen_event_ids)[self._max_seen_ids // 2 :])
        return False

    def _is_on_cooldown(self, camera_name: str) -> bool:
        last = self._last_processed.get(camera_name)
        if last is None:
            return False
        return (datetime.now() - last).total_seconds() < settings.ring.cooldown_seconds

    @staticmethod
    def _detection_summary(df_dets) -> str:
        if len(df_dets) == 0:
            return ""
        counts = df_dets["class_name"].value_counts()
        return ", ".join(f"{cls} x{n}" if n > 1 else cls for cls, n in counts.items())

    # --- Event Processing ---

    async def _process_events(self) -> None:
        while not self._shutdown.is_set():
            try:
                kind, event = await asyncio.wait_for(self._event_queue.get(), timeout=10.0)
                if self._is_duplicate(event):
                    continue

                cam = getattr(event, "device_name", None) or settings.ring.camera_name or "Camera"

                if kind == "ding":
                    snap = await ring_api.download_snapshot(self.ring, cam)
                    notify_ding(cam, str(snap) if snap else None)
                elif kind == "motion":
                    await self._handle_motion(cam)

            except TimeoutError:
                continue
            except Exception:
                log.exception("Error processing event")

    async def _handle_motion(self, cam_name: str) -> None:
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")

        # Extend active visits instead of re-detecting
        active = get_active_visits(self.session)
        if active:
            for v in active:
                extend_visit(self.session, v)
            log.info(
                "Motion extends %d visit(s): %s",
                len(active),
                ", ".join(v.display_name for v in active),
            )
            return

        if self._is_on_cooldown(cam_name):
            log.debug("Cooldown active for %s", cam_name)
            return
        self._last_processed[cam_name] = now

        log.info("Processing motion at %s (%s)", cam_name, ts)

        # Snapshot
        snapshot_path = await ring_api.download_snapshot(self.ring, cam_name)
        snap_str = str(snapshot_path) if snapshot_path else None
        if not snapshot_path:
            notify_motion(cam_name, ts)
            return

        # Archive video in background
        asyncio.create_task(self._archive_video(cam_name))

        # Detection
        paths, resized, padded = prepare_batch([snap_str])
        if not paths:
            notify_motion(cam_name, ts, snap_str)
            return

        _meta, df_dets, img_crops = run_detection(self.models, resized, paths)
        summary = self._detection_summary(df_dets)

        # VLM caption (replaces class list when available)
        caption = caption_image(snapshot_path) if len(df_dets) > 0 else None
        if caption:
            summary = caption

        if len(df_dets) == 0:
            notify_motion(cam_name, ts, snap_str)
            return

        # Vehicle matching
        vehicle_dets = df_dets[df_dets["class_id"].isin({2, 7})]
        person_dets = df_dets[df_dets["class_name"] == "person"]
        matched_refs = []

        if len(vehicle_dets) > 0 and img_crops:
            indices = vehicle_dets.index.tolist()
            crops = [img_crops[i] for i in indices if i < len(img_crops)]
            if crops:
                padded_crops = [pad_to_square(c) for c in crops]
                embeddings = compute_clip_embeddings(self.models, padded_crops)
                matches = match_against_references(self.session, embeddings)
                seen = set()
                for m in matches:
                    if m["reference_name"] not in seen:
                        matched_refs.append(m)
                        seen.add(m["reference_name"])
                clear_gpu_memory()

        # Face detection + matching
        face_matches: list[dict] = []
        unmatched_face_embeddings: list[list[float]] = []

        if settings.face.enabled and self.models.face_detector is not None:
            face_crops, face_embeddings = detect_faces_simple(
                self.models, padded[0], settings.face.min_face_size
            )
            if face_embeddings:
                face_matches = match_against_face_profiles(
                    self.session, face_embeddings, settings.face.match_threshold
                )
                matched_indices = {m["vector_index"] for m in face_matches}
                unmatched_face_embeddings = [
                    emb for i, emb in enumerate(face_embeddings) if i not in matched_indices
                ]
                log.info(
                    "Face detection: %d face(s) found, %d matched",
                    len(face_embeddings),
                    len(face_matches),
                )

        # Store unmatched face embeddings for later labeling
        for emb in unmatched_face_embeddings:
            try:
                store_watcher_face_embedding(self.session, snap_str, emb)
            except Exception:
                log.warning("Failed to store face embedding (non-fatal)", exc_info=True)

        # Build combined summary: append known face names to detection summary
        known_face_names = [m["display_name"] for m in face_matches]
        if known_face_names and summary:
            combined_summary = f"{summary}, {', '.join(known_face_names)}"
        elif known_face_names:
            combined_summary = ", ".join(known_face_names)
        else:
            combined_summary = summary

        # --- Vehicle arrivals (with face names folded into summary) ---
        for m in matched_refs:
            existing = get_active_visit_by_reference(self.session, m["reference_name"])
            if existing:
                extend_visit(self.session, existing)
            else:
                record_arrival(
                    self.session,
                    m["reference_name"],
                    m["display_name"],
                    cam_name,
                    snap_str,
                )
                notify_arrival(m["display_name"], cam_name, combined_summary, snap_str)

        # --- Face-only arrivals (no vehicle match) ---
        if not matched_refs:
            seen_faces: set[str] = set()
            for fm in face_matches:
                if fm["profile_name"] in seen_faces:
                    continue
                seen_faces.add(fm["profile_name"])
                face_ref = f"face:{fm['profile_name']}"
                existing = get_active_visit_by_reference(self.session, face_ref)
                if existing:
                    extend_visit(self.session, existing)
                else:
                    record_arrival(
                        self.session, face_ref, fm["display_name"], cam_name, snap_str
                    )
                    notify_known_person(fm["display_name"], cam_name, summary, snap_str)

        # --- Unknown visitor (no known vehicle or face) ---
        if not matched_refs and not face_matches:
            if len(person_dets) > 0 or unmatched_face_embeddings:
                notify_unknown_visitor(cam_name, summary, snap_str)
            else:
                notify_motion(cam_name, f"{ts} — {summary}", snap_str)

    # --- Background Tasks ---

    async def _wait_or_shutdown(self, seconds: float) -> bool:
        """Sleep for `seconds`, waking early if shutdown is requested.

        Returns True if shutdown was triggered, False if sleep completed normally.
        """
        try:
            await asyncio.wait_for(self._shutdown.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    async def _check_departures(self) -> None:
        timeout = settings.ring.departure_timeout
        while not self._shutdown.is_set():
            if await self._wait_or_shutdown(60):
                break
            try:
                now = datetime.now()
                for v in get_active_visits(self.session):
                    elapsed = (now - v.last_motion_at).total_seconds()
                    if elapsed > timeout:
                        dur = record_departure(self.session, v)
                        notify_departure(v.display_name, v.camera_name, dur)
            except Exception:
                log.exception("Error checking departures")

    async def _refresh_ring_session(self) -> None:
        while not self._shutdown.is_set():
            if await self._wait_or_shutdown(3600):
                break
            try:
                await self.ring.async_update_data()
            except Exception:
                log.exception("Failed to refresh Ring session")

    async def _archive_video(self, camera_name: str | None = None) -> None:
        try:
            await ring_api.download_latest_video(self.ring, camera_name)
        except Exception:
            log.exception("Failed to archive video")

    # --- Main ---

    async def run(self) -> None:
        await self.startup()
        started = await self.listener.start(timeout=10)
        if not started:
            raise RuntimeError("Firebase listener failed to start (FCM registration failed)")
        log.info("Listening for events (Firebase push)")

        tasks = [
            asyncio.create_task(self._process_events(), name="events"),
            asyncio.create_task(self._check_departures(), name="departures"),
            asyncio.create_task(self._refresh_ring_session(), name="refresh"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._shutdown.set()
            await self.listener.stop()
            log.info("Ring Watcher stopped")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    watcher = RingWatcher()

    def _shutdown(sig, _):
        log.info("Received %s, shutting down...", signal.Signals(sig).name)
        watcher._shutdown.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        log.info("Shutting down")


if __name__ == "__main__":
    main()
