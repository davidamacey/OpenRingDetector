"""Main watcher: push-based Ring events → detect → identify → track → notify.

Uses RingEventListener (Firebase push) for near-instant motion alerts.
Downloads and analyzes full video recordings for comprehensive detection,
with snapshot fallback when video is unavailable.
Tracks arrival/departure of known references (vehicles) and face-matched persons,
sending notifications with snapshot images via ntfy.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from ring_detector import ring_api
from ring_detector.captioner import caption_frames, caption_image
from ring_detector.config import settings
from ring_detector.database import (
    extend_visit,
    get_active_visit_by_reference,
    get_active_visits,
    get_all_references,
    get_session,
    match_against_face_profiles,
    match_against_references,
    record_arrival,
    record_departure,
    record_event,
    run_migrations,
    store_watcher_face_embedding,
)
from ring_detector.detector import (
    clear_gpu_memory,
    compute_clip_embeddings,
    detect_faces_simple,
    run_detection,
)
from ring_detector.image_utils import extract_key_frames, pad_to_square, prepare_batch
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


@dataclass
class FrameAnalysisResult:
    """Aggregated detection results across multiple video frames."""

    matched_refs: list[dict] = field(default_factory=list)
    face_matches: list[dict] = field(default_factory=list)
    unmatched_face_embeddings: list[list[float]] = field(default_factory=list)
    detection_summary: str = ""
    caption: str | None = None
    best_frame_path: str | None = None
    has_persons: bool = False


def _cluster_embeddings(
    embeddings: list[list[float]],
    threshold: float = 0.90,
) -> list[list[float]]:
    """Group similar embeddings and return cluster centroids.

    Simple greedy clustering — good enough for the small counts we see per video.
    """
    if not embeddings:
        return []

    arr = np.array(embeddings)
    used = set()
    centroids = []

    for i in range(len(arr)):
        if i in used:
            continue
        cluster = [arr[i]]
        used.add(i)
        for j in range(i + 1, len(arr)):
            if j in used:
                continue
            sim = float(np.dot(arr[i], arr[j]) / (np.linalg.norm(arr[i]) * np.linalg.norm(arr[j])))
            if sim > threshold:
                cluster.append(arr[j])
                used.add(j)
        centroid = np.mean(cluster, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # Re-normalize
        centroids.append(centroid.tolist())

    return centroids


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

        run_migrations()
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

    async def _wait_or_shutdown(self, seconds: float) -> bool:
        """Sleep for `seconds` or until shutdown. Returns True if shutdown was requested."""
        try:
            await asyncio.wait_for(asyncio.shield(self._shutdown.wait()), timeout=seconds)
            return True
        except TimeoutError:
            return False

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
                    snap_str = str(snap) if snap else None
                    record_event(
                        self.session,
                        event_type="ding",
                        camera_name=cam,
                        snapshot_path=snap_str,
                    )
                    notify_ding(cam, snap_str)
                elif kind == "motion":
                    await self._handle_motion(cam)

            except TimeoutError:
                continue
            except Exception:
                log.exception("Error processing event")

    def _analyze_frames(
        self,
        frames: list[np.ndarray],
        frame_labels: list[str],
        cam_name: str,
    ) -> FrameAnalysisResult:
        """Run YOLO/CLIP/face detection across multiple frames, aggregate and deduplicate.

        YOLO runs as a single batch for speed (~6x faster than sequential).
        CLIP and face detection run per-frame on detected crops.
        """
        result = FrameAnalysisResult()

        all_vehicle_embeddings: list[list[float]] = []
        all_face_embeddings: list[list[float]] = []
        all_class_counts: dict[str, int] = {}
        best_frame_idx = 0
        best_frame_det_count = 0

        # Prepare all frames for batched YOLO
        resized_all = [cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))) for f in frames]
        padded_all = [pad_to_square(f) for f in frames]

        # Batched YOLO detection
        _meta, df_dets, img_crops = run_detection(self.models, resized_all, frame_labels)

        # Process YOLO results per frame for vehicle/face analysis
        if len(df_dets) > 0:
            for i, frame_label in enumerate(frame_labels):
                # Get detections for this frame via file_uuid linkage
                frame_meta = _meta[_meta["path"] == frame_label]
                if frame_meta.empty:
                    continue
                frame_uuid = frame_meta["file_uuid"].iloc[0]
                frame_dets = df_dets[df_dets["file_uuid"] == frame_uuid]

                # Track best frame
                if len(frame_dets) > best_frame_det_count:
                    best_frame_det_count = len(frame_dets)
                    best_frame_idx = i

                # Accumulate class counts
                for cls_name in frame_dets["class_name"]:
                    all_class_counts[cls_name] = all_class_counts.get(cls_name, 0) + 1
                if "person" in frame_dets["class_name"].values:
                    result.has_persons = True

                # Collect vehicle crop embeddings
                vehicle_dets = frame_dets[frame_dets["class_id"].isin({2, 7})]
                if len(vehicle_dets) > 0 and img_crops:
                    indices = vehicle_dets.index.tolist()
                    crops = [img_crops[j] for j in indices if j < len(img_crops)]
                    if crops:
                        padded_crops = [pad_to_square(c) for c in crops]
                        embs = compute_clip_embeddings(self.models, padded_crops)
                        all_vehicle_embeddings.extend(embs)

        clear_gpu_memory()

        # Face detection per frame (SCRFD doesn't batch well)
        if settings.face.enabled and self.models.face_detector is not None:
            for padded in padded_all:
                _face_crops, face_embs = detect_faces_simple(
                    self.models, padded, settings.face.min_face_size
                )
                all_face_embeddings.extend(face_embs)
            clear_gpu_memory()

        # Save best frame as snapshot
        if frames:
            best_frame = frames[best_frame_idx]
            snap_dir = settings.storage.snapshot_dir()
            date_dir = snap_dir / datetime.now().strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            best_path = date_dir / f"{cam_name}_best_{ts}.jpg"
            cv2.imwrite(str(best_path), best_frame)
            result.best_frame_path = str(best_path)
            log.info("Best frame saved: %s (%d detections)", best_path, best_frame_det_count)

        # Cluster and match vehicle embeddings
        if all_vehicle_embeddings:
            clustered = _cluster_embeddings(all_vehicle_embeddings, threshold=0.90)
            matches = match_against_references(self.session, clustered)
            seen = set()
            for m in matches:
                if m["reference_name"] not in seen:
                    result.matched_refs.append(m)
                    seen.add(m["reference_name"])

        # Cluster and match face embeddings
        if all_face_embeddings:
            clustered_faces = _cluster_embeddings(all_face_embeddings, threshold=0.80)
            result.face_matches = match_against_face_profiles(
                self.session, clustered_faces, settings.face.match_threshold
            )
            matched_indices = {m["vector_index"] for m in result.face_matches}
            result.unmatched_face_embeddings = [
                emb for i, emb in enumerate(clustered_faces) if i not in matched_indices
            ]
            log.info(
                "Video face detection: %d face cluster(s), %d matched",
                len(clustered_faces),
                len(result.face_matches),
            )

        # Build detection summary from aggregated counts (deduplicated via clustering)
        if all_class_counts:
            # Use unique entity count (vehicles from clusters, others from max per-frame)
            summary_parts = []
            for cls, _count in sorted(all_class_counts.items(), key=lambda x: -x[1]):
                # For vehicles, use cluster count; for others, cap at reasonable max
                if cls in ("car", "truck"):
                    n = len([r for r in result.matched_refs]) or max(
                        1, len(all_vehicle_embeddings) // max(1, len(frames))
                    )
                else:
                    n = min(_count, len(frames))  # Rough unique count
                    n = max(1, n // max(1, len(frames)))  # Average per frame
                if n > 1:
                    summary_parts.append(f"{cls} x{n}")
                else:
                    summary_parts.append(cls)
            result.detection_summary = ", ".join(summary_parts)

        # VLM caption — send multiple frames for richer context
        if result.detection_summary and frames:
            # Save key frames as temp JPGs for the captioner
            frame_paths = []
            step = max(1, len(frames) // 5)
            for idx in range(0, len(frames), step):
                tmp_path = settings.storage.snapshot_dir() / f"_caption_tmp_{idx}.jpg"
                cv2.imwrite(str(tmp_path), frames[idx])
                frame_paths.append(str(tmp_path))

            cap = caption_frames(frame_paths, max_frames=5)
            # Clean up temp files
            for p in frame_paths:
                with contextlib.suppress(OSError):
                    os.unlink(p)

            if not cap and result.best_frame_path:
                cap = caption_image(result.best_frame_path)
            if cap:
                result.caption = cap
                result.detection_summary = cap

        return result

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

        # Start snapshot download in background (fallback if video fails)
        snapshot_task = asyncio.create_task(ring_api.download_snapshot(self.ring, cam_name))

        # --- Video-first path ---
        if settings.video.enabled:
            log.info("Waiting for video from %s...", cam_name)
            video_path = await ring_api.download_video_with_retry(
                self.ring,
                cam_name,
                timeout=settings.video.wait_timeout,
                retry_delay=settings.video.retry_delay,
                shutdown_event=self._shutdown,
            )

            if video_path:
                key_frames = extract_key_frames(
                    video_path,
                    frame_interval=settings.video.frame_interval,
                    max_frames=settings.video.max_frames,
                )
                if key_frames:
                    frame_images = [f[1] for f in key_frames]
                    frame_labels = [f"{cam_name}:frame{f[0]}" for f in key_frames]

                    analysis = self._analyze_frames(frame_images, frame_labels, cam_name)
                    snap_str = analysis.best_frame_path

                    # Cancel snapshot task — we have a better frame
                    snapshot_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await snapshot_task

                    self._apply_analysis_results(analysis, cam_name, ts, snap_str)
                    return

            log.warning("Video unavailable for %s, falling back to snapshot", cam_name)

        # --- Snapshot fallback ---
        snapshot_path = await snapshot_task
        snap_str = str(snapshot_path) if snapshot_path else None
        if not snapshot_path:
            notify_motion(cam_name, ts)
            return

        # Single-frame detection (original behavior)
        paths, resized, padded = prepare_batch([snap_str])
        if not paths:
            notify_motion(cam_name, ts, snap_str)
            return

        _meta, df_dets, img_crops = run_detection(self.models, resized, paths)
        summary = self._detection_summary(df_dets)

        caption = caption_image(snapshot_path) if len(df_dets) > 0 else None
        if caption:
            summary = caption

        if len(df_dets) == 0:
            notify_motion(cam_name, ts, snap_str)
            return

        # Build a FrameAnalysisResult from single-frame detection
        analysis = FrameAnalysisResult(
            detection_summary=summary,
            caption=caption,
            best_frame_path=snap_str,
            has_persons="person" in df_dets["class_name"].values if len(df_dets) > 0 else False,
        )

        # Vehicle matching
        vehicle_dets = df_dets[df_dets["class_id"].isin({2, 7})]
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
                        analysis.matched_refs.append(m)
                        seen.add(m["reference_name"])
                clear_gpu_memory()

        # Face detection
        if settings.face.enabled and self.models.face_detector is not None:
            face_crops, face_embeddings = detect_faces_simple(
                self.models, padded[0], settings.face.min_face_size
            )
            if face_embeddings:
                analysis.face_matches = match_against_face_profiles(
                    self.session, face_embeddings, settings.face.match_threshold
                )
                matched_indices = {m["vector_index"] for m in analysis.face_matches}
                analysis.unmatched_face_embeddings = [
                    emb for i, emb in enumerate(face_embeddings) if i not in matched_indices
                ]

        self._apply_analysis_results(analysis, cam_name, ts, snap_str)

    def _apply_analysis_results(
        self,
        analysis: FrameAnalysisResult,
        cam_name: str,
        ts: str,
        snap_str: str | None,
    ) -> None:
        """Apply detection results: record events, track visits, send notifications."""
        summary = analysis.detection_summary
        caption = analysis.caption

        # Store unmatched face embeddings for later labeling
        for emb in analysis.unmatched_face_embeddings:
            try:
                store_watcher_face_embedding(self.session, snap_str, emb)
            except Exception:
                log.warning("Failed to store face embedding (non-fatal)", exc_info=True)

        # Build combined summary with face names
        known_face_names = [m["display_name"] for m in analysis.face_matches]
        if known_face_names and summary:
            combined_summary = f"{summary}, {', '.join(known_face_names)}"
        elif known_face_names:
            combined_summary = ", ".join(known_face_names)
        else:
            combined_summary = summary

        # --- Vehicle arrivals ---
        for m in analysis.matched_refs:
            existing = get_active_visit_by_reference(self.session, m["reference_name"])
            if existing:
                extend_visit(self.session, existing)
                record_event(
                    self.session,
                    event_type="motion",
                    camera_name=cam_name,
                    snapshot_path=snap_str,
                    detection_summary=summary,
                    reference_name=m["reference_name"],
                    display_name=m["display_name"],
                    visit_event_id=existing.id,
                    caption=caption,
                )
            else:
                visit = record_arrival(
                    self.session,
                    m["reference_name"],
                    m["display_name"],
                    cam_name,
                    snap_str,
                )
                record_event(
                    self.session,
                    event_type="arrival",
                    camera_name=cam_name,
                    snapshot_path=snap_str,
                    detection_summary=summary,
                    reference_name=m["reference_name"],
                    display_name=m["display_name"],
                    visit_event_id=visit.id,
                    caption=caption,
                )
                notify_arrival(m["display_name"], cam_name, combined_summary, snap_str)

        # --- Face-only arrivals (no vehicle match) ---
        if not analysis.matched_refs:
            seen_faces: set[str] = set()
            for fm in analysis.face_matches:
                if fm["profile_name"] in seen_faces:
                    continue
                seen_faces.add(fm["profile_name"])
                face_ref = f"face:{fm['profile_name']}"
                existing = get_active_visit_by_reference(self.session, face_ref)
                if existing:
                    extend_visit(self.session, existing)
                else:
                    visit = record_arrival(
                        self.session, face_ref, fm["display_name"], cam_name, snap_str
                    )
                    record_event(
                        self.session,
                        event_type="arrival",
                        camera_name=cam_name,
                        snapshot_path=snap_str,
                        detection_summary=summary,
                        display_name=fm["display_name"],
                        visit_event_id=visit.id,
                        caption=caption,
                    )
                    notify_known_person(fm["display_name"], cam_name, summary, snap_str)

        # --- Unknown visitor (no known vehicle or face) ---
        if not analysis.matched_refs and not analysis.face_matches:
            record_event(
                self.session,
                event_type="motion",
                camera_name=cam_name,
                snapshot_path=snap_str,
                detection_summary=summary,
                caption=caption,
            )
            if analysis.has_persons or analysis.unmatched_face_embeddings:
                notify_unknown_visitor(cam_name, summary or ts, snap_str)
            else:
                notify_motion(cam_name, f"{ts} — {summary}" if summary else ts, snap_str)

    # --- Background Tasks ---

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
                        record_event(
                            self.session,
                            event_type="departure",
                            camera_name=v.camera_name,
                            snapshot_path=v.snapshot_path,
                            reference_name=v.reference_name,
                            display_name=v.display_name,
                            visit_event_id=v.id,
                        )
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

    async def _monitor_listener(self) -> None:
        """Restart the Firebase listener if it drops (network hiccup, etc.)."""
        while not self._shutdown.is_set():
            if await self._wait_or_shutdown(300):
                break
            if self.listener and not self.listener.started:
                log.warning("Event listener not active — attempting restart")
                try:
                    await self.listener.start(timeout=10)
                    log.info("Event listener restarted")
                except Exception:
                    log.exception("Failed to restart event listener")

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
            raise RuntimeError(
                "Failed to start Ring event listener (FCM registration failed). "
                "Check your Ring token and network connection."
            )
        log.info("Listening for events (Firebase push)")

        tasks = [
            asyncio.create_task(self._process_events(), name="events"),
            asyncio.create_task(self._check_departures(), name="departures"),
            asyncio.create_task(self._refresh_ring_session(), name="refresh"),
            asyncio.create_task(self._monitor_listener(), name="monitor"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            self._shutdown.set()
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.listener:
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
