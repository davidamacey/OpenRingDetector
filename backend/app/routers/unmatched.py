"""Unmatched detections router."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.schemas import UnmatchedFace, UnmatchedVehicle
from ring_detector.database import Event, FaceEmbedding, get_session

router = APIRouter(tags=["unmatched"])


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("/unmatched/vehicles", response_model=list[UnmatchedVehicle])
def unmatched_vehicles(
    min_sightings: int = Query(1, ge=1),
    days: int = Query(30, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(_get_db),
):
    """Return motion events without a known reference_name as proxy for unmatched vehicles."""
    cutoff = datetime.now() - timedelta(days=days)
    rows = (
        session.query(Event)
        .filter(
            Event.event_type == "motion",
            Event.reference_name.is_(None),
            Event.snapshot_path.isnot(None),
            Event.occurred_at >= cutoff,
        )
        .order_by(Event.occurred_at.desc())
        .limit(limit)
        .all()
    )

    # Return each unmatched event as its own "cluster" for simplicity
    result = []
    for ev in rows:
        result.append(
            UnmatchedVehicle(
                cluster_id=str(ev.id),
                sighting_count=1,
                first_seen=ev.occurred_at,
                last_seen=ev.occurred_at,
                camera_name=ev.camera_name,
                representative_crop_url=(
                    f"/api/images/{ev.snapshot_path}" if ev.snapshot_path else None
                ),
                all_crop_urls=([f"/api/images/{ev.snapshot_path}"] if ev.snapshot_path else []),
            )
        )
    return result


@router.get("/unmatched/faces", response_model=list[UnmatchedFace])
def unmatched_faces(
    days: int = Query(30, ge=1),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(_get_db),
):
    """Return face embeddings with person_name == 'unknown'."""
    rows = (
        session.query(FaceEmbedding)
        .filter(FaceEmbedding.person_name == "unknown")
        .limit(limit)
        .all()
    )
    return [
        UnmatchedFace(
            face_embedding_uuid=r.uuid,
            sighting_count=1,
            last_seen=datetime.now(),
            camera_name="unknown",
            crop_url=f"/api/images/{r.img_path}" if r.img_path else None,
        )
        for r in rows
    ]


@router.post("/unmatched/dismiss")
def dismiss_unmatched(body: dict, session: Session = Depends(_get_db)):
    # Minimal implementation — dismiss is a no-op for now
    return {"success": True}
