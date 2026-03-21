"""Events router — GET /api/events, GET /api/events/{id}."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.schemas import DetectionResponse, EventListResponse, EventResponse
from ring_detector.database import Detection, Event, get_session

router = APIRouter(tags=["events"])


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _snapshot_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"/api/images/{quote(path, safe='/')}"


def _build_event_response(event: Event, session: Session) -> EventResponse:
    detections: list[DetectionResponse] = []
    if event.file_uuid:
        rows = session.query(Detection).filter_by(file_uuid=event.file_uuid).all()
        detections = [DetectionResponse.model_validate(d) for d in rows]

    return EventResponse(
        id=event.id,
        event_type=event.event_type,
        camera_name=event.camera_name,
        occurred_at=event.occurred_at,
        snapshot_url=_snapshot_url(event.snapshot_path),
        detection_summary=event.detection_summary,
        reference_name=event.reference_name,
        display_name=event.display_name,
        caption=event.caption,
        detections=detections,
        visit_id=event.visit_event_id,
    )


@router.get("/events", response_model=EventListResponse)
def list_events(
    camera: str | None = Query(None),
    type: str | None = Query(None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    session: Session = Depends(_get_db),
):
    from ring_detector.database import get_recent_events

    items, total = get_recent_events(
        session,
        limit=limit,
        offset=offset,
        camera=camera,
        event_type=type,
        from_dt=from_dt,
        to_dt=to_dt,
    )
    return EventListResponse(
        total=total,
        items=[_build_event_response(e, session) for e in items],
    )


@router.get("/events/{event_id}", response_model=EventResponse)
def get_event(event_id: int, session: Session = Depends(_get_db)):
    event = session.query(Event).filter_by(id=event_id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return _build_event_response(event, session)
