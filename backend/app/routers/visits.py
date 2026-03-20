"""Visits router — GET /api/visits, /api/visits/active, /api/visits/{id}."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.schemas import VisitListResponse, VisitResponse
from ring_detector.database import VisitEvent, get_session

router = APIRouter(tags=["visits"])


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


def _build_visit(v: VisitEvent) -> VisitResponse:
    duration = None
    if v.departed_at and v.arrived_at:
        duration = int((v.departed_at - v.arrived_at).total_seconds() / 60)
    return VisitResponse(
        id=v.id,
        reference_name=v.reference_name,
        display_name=v.display_name,
        camera_name=v.camera_name,
        arrived_at=v.arrived_at,
        last_motion_at=v.last_motion_at,
        departed_at=v.departed_at,
        duration_minutes=duration,
        snapshot_url=_snapshot_url(v.snapshot_path),
        is_active=v.departed_at is None,
    )


@router.get("/visits/active", response_model=list[VisitResponse])
def active_visits(session: Session = Depends(_get_db)):
    from ring_detector.database import get_active_visits

    return [_build_visit(v) for v in get_active_visits(session)]


@router.get("/visits", response_model=VisitListResponse)
def list_visits(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    name: str | None = Query(None),
    active_only: bool = Query(False),
    type: str | None = Query(None),
    session: Session = Depends(_get_db),
):
    q = session.query(VisitEvent)
    if active_only:
        q = q.filter(VisitEvent.departed_at.is_(None))
    if from_dt:
        q = q.filter(VisitEvent.arrived_at >= from_dt)
    if to_dt:
        q = q.filter(VisitEvent.arrived_at <= to_dt)
    if name:
        q = q.filter(VisitEvent.display_name.ilike(f"%{name}%"))
    total = q.count()
    items = q.order_by(VisitEvent.arrived_at.desc()).offset(offset).limit(limit).all()
    return VisitListResponse(total=total, items=[_build_visit(v) for v in items])


@router.get("/visits/{visit_id}", response_model=VisitResponse)
def get_visit(visit_id: int, session: Session = Depends(_get_db)):
    v = session.query(VisitEvent).filter_by(id=visit_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Visit not found")
    return _build_visit(v)
