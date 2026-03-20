"""References (vehicle profiles) router."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.schemas import ReferenceResponse
from ring_detector.database import Reference, VisitEvent, get_session

router = APIRouter(tags=["references"])


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _build_ref(ref: Reference, session: Session) -> ReferenceResponse:
    visit_count = (
        session.query(func.count(VisitEvent.id))
        .filter(VisitEvent.reference_name == ref.name)
        .scalar()
        or 0
    )
    last_seen_row = (
        session.query(VisitEvent.arrived_at)
        .filter(VisitEvent.reference_name == ref.name)
        .order_by(VisitEvent.arrived_at.desc())
        .first()
    )
    last_seen = last_seen_row[0] if last_seen_row else None
    return ReferenceResponse(
        uuid=ref.uuid,
        name=ref.name,
        display_name=ref.display_name,
        category=ref.category,
        visit_count=visit_count,
        last_seen=last_seen,
        sample_image_url=None,
    )


@router.get("/references", response_model=list[ReferenceResponse])
def list_references(session: Session = Depends(_get_db)):
    refs = session.query(Reference).all()
    return [_build_ref(r, session) for r in refs]


@router.put("/references/{name}", response_model=ReferenceResponse)
def update_reference(
    name: str,
    body: dict,
    session: Session = Depends(_get_db),
):
    ref = session.query(Reference).filter_by(name=name).first()
    if not ref:
        raise HTTPException(status_code=404, detail="Reference not found")
    if "display_name" in body:
        ref.display_name = body["display_name"]
    if "category" in body:
        ref.category = body["category"]
    session.commit()
    return _build_ref(ref, session)


@router.delete("/references/{name}")
def delete_reference(name: str, session: Session = Depends(_get_db)):
    ref = session.query(Reference).filter_by(name=name).first()
    if not ref:
        raise HTTPException(status_code=404, detail="Reference not found")
    session.delete(ref)
    session.commit()
    return {"success": True}
