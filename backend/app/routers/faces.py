"""Face profiles router — CRUD for known people."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.schemas import FaceProfileResponse
from ring_detector.database import FaceEmbedding, VisitEvent, get_session

router = APIRouter(tags=["faces"])


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _build_face_profile(name: str, session: Session) -> FaceProfileResponse:
    """Build a face profile response from face_embeddings rows sharing a person_name."""
    rows = session.query(FaceEmbedding).filter_by(person_name=name).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Face profile not found")

    display_name = rows[0].label if rows[0].label != "face" else name
    created_at = min(r.label for r in rows)  # fallback — no created_at on FaceEmbedding

    # Use visit events keyed as "face:{name}"
    visit_count = (
        session.query(func.count(VisitEvent.id))
        .filter(VisitEvent.reference_name == f"face:{name}")
        .scalar()
        or 0
    )
    last_seen_row = (
        session.query(VisitEvent.arrived_at)
        .filter(VisitEvent.reference_name == f"face:{name}")
        .order_by(VisitEvent.arrived_at.desc())
        .first()
    )
    last_seen = last_seen_row[0] if last_seen_row else None
    sample_path = rows[0].img_path if rows else None

    from datetime import datetime

    return FaceProfileResponse(
        uuid=rows[0].uuid,
        name=name,
        display_name=display_name,
        created_at=datetime.now(),  # FaceEmbedding has no created_at column
        visit_count=visit_count,
        last_seen=last_seen,
        sample_image_url=f"/api/images/{sample_path}" if sample_path else None,
    )


@router.get("/faces", response_model=list[FaceProfileResponse])
def list_faces(session: Session = Depends(_get_db)):
    names = (
        session.query(FaceEmbedding.person_name)
        .filter(FaceEmbedding.person_name != "unknown")
        .distinct()
        .all()
    )
    profiles = []
    for (name,) in names:
        try:
            profiles.append(_build_face_profile(name, session))
        except HTTPException:
            continue
    return profiles


@router.get("/faces/{name}", response_model=FaceProfileResponse)
def get_face(name: str, session: Session = Depends(_get_db)):
    return _build_face_profile(name, session)


@router.put("/faces/{name}", response_model=FaceProfileResponse)
def update_face(name: str, body: dict, session: Session = Depends(_get_db)):
    rows = session.query(FaceEmbedding).filter_by(person_name=name).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Face profile not found")
    if "display_name" in body:
        for r in rows:
            r.label = body["display_name"]
    session.commit()
    return _build_face_profile(name, session)


@router.delete("/faces/{name}")
def delete_face(name: str, session: Session = Depends(_get_db)):
    rows = session.query(FaceEmbedding).filter_by(person_name=name).all()
    if not rows:
        raise HTTPException(status_code=404, detail="Face profile not found")
    for r in rows:
        session.delete(r)
    session.commit()
    return {"success": True}
