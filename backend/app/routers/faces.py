"""Face profiles router — CRUD for known people."""

from __future__ import annotations

import logging
from typing import Annotated
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.schemas import FaceProfileResponse
from ring_detector.database import (
    FaceEmbedding,
    FaceProfile,
    VisitEvent,
    get_session,
    upsert_face_profile,
)

router = APIRouter(tags=["faces"])
log = logging.getLogger(__name__)


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def _build_face_response(profile: FaceProfile, session: Session) -> FaceProfileResponse:
    """Build response from the face_profiles table (canonical source)."""
    visit_count = (
        session.query(func.count(VisitEvent.id))
        .filter(VisitEvent.reference_name == f"face:{profile.name}")
        .scalar()
        or 0
    )
    last_seen_row = (
        session.query(VisitEvent.arrived_at)
        .filter(VisitEvent.reference_name == f"face:{profile.name}")
        .order_by(VisitEvent.arrived_at.desc())
        .first()
    )
    last_seen = last_seen_row[0] if last_seen_row else None

    # Look for a face profile thumbnail in archive/faces/{name}.jpg
    from pathlib import Path

    from ring_detector.config import settings

    sample_url = None
    for ext in (".jpg", ".jpeg", ".png"):
        thumb = Path(settings.storage.archive_dir) / "faces" / f"{profile.name}{ext}"
        if thumb.exists():
            clean = str(thumb).lstrip("/")
            sample_url = f"/api/images/{quote(clean, safe='/')}"
            break

    # Fallback: sample from face_embeddings table
    if not sample_url:
        sample = (
            session.query(FaceEmbedding.img_path)
            .filter(FaceEmbedding.person_name == profile.name)
            .first()
        )
        if sample and sample[0]:
            clean = sample[0].lstrip("/")
            sample_url = f"/api/images/{quote(clean, safe='/')}"

    return FaceProfileResponse(
        uuid=profile.uuid,
        name=profile.name,
        display_name=profile.display_name,
        created_at=profile.created_at,
        visit_count=visit_count,
        last_seen=last_seen,
        sample_image_url=sample_url,
    )


@router.get("/faces", response_model=list[FaceProfileResponse])
def list_faces(session: Session = Depends(_get_db)):
    profiles = session.query(FaceProfile).all()
    return [_build_face_response(p, session) for p in profiles]


@router.get("/faces/{name}", response_model=FaceProfileResponse)
def get_face(name: str, session: Session = Depends(_get_db)):
    profile = session.query(FaceProfile).filter_by(name=name).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Face profile not found")
    return _build_face_response(profile, session)


@router.post("/faces", response_model=FaceProfileResponse)
def create_face(
    name: Annotated[str, Form()],
    display_name: Annotated[str, Form()],
    image: UploadFile = File(...),
    session: Session = Depends(_get_db),
):
    """Create/update a face profile from an uploaded photo.

    Auto-detects the face using SCRFD and computes ArcFace embedding.
    """
    import cv2

    from ring_detector.config import settings
    from ring_detector.face_detector import create_face_detector

    data = image.file.read()
    arr = np.frombuffer(data, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    face_detector = create_face_detector(settings)
    if face_detector is None:
        raise HTTPException(status_code=503, detail="Face detection not available")

    results = face_detector.detect_and_embed(img_bgr, min_face_size=30)
    if not results:
        raise HTTPException(status_code=400, detail="No face detected in the image")

    # Use the highest-scoring face
    best = max(results, key=lambda r: r.score)
    vector = best.embedding

    # Save uploaded photo as face profile thumbnail
    faces_dir = settings.storage.archive_dir / "faces"
    faces_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = faces_dir / f"{name}.jpg"
    thumb_path.write_bytes(data)
    log.info("Face profile thumbnail saved: %s", thumb_path)

    upsert_face_profile(session, name, display_name, vector)
    profile = session.query(FaceProfile).filter_by(name=name).first()
    return _build_face_response(profile, session)


@router.put("/faces/{name}", response_model=FaceProfileResponse)
def update_face(name: str, body: dict, session: Session = Depends(_get_db)):
    profile = session.query(FaceProfile).filter_by(name=name).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Face profile not found")
    if "display_name" in body:
        profile.display_name = body["display_name"]
    session.commit()
    return _build_face_response(profile, session)


@router.delete("/faces/{name}")
def delete_face(name: str, session: Session = Depends(_get_db)):
    from ring_detector.database import delete_face_profile

    if not delete_face_profile(session, name):
        raise HTTPException(status_code=404, detail="Face profile not found")
    return {"success": True}
