"""References (vehicle profiles) router."""

from __future__ import annotations

import logging
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.schemas import ReferenceResponse
from ring_detector.database import Reference, VisitEvent, get_session, upsert_reference

router = APIRouter(tags=["references"])
log = logging.getLogger(__name__)

# Lazy-loaded CLIP model for embedding computation
_clip_model = None
_clip_preprocess = None
_clip_device = None


def _get_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        import open_clip
        import torch

        from ring_detector.config import settings

        device = torch.device(settings.model.device if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model = model.eval().to(device)
        _clip_preprocess = preprocess
        _clip_device = device
        log.info("CLIP ViT-B/32 loaded for API reference creation")
    return _clip_model, _clip_preprocess, _clip_device


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

    # Look for a reference thumbnail in archive/references/{name}.jpg
    from pathlib import Path
    from urllib.parse import quote

    from ring_detector.config import settings

    sample_url = None
    for ext in (".jpg", ".jpeg", ".png"):
        thumb = Path(settings.storage.archive_dir) / "references" / f"{ref.name}{ext}"
        if thumb.exists():
            clean = str(thumb).lstrip("/")
            sample_url = f"/api/images/{quote(clean, safe='/')}"
            break

    # Fallback: use latest visit snapshot
    if not sample_url:
        from ring_detector.database import Event

        snap_row = (
            session.query(Event.snapshot_path)
            .filter(Event.reference_name == ref.name, Event.snapshot_path.isnot(None))
            .order_by(Event.occurred_at.desc())
            .first()
        )
        if snap_row and snap_row[0]:
            clean = snap_row[0].lstrip("/")
            sample_url = f"/api/images/{quote(clean, safe='/')}"

    return ReferenceResponse(
        uuid=ref.uuid,
        name=ref.name,
        display_name=ref.display_name,
        category=ref.category,
        visit_count=visit_count,
        last_seen=last_seen,
        sample_image_url=sample_url,
    )


@router.get("/references", response_model=list[ReferenceResponse])
def list_references(session: Session = Depends(_get_db)):
    refs = session.query(Reference).all()
    return [_build_ref(r, session) for r in refs]


@router.post("/references", response_model=ReferenceResponse)
def create_reference(
    name: Annotated[str, Form()],
    display_name: Annotated[str, Form()],
    category: Annotated[str, Form()] = "vehicle",
    images: list[UploadFile] = File(...),
    session: Session = Depends(_get_db),
):
    """Create/update a vehicle reference from uploaded images.

    Computes CLIP ViT-B/32 embeddings for each image and averages them.
    """
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image as PILImage

    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    clip_model, clip_preprocess, device = _get_clip()

    tensors = []
    first_image_data: bytes | None = None
    for upload in images:
        data = upload.file.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        if first_image_data is None:
            first_image_data = data
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        tensors.append(clip_preprocess(pil_img))

    if not tensors:
        raise HTTPException(status_code=400, detail="No valid images provided")

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(batch)
    normed = F.normalize(features.float(), p=2, dim=1)
    avg = normed.mean(dim=0)
    avg = F.normalize(avg.unsqueeze(0), p=2, dim=1).squeeze(0)
    vector = avg.cpu().tolist()

    # Save first uploaded image as reference thumbnail
    from ring_detector.config import settings as cfg

    ref_dir = cfg.storage.archive_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = ref_dir / f"{name}.jpg"
    thumb_path.write_bytes(first_image_data)
    log.info("Reference thumbnail saved: %s", thumb_path)

    upsert_reference(session, name, display_name, vector, category)
    ref = session.query(Reference).filter_by(name=name).first()
    return _build_ref(ref, session)


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
