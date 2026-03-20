"""PostgreSQL + pgvector database layer.

Combines structured metadata, detection records, reference vectors,
and embedding similarity search in a single PostgreSQL database.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import uuid4

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from ring_detector.config import settings

log = logging.getLogger(__name__)

CLIP_EMBED_DIM = 512   # CLIP ViT-B/32 visual encoder output
FACE_EMBED_DIM = 512   # ArcFace r100 output (InsightFace buffalo_l)


class Base(DeclarativeBase):
    pass


class Metadata(Base):
    __tablename__ = "metadata"

    file_uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    created_at = Column(DateTime, default=datetime.now)
    path = Column(String, unique=True)
    file_name = Column(String)
    height = Column(Integer)
    width = Column(Integer)

    detections = relationship("Detection", back_populates="file_metadata", cascade="all, delete")
    embeddings = relationship("Embedding", back_populates="file_metadata", cascade="all, delete")


class Detection(Base):
    __tablename__ = "detections"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    file_uuid = Column(String, ForeignKey("metadata.file_uuid", ondelete="CASCADE"))
    class_name = Column(String)
    class_id = Column(Integer)
    confidence = Column(Float)
    xcenter = Column(Float)
    ycenter = Column(Float)
    width = Column(Float)
    height = Column(Float)

    file_metadata = relationship("Metadata", back_populates="detections")


class Embedding(Base):
    __tablename__ = "embeddings"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    file_uuid = Column(String, ForeignKey("metadata.file_uuid", ondelete="CASCADE"))
    embed_type = Column(String)  # full_image, detection, reference
    label = Column(String, default="none")
    vector = Column(Vector(CLIP_EMBED_DIM))

    file_metadata = relationship("Metadata", back_populates="embeddings")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    file_uuid = Column(String, ForeignKey("metadata.file_uuid", ondelete="CASCADE"))
    img_path = Column(String)
    label = Column(String, default="none")
    person_name = Column(String, default="unknown")
    vector = Column(Vector(FACE_EMBED_DIM))


class Reference(Base):
    __tablename__ = "references"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    name = Column(String, unique=True)
    display_name = Column(String)
    category = Column(String, default="vehicle")
    vector = Column(Vector(CLIP_EMBED_DIM))


class FaceProfile(Base):
    """Named face reference for person recognition (InceptionResnetV1 embeddings)."""

    __tablename__ = "face_profiles"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    name = Column(String, unique=True)
    display_name = Column(String)
    vector = Column(Vector(FACE_EMBED_DIM))
    created_at = Column(DateTime, default=datetime.now)


class VisitEvent(Base):
    """Tracks arrival/departure of known references."""

    __tablename__ = "visit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reference_name = Column(String, index=True)
    display_name = Column(String)
    camera_name = Column(String)
    arrived_at = Column(DateTime)
    last_motion_at = Column(DateTime)  # updated on each motion during visit
    departed_at = Column(DateTime, nullable=True)
    snapshot_path = Column(String, nullable=True)


# --- Engine / Session ---

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db.url, pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def create_tables() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    log.info("Database tables ready (pgvector enabled)")


# --- Bulk Inserts ---


def insert_metadata_bulk(session: Session, data: list[dict]) -> None:
    session.execute(Metadata.__table__.insert(), data)
    session.commit()


def insert_detections_bulk(session: Session, data: list[dict]) -> None:
    session.execute(Detection.__table__.insert(), data)
    session.commit()


def insert_embeddings_bulk(
    session: Session,
    file_uuids: list[str],
    vectors: list[list[float]],
    embed_type: str = "full_image",
    labels: list[str] | None = None,
) -> None:
    labels = labels or ["none"] * len(file_uuids)
    data = [
        {
            "uuid": uuid4().hex,
            "file_uuid": fid,
            "embed_type": embed_type,
            "label": label,
            "vector": vec,
        }
        for fid, vec, label in zip(file_uuids, vectors, labels, strict=True)
    ]
    session.execute(Embedding.__table__.insert(), data)
    session.commit()


def insert_face_embeddings_bulk(
    session: Session,
    file_uuids: list[str],
    img_paths: list[str],
    vectors: list[list[float]],
    labels: list[str] | None = None,
) -> None:
    labels = labels or ["face"] * len(file_uuids)
    data = [
        {
            "uuid": uuid4().hex,
            "file_uuid": fid,
            "img_path": path,
            "label": label,
            "vector": vec,
        }
        for fid, path, vec, label in zip(file_uuids, img_paths, vectors, labels, strict=True)
    ]
    session.execute(FaceEmbedding.__table__.insert(), data)
    session.commit()


# --- References ---


def upsert_reference(
    session: Session,
    name: str,
    display_name: str,
    vector: list[float],
    category: str = "vehicle",
) -> None:
    existing = session.query(Reference).filter_by(name=name).first()
    if existing:
        existing.vector = vector
        existing.display_name = display_name
        existing.category = category
    else:
        session.add(
            Reference(
                uuid=uuid4().hex,
                name=name,
                display_name=display_name,
                category=category,
                vector=vector,
            )
        )
    session.commit()
    log.info("Reference '%s' saved", display_name)


def get_all_references(session: Session) -> list[Reference]:
    return session.query(Reference).all()


def get_reference_by_name(session: Session, name: str) -> Reference | None:
    return session.query(Reference).filter_by(name=name).first()


# --- Face Profiles ---


def upsert_face_profile(
    session: Session,
    name: str,
    display_name: str,
    vector: list[float],
) -> None:
    """Insert or update a named face reference profile."""
    existing = session.query(FaceProfile).filter_by(name=name).first()
    if existing:
        existing.vector = vector
        existing.display_name = display_name
    else:
        session.add(
            FaceProfile(
                uuid=uuid4().hex,
                name=name,
                display_name=display_name,
                vector=vector,
            )
        )
    session.commit()
    log.info("Face profile '%s' saved", display_name)


def get_all_face_profiles(session: Session) -> list[FaceProfile]:
    return session.query(FaceProfile).all()


def delete_face_profile(session: Session, name: str) -> bool:
    """Delete a face profile by name. Returns True if it existed."""
    profile = session.query(FaceProfile).filter_by(name=name).first()
    if not profile:
        return False
    session.delete(profile)
    session.commit()
    log.info("Face profile '%s' deleted", name)
    return True


def match_against_face_profiles(
    session: Session,
    vectors: list[list[float]],
    threshold: float = 0.6,
) -> list[dict]:
    """Match face embeddings against known face profiles.

    Returns one best match per embedding vector (highest similarity above threshold).
    Deduplicates by profile name so the same person isn't returned multiple times.
    """
    profiles = get_all_face_profiles(session)
    if not profiles:
        return []

    matches = []
    seen_profiles: set[str] = set()

    for i, vec in enumerate(vectors):
        vec_np = np.array(vec)
        best: dict | None = None
        best_sim = threshold  # minimum to qualify

        for profile in profiles:
            ref_np = np.array(profile.vector)
            denom = np.linalg.norm(vec_np) * np.linalg.norm(ref_np)
            if denom == 0:
                continue
            similarity = float(np.dot(vec_np, ref_np) / denom)
            if similarity > best_sim:
                best_sim = similarity
                best = {
                    "vector_index": i,
                    "profile_name": profile.name,
                    "display_name": profile.display_name,
                    "similarity": similarity,
                }

        if best and best["profile_name"] not in seen_profiles:
            matches.append(best)
            seen_profiles.add(best["profile_name"])

    return matches


def store_watcher_face_embedding(
    session: Session,
    snapshot_path: str,
    vector: list[float],
    person_name: str = "unknown",
) -> None:
    """Persist an unmatched face embedding (for later labeling via ring-face add)."""
    from pathlib import Path as _Path

    existing_meta = session.query(Metadata).filter_by(path=snapshot_path).first()
    if existing_meta:
        file_uuid = existing_meta.file_uuid
    else:
        file_uuid = uuid4().hex
        session.add(
            Metadata(
                file_uuid=file_uuid,
                path=snapshot_path,
                file_name=_Path(snapshot_path).name,
            )
        )
        session.flush()

    session.add(
        FaceEmbedding(
            uuid=uuid4().hex,
            file_uuid=file_uuid,
            img_path=snapshot_path,
            label="watcher_detection",
            person_name=person_name,
            vector=vector,
        )
    )
    session.commit()


# --- Vector Search ---


def find_similar_embeddings(
    session: Session,
    query_vector: list[float],
    embed_type: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Find most similar embeddings using pgvector cosine distance."""
    q = session.query(
        Embedding.uuid,
        Embedding.file_uuid,
        Embedding.label,
        Embedding.vector.cosine_distance(query_vector).label("distance"),
    )
    if embed_type:
        q = q.filter(Embedding.embed_type == embed_type)
    q = q.order_by("distance").limit(limit)

    return [
        {
            "uuid": row.uuid,
            "file_uuid": row.file_uuid,
            "label": row.label,
            "distance": row.distance,
        }
        for row in q.all()
    ]


def match_against_references(
    session: Session,
    vectors: list[list[float]],
    threshold: float = 0.15,
) -> list[dict]:
    """Compare vectors against all references. Returns matches above threshold."""
    refs = get_all_references(session)
    if not refs:
        return []

    matches = []
    for i, vec in enumerate(vectors):
        vec_np = np.array(vec)
        for ref in refs:
            ref_np = np.array(ref.vector)
            denom = np.linalg.norm(vec_np) * np.linalg.norm(ref_np)
            if denom == 0:
                continue
            similarity = float(np.dot(vec_np, ref_np) / denom)
            if similarity > (1 - threshold):
                matches.append(
                    {
                        "vector_index": i,
                        "reference_name": ref.name,
                        "display_name": ref.display_name,
                        "category": ref.category,
                        "similarity": similarity,
                    }
                )
    return matches


# --- Query Helpers ---


def get_all_image_paths(session: Session) -> list[str]:
    return [p[0] for p in session.query(Metadata.path).all()]


def get_new_paths(session: Session, candidate_paths: list[str]) -> list[str]:
    existing = set(p[0] for p in session.query(Metadata.path).all())
    return [p for p in candidate_paths if p not in existing]


# --- Visit Tracking ---


def record_arrival(
    session: Session,
    reference_name: str,
    display_name: str,
    camera_name: str,
    snapshot_path: str | None = None,
) -> VisitEvent:
    now = datetime.now()
    visit = VisitEvent(
        reference_name=reference_name,
        display_name=display_name,
        camera_name=camera_name,
        arrived_at=now,
        last_motion_at=now,
        snapshot_path=snapshot_path,
    )
    session.add(visit)
    session.commit()
    log.info("Visit started: %s at %s", display_name, camera_name)
    return visit


def extend_visit(session: Session, visit: VisitEvent) -> None:
    """Update last_motion_at to push back departure timeout."""
    visit.last_motion_at = datetime.now()
    session.commit()


def record_departure(session: Session, visit: VisitEvent) -> int:
    """Mark visit as departed. Returns duration in minutes."""
    visit.departed_at = datetime.now()
    session.commit()
    duration = int((visit.departed_at - visit.arrived_at).total_seconds() / 60)
    log.info("%s departed after %d min", visit.display_name, duration)
    return duration


def get_active_visits(session: Session) -> list[VisitEvent]:
    return session.query(VisitEvent).filter(VisitEvent.departed_at.is_(None)).all()


def get_active_visit_by_reference(session: Session, reference_name: str) -> VisitEvent | None:
    return (
        session.query(VisitEvent)
        .filter(
            VisitEvent.reference_name == reference_name,
            VisitEvent.departed_at.is_(None),
        )
        .first()
    )
