"""PostgreSQL + pgvector database layer.

Combines structured metadata, detection records, reference vectors,
and embedding similarity search in a single PostgreSQL database.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    Date,
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

# Embedding dimensions
YOLO_EMBED_DIM = 576
FACE_EMBED_DIM = 512


class Base(DeclarativeBase):
    pass


class Metadata(Base):
    __tablename__ = "metadata"

    file_uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    date = Column(Date)
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
    """Vector embeddings for images and detected objects."""

    __tablename__ = "embeddings"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    file_uuid = Column(String, ForeignKey("metadata.file_uuid", ondelete="CASCADE"))
    embed_type = Column(String)  # "full_image", "detection", "face"
    label = Column(String, default="none")
    vector = Column(Vector(YOLO_EMBED_DIM))  # pgvector column

    file_metadata = relationship("Metadata", back_populates="embeddings")


class FaceEmbedding(Base):
    """Face embeddings (different dimension than YOLO embeddings)."""

    __tablename__ = "face_embeddings"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    file_uuid = Column(String, ForeignKey("metadata.file_uuid", ondelete="CASCADE"))
    img_path = Column(String)
    label = Column(String, default="none")
    person_name = Column(String, default="unknown")
    vector = Column(Vector(FACE_EMBED_DIM))


class Reference(Base):
    """Named reference vectors for known vehicles/objects."""

    __tablename__ = "references"

    uuid = Column(String, primary_key=True, default=lambda: uuid4().hex)
    name = Column(String, unique=True)  # e.g. "cleaners_car", "yard_guy"
    display_name = Column(String)  # e.g. "Cleaner's Car"
    category = Column(String, default="vehicle")  # "vehicle", "person", "other"
    vector = Column(Vector(YOLO_EMBED_DIM))


class VisitEvent(Base):
    """Tracks arrival/departure of known references (cleaner, yard guy, etc.)."""

    __tablename__ = "visit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    reference_name = Column(String)  # FK-ish to references.name
    display_name = Column(String)
    camera_name = Column(String)
    arrived_at = Column(DateTime)
    departed_at = Column(DateTime, nullable=True)
    snapshot_path = Column(String, nullable=True)
    notified_departure = Column(Integer, default=0)  # 0=no, 1=yes


# --- Engine / Session Management ---

_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db.url, echo=False)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


def create_tables() -> None:
    """Create all tables and enable pgvector extension."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    log.info("Database tables created (pgvector enabled)")


# --- Bulk Operations ---


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
    """Bulk insert embeddings with pgvector."""
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


# --- Reference Management ---


def upsert_reference(
    session: Session,
    name: str,
    display_name: str,
    vector: list[float],
    category: str = "vehicle",
) -> None:
    """Insert or update a reference vector."""
    existing = session.query(Reference).filter_by(name=name).first()
    if existing:
        existing.vector = vector
        existing.display_name = display_name
        existing.category = category
    else:
        ref = Reference(
            uuid=uuid4().hex,
            name=name,
            display_name=display_name,
            category=category,
            vector=vector,
        )
        session.add(ref)
    session.commit()
    log.info("Reference '%s' saved", display_name)


def get_all_references(session: Session) -> list[Reference]:
    return session.query(Reference).all()


def get_reference_by_name(session: Session, name: str) -> Reference | None:
    return session.query(Reference).filter_by(name=name).first()


# --- Vector Similarity Search ---


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
        {"uuid": row.uuid, "file_uuid": row.file_uuid, "label": row.label, "distance": row.distance}
        for row in q.all()
    ]


def match_against_references(
    session: Session,
    vectors: list[list[float]],
    threshold: float = 0.15,  # cosine distance (lower = more similar)
) -> list[dict]:
    """Check a list of vectors against all references. Returns matches."""
    refs = get_all_references(session)
    if not refs:
        return []

    matches = []
    for i, vec in enumerate(vectors):
        vec_np = np.array(vec)
        for ref in refs:
            ref_np = np.array(ref.vector)
            # Cosine similarity
            norm_product = np.linalg.norm(vec_np) * np.linalg.norm(ref_np)
            similarity = float(np.dot(vec_np, ref_np) / norm_product)
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
    paths = session.query(Metadata.path).all()
    return [p[0] for p in paths]


def get_new_paths(session: Session, candidate_paths: list[str]) -> list[str]:
    """Return paths not already in the database."""
    existing = set(p[0] for p in session.query(Metadata.path).all())
    return [p for p in candidate_paths if p not in existing]


# --- Visit Event Tracking ---


def record_arrival(
    session: Session,
    reference_name: str,
    display_name: str,
    camera_name: str,
    snapshot_path: str | None = None,
) -> VisitEvent:
    """Record that a known reference arrived. Returns the visit event."""
    from datetime import datetime

    visit = VisitEvent(
        reference_name=reference_name,
        display_name=display_name,
        camera_name=camera_name,
        arrived_at=datetime.now(),
        snapshot_path=snapshot_path,
    )
    session.add(visit)
    session.commit()
    log.info("Visit recorded: %s arrived at %s", display_name, camera_name)
    return visit


def record_departure(session: Session, visit: VisitEvent) -> int:
    """Mark a visit as departed. Returns visit duration in minutes."""
    from datetime import datetime

    visit.departed_at = datetime.now()
    visit.notified_departure = 1
    session.commit()
    duration = int((visit.departed_at - visit.arrived_at).total_seconds() / 60)
    log.info("%s departed after %d min", visit.display_name, duration)
    return duration


def get_active_visits(session: Session) -> list[VisitEvent]:
    """Get all visits that haven't departed yet."""
    return session.query(VisitEvent).filter(VisitEvent.departed_at.is_(None)).all()


def get_active_visit_by_reference(session: Session, reference_name: str) -> VisitEvent | None:
    """Get active (not departed) visit for a specific reference."""
    return (
        session.query(VisitEvent)
        .filter(
            VisitEvent.reference_name == reference_name,
            VisitEvent.departed_at.is_(None),
        )
        .first()
    )
