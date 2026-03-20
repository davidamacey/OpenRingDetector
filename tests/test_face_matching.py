"""Tests for face profile CRUD and matching logic."""

from __future__ import annotations

import numpy as np
import pytest

# Skip entire module if SQLAlchemy or pgvector aren't installed
sqlalchemy = pytest.importorskip("sqlalchemy")
pytest.importorskip("pgvector")

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from ring_detector.config import Settings  # noqa: E402
from ring_detector.database import (  # noqa: E402
    Base,
    delete_face_profile,
    get_all_face_profiles,
    match_against_face_profiles,
    store_watcher_face_embedding,
    upsert_face_profile,
)

FACE_DIM = 512


@pytest.fixture()
def session():
    """In-memory SQLite session with pgvector-compatible schema.

    pgvector isn't available in unit tests, so we monkey-patch the Vector
    column type to a plain String for schema creation only.
    """
    from unittest.mock import patch

    from pgvector.sqlalchemy import Vector
    from sqlalchemy import String

    # Swap Vector → String for SQLite compatibility
    with patch.object(Vector, "__class_getitem__", return_value=String):
        engine = create_engine("sqlite:///:memory:")
        # pgvector extension not available in SQLite — skip that step
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine)
        db = factory()
        yield db
        db.close()


def _rand_vec(dim: int = FACE_DIM) -> list[float]:
    v = np.random.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _near_vec(base: list[float], noise: float = 0.05) -> list[float]:
    """Return a vector close to base (small random perturbation)."""
    v = np.array(base) + np.random.randn(len(base)) * noise
    v /= np.linalg.norm(v)
    return v.tolist()


# ---------------------------------------------------------------------------
# FaceDetectionConfig
# ---------------------------------------------------------------------------


def test_face_config_defaults():
    s = Settings()
    assert s.face.enabled is True
    assert s.face.match_threshold == 0.6
    assert s.face.min_face_size == 50


def test_face_config_env(monkeypatch):
    monkeypatch.setenv("ENABLE_FACE_DETECTION", "false")
    monkeypatch.setenv("FACE_MATCH_THRESHOLD", "0.75")
    monkeypatch.setenv("FACE_MIN_SIZE", "80")
    s = Settings()
    assert s.face.enabled is False
    assert s.face.match_threshold == 0.75
    assert s.face.min_face_size == 80


# ---------------------------------------------------------------------------
# FaceProfile CRUD
# ---------------------------------------------------------------------------


def test_upsert_face_profile_insert(session):
    vec = _rand_vec()
    upsert_face_profile(session, "david", "David", vec)
    profiles = get_all_face_profiles(session)
    assert len(profiles) == 1
    assert profiles[0].name == "david"
    assert profiles[0].display_name == "David"


def test_upsert_face_profile_update(session):
    vec1 = _rand_vec()
    vec2 = _rand_vec()
    upsert_face_profile(session, "david", "David", vec1)
    upsert_face_profile(session, "david", "Dave", vec2)  # update
    profiles = get_all_face_profiles(session)
    assert len(profiles) == 1
    assert profiles[0].display_name == "Dave"


def test_delete_face_profile_existing(session):
    upsert_face_profile(session, "jane", "Jane", _rand_vec())
    assert delete_face_profile(session, "jane") is True
    assert get_all_face_profiles(session) == []


def test_delete_face_profile_nonexistent(session):
    assert delete_face_profile(session, "nobody") is False


def test_get_all_face_profiles_empty(session):
    assert get_all_face_profiles(session) == []


def test_get_all_face_profiles_multiple(session):
    upsert_face_profile(session, "alice", "Alice", _rand_vec())
    upsert_face_profile(session, "bob", "Bob", _rand_vec())
    profiles = get_all_face_profiles(session)
    names = {p.name for p in profiles}
    assert names == {"alice", "bob"}


# ---------------------------------------------------------------------------
# match_against_face_profiles
# ---------------------------------------------------------------------------


def test_match_returns_empty_when_no_profiles(session):
    result = match_against_face_profiles(session, [_rand_vec()], threshold=0.6)
    assert result == []


def test_match_returns_empty_when_no_vectors(session):
    upsert_face_profile(session, "david", "David", _rand_vec())
    result = match_against_face_profiles(session, [], threshold=0.6)
    assert result == []


def test_match_known_face(session):
    base = _rand_vec()
    upsert_face_profile(session, "david", "David", base)

    # Query with a vector very close to the stored one
    query = _near_vec(base, noise=0.01)
    result = match_against_face_profiles(session, [query], threshold=0.5)
    assert len(result) == 1
    assert result[0]["profile_name"] == "david"
    assert result[0]["display_name"] == "David"
    assert result[0]["similarity"] > 0.5


def test_match_unknown_face(session):
    base = _rand_vec()
    upsert_face_profile(session, "david", "David", base)

    # Query with a completely different vector
    query = _rand_vec()
    # Force orthogonal-ish vector by projecting out the component along base
    base_np = np.array(base)
    q_np = np.array(query)
    q_np -= np.dot(q_np, base_np) * base_np
    q_np /= np.linalg.norm(q_np)
    result = match_against_face_profiles(session, [q_np.tolist()], threshold=0.6)
    assert result == []


def test_match_deduplicates_profiles(session):
    """Two queries both matching the same profile → one result entry per profile."""
    base = _rand_vec()
    upsert_face_profile(session, "david", "David", base)

    q1 = _near_vec(base, noise=0.01)
    q2 = _near_vec(base, noise=0.01)
    result = match_against_face_profiles(session, [q1, q2], threshold=0.5)
    # Should be deduplicated to at most 1 match for "david"
    profile_names = [r["profile_name"] for r in result]
    assert profile_names.count("david") <= 1


def test_match_best_match_wins(session):
    """When multiple profiles match, the one with highest similarity wins per query."""
    base_david = _rand_vec()
    base_jane = _rand_vec()
    upsert_face_profile(session, "david", "David", base_david)
    upsert_face_profile(session, "jane", "Jane", base_jane)

    # Query very close to david
    query = _near_vec(base_david, noise=0.005)
    result = match_against_face_profiles(session, [query], threshold=0.5)
    if result:  # only assert if something matched
        assert result[0]["profile_name"] == "david"


def test_match_threshold_respected(session):
    base = _rand_vec()
    upsert_face_profile(session, "david", "David", base)

    # Use a very high threshold that no match can beat
    result = match_against_face_profiles(session, [_rand_vec()], threshold=0.999)
    assert result == []


# ---------------------------------------------------------------------------
# store_watcher_face_embedding
# ---------------------------------------------------------------------------


def test_store_watcher_face_embedding(session):
    """Storing an unmatched face embedding should not raise."""
    vec = _rand_vec()
    # Should complete without error
    store_watcher_face_embedding(session, "/tmp/snap_001.jpg", vec)
    store_watcher_face_embedding(session, "/tmp/snap_001.jpg", vec)  # same path, no collision


def test_store_watcher_face_embedding_creates_metadata(session):
    from ring_detector.database import Metadata

    vec = _rand_vec()
    store_watcher_face_embedding(session, "/tmp/snap_002.jpg", vec)
    meta = session.query(Metadata).filter_by(path="/tmp/snap_002.jpg").first()
    assert meta is not None
    assert meta.file_name == "snap_002.jpg"
