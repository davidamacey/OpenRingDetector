"""Initial schema — all tables

Revision ID: 001
Revises:
Create Date: 2026-03-20

Creates the complete ring-detector schema:
  - metadata, detections, embeddings (512-dim CLIP ViT-B/32)
  - face_embeddings, face_profiles (512-dim ArcFace)
  - references (512-dim CLIP vehicle references)
  - visit_events (arrival/departure tracking)
  - events (raw event log for dashboard)

For existing databases bootstrapped via create_tables() / initdb.sql, stamp first:
    alembic stamp 001
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # pgvector extension must exist before any vector() columns are created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "metadata",
        sa.Column("file_uuid", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("path", sa.String(), nullable=True),
        sa.Column("file_name", sa.String(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.UniqueConstraint("path"),
    )

    op.create_table(
        "detections",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column(
            "file_uuid",
            sa.String(),
            sa.ForeignKey("metadata.file_uuid", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("class_name", sa.String(), nullable=True),
        sa.Column("class_id", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("xcenter", sa.Float(), nullable=True),
        sa.Column("ycenter", sa.Float(), nullable=True),
        sa.Column("width", sa.Float(), nullable=True),
        sa.Column("height", sa.Float(), nullable=True),
    )

    op.create_table(
        "embeddings",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column(
            "file_uuid",
            sa.String(),
            sa.ForeignKey("metadata.file_uuid", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("embed_type", sa.String(), nullable=True),
        sa.Column("label", sa.String(), server_default="none", nullable=True),
        sa.Column("vector", Vector(512), nullable=True),  # CLIP ViT-B/32
    )

    op.create_table(
        "face_embeddings",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column(
            "file_uuid",
            sa.String(),
            sa.ForeignKey("metadata.file_uuid", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("img_path", sa.String(), nullable=True),
        sa.Column("label", sa.String(), server_default="none", nullable=True),
        sa.Column("person_name", sa.String(), server_default="unknown", nullable=True),
        sa.Column("vector", Vector(512), nullable=True),  # ArcFace w600k_r50
    )

    op.create_table(
        "references",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("category", sa.String(), server_default="vehicle", nullable=True),
        sa.Column("vector", Vector(512), nullable=True),  # CLIP ViT-B/32
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "face_profiles",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("vector", Vector(512), nullable=True),  # ArcFace w600k_r50
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("name"),
    )

    op.create_table(
        "visit_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("reference_name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("camera_name", sa.String(), nullable=True),
        sa.Column("arrived_at", sa.DateTime(), nullable=True),
        sa.Column("last_motion_at", sa.DateTime(), nullable=True),
        sa.Column("departed_at", sa.DateTime(), nullable=True),
        sa.Column("snapshot_path", sa.String(), nullable=True),
    )
    op.create_index("ix_visit_events_reference_name", "visit_events", ["reference_name"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("camera_name", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=True),
        sa.Column(
            "file_uuid",
            sa.String(),
            sa.ForeignKey("metadata.file_uuid", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("snapshot_path", sa.String(), nullable=True),
        sa.Column("detection_summary", sa.String(), nullable=True),
        sa.Column("reference_name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column(
            "visit_event_id",
            sa.Integer(),
            sa.ForeignKey("visit_events.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("caption", sa.String(), nullable=True),
    )
    op.create_index("ix_events_occurred_at", "events", ["occurred_at"])


def downgrade() -> None:
    op.drop_index("ix_events_occurred_at", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_visit_events_reference_name", table_name="visit_events")
    op.drop_table("visit_events")
    op.drop_table("face_profiles")
    op.drop_table("references")
    op.drop_table("face_embeddings")
    op.drop_table("embeddings")
    op.drop_table("detections")
    op.drop_table("metadata")
    op.execute("DROP EXTENSION IF EXISTS vector")
