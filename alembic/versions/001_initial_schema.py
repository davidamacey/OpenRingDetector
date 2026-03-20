"""Initial schema — all tables

Revision ID: 001
Revises:
Create Date: 2026-03-20

Creates the complete ring-detector schema:
  - metadata, detections, embeddings (576-dim YOLO)
  - face_embeddings (512-dim ArcFace)
  - references (vehicle/person reference vectors)
  - visit_events (arrival/departure tracking)

For existing databases bootstrapped via create_tables() / initdb.sql, stamp first:
    alembic stamp 001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

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
        sa.Column("vector", Vector(576), nullable=True),
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
        sa.Column("vector", Vector(512), nullable=True),
    )

    op.create_table(
        "references",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("category", sa.String(), server_default="vehicle", nullable=True),
        sa.Column("vector", Vector(576), nullable=True),
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


def downgrade() -> None:
    op.drop_index("ix_visit_events_reference_name", table_name="visit_events")
    op.drop_table("visit_events")
    op.drop_table("references")
    op.drop_table("face_embeddings")
    op.drop_table("embeddings")
    op.drop_table("detections")
    op.drop_table("metadata")
    op.execute("DROP EXTENSION IF EXISTS vector")
