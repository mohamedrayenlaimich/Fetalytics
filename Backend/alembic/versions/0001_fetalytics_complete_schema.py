"""Fetalytics complete cloud storage, dataset, training, and clinical analysis schema

Revision ID: 0001_fetalytics_complete_schema
Revises: None
Create Date: 2026-10-29

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

from Backend.app.core.database import Base
import Backend.app.models  # noqa: F401

revision: str = "0001_fetalytics_complete_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)

    # Ensure existing patients table has the new pseudonymous subject columns if upgraded in-place
    inspector = sa.inspect(bind)
    if "patients" in inspector.get_table_names():
        existing_cols = {c["name"] for c in inspector.get_columns("patients")}
        if "uuid" not in existing_cols:
            op.add_column("patients", sa.Column("uuid", sa.String(length=36), nullable=True))
        if "dataset_source" not in existing_cols:
            op.add_column("patients", sa.Column("dataset_source", sa.String(length=100), nullable=True))
        if "subject_metadata" not in existing_cols:
            op.add_column("patients", sa.Column("subject_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in [
        "analysis_reports",
        "measurements",
        "analysis_jobs",
        "model_versions",
        "training_runs",
        "annotations",
        "extracted_frames",
        "ultrasound_videos",
        "dataset_files",
        "examinations",
        "datasets",
    ]:
        op.drop_table(table_name)
