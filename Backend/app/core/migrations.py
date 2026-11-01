import uuid
from sqlalchemy import Engine, inspect, text
from Backend.app.core.database import Base
import Backend.app.models  # noqa: F401


def run_schema_migrations(engine: Engine) -> None:
    """
    Idempotently create all tables and apply non-destructive column migrations
    for existing tables in Neon PostgreSQL or SQLite test databases.
    """
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())

    if "patients" in table_names:
        existing_cols = {c["name"] for c in inspector.get_columns("patients")}
        with engine.begin() as conn:
            if "uuid" not in existing_cols:
                conn.execute(text("ALTER TABLE patients ADD COLUMN uuid VARCHAR(36)"))
            if "dataset_source" not in existing_cols:
                conn.execute(text("ALTER TABLE patients ADD COLUMN dataset_source VARCHAR(100)"))
            if "subject_metadata" not in existing_cols:
                conn.execute(text("ALTER TABLE patients ADD COLUMN subject_metadata JSON"))

            # Backfill any NULL uuid rows in patients
            rows = conn.execute(text("SELECT id FROM patients WHERE uuid IS NULL")).fetchall()
            for (pid,) in rows:
                conn.execute(
                    text("UPDATE patients SET uuid = :u WHERE id = :pid"),
                    {"u": str(uuid.uuid4()), "pid": pid},
                )
