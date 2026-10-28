import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text, inspect
from Backend.app.core.database import engine, Base
from Backend.app.core.config import settings
import Backend.app.models  # Load all models so Base metadata is populated

def sync_and_verify_database():
    print("=" * 65)
    print("Fetalytics Database Schema Sync & Verification")
    print("=" * 65)
    
    if not settings.DATABASE_URL:
        print("[ERROR] DATABASE_URL is not set in .env!")
        sys.exit(1)

    try:
        with engine.connect() as conn:
            # Query server version
            server_version = conn.execute(text("SELECT version();")).scalar()
            print(f"[OK] Connected to PostgreSQL:\n  {server_version}\n")

            # Check existing public tables
            existing = conn.execute(text(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )).fetchall()
            existing_tables = [r[0] for r in existing]
            print(f"Current tables before sync: {existing_tables}")

            # Recreate tables to guarantee matching constraints and foreign keys
            print("\nDropping old draft tables if any and synchronizing full clinical schema...")
            # Drop in correct foreign-key order
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            print("[OK] Schema synchronized successfully!\n")

        # Inspect and print newly created tables & columns
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        print(f"Created {len(table_names)} tables in Neon PostgreSQL:")
        for tbl in table_names:
            cols = [col["name"] for col in inspector.get_columns(tbl)]
            print(f"  • {tbl}: {', '.join(cols)}")

        print("\n" + "=" * 65)
        print("Database schema is fully synchronized, verified, and ready!")
        print("=" * 65)

    except Exception as e:
        print(f"\n[ERROR] Database sync failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    sync_and_verify_database()
