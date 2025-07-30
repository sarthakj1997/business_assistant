import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from database.setup_db import SessionLocal
from database.models.users import User

def clear_all_data():
    db = SessionLocal()
    try:
        db.execute(text("TRUNCATE TABLE line_items RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE invoices RESTART IDENTITY CASCADE;"))
        db.execute(text("TRUNCATE TABLE users RESTART IDENTITY CASCADE;"))
        db.commit()
        print("All PostgreSQL tables have been cleared.")
    except Exception as e:
        db.rollback()
        print(f"Failed to clear tables: {e}")
    finally:
        db.close()

def create_default_user():
    db = SessionLocal()
    try:
        default_user = User(
            name="Default User",
            email="default@example.com"
        )
        db.add(default_user)
        db.commit()
        print(f"Default user created with ID: {default_user.id}")
        return default_user.id
    except Exception as e:
        db.rollback()
        print(f"Failed to create default user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    clear_all_data()
    create_default_user()
