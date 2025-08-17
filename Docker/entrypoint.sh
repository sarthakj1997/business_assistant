#!/bin/bash
set -e

echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -p 5432 -U sarthak; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done

echo "PostgreSQL is up - running migrations"
alembic upgrade head

echo "Creating default user if not exists"
python -c "
import sys
sys.path.append('/app')
from database.models.users import User
from database.setup_db import SessionLocal

session = SessionLocal()
try:
    existing_user = session.query(User).filter(User.id == 1).first()
    if not existing_user:
        default_user = User(id=1, name='Default User', email='user@example.com')
        session.add(default_user)
        session.commit()
        print('Default user created')
    else:
        print('Default user already exists')
except Exception as e:
    print(f'Error creating user: {e}')
    session.rollback()
finally:
    session.close()
"

echo "Starting FastAPI server"
exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
