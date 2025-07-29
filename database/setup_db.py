# database/setup_db.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., postgres://user:pass@localhost/db
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)