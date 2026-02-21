"""
Database models for Fashion Virtual Try-On
"""
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fashion_tryon.db")

# Render uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Garment(Base):
    __tablename__ = "garments"

    id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)  # tops, bottoms, one-pieces
    gender = Column(String(20), nullable=False)     # hombre, mujer
    size = Column(String(10), nullable=False)       # S, M, L
    price = Column(Float, default=0)
    image_url = Column(Text, nullable=False)        # URL or base64
    image_path = Column(Text, nullable=True)        # local path (dev only)
    created_at = Column(DateTime, default=datetime.utcnow)


class TryOnResult(Base):
    __tablename__ = "tryon_results"

    id = Column(String(20), primary_key=True)
    garment_ids = Column(Text, nullable=False)      # comma-separated garment IDs
    model_image_url = Column(Text, nullable=True)
    result_image_url = Column(Text, nullable=False)
    video_url = Column(Text, nullable=True)
    gender = Column(String(20), nullable=True)
    size = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        print(f"Database tables may already exist: {e}")
        # Tables already exist, that's fine
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
