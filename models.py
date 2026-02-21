"""
Database models for Fashion Virtual Try-On + Admin Dashboard
"""
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./fashion_tryon.db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Garment(Base):
    __tablename__ = "garments"
    id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    gender = Column(String(20), nullable=False)
    size = Column(String(10), nullable=False)
    price = Column(Float, default=0)
    image_url = Column(Text, nullable=False)
    image_path = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class TryOnResult(Base):
    __tablename__ = "tryon_results"
    id = Column(String(20), primary_key=True)
    garment_ids = Column(Text, nullable=False)
    model_image_url = Column(Text, nullable=True)
    result_image_url = Column(Text, nullable=False)
    video_url = Column(Text, nullable=True)
    gender = Column(String(20), nullable=True)
    size = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Client(Base):
    __tablename__ = "clients"
    id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), nullable=True)
    phone = Column(String(50), nullable=True)
    price_per_outfit = Column(Float, default=2000)
    price_per_video = Column(Float, default=5000)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(String(20), primary_key=True)
    client_id = Column(String(20), nullable=False)
    usage_type = Column(String(30), nullable=False)  # tryon, video-480p, video-720p, video-1080p
    garments_desc = Column(Text, nullable=True)
    credits_used = Column(Integer, default=1)
    cost_usd = Column(Float, default=0)
    charge_cop = Column(Float, default=0)
    result_id = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AdminSettings(Base):
    __tablename__ = "admin_settings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    fashn_plan = Column(String(20), default="tier1")  # ondemand, tier1, tier2, tier3
    cop_rate = Column(Float, default=4200)
    updated_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        print(f"Database tables may already exist: {e}")
        pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
