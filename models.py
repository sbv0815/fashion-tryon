"""
Database models for Fashion Virtual Try-On
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


class Store(Base):
    __tablename__ = "stores"
    id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), nullable=False, unique=True)
    logo_url = Column(Text, nullable=True)
    logo_path = Column(Text, nullable=True)
    primary_color = Column(String(10), default="#c9a55a")
    phone = Column(String(50), nullable=True)
    email = Column(String(200), nullable=True)
    address = Column(Text, nullable=True)
    instagram = Column(String(100), nullable=True)
    website = Column(String(300), nullable=True)
    facebook = Column(String(200), nullable=True)
    tiktok = Column(String(100), nullable=True)
    twitter = Column(String(100), nullable=True)
    panel_password = Column(String(100), nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


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
    image_data = Column(Text, nullable=True)
    store_id = Column(String(20), nullable=True)
    reference = Column(String(100), nullable=True)
    color = Column(String(50), nullable=True)
    material = Column(String(100), nullable=True)
    tryon_enabled = Column(Boolean, default=False)
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
    store_id = Column(String(20), nullable=True)
    customer_id = Column(String(20), nullable=True)
    shared = Column(Boolean, default=False)
    show_on_wall = Column(Boolean, default=False)
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


class CustomerProfile(Base):
    __tablename__ = "customer_profiles"
    id = Column(String(20), primary_key=True)
    name = Column(String(200), nullable=False)
    phone = Column(String(50), nullable=True)
    access_token = Column(String(40), nullable=False, unique=True)
    gender = Column(String(20), nullable=True)
    size = Column(String(10), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(String(20), primary_key=True)
    client_id = Column(String(20), nullable=False)
    usage_type = Column(String(30), nullable=False)
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
    fashn_plan = Column(String(20), default="tier1")
    cop_rate = Column(Float, default=4200)
    updated_at = Column(DateTime, default=datetime.utcnow)


class ShareLog(Base):
    __tablename__ = "share_logs"
    id = Column(String(20), primary_key=True)
    result_id = Column(String(20), nullable=True)
    garment_id = Column(String(20), nullable=True)
    store_id = Column(String(20), nullable=True)
    platform = Column(String(30), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class TryOnView(Base):
    __tablename__ = "tryon_views"
    id = Column(String(20), primary_key=True)
    garment_id = Column(String(20), nullable=False)
    store_id = Column(String(20), nullable=True)
    source = Column(String(30), default="qr")
    created_at = Column(DateTime, default=datetime.utcnow)


class Collection(Base):
    __tablename__ = "collections"
    id = Column(String(20), primary_key=True)
    store_id = Column(String(20), nullable=False)
    name = Column(String(200), nullable=False)
    slug = Column(String(150), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    cover_color = Column(String(10), default="#1a1a1a")
    garment_ids = Column(Text, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class GarmentFeedback(Base):
    __tablename__ = "garment_feedback"
    id = Column(String(20), primary_key=True)
    result_id = Column(String(20), nullable=True)
    garment_id = Column(String(20), nullable=True)
    store_id = Column(String(20), nullable=True)
    rating = Column(Integer, default=3)
    comment = Column(Text, nullable=True)
    would_buy = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        print(f"Database tables may already exist: {e}")

    try:
        from sqlalchemy import text, inspect
        insp = inspect(engine)
        tables = insp.get_table_names()

        def add_col(table, column, col_type):
            if table in tables:
                existing = [c['name'] for c in insp.get_columns(table)]
                if column not in existing:
                    with engine.connect() as conn:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"))
                        conn.commit()
                    print(f"  Added {table}.{column}")

        add_col('stores', 'facebook', 'VARCHAR(200)')
        add_col('stores', 'tiktok', 'VARCHAR(100)')
        add_col('stores', 'twitter', 'VARCHAR(100)')
        add_col('tryon_results', 'customer_id', 'VARCHAR(20)')
        add_col('tryon_results', 'show_on_wall', 'BOOLEAN DEFAULT FALSE')
        add_col('tryon_results', 'shared', 'BOOLEAN DEFAULT FALSE')
        add_col('garments', 'image_data', 'TEXT')
        add_col('garments', 'store_id', 'VARCHAR(20)')
        add_col('garments', 'reference', 'VARCHAR(100)')
        add_col('garments', 'color', 'VARCHAR(50)')
        add_col('garments', 'material', 'VARCHAR(100)')
        add_col('garments', 'tryon_enabled', 'BOOLEAN DEFAULT FALSE')

        for tname in ['customer_profiles', 'share_logs', 'tryon_views',
                      'collections', 'garment_feedback', 'stores']:
            if tname not in tables and tname in Base.metadata.tables:
                Base.metadata.tables[tname].create(bind=engine)
                print(f"  Created table {tname}")

    except Exception as e:
        print(f"Migration note: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()