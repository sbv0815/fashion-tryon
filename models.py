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
    website = Column(Text, nullable=True)
    panel_password = Column(String(100), nullable=True)  # simple password for store panel
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
    shared = Column(Boolean, default=False)
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
    """Track when users share their try-on results on social media."""
    __tablename__ = "share_logs"
    id = Column(String(20), primary_key=True)
    result_id = Column(String(20), nullable=False)
    garment_id = Column(String(20), nullable=True)
    store_id = Column(String(20), nullable=True)
    platform = Column(String(30), nullable=False)  # instagram, whatsapp, tiktok, facebook, copy_link
    created_at = Column(DateTime, default=datetime.utcnow)


class TryOnView(Base):
    """Track each time someone opens the try-on for a garment (from QR scan etc)."""
    __tablename__ = "tryon_views"
    id = Column(String(20), primary_key=True)
    garment_id = Column(String(20), nullable=False)
    store_id = Column(String(20), nullable=True)
    source = Column(String(30), nullable=True)  # qr, collection_page, direct
    created_at = Column(DateTime, default=datetime.utcnow)


class GarmentFeedback(Base):
    """Customer feedback/rating after trying on a garment."""
    __tablename__ = "garment_feedback"
    id = Column(String(20), primary_key=True)
    result_id = Column(String(20), nullable=False)
    garment_id = Column(String(20), nullable=False)
    store_id = Column(String(20), nullable=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    comment = Column(Text, nullable=True)
    would_buy = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Collection(Base):
    """A curated collection/lookbook for a store."""
    __tablename__ = "collections"
    id = Column(String(20), primary_key=True)
    store_id = Column(String(20), nullable=False)
    name = Column(String(200), nullable=False)
    slug = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    cover_color = Column(String(10), default="#1a1a1a")
    active = Column(Boolean, default=True)
    garment_ids = Column(Text, nullable=True)  # comma-separated garment IDs
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
    except Exception as e:
        print(f"Database tables may already exist: {e}")
        pass

    # Auto-migrate: add missing columns/tables
    try:
        from sqlalchemy import text, inspect
        insp = inspect(engine)
        existing_tables = insp.get_table_names()

        # Create new tables if not exist
        for table_name in ['stores', 'clients', 'usage_logs', 'admin_settings',
                           'share_logs', 'tryon_views', 'collections', 'garment_feedback']:
            if table_name not in existing_tables:
                Base.metadata.tables[table_name].create(bind=engine)
                print(f"Created {table_name} table")

        # Check garments table for new columns
        if 'garments' in existing_tables:
            existing_cols = [c['name'] for c in insp.get_columns('garments')]
            with engine.connect() as conn:
                new_cols = {
                    'image_data': 'TEXT', 'store_id': 'VARCHAR(20)',
                    'reference': 'VARCHAR(100)', 'color': 'VARCHAR(50)',
                    'material': 'VARCHAR(100)', 'tryon_enabled': 'BOOLEAN DEFAULT FALSE',
                }
                for col_name, col_type in new_cols.items():
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE garments ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        print(f"Added {col_name} column to garments")

        # Check tryon_results for new columns
        if 'tryon_results' in existing_tables:
            existing_cols = [c['name'] for c in insp.get_columns('tryon_results')]
            with engine.connect() as conn:
                for col_name, col_type in {'store_id': 'VARCHAR(20)', 'shared': 'BOOLEAN DEFAULT FALSE'}.items():
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE tryon_results ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        print(f"Added {col_name} to tryon_results")

        # Check stores for new columns
        if 'stores' in existing_tables:
            existing_cols = [c['name'] for c in insp.get_columns('stores')]
            with engine.connect() as conn:
                for col_name, col_type in {'instagram': 'VARCHAR(100)', 'website': 'TEXT', 'panel_password': 'VARCHAR(100)'}.items():
                    if col_name not in existing_cols:
                        conn.execute(text(f"ALTER TABLE stores ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                        print(f"Added {col_name} to stores")

    except Exception as e:
        print(f"Migration note: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()