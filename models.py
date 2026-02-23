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
    slug = Column(String(100), nullable=False, unique=True)  # URL-friendly name: /tienda/slug
    logo_url = Column(Text, nullable=True)
    logo_path = Column(Text, nullable=True)
    primary_color = Column(String(10), default="#c9a55a")  # brand color
    phone = Column(String(50), nullable=True)
    email = Column(String(200), nullable=True)
    address = Column(Text, nullable=True)
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
    image_data = Column(Text, nullable=True)  # base64 encoded image for persistence
    store_id = Column(String(20), nullable=True)  # links to Store
    reference = Column(String(100), nullable=True)  # product reference/SKU
    color = Column(String(50), nullable=True)
    material = Column(String(100), nullable=True)
    tryon_enabled = Column(Boolean, default=False)  # Only enabled garments allow virtual try-on
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

    # Auto-migrate: add missing columns/tables
    try:
        from sqlalchemy import text, inspect
        insp = inspect(engine)

        # Create stores table if not exists
        if 'stores' not in insp.get_table_names():
            Base.metadata.tables['stores'].create(bind=engine)
            print("Created stores table")

        # Check garments table for new columns
        if 'garments' in insp.get_table_names():
            existing_cols = [c['name'] for c in insp.get_columns('garments')]
            with engine.connect() as conn:
                if 'image_data' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN image_data TEXT"))
                    conn.commit()
                    print("Added image_data column to garments")
                if 'store_id' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN store_id VARCHAR(20)"))
                    conn.commit()
                    print("Added store_id column to garments")
                if 'reference' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN reference VARCHAR(100)"))
                    conn.commit()
                    print("Added reference column to garments")
                if 'color' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN color VARCHAR(50)"))
                    conn.commit()
                    print("Added color column to garments")
                if 'material' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN material VARCHAR(100)"))
                    conn.commit()
                    print("Added material column to garments")
                if 'tryon_enabled' not in existing_cols:
                    conn.execute(text("ALTER TABLE garments ADD COLUMN tryon_enabled BOOLEAN DEFAULT FALSE"))
                    conn.commit()
                    print("Added tryon_enabled column to garments")

        # Check for other tables
        if 'clients' not in insp.get_table_names():
            Base.metadata.tables['clients'].create(bind=engine)
            print("Created clients table")
        if 'usage_logs' not in insp.get_table_names():
            Base.metadata.tables['usage_logs'].create(bind=engine)
            print("Created usage_logs table")
        if 'admin_settings' not in insp.get_table_names():
            Base.metadata.tables['admin_settings'].create(bind=engine)
            print("Created admin_settings table")

    except Exception as e:
        print(f"Migration note: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()