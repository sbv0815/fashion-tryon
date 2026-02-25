"""
Fashion Virtual Try-On Platform + Admin Dashboard
FastAPI backend with FASHN.ai API integration
Production-ready with PostgreSQL database
"""
import os
import time
import uuid
import base64
import httpx
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv

load_dotenv()

from models import (init_db, get_db, Garment, TryOnResult, Client, UsageLog,
                    AdminSettings, Store, ShareLog, TryOnView, Collection, GarmentFeedback)

# ============================================================
# CONFIGURATION
# ============================================================
FASHN_API_KEY = os.getenv("FASHN_API_KEY", "YOUR_API_KEY_HERE")
FASHN_BASE_URL = "https://api.fashn.ai/v1"

UPLOAD_DIR = Path("static/uploads")
RESULTS_DIR = Path("static/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PRICE_PER_CREDIT = {
    "ondemand": 0.075,
    "tier1": 0.0675,
    "tier2": 0.06,
    "tier3": 0.0488,
}

SIZE_CHART = {
    "mujer": {
        "S":  {"min_chest": 80, "max_chest": 88,  "min_waist": 60, "max_waist": 68},
        "M":  {"min_chest": 88, "max_chest": 96,  "min_waist": 68, "max_waist": 76},
        "L":  {"min_chest": 96, "max_chest": 104, "min_waist": 76, "max_waist": 84},
    },
    "hombre": {
        "S":  {"min_chest": 88, "max_chest": 96,  "min_waist": 72, "max_waist": 80},
        "M":  {"min_chest": 96, "max_chest": 104, "min_waist": 80, "max_waist": 88},
        "L":  {"min_chest": 104, "max_chest": 112, "min_waist": 88, "max_waist": 96},
    }
}

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="Fashion Virtual Try-On", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def on_startup():
    init_db()
    db = next(get_db())
    try:
        settings = db.query(AdminSettings).first()
        if not settings:
            db.add(AdminSettings(fashn_plan="tier1", cop_rate=4200))
            db.commit()
    finally:
        db.close()


# ============================================================
# HELPERS
# ============================================================
def get_settings(db: Session) -> AdminSettings:
    settings = db.query(AdminSettings).first()
    if not settings:
        settings = AdminSettings(fashn_plan="tier1", cop_rate=4200)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


def log_usage(db: Session, client_id: str, usage_type: str, credits: int,
              garments_desc: str = "", result_id: str = "", notes: str = ""):
    settings = get_settings(db)
    cost_usd = credits * PRICE_PER_CREDIT.get(settings.fashn_plan, 0.0675)
    client = db.query(Client).filter(Client.id == client_id).first()
    charge_cop = 0
    if client:
        if usage_type == "tryon":
            charge_cop = client.price_per_outfit * credits
        else:
            charge_cop = client.price_per_video
    log = UsageLog(
        id=uuid.uuid4().hex[:12], client_id=client_id, usage_type=usage_type,
        garments_desc=garments_desc, credits_used=credits,
        cost_usd=round(cost_usd, 4), charge_cop=charge_cop,
        result_id=result_id, notes=notes, created_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    return log


def track_view(db: Session, garment_id: str, store_id: str = None, source: str = "qr"):
    """Track a garment view/interaction."""
    view = TryOnView(
        id=uuid.uuid4().hex[:12], garment_id=garment_id,
        store_id=store_id, source=source, created_at=datetime.utcnow()
    )
    db.add(view)
    db.commit()


async def fashn_run(model_name: str, inputs: dict, timeout: int = 120) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FASHN_API_KEY}"
    }
    payload = {"model_name": model_name, "inputs": inputs}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(f"{FASHN_BASE_URL}/run", json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"FASHN API error: {resp.text}")
        data = resp.json()
        prediction_id = data.get("id")
        if not prediction_id:
            raise HTTPException(status_code=500, detail="No prediction ID returned")
        start = time.time()
        while time.time() - start < timeout:
            status_resp = await client.get(f"{FASHN_BASE_URL}/status/{prediction_id}", headers=headers)
            status_data = status_resp.json()
            status = status_data.get("status")
            if status == "completed":
                return status_data
            elif status == "failed":
                error_msg = status_data.get("error", {}).get("message", "Unknown error")
                raise HTTPException(status_code=500, detail=f"FASHN generation failed: {error_msg}")
            await asyncio.sleep(1.5)
        raise HTTPException(status_code=504, detail="FASHN API timeout")


async def save_upload(file: UploadFile, prefix: str = "") -> str:
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{prefix}{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename
    content = await file.read()
    filepath.write_bytes(content)
    return str(filepath)


def image_to_base64_url(filepath: str) -> str:
    data = Path(filepath).read_bytes()
    b64 = base64.b64encode(data).decode()
    ext = Path(filepath).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64}"


def ensure_garment_file(garment, db: Session) -> str:
    if garment.image_path and Path(garment.image_path).exists():
        return garment.image_path
    if garment.image_data:
        import re
        match = re.match(r'data:(image/\w+);base64,(.+)', garment.image_data)
        if match:
            mime_type = match.group(1)
            img_bytes = base64.b64decode(match.group(2))
            ext = ".jpg" if "jpeg" in mime_type else ".png"
            restored_path = UPLOAD_DIR / f"restored_{garment.id}{ext}"
            restored_path.write_bytes(img_bytes)
            garment.image_path = str(restored_path)
            db.commit()
            return str(restored_path)
    raise HTTPException(status_code=404, detail=f"Image for garment {garment.id} not found")


# ============================================================
# ROUTES - PAGES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tienda", response_class=HTMLResponse)
async def store_view(request: Request):
    return templates.TemplateResponse("store.html", {"request": request})


@app.get("/tienda/{store_slug}", response_class=HTMLResponse)
async def store_branded_view(store_slug: str, request: Request, db: Session = Depends(get_db)):
    store = db.query(Store).filter(Store.slug == store_slug, Store.active == True).first()
    if not store:
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    return templates.TemplateResponse("store_branded.html", {"request": request, "store": store})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


# ============================================================
# STORE PANEL (client-facing dashboard)
# ============================================================
@app.get("/panel/{store_slug}", response_class=HTMLResponse)
async def store_panel(store_slug: str, request: Request, db: Session = Depends(get_db)):
    """Store owner panel - requires password auth via JS."""
    store = db.query(Store).filter(Store.slug == store_slug).first()
    if not store:
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    return templates.TemplateResponse("store_panel.html", {"request": request, "store": store})


@app.post("/api/panel/auth")
async def panel_auth(
    store_slug: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    """Authenticate store owner for panel access."""
    store = db.query(Store).filter(Store.slug == store_slug).first()
    if not store:
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    if store.panel_password and store.panel_password == password:
        return {"success": True, "store_id": store.id, "store_name": store.name}
    raise HTTPException(status_code=401, detail="Contraseña incorrecta")


# ============================================================
# ROUTES - STORES (COMPANIES)
# ============================================================
@app.post("/api/store")
async def create_store(
    name: str = Form(...), slug: str = Form(...),
    phone: str = Form(""), email: str = Form(""),
    address: str = Form(""), primary_color: str = Form("#c9a55a"),
    instagram: str = Form(""), website: str = Form(""),
    panel_password: str = Form(""),
    logo: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    import re
    slug = re.sub(r'[^a-z0-9-]', '', slug.lower().replace(' ', '-'))
    existing = db.query(Store).filter(Store.slug == slug).first()
    if existing:
        raise HTTPException(status_code=400, detail="Slug ya existe")
    store_id = uuid.uuid4().hex[:12]
    # Auto-generate password if not provided
    if not panel_password:
        panel_password = uuid.uuid4().hex[:8]
    logo_path = None
    if logo and logo.filename:
        ext = Path(logo.filename).suffix or ".png"
        logo_path = str(UPLOAD_DIR / f"logo_{store_id}{ext}")
        content = await logo.read()
        Path(logo_path).write_bytes(content)
    store = Store(
        id=store_id, name=name, slug=slug,
        phone=phone, email=email, address=address,
        primary_color=primary_color, instagram=instagram, website=website,
        panel_password=panel_password,
        logo_path=logo_path,
        logo_url=f"/api/store-logo/{store_id}" if logo_path else None
    )
    db.add(store)
    db.commit()
    return {"success": True, "store": {
        "id": store.id, "name": store.name, "slug": store.slug,
        "panel_password": panel_password,
        "panel_url": f"/panel/{store.slug}"
    }}


@app.get("/api/stores")
async def list_stores(db: Session = Depends(get_db)):
    stores = db.query(Store).order_by(Store.created_at.desc()).all()
    return [{"id": s.id, "name": s.name, "slug": s.slug, "phone": s.phone,
             "email": s.email, "primary_color": s.primary_color,
             "instagram": s.instagram, "website": s.website,
             "panel_password": s.panel_password,
             "logo_url": s.logo_url, "active": s.active,
             "garment_count": db.query(Garment).filter(Garment.store_id == s.id).count(),
             "store_url": f"/tienda/{s.slug}",
             "panel_url": f"/panel/{s.slug}",
             } for s in stores]


@app.get("/api/store-logo/{store_id}")
async def get_store_logo(store_id: str, db: Session = Depends(get_db)):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store or not store.logo_path or not Path(store.logo_path).exists():
        raise HTTPException(status_code=404, detail="Logo not found")
    from fastapi.responses import FileResponse
    return FileResponse(store.logo_path)


@app.delete("/api/store/{store_id}")
async def delete_store(store_id: str, db: Session = Depends(get_db)):
    store = db.query(Store).filter(Store.id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    db.delete(store)
    db.commit()
    return {"success": True}


@app.get("/api/catalog/{store_id}")
async def get_store_catalog(store_id: str, gender: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Garment).filter(Garment.store_id == store_id)
    if gender:
        query = query.filter(Garment.gender == gender)
    garments = query.order_by(Garment.created_at.desc()).all()
    return {"garments": [{"id": g.id, "name": g.name, "category": g.category,
        "gender": g.gender, "size": g.size, "price": g.price,
        "image_url": g.image_url, "reference": g.reference, "color": g.color,
    } for g in garments]}


# ============================================================
# ROUTES - GARMENT CATALOG
# ============================================================
@app.post("/api/upload-garment")
async def upload_garment(
    name: str = Form(...), category: str = Form(...),
    gender: str = Form(...), size: str = Form(...),
    price: float = Form(0), image: UploadFile = File(...),
    store_id: str = Form(""), reference: str = Form(""),
    color: str = Form(""), material: str = Form(""),
    db: Session = Depends(get_db)
):
    filepath = await save_upload(image, prefix=f"garment_{gender}_{size}_")
    img_bytes = Path(filepath).read_bytes()
    ext = Path(filepath).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    img_b64 = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"
    garment_id = uuid.uuid4().hex[:12]
    garment = Garment(
        id=garment_id, name=name, category=category,
        gender=gender, size=size, price=price,
        image_url=f"/api/garment-image/{garment_id}",
        image_path=filepath, image_data=img_b64,
        store_id=store_id if store_id else None,
        reference=reference if reference else None,
        color=color if color else None,
        material=material if material else None,
    )
    db.add(garment)
    db.commit()
    db.refresh(garment)
    return {"success": True, "garment": {
        "id": garment.id, "name": garment.name, "category": garment.category,
        "gender": garment.gender, "size": garment.size, "price": garment.price,
        "image_url": garment.image_url,
    }}


@app.get("/api/garment-image/{garment_id}")
async def get_garment_image(garment_id: str, db: Session = Depends(get_db)):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    if garment.image_path and Path(garment.image_path).exists():
        from fastapi.responses import FileResponse
        return FileResponse(garment.image_path)
    if garment.image_data:
        import re
        match = re.match(r'data:(image/\w+);base64,(.+)', garment.image_data)
        if match:
            mime_type = match.group(1)
            img_bytes = base64.b64decode(match.group(2))
            ext = ".jpg" if "jpeg" in mime_type else ".png"
            restored_path = UPLOAD_DIR / f"restored_{garment_id}{ext}"
            restored_path.write_bytes(img_bytes)
            garment.image_path = str(restored_path)
            db.commit()
            from fastapi.responses import Response
            return Response(content=img_bytes, media_type=mime_type)
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/catalog")
async def get_catalog(gender: Optional[str] = None, size: Optional[str] = None,
                      store_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Garment)
    if store_id:
        query = query.filter(Garment.store_id == store_id)
    if gender:
        query = query.filter(Garment.gender == gender)
    if size:
        query = query.filter(Garment.size == size)
    garments = query.order_by(Garment.created_at.desc()).all()
    store_name = None
    if store_id:
        client = db.query(Client).filter(Client.id == store_id).first()
        if client:
            store_name = client.name
    return {"garments": [{
        "id": g.id, "name": g.name, "category": g.category,
        "gender": g.gender, "size": g.size, "price": g.price,
        "image_url": g.image_url, "reference": g.reference,
        "store_id": g.store_id, "tryon_enabled": g.tryon_enabled if g.tryon_enabled else False,
    } for g in garments], "store_name": store_name}


@app.get("/api/store-info/{store_id}")
async def get_store_info(store_id: str, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == store_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    return {"id": client.id, "name": client.name, "phone": client.phone, "email": client.email}


@app.delete("/api/garment/{garment_id}")
async def delete_garment(garment_id: str, db: Session = Depends(get_db)):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    if garment.image_path and Path(garment.image_path).exists():
        Path(garment.image_path).unlink()
    db.delete(garment)
    db.commit()
    return {"success": True}


@app.put("/api/garment/{garment_id}")
async def update_garment(
    garment_id: str,
    name: str = Form(None), category: str = Form(None),
    size: str = Form(None), price: float = Form(None),
    gender: str = Form(None), tryon_enabled: str = Form(None),
    db: Session = Depends(get_db)
):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    if name: garment.name = name
    if category: garment.category = category
    if size: garment.size = size
    if price is not None: garment.price = price
    if gender: garment.gender = gender
    if tryon_enabled is not None:
        garment.tryon_enabled = tryon_enabled.lower() in ('true', '1', 'yes')
    db.commit()
    return {"success": True, "garment": {"id": garment.id, "name": garment.name,
            "category": garment.category, "tryon_enabled": garment.tryon_enabled}}


# ============================================================
# QR CODE GENERATION
# ============================================================
@app.get("/api/garment/{garment_id}/qr")
async def get_garment_qr(garment_id: str, request: Request, db: Session = Depends(get_db)):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    import qrcode
    from io import BytesIO
    base_url = str(request.base_url).rstrip('/')
    garment_url = f"{base_url}/prenda/{garment_id}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=2)
    qr.add_data(garment_url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="#1a1a1a", back_color="#ffffff")
    from PIL import Image, ImageDraw, ImageFont
    qr_size = img.size[0]
    label_height = 50
    final = Image.new('RGB', (qr_size, qr_size + label_height), '#ffffff')
    final.paste(img, (0, 0))
    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    text = garment.name[:30]
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (qr_size - tw) // 2
    draw.text((x, qr_size + 10), text, fill="#1a1a1a", font=font)
    buf = BytesIO()
    final.save(buf, format='PNG')
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=qr_{garment.name}_{garment_id}.png"})


@app.get("/api/garments/qr-all")
async def get_all_qr_codes(request: Request, store_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Garment)
    if store_id:
        query = query.filter(Garment.store_id == store_id)
    garments = query.all()
    if not garments:
        raise HTTPException(status_code=404, detail="No garments found")
    import qrcode
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO
    base_url = str(request.base_url).rstrip('/')
    store_name = ""
    if store_id:
        store = db.query(Store).filter(Store.id == store_id).first()
        if store:
            store_name = store.name
    qr_w, qr_h = 250, 310
    cols = 3
    rows = (len(garments) + cols - 1) // cols
    title_h = 60 if store_name else 0
    sheet = Image.new('RGB', (cols * qr_w + 40, rows * qr_h + 40 + title_h), '#ffffff')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = font_bold = font_small = ImageFont.load_default()
    if store_name:
        bbox = draw.textbbox((0, 0), store_name, font=font_bold)
        tw = bbox[2] - bbox[0]
        draw.text(((cols * qr_w + 40 - tw) // 2, 15), store_name, fill="#1a1a1a", font=font_bold)
    for i, g in enumerate(garments):
        col = i % cols
        row = i // cols
        x = 20 + col * qr_w
        y = 20 + row * qr_h + title_h
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=6, border=2)
        qr.add_data(f"{base_url}/prenda/{g.id}")
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#1a1a1a", back_color="#ffffff").resize((220, 220))
        sheet.paste(qr_img, (x + 15, y))
        text = g.name[:25]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (qr_w - tw) // 2, y + 225), text, fill="#1a1a1a", font=font)
        meta = f"{g.size} · {g.category}"
        if g.reference:
            meta += f" · Ref: {g.reference}"
        bbox2 = draw.textbbox((0, 0), meta, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text((x + (qr_w - tw2) // 2, y + 248), meta, fill="#666666", font=font_small)
        if g.price and g.price > 0:
            price_text = f"${g.price:,.0f}"
            bbox3 = draw.textbbox((0, 0), price_text, font=font)
            tw3 = bbox3[2] - bbox3[0]
            draw.text((x + (qr_w - tw3) // 2, y + 268), price_text, fill="#1a7a3a", font=font)
    buf = BytesIO()
    sheet.save(buf, format='PNG')
    buf.seek(0)
    filename = f"qr_codes_{store_name or 'all'}.png".replace(' ', '_')
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"})


# ============================================================
# GARMENT LANDING PAGE (from QR scan) — with view tracking
# ============================================================
@app.get("/prenda/{garment_id}")
async def garment_landing(garment_id: str, request: Request, db: Session = Depends(get_db)):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Prenda no encontrada")

    # Track the view
    track_view(db, garment_id, garment.store_id, source="qr")

    store = None
    if garment.store_id:
        store = db.query(Store).filter(Store.id == garment.store_id).first()

    return templates.TemplateResponse("garment_landing.html", {
        "request": request, "garment": garment,
        "store_name": store.name if store else "",
        "store": store,
    })


# ============================================================
# ROUTES - SIZE ESTIMATION
# ============================================================
@app.post("/api/estimate-size")
async def estimate_size(
    gender: str = Form(...), height_cm: float = Form(...),
    weight_kg: float = Form(...), chest_cm: Optional[float] = Form(None),
    waist_cm: Optional[float] = Form(None),
):
    if not chest_cm:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        chest_cm = (72 + bmi * 1.1) if gender == "mujer" else (78 + bmi * 1.2)
    if not waist_cm:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        waist_cm = (52 + bmi * 1.0) if gender == "mujer" else (62 + bmi * 1.1)
    chart = SIZE_CHART.get(gender, SIZE_CHART["mujer"])
    estimated_size = "M"
    for size_name, ranges in chart.items():
        if ranges["min_chest"] <= chest_cm <= ranges["max_chest"]:
            estimated_size = size_name
            break
    else:
        if chest_cm < chart["S"]["min_chest"]:
            estimated_size = "S"
        elif chest_cm > chart["L"]["max_chest"]:
            estimated_size = "L"
    return {
        "estimated_size": estimated_size, "gender": gender,
        "measurements": {
            "chest_cm": round(chest_cm, 1), "waist_cm": round(waist_cm, 1),
            "height_cm": height_cm, "weight_kg": weight_kg
        }
    }


# ============================================================
# ROUTES - VIRTUAL TRY-ON
# ============================================================
@app.post("/api/try-on")
async def virtual_try_on(
    model_image: UploadFile = File(...),
    garment_id: str = Form(...),
    client_id: str = Form(""),
    db: Session = Depends(get_db)
):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    if not client_id and garment.store_id:
        client_id = garment.store_id
    model_path = await save_upload(model_image, prefix="model_")
    model_b64 = image_to_base64_url(model_path)
    garment_filepath = ensure_garment_file(garment, db)
    garment_b64 = image_to_base64_url(garment_filepath)
    result = await fashn_run("tryon-v1.6", {
        "model_image": model_b64, "garment_image": garment_b64,
        "category": garment.category, "mode": "performance", "garment_photo_type": "auto"
    })
    output_urls = result.get("output", [])
    if not output_urls:
        raise HTTPException(status_code=500, detail="No output generated")
    result_url = output_urls[0]
    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"tryon_{result_id}.png"
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(result_url)
        result_path.write_bytes(img_resp.content)
    tryon = TryOnResult(
        id=result_id, garment_ids=garment_id,
        result_image_url=f"/static/results/tryon_{result_id}.png",
        store_id=garment.store_id
    )
    db.add(tryon)
    db.commit()
    if client_id:
        log_usage(db, client_id, "tryon", 1, garments_desc=garment.name,
                  result_id=result_id, notes="Try-on individual")

    # Get store info for share page
    store = None
    if garment.store_id:
        store = db.query(Store).filter(Store.id == garment.store_id).first()

    return {
        "success": True,
        "result_image": f"/static/results/tryon_{result_id}.png",
        "result_id": result_id,
        "share_url": f"/resultado/{result_id}",
        "fashn_url": result_url,
        "garment": {"id": garment.id, "name": garment.name, "category": garment.category,
                    "size": garment.size, "price": garment.price},
        "store": {"name": store.name, "instagram": store.instagram, "phone": store.phone,
                  "primary_color": store.primary_color} if store else None,
    }


@app.post("/api/try-on-outfit")
async def virtual_try_on_outfit(
    model_image: UploadFile = File(...),
    top_id: str = Form(None), bottom_id: str = Form(None),
    onepiece_id: str = Form(None), client_id: str = Form(""),
    db: Session = Depends(get_db)
):
    if onepiece_id:
        garment = db.query(Garment).filter(Garment.id == onepiece_id).first()
        if not garment:
            raise HTTPException(status_code=404, detail="One-piece not found")
        if not client_id and garment.store_id:
            client_id = garment.store_id
        model_path = await save_upload(model_image, prefix="model_")
        result = await fashn_run("tryon-v1.6", {
            "model_image": image_to_base64_url(model_path),
            "garment_image": image_to_base64_url(ensure_garment_file(garment, db)),
            "category": "one-pieces", "mode": "performance", "garment_photo_type": "auto"
        })
        output_urls = result.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="No output generated")
        result_url = output_urls[0]
        result_id = uuid.uuid4().hex[:8]
        result_path = RESULTS_DIR / f"outfit_{result_id}.png"
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(result_url)
            result_path.write_bytes(img_resp.content)
        db.add(TryOnResult(id=result_id, garment_ids=onepiece_id,
               result_image_url=f"/static/results/outfit_{result_id}.png", store_id=garment.store_id))
        db.commit()
        if client_id:
            log_usage(db, client_id, "tryon", 1, garments_desc=garment.name,
                      result_id=result_id, notes="One-piece")
        store = db.query(Store).filter(Store.id == garment.store_id).first() if garment.store_id else None
        return {
            "success": True, "result_image": f"/static/results/outfit_{result_id}.png",
            "result_id": result_id, "share_url": f"/resultado/{result_id}",
            "fashn_url": result_url,
            "garments_used": [{"id": garment.id, "name": garment.name, "category": garment.category,
                               "size": garment.size, "price": garment.price}],
            "store": {"name": store.name, "instagram": store.instagram, "phone": store.phone,
                      "primary_color": store.primary_color} if store else None,
        }

    if not top_id and not bottom_id:
        raise HTTPException(status_code=400, detail="Select at least one garment")
    if not client_id:
        check_id = top_id or bottom_id
        if check_id:
            check_garment = db.query(Garment).filter(Garment.id == check_id).first()
            if check_garment and check_garment.store_id:
                client_id = check_garment.store_id
    model_path = await save_upload(model_image, prefix="model_")
    current_image_b64 = image_to_base64_url(model_path)
    current_fashn_url = None
    garments_used = []
    total_credits = 0
    store_id_for_result = None

    if top_id:
        top = db.query(Garment).filter(Garment.id == top_id).first()
        if not top:
            raise HTTPException(status_code=404, detail="Top not found")
        store_id_for_result = top.store_id
        result_top = await fashn_run("tryon-v1.6", {
            "model_image": current_image_b64,
            "garment_image": image_to_base64_url(ensure_garment_file(top, db)),
            "category": top.category if top.category in ["tops","bottoms","one-pieces"] else "tops",
            "mode": "performance", "garment_photo_type": "auto"
        })
        output_urls = result_top.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating top")
        current_fashn_url = output_urls[0]
        garments_used.append({"id": top.id, "name": top.name, "category": top.category,
                              "size": top.size, "price": top.price})
        total_credits += 1
        intermediate_path = RESULTS_DIR / f"intermediate_{uuid.uuid4().hex[:8]}.png"
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(current_fashn_url)
            intermediate_path.write_bytes(img_resp.content)
        current_image_b64 = image_to_base64_url(str(intermediate_path))

    if bottom_id:
        bottom = db.query(Garment).filter(Garment.id == bottom_id).first()
        if not bottom:
            raise HTTPException(status_code=404, detail="Bottom not found")
        if not store_id_for_result:
            store_id_for_result = bottom.store_id
        result_bottom = await fashn_run("tryon-v1.6", {
            "model_image": current_image_b64,
            "garment_image": image_to_base64_url(ensure_garment_file(bottom, db)),
            "category": bottom.category if bottom.category in ["tops","bottoms","one-pieces"] else "bottoms",
            "mode": "performance", "garment_photo_type": "auto"
        })
        output_urls = result_bottom.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating bottom")
        current_fashn_url = output_urls[0]
        garments_used.append({"id": bottom.id, "name": bottom.name, "category": bottom.category,
                              "size": bottom.size, "price": bottom.price})
        total_credits += 1

    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"outfit_{result_id}.png"
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(current_fashn_url)
        result_path.write_bytes(img_resp.content)
    db.add(TryOnResult(id=result_id, garment_ids=",".join([g["id"] for g in garments_used]),
           result_image_url=f"/static/results/outfit_{result_id}.png", store_id=store_id_for_result))
    db.commit()
    if client_id:
        garment_names = " + ".join([g["name"] for g in garments_used])
        log_usage(db, client_id, "tryon", total_credits, garments_desc=garment_names,
                  result_id=result_id, notes=f"Outfit ({total_credits} prendas)")

    store = db.query(Store).filter(Store.id == store_id_for_result).first() if store_id_for_result else None
    return {
        "success": True, "result_image": f"/static/results/outfit_{result_id}.png",
        "result_id": result_id, "share_url": f"/resultado/{result_id}",
        "fashn_url": current_fashn_url, "garments_used": garments_used,
        "store": {"name": store.name, "instagram": store.instagram, "phone": store.phone,
                  "primary_color": store.primary_color} if store else None,
    }


# ============================================================
# SHARE RESULT PAGE — branded frame + social sharing
# ============================================================
@app.get("/resultado/{result_id}", response_class=HTMLResponse)
async def share_result_page(result_id: str, request: Request, db: Session = Depends(get_db)):
    """Public page showing try-on result with branded frame and share buttons."""
    result = db.query(TryOnResult).filter(TryOnResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Resultado no encontrado")

    # Get garment info
    garment_ids = result.garment_ids.split(",") if result.garment_ids else []
    garments = db.query(Garment).filter(Garment.id.in_(garment_ids)).all() if garment_ids else []

    # Get store info
    store = None
    if result.store_id:
        store = db.query(Store).filter(Store.id == result.store_id).first()

    base_url = str(request.base_url).rstrip('/')
    share_url = f"{base_url}/resultado/{result_id}"

    return templates.TemplateResponse("share_result.html", {
        "request": request, "result": result, "garments": garments,
        "store": store, "share_url": share_url, "base_url": base_url,
    })


@app.post("/api/share/{result_id}")
async def log_share(result_id: str, platform: str = Form(...), db: Session = Depends(get_db)):
    """Log when a user shares their result on social media."""
    result = db.query(TryOnResult).filter(TryOnResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    # Get garment/store info
    garment_id = result.garment_ids.split(",")[0] if result.garment_ids else None

    share = ShareLog(
        id=uuid.uuid4().hex[:12], result_id=result_id,
        garment_id=garment_id, store_id=result.store_id,
        platform=platform, created_at=datetime.utcnow()
    )
    db.add(share)
    result.shared = True
    db.commit()
    return {"success": True}


@app.get("/api/share-image/{result_id}")
async def get_branded_share_image(result_id: str, db: Session = Depends(get_db)):
    """Generate a branded image with store frame for sharing."""
    result = db.query(TryOnResult).filter(TryOnResult.id == result_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")

    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO

    # Load result image
    result_path = Path(result.result_image_url.lstrip('/'))
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    img = Image.open(result_path).convert('RGB')

    # Get store info
    store = None
    if result.store_id:
        store = db.query(Store).filter(Store.id == result.store_id).first()

    # Create branded frame
    padding = 40
    footer_h = 80
    w, h = img.size
    canvas = Image.new('RGB', (w + padding * 2, h + padding + footer_h), '#ffffff')
    canvas.paste(img, (padding, padding // 2))

    draw = ImageDraw.Draw(canvas)

    try:
        font_name = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_tag = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font_name = font_sub = font_tag = ImageFont.load_default()

    footer_y = h + padding // 2 + 8

    if store:
        # Store name
        draw.text((padding, footer_y), store.name, fill="#1a1a1a", font=font_name)
        # Instagram handle
        if store.instagram:
            draw.text((padding, footer_y + 30), f"@{store.instagram}", fill="#666666", font=font_sub)
        # Hashtag
        draw.text((padding, footer_y + 52), "Probado con IA ✨ #FashionTryOn", fill="#999999", font=font_tag)
    else:
        draw.text((padding, footer_y), "Fashion Try-On", fill="#1a1a1a", font=font_name)
        draw.text((padding, footer_y + 30), "Probado con IA ✨", fill="#999999", font=font_sub)

    # Color accent bar at top
    accent_color = store.primary_color if store else "#c9a55a"
    draw.rectangle([(0, 0), (canvas.size[0], 4)], fill=accent_color)

    buf = BytesIO()
    canvas.save(buf, format='JPEG', quality=92)
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename=tryon_{result_id}.jpg"})


# ============================================================
# GARMENT FEEDBACK (rating + comment + buy intent)
# ============================================================
@app.post("/api/feedback")
async def submit_feedback(
    result_id: str = Form(...),
    garment_ids: str = Form(...),
    rating: int = Form(...),
    comment: str = Form(""),
    would_buy: str = Form("false"),
    db: Session = Depends(get_db)
):
    """Submit customer feedback after trying on garments."""
    result = db.query(TryOnResult).filter(TryOnResult.id == result_id).first()
    store_id = result.store_id if result else None

    for gid in garment_ids.split(","):
        gid = gid.strip()
        if not gid:
            continue
        fb = GarmentFeedback(
            id=uuid.uuid4().hex[:12],
            result_id=result_id,
            garment_id=gid,
            store_id=store_id,
            rating=max(1, min(5, rating)),
            comment=comment if comment else None,
            would_buy=would_buy.lower() in ('true', '1', 'yes'),
            created_at=datetime.utcnow()
        )
        db.add(fb)
    db.commit()
    return {"success": True}


@app.get("/api/feedback/{store_id}")
async def get_store_feedback(store_id: str, db: Session = Depends(get_db)):
    """Get all feedback for a store's garments."""
    feedbacks = db.query(GarmentFeedback).filter(
        GarmentFeedback.store_id == store_id
    ).order_by(GarmentFeedback.created_at.desc()).limit(200).all()

    result = []
    for f in feedbacks:
        garment = db.query(Garment).filter(Garment.id == f.garment_id).first()
        result.append({
            "id": f.id, "garment_id": f.garment_id,
            "garment_name": garment.name if garment else "—",
            "garment_image": garment.image_url if garment else "",
            "rating": f.rating, "comment": f.comment,
            "would_buy": f.would_buy,
            "created_at": f.created_at.isoformat() if f.created_at else "",
        })
    return {"feedback": result}


# ============================================================
# COLLECTIONS / LOOKBOOK
# ============================================================
@app.post("/api/collection")
async def create_collection(
    store_id: str = Form(...), name: str = Form(...),
    slug: str = Form(""), description: str = Form(""),
    cover_color: str = Form("#1a1a1a"), garment_ids: str = Form(""),
    db: Session = Depends(get_db)
):
    import re
    if not slug:
        slug = re.sub(r'[^a-z0-9-]', '', name.lower().replace(' ', '-'))
    slug = f"{slug}-{uuid.uuid4().hex[:4]}"

    collection = Collection(
        id=uuid.uuid4().hex[:12], store_id=store_id, name=name,
        slug=slug, description=description, cover_color=cover_color,
        garment_ids=garment_ids, active=True
    )
    db.add(collection)
    db.commit()
    return {"success": True, "collection": {
        "id": collection.id, "name": collection.name, "slug": collection.slug,
        "url": f"/coleccion/{collection.slug}"
    }}


@app.get("/api/collections")
async def list_collections(store_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Collection).filter(Collection.active == True)
    if store_id:
        query = query.filter(Collection.store_id == store_id)
    collections = query.order_by(Collection.created_at.desc()).all()
    result = []
    for c in collections:
        garment_count = len(c.garment_ids.split(",")) if c.garment_ids else 0
        store = db.query(Store).filter(Store.id == c.store_id).first()
        result.append({
            "id": c.id, "name": c.name, "slug": c.slug,
            "description": c.description, "cover_color": c.cover_color,
            "store_name": store.name if store else "", "garment_count": garment_count,
            "url": f"/coleccion/{c.slug}", "created_at": c.created_at.isoformat() if c.created_at else "",
        })
    return {"collections": result}


@app.delete("/api/collection/{collection_id}")
async def delete_collection(collection_id: str, db: Session = Depends(get_db)):
    coll = db.query(Collection).filter(Collection.id == collection_id).first()
    if not coll:
        raise HTTPException(status_code=404, detail="Collection not found")
    coll.active = False
    db.commit()
    return {"success": True}


@app.get("/coleccion/{collection_slug}", response_class=HTMLResponse)
async def collection_page(collection_slug: str, request: Request, db: Session = Depends(get_db)):
    """Public collection/lookbook page."""
    collection = db.query(Collection).filter(
        Collection.slug == collection_slug, Collection.active == True
    ).first()
    if not collection:
        raise HTTPException(status_code=404, detail="Colección no encontrada")

    store = db.query(Store).filter(Store.id == collection.store_id).first()

    # Get garments in collection
    garment_ids = [gid.strip() for gid in collection.garment_ids.split(",") if gid.strip()] if collection.garment_ids else []
    if garment_ids:
        garments = db.query(Garment).filter(Garment.id.in_(garment_ids)).all()
    else:
        # If no specific garments, show all from store
        garments = db.query(Garment).filter(Garment.store_id == collection.store_id).all()

    # Track views for analytics
    for g in garments:
        track_view(db, g.id, collection.store_id, source="collection_page")

    return templates.TemplateResponse("collection.html", {
        "request": request, "collection": collection,
        "store": store, "garments": garments,
    })


# ============================================================
# ANALYTICS API
# ============================================================
@app.get("/api/analytics/{store_id}")
async def get_store_analytics(store_id: str, days: int = 30, db: Session = Depends(get_db)):
    """Get analytics data for a store."""
    since = datetime.utcnow() - timedelta(days=days)

    # Views per garment
    views = db.query(
        TryOnView.garment_id, func.count(TryOnView.id).label('views')
    ).filter(
        TryOnView.store_id == store_id, TryOnView.created_at >= since
    ).group_by(TryOnView.garment_id).all()

    views_dict = {v.garment_id: v.views for v in views}

    # Try-ons per garment
    results = db.query(TryOnResult).filter(
        TryOnResult.store_id == store_id, TryOnResult.created_at >= since
    ).all()

    tryon_count = {}
    for r in results:
        for gid in r.garment_ids.split(","):
            gid = gid.strip()
            tryon_count[gid] = tryon_count.get(gid, 0) + 1

    # Shares per garment
    shares = db.query(
        ShareLog.garment_id, ShareLog.platform, func.count(ShareLog.id).label('count')
    ).filter(
        ShareLog.store_id == store_id, ShareLog.created_at >= since
    ).group_by(ShareLog.garment_id, ShareLog.platform).all()

    shares_by_garment = {}
    shares_by_platform = {}
    for s in shares:
        if s.garment_id:
            shares_by_garment[s.garment_id] = shares_by_garment.get(s.garment_id, 0) + s.count
        shares_by_platform[s.platform] = shares_by_platform.get(s.platform, 0) + s.count

    # Build garment ranking
    garments = db.query(Garment).filter(Garment.store_id == store_id).all()
    ranking = []
    for g in garments:
        ranking.append({
            "id": g.id, "name": g.name, "category": g.category,
            "reference": g.reference, "price": g.price,
            "image_url": g.image_url,
            "views": views_dict.get(g.id, 0),
            "tryons": tryon_count.get(g.id, 0),
            "shares": shares_by_garment.get(g.id, 0),
            "engagement": views_dict.get(g.id, 0) + tryon_count.get(g.id, 0) * 3 + shares_by_garment.get(g.id, 0) * 5,
        })
    ranking.sort(key=lambda x: x['engagement'], reverse=True)

    total_views = sum(v.views for v in views)
    total_tryons = sum(tryon_count.values())
    total_shares = sum(shares_by_platform.values())

    # Feedback stats
    feedbacks = db.query(GarmentFeedback).filter(
        GarmentFeedback.store_id == store_id, GarmentFeedback.created_at >= since
    ).all()

    avg_rating = round(sum(f.rating for f in feedbacks) / len(feedbacks), 1) if feedbacks else 0
    buy_intent = len([f for f in feedbacks if f.would_buy])
    total_feedback = len(feedbacks)

    # Per-garment feedback
    feedback_by_garment = {}
    for f in feedbacks:
        if f.garment_id not in feedback_by_garment:
            feedback_by_garment[f.garment_id] = {"ratings": [], "comments": [], "buy_count": 0}
        feedback_by_garment[f.garment_id]["ratings"].append(f.rating)
        if f.comment:
            feedback_by_garment[f.garment_id]["comments"].append(f.comment)
        if f.would_buy:
            feedback_by_garment[f.garment_id]["buy_count"] += 1

    # Enrich ranking with feedback
    for item in ranking:
        fb = feedback_by_garment.get(item["id"], {})
        ratings = fb.get("ratings", [])
        item["avg_rating"] = round(sum(ratings) / len(ratings), 1) if ratings else 0
        item["feedback_count"] = len(ratings)
        item["buy_intent"] = fb.get("buy_count", 0)
        item["comments"] = fb.get("comments", [])[:5]  # last 5

    return {
        "period_days": days,
        "totals": {
            "views": total_views, "tryons": total_tryons,
            "shares": total_shares,
            "conversion_rate": round(total_tryons / total_views * 100, 1) if total_views > 0 else 0,
            "avg_rating": avg_rating,
            "total_feedback": total_feedback,
            "buy_intent": buy_intent,
        },
        "shares_by_platform": shares_by_platform,
        "garment_ranking": ranking,
    }


@app.get("/analytics/{store_slug}", response_class=HTMLResponse)
async def analytics_page(store_slug: str, request: Request, db: Session = Depends(get_db)):
    """Client-facing analytics report page."""
    store = db.query(Store).filter(Store.slug == store_slug).first()
    if not store:
        raise HTTPException(status_code=404, detail="Tienda no encontrada")
    return templates.TemplateResponse("analytics.html", {"request": request, "store": store})


# ============================================================
# ROUTES - VIDEO
# ============================================================
@app.post("/api/generate-video")
async def generate_video(
    image_url: str = Form(...), client_id: str = Form(""),
    resolution: str = Form("720p"), db: Session = Depends(get_db)
):
    result = await fashn_run("image-to-video", {"image": image_url}, timeout=180)
    output = result.get("output", [])
    if not output:
        raise HTTPException(status_code=500, detail="No video generated")
    video_url = output[0] if isinstance(output, list) else output
    if client_id:
        credit_map = {"480p": 1, "720p": 3, "1080p": 6}
        credits = credit_map.get(resolution, 3)
        log_usage(db, client_id, f"video-{resolution}", credits, notes=f"Video desfile {resolution}")
    return {"success": True, "video_url": video_url}


@app.get("/api/credits")
async def check_credits():
    headers = {"Authorization": f"Bearer {FASHN_API_KEY}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{FASHN_BASE_URL}/credits", headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Could not fetch credits", "status": resp.status_code}


# ============================================================
# ADMIN API - CLIENTS
# ============================================================
@app.get("/api/admin/clients")
async def get_clients(db: Session = Depends(get_db)):
    clients = db.query(Client).filter(Client.active == True).order_by(Client.created_at.desc()).all()
    result = []
    for c in clients:
        logs = db.query(UsageLog).filter(UsageLog.client_id == c.id).all()
        total_credits = sum(l.credits_used for l in logs)
        total_cost_usd = sum(l.cost_usd for l in logs)
        total_charge_cop = sum(l.charge_cop for l in logs)
        fotos = len([l for l in logs if l.usage_type == "tryon"])
        videos = len([l for l in logs if l.usage_type.startswith("video")])
        settings = get_settings(db)
        cost_cop = total_cost_usd * settings.cop_rate
        result.append({
            "id": c.id, "name": c.name, "email": c.email, "phone": c.phone,
            "price_per_outfit": c.price_per_outfit, "price_per_video": c.price_per_video,
            "created_at": c.created_at.isoformat() if c.created_at else "",
            "stats": {
                "total_generations": len(logs), "fotos": fotos, "videos": videos,
                "total_credits": total_credits, "cost_usd": round(total_cost_usd, 4),
                "cost_cop": round(cost_cop), "charge_cop": round(total_charge_cop),
                "profit_cop": round(total_charge_cop - cost_cop),
            }
        })
    return {"clients": result}


@app.post("/api/admin/clients")
async def create_client(
    name: str = Form(...), email: str = Form(""),
    phone: str = Form(""), price_per_outfit: float = Form(2000),
    price_per_video: float = Form(5000),
    db: Session = Depends(get_db)
):
    client = Client(
        id=uuid.uuid4().hex[:12], name=name, email=email, phone=phone,
        price_per_outfit=price_per_outfit, price_per_video=price_per_video,
    )
    db.add(client)
    db.commit()
    db.refresh(client)
    return {"success": True, "client": {
        "id": client.id, "name": client.name, "email": client.email,
        "price_per_outfit": client.price_per_outfit, "price_per_video": client.price_per_video,
    }}


@app.put("/api/admin/clients/{client_id}")
async def update_client(
    client_id: str,
    name: str = Form(None), email: str = Form(None), phone: str = Form(None),
    price_per_outfit: float = Form(None), price_per_video: float = Form(None),
    db: Session = Depends(get_db)
):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    if name is not None: client.name = name
    if email is not None: client.email = email
    if phone is not None: client.phone = phone
    if price_per_outfit is not None: client.price_per_outfit = price_per_outfit
    if price_per_video is not None: client.price_per_video = price_per_video
    db.commit()
    return {"success": True}


@app.delete("/api/admin/clients/{client_id}")
async def delete_client(client_id: str, db: Session = Depends(get_db)):
    client = db.query(Client).filter(Client.id == client_id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    client.active = False
    db.commit()
    return {"success": True}


# ============================================================
# ADMIN API - USAGE LOGS
# ============================================================
@app.get("/api/admin/usage")
async def get_usage_logs(client_id: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(UsageLog)
    if client_id:
        query = query.filter(UsageLog.client_id == client_id)
    logs = query.order_by(UsageLog.created_at.desc()).limit(200).all()
    settings = get_settings(db)
    result = []
    for l in logs:
        client = db.query(Client).filter(Client.id == l.client_id).first()
        result.append({
            "id": l.id, "client_id": l.client_id,
            "client_name": client.name if client else "Desconocido",
            "usage_type": l.usage_type, "garments_desc": l.garments_desc,
            "credits_used": l.credits_used, "cost_usd": l.cost_usd,
            "cost_cop": round(l.cost_usd * settings.cop_rate),
            "charge_cop": l.charge_cop, "notes": l.notes,
            "created_at": l.created_at.isoformat() if l.created_at else "",
        })
    return {"logs": result}


@app.post("/api/admin/usage")
async def manual_log_usage(
    client_id: str = Form(...), usage_type: str = Form(...),
    credits_used: int = Form(1), garments_desc: str = Form(""),
    notes: str = Form(""), db: Session = Depends(get_db)
):
    log = log_usage(db, client_id, usage_type, credits_used,
                    garments_desc=garments_desc, notes=notes)
    return {"success": True, "log_id": log.id}


@app.delete("/api/admin/usage/{log_id}")
async def delete_usage_log(log_id: str, db: Session = Depends(get_db)):
    log = db.query(UsageLog).filter(UsageLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    db.delete(log)
    db.commit()
    return {"success": True}


# ============================================================
# ADMIN API - DASHBOARD STATS
# ============================================================
@app.get("/api/admin/dashboard")
async def admin_dashboard_stats(db: Session = Depends(get_db)):
    settings = get_settings(db)
    logs = db.query(UsageLog).all()
    clients = db.query(Client).filter(Client.active == True).all()
    total_credits = sum(l.credits_used for l in logs)
    total_cost_usd = sum(l.cost_usd for l in logs)
    total_charge_cop = sum(l.charge_cop for l in logs)
    total_cost_cop = total_cost_usd * settings.cop_rate
    fotos = len([l for l in logs if l.usage_type == "tryon"])
    videos = len([l for l in logs if l.usage_type.startswith("video")])
    return {
        "plan": settings.fashn_plan, "cop_rate": settings.cop_rate,
        "price_per_credit": PRICE_PER_CREDIT.get(settings.fashn_plan, 0.0675),
        "total_clients": len(clients), "total_generations": len(logs),
        "total_fotos": fotos, "total_videos": videos,
        "total_credits": total_credits,
        "total_cost_usd": round(total_cost_usd, 2),
        "total_cost_cop": round(total_cost_cop),
        "total_charge_cop": round(total_charge_cop),
        "total_profit_cop": round(total_charge_cop - total_cost_cop),
        "margin_pct": round(((total_charge_cop - total_cost_cop) / total_charge_cop * 100) if total_charge_cop > 0 else 0),
    }


# ============================================================
# ADMIN API - SETTINGS
# ============================================================
@app.put("/api/admin/settings")
async def update_settings(
    fashn_plan: str = Form(None), cop_rate: float = Form(None),
    db: Session = Depends(get_db)
):
    settings = get_settings(db)
    if fashn_plan: settings.fashn_plan = fashn_plan
    if cop_rate: settings.cop_rate = cop_rate
    settings.updated_at = datetime.utcnow()
    db.commit()
    return {"success": True, "plan": settings.fashn_plan, "cop_rate": settings.cop_rate}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.delete("/api/admin/cleanup-garments")
async def cleanup_garments(db: Session = Depends(get_db)):
    old_garments = db.query(Garment).filter(Garment.image_data == None).all()
    count = len(old_garments)
    for g in old_garments:
        db.delete(g)
    db.commit()
    return {"success": True, "deleted": count, "message": f"Eliminadas {count} prendas sin imagen"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)