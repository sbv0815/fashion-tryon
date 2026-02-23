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
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from dotenv import load_dotenv

load_dotenv()

from models import init_db, get_db, Garment, TryOnResult, Client, UsageLog, AdminSettings

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
app = FastAPI(title="Fashion Virtual Try-On", version="2.0.0")

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
    # Ensure admin settings exist
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
    """Automatically log usage and calculate costs."""
    settings = get_settings(db)
    cost_usd = credits * PRICE_PER_CREDIT.get(settings.fashn_plan, 0.0675)

    client = db.query(Client).filter(Client.id == client_id).first()
    charge_cop = 0
    if client:
        if usage_type == "tryon":
            charge_cop = client.price_per_outfit
        else:
            charge_cop = client.price_per_video

    log = UsageLog(
        id=uuid.uuid4().hex[:12],
        client_id=client_id,
        usage_type=usage_type,
        garments_desc=garments_desc,
        credits_used=credits,
        cost_usd=round(cost_usd, 4),
        charge_cop=charge_cop,
        result_id=result_id,
        notes=notes,
        created_at=datetime.utcnow(),
    )
    db.add(log)
    db.commit()
    return log


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
    """Ensure garment image exists as local file. Restore from DB if needed."""
    if garment.image_path and Path(garment.image_path).exists():
        return garment.image_path

    # Restore from base64 in database
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
    """Admin/management view"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tienda", response_class=HTMLResponse)
async def store_view(request: Request):
    """Client-facing store view - no admin elements"""
    return templates.TemplateResponse("store.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    return templates.TemplateResponse("admin.html", {"request": request})


# ============================================================
# ROUTES - GARMENT CATALOG
# ============================================================
@app.post("/api/upload-garment")
async def upload_garment(
    name: str = Form(...), category: str = Form(...),
    gender: str = Form(...), size: str = Form(...),
    price: float = Form(0), image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Save file locally for FASHN API use
    filepath = await save_upload(image, prefix=f"garment_{gender}_{size}_")

    # Also store as base64 in database for persistence across deploys
    img_bytes = Path(filepath).read_bytes()
    ext = Path(filepath).suffix.lower()
    mime = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
    img_b64 = f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

    garment_id = uuid.uuid4().hex[:12]
    garment = Garment(
        id=garment_id, name=name, category=category,
        gender=gender, size=size, price=price,
        image_url=f"/api/garment-image/{garment_id}",
        image_path=filepath,
        image_data=img_b64,
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
    """Serve garment image from database (persistent) or local file."""
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")

    # Try local file first (faster)
    if garment.image_path and Path(garment.image_path).exists():
        from fastapi.responses import FileResponse
        return FileResponse(garment.image_path)

    # Fall back to base64 from database
    if garment.image_data:
        import re
        match = re.match(r'data:(image/\w+);base64,(.+)', garment.image_data)
        if match:
            mime_type = match.group(1)
            img_bytes = base64.b64decode(match.group(2))

            # Restore local file for FASHN API use
            ext = ".jpg" if "jpeg" in mime_type else ".png"
            restored_path = UPLOAD_DIR / f"restored_{garment_id}{ext}"
            restored_path.write_bytes(img_bytes)
            garment.image_path = str(restored_path)
            db.commit()

            from fastapi.responses import Response
            return Response(content=img_bytes, media_type=mime_type)

    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/api/catalog")
async def get_catalog(gender: Optional[str] = None, size: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Garment)
    if gender:
        query = query.filter(Garment.gender == gender)
    if size:
        query = query.filter(Garment.size == size)
    garments = query.order_by(Garment.created_at.desc()).all()
    return {"garments": [{
        "id": g.id, "name": g.name, "category": g.category,
        "gender": g.gender, "size": g.size, "price": g.price,
        "image_url": g.image_url,
    } for g in garments]}


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


# ============================================================
# QR CODE GENERATION
# ============================================================
@app.get("/api/garment/{garment_id}/qr")
async def get_garment_qr(garment_id: str, request: Request, db: Session = Depends(get_db)):
    """Generate QR code for a garment that links to its try-on page."""
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")

    import qrcode
    from io import BytesIO

    # Build URL to garment landing page
    base_url = str(request.base_url).rstrip('/')
    garment_url = f"{base_url}/prenda/{garment_id}"

    # Generate QR
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=2)
    qr.add_data(garment_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#1a1a1a", back_color="#ffffff")

    # Add garment name as label below QR
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
async def get_all_qr_codes(request: Request, db: Session = Depends(get_db)):
    """Generate a printable sheet with all garment QR codes."""
    garments = db.query(Garment).all()
    if not garments:
        raise HTTPException(status_code=404, detail="No garments found")

    import qrcode
    from PIL import Image, ImageDraw, ImageFont
    from io import BytesIO

    base_url = str(request.base_url).rstrip('/')

    # Each QR is 250x290 (250 qr + 40 label)
    qr_w, qr_h = 250, 290
    cols = 3
    rows = (len(garments) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * qr_w + 40, rows * qr_h + 40), '#ffffff')
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()

    for i, g in enumerate(garments):
        col = i % cols
        row = i // cols
        x = 20 + col * qr_w
        y = 20 + row * qr_h

        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=6, border=2)
        qr.add_data(f"{base_url}/prenda/{g.id}")
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="#1a1a1a", back_color="#ffffff").resize((220, 220))
        sheet.paste(qr_img, (x + 15, y))

        # Label
        text = g.name[:25]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (qr_w - tw) // 2, y + 225), text, fill="#1a1a1a", font=font)
        draw.text((x + (qr_w - 40) // 2, y + 245), f"{g.size} · {g.category}", fill="#666666", font=font)

    buf = BytesIO()
    sheet.save(buf, format='PNG')
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=qr_codes_all.png"})


# ============================================================
# GARMENT LANDING PAGE (from QR scan)
# ============================================================
@app.get("/prenda/{garment_id}")
async def garment_landing(garment_id: str, request: Request, db: Session = Depends(get_db)):
    """Landing page when someone scans a garment QR code."""
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Prenda no encontrada")

    return templates.TemplateResponse("garment_landing.html", {
        "request": request,
        "garment": garment,
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
# ROUTES - VIRTUAL TRY-ON (with auto usage logging)
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

    tryon = TryOnResult(id=result_id, garment_ids=garment_id, result_image_url=f"/static/results/tryon_{result_id}.png")
    db.add(tryon)
    db.commit()

    # Auto log usage
    if client_id:
        log_usage(db, client_id, "tryon", 1, garments_desc=garment.name, result_id=result_id, notes="Try-on individual")

    return {
        "success": True, "result_image": f"/static/results/tryon_{result_id}.png",
        "fashn_url": result_url,
        "garment": {"id": garment.id, "name": garment.name, "category": garment.category, "size": garment.size, "price": garment.price}
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

        db.add(TryOnResult(id=result_id, garment_ids=onepiece_id, result_image_url=f"/static/results/outfit_{result_id}.png"))
        db.commit()

        if client_id:
            log_usage(db, client_id, "tryon", 1, garments_desc=garment.name, result_id=result_id, notes="One-piece")

        return {
            "success": True, "result_image": f"/static/results/outfit_{result_id}.png",
            "fashn_url": result_url,
            "garments_used": [{"id": garment.id, "name": garment.name, "category": garment.category, "size": garment.size, "price": garment.price}]
        }

    if not top_id and not bottom_id:
        raise HTTPException(status_code=400, detail="Select at least one garment")

    model_path = await save_upload(model_image, prefix="model_")
    current_image_b64 = image_to_base64_url(model_path)
    current_fashn_url = None
    garments_used = []
    total_credits = 0

    if top_id:
        top = db.query(Garment).filter(Garment.id == top_id).first()
        if not top:
            raise HTTPException(status_code=404, detail="Top not found")
        result_top = await fashn_run("tryon-v1.6", {
            "model_image": current_image_b64,
            "garment_image": image_to_base64_url(ensure_garment_file(top, db)),
            "category": "tops", "mode": "performance", "garment_photo_type": "auto"
        })
        output_urls = result_top.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating top")
        current_fashn_url = output_urls[0]
        garments_used.append({"id": top.id, "name": top.name, "category": top.category, "size": top.size, "price": top.price})
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
        result_bottom = await fashn_run("tryon-v1.6", {
            "model_image": current_image_b64,
            "garment_image": image_to_base64_url(ensure_garment_file(bottom, db)),
            "category": "bottoms", "mode": "performance", "garment_photo_type": "auto"
        })
        output_urls = result_bottom.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating bottom")
        current_fashn_url = output_urls[0]
        garments_used.append({"id": bottom.id, "name": bottom.name, "category": bottom.category, "size": bottom.size, "price": bottom.price})
        total_credits += 1

    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"outfit_{result_id}.png"
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(current_fashn_url)
        result_path.write_bytes(img_resp.content)

    db.add(TryOnResult(id=result_id, garment_ids=",".join([g["id"] for g in garments_used]), result_image_url=f"/static/results/outfit_{result_id}.png"))
    db.commit()

    # Auto log usage
    if client_id:
        garment_names = " + ".join([g["name"] for g in garments_used])
        log_usage(db, client_id, "tryon", total_credits, garments_desc=garment_names, result_id=result_id, notes=f"Outfit ({total_credits} prendas)")

    return {
        "success": True, "result_image": f"/static/results/outfit_{result_id}.png",
        "fashn_url": current_fashn_url, "garments_used": garments_used
    }


# ============================================================
# ROUTES - VIDEO (with auto usage logging)
# ============================================================
@app.post("/api/generate-video")
async def generate_video(
    image_url: str = Form(...),
    client_id: str = Form(""),
    resolution: str = Form("720p"),
    db: Session = Depends(get_db)
):
    result = await fashn_run("image-to-video", {"image": image_url}, timeout=180)
    output = result.get("output", [])
    if not output:
        raise HTTPException(status_code=500, detail="No video generated")

    video_url = output[0] if isinstance(output, list) else output

    # Auto log usage
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
async def get_usage_logs(
    client_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
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
    notes: str = Form(""),
    db: Session = Depends(get_db)
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
        "plan": settings.fashn_plan,
        "cop_rate": settings.cop_rate,
        "price_per_credit": PRICE_PER_CREDIT.get(settings.fashn_plan, 0.0675),
        "total_clients": len(clients),
        "total_generations": len(logs),
        "total_fotos": fotos,
        "total_videos": videos,
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
    """Remove garments that have no image_data (old uploads lost after deploy)."""
    old_garments = db.query(Garment).filter(Garment.image_data == None).all()
    count = len(old_garments)
    for g in old_garments:
        db.delete(g)
    db.commit()
    return {"success": True, "deleted": count, "message": f"Eliminadas {count} prendas sin imagen"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)