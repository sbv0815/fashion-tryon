"""
Fashion Virtual Try-On Platform
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
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from models import init_db, get_db, Garment, TryOnResult

# ============================================================
# CONFIGURATION
# ============================================================
FASHN_API_KEY = os.getenv("FASHN_API_KEY", "YOUR_API_KEY_HERE")
FASHN_BASE_URL = "https://api.fashn.ai/v1"

UPLOAD_DIR = Path("static/uploads")
RESULTS_DIR = Path("static/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# ============================================================
# FASHN API HELPERS
# ============================================================
async def fashn_run(model_name: str, inputs: dict, timeout: int = 120) -> dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FASHN_API_KEY}"
    }
    payload = {"model_name": model_name, "inputs": inputs}

    async with httpx.AsyncClient(timeout=30) as client:
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

            await asyncio.sleep(2)

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


# ============================================================
# ROUTES - PAGES
# ============================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================================
# ROUTES - GARMENT CATALOG
# ============================================================
@app.post("/api/upload-garment")
async def upload_garment(
    name: str = Form(...),
    category: str = Form(...),
    gender: str = Form(...),
    size: str = Form(...),
    price: float = Form(0),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    filepath = await save_upload(image, prefix=f"garment_{gender}_{size}_")
    garment_id = uuid.uuid4().hex[:12]
    garment = Garment(
        id=garment_id, name=name, category=category,
        gender=gender, size=size, price=price,
        image_url=f"/{filepath}", image_path=filepath,
    )
    db.add(garment)
    db.commit()
    db.refresh(garment)

    return {"success": True, "garment": {
        "id": garment.id, "name": garment.name, "category": garment.category,
        "gender": garment.gender, "size": garment.size, "price": garment.price,
        "image_url": garment.image_url,
    }}


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
    db: Session = Depends(get_db)
):
    garment = db.query(Garment).filter(Garment.id == garment_id).first()
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")

    model_path = await save_upload(model_image, prefix="model_")
    model_b64 = image_to_base64_url(model_path)
    garment_b64 = image_to_base64_url(garment.image_path)

    result = await fashn_run("tryon-v1.6", {
        "model_image": model_b64, "garment_image": garment_b64,
        "category": garment.category, "mode": "balanced", "garment_photo_type": "flat-lay"
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

    return {
        "success": True, "result_image": f"/static/results/tryon_{result_id}.png",
        "fashn_url": result_url,
        "garment": {"id": garment.id, "name": garment.name, "category": garment.category, "size": garment.size, "price": garment.price}
    }


@app.post("/api/try-on-outfit")
async def virtual_try_on_outfit(
    model_image: UploadFile = File(...),
    top_id: str = Form(None), bottom_id: str = Form(None),
    onepiece_id: str = Form(None),
    db: Session = Depends(get_db)
):
    if onepiece_id:
        garment = db.query(Garment).filter(Garment.id == onepiece_id).first()
        if not garment:
            raise HTTPException(status_code=404, detail="One-piece not found")

        model_path = await save_upload(model_image, prefix="model_")
        result = await fashn_run("tryon-v1.6", {
            "model_image": image_to_base64_url(model_path),
            "garment_image": image_to_base64_url(garment.image_path),
            "category": "one-pieces", "mode": "balanced", "garment_photo_type": "flat-lay"
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

    if top_id:
        top = db.query(Garment).filter(Garment.id == top_id).first()
        if not top:
            raise HTTPException(status_code=404, detail="Top not found")

        result_top = await fashn_run("tryon-v1.6", {
            "model_image": current_image_b64,
            "garment_image": image_to_base64_url(top.image_path),
            "category": "tops", "mode": "balanced", "garment_photo_type": "flat-lay"
        })
        output_urls = result_top.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating top")

        current_fashn_url = output_urls[0]
        garments_used.append({"id": top.id, "name": top.name, "category": top.category, "size": top.size, "price": top.price})

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
            "garment_image": image_to_base64_url(bottom.image_path),
            "category": "bottoms", "mode": "balanced", "garment_photo_type": "flat-lay"
        })
        output_urls = result_bottom.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating bottom")

        current_fashn_url = output_urls[0]
        garments_used.append({"id": bottom.id, "name": bottom.name, "category": bottom.category, "size": bottom.size, "price": bottom.price})

    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"outfit_{result_id}.png"
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(current_fashn_url)
        result_path.write_bytes(img_resp.content)

    db.add(TryOnResult(id=result_id, garment_ids=",".join([g["id"] for g in garments_used]), result_image_url=f"/static/results/outfit_{result_id}.png"))
    db.commit()

    return {
        "success": True, "result_image": f"/static/results/outfit_{result_id}.png",
        "fashn_url": current_fashn_url, "garments_used": garments_used
    }


# ============================================================
# ROUTES - VIDEO & CREDITS
# ============================================================
@app.post("/api/generate-video")
async def generate_video(image_url: str = Form(...)):
    result = await fashn_run("image-to-video", {"image": image_url}, timeout=180)
    output = result.get("output", [])
    if not output:
        raise HTTPException(status_code=500, detail="No video generated")
    return {"success": True, "video_url": output[0] if isinstance(output, list) else output}


@app.get("/api/credits")
async def check_credits():
    headers = {"Authorization": f"Bearer {FASHN_API_KEY}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{FASHN_BASE_URL}/credits", headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Could not fetch credits", "status": resp.status_code}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)