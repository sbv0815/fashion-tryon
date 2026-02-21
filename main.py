"""
Fashion Virtual Try-On Platform
FastAPI backend with FASHN.ai API integration
"""
import os
import time
import uuid
import base64
import json
import httpx
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ============================================================
# CONFIGURATION
# ============================================================
FASHN_API_KEY ="fa-5OaGApRs978K-ViF5znttMmol3ZaU37fNA7E6"
FASHN_BASE_URL = "https://api.fashn.ai/v1"

UPLOAD_DIR = Path("static/uploads")
RESULTS_DIR = Path("static/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Size charts (chest cm ranges)
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

# In-memory catalog (in production, use a database)
CATALOG = {}

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(title="Fashion Virtual Try-On", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============================================================
# FASHN API HELPERS
# ============================================================
async def fashn_run(model_name: str, inputs: dict, timeout: int = 120) -> dict:
    """Submit a job to FASHN API and poll until complete."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FASHN_API_KEY}"
    }
    payload = {
        "model_name": model_name,
        "inputs": inputs
    }

    async with httpx.AsyncClient(timeout=30) as client:
        # Submit job
        resp = await client.post(f"{FASHN_BASE_URL}/run", json=payload, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"FASHN API error: {resp.text}")
        
        data = resp.json()
        prediction_id = data.get("id")
        if not prediction_id:
            raise HTTPException(status_code=500, detail="No prediction ID returned")

        # Poll for result
        start = time.time()
        while time.time() - start < timeout:
            status_resp = await client.get(
                f"{FASHN_BASE_URL}/status/{prediction_id}",
                headers=headers
            )
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
    """Save uploaded file and return its path."""
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{prefix}{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename
    content = await file.read()
    filepath.write_bytes(content)
    return str(filepath)


def image_to_base64_url(filepath: str) -> str:
    """Convert local image to base64 data URL for API."""
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
# ROUTES - API
# ============================================================
@app.post("/api/upload-garment")
async def upload_garment(
    name: str = Form(...),
    category: str = Form(...),  # tops, bottoms, one-pieces
    gender: str = Form(...),    # hombre, mujer
    size: str = Form(...),      # S, M, L
    price: float = Form(0),
    image: UploadFile = File(...)
):
    """Upload a garment to the catalog."""
    filepath = await save_upload(image, prefix=f"garment_{gender}_{size}_")
    
    garment_id = uuid.uuid4().hex[:12]
    CATALOG[garment_id] = {
        "id": garment_id,
        "name": name,
        "category": category,
        "gender": gender,
        "size": size,
        "price": price,
        "image_path": filepath,
        "image_url": f"/{filepath}"
    }
    
    return {"success": True, "garment": CATALOG[garment_id]}


@app.get("/api/catalog")
async def get_catalog(gender: Optional[str] = None, size: Optional[str] = None):
    """Get catalog, optionally filtered."""
    items = list(CATALOG.values())
    if gender:
        items = [i for i in items if i["gender"] == gender]
    if size:
        items = [i for i in items if i["size"] == size]
    return {"garments": items}


@app.post("/api/estimate-size")
async def estimate_size(
    gender: str = Form(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    chest_cm: Optional[float] = Form(None),
    waist_cm: Optional[float] = Form(None),
):
    """
    Estimate clothing size based on measurements.
    If chest/waist not provided, estimate from height+weight.
    """
    # Simple estimation if no direct measurements
    if not chest_cm:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if gender == "mujer":
            chest_cm = 72 + (bmi * 1.1)
        else:
            chest_cm = 78 + (bmi * 1.2)
    
    if not waist_cm:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if gender == "mujer":
            waist_cm = 52 + (bmi * 1.0)
        else:
            waist_cm = 62 + (bmi * 1.1)

    # Determine size
    chart = SIZE_CHART.get(gender, SIZE_CHART["mujer"])
    estimated_size = "M"  # default
    
    for size_name, ranges in chart.items():
        if ranges["min_chest"] <= chest_cm <= ranges["max_chest"]:
            estimated_size = size_name
            break
    else:
        # If outside ranges, pick closest
        if chest_cm < chart["S"]["min_chest"]:
            estimated_size = "S"
        elif chest_cm > chart["L"]["max_chest"]:
            estimated_size = "L"

    return {
        "estimated_size": estimated_size,
        "gender": gender,
        "measurements": {
            "chest_cm": round(chest_cm, 1),
            "waist_cm": round(waist_cm, 1),
            "height_cm": height_cm,
            "weight_kg": weight_kg
        }
    }


@app.post("/api/try-on")
async def virtual_try_on(
    model_image: UploadFile = File(...),
    garment_id: str = Form(...)
):
    """
    Virtual try-on: combine user/model photo with a garment.
    Returns the generated image URL.
    """
    if garment_id not in CATALOG:
        raise HTTPException(status_code=404, detail="Garment not found")
    
    garment = CATALOG[garment_id]
    
    # Save model image
    model_path = await save_upload(model_image, prefix="model_")
    
    # Convert both images to base64 for API
    model_b64 = image_to_base64_url(model_path)
    garment_b64 = image_to_base64_url(garment["image_path"])
    
    # Call FASHN Virtual Try-On
    result = await fashn_run(
        model_name="tryon-v1.6",
        inputs={
            "model_image": model_b64,
            "garment_image": garment_b64,
            "category": garment["category"],
            "mode": "balanced",
            "garment_photo_type": "flat-lay"
        }
    )
    
    output_urls = result.get("output", [])
    if not output_urls:
        raise HTTPException(status_code=500, detail="No output generated")
    
    # Download and save result locally
    result_url = output_urls[0]
    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"tryon_{result_id}.png"
    
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(result_url)
        result_path.write_bytes(img_resp.content)
    
    return {
        "success": True,
        "result_image": f"/static/results/tryon_{result_id}.png",
        "result_local_path": str(result_path),
        "fashn_url": result_url,
        "garment": garment
    }


@app.post("/api/try-on-outfit")
async def virtual_try_on_outfit(
    model_image: UploadFile = File(...),
    top_id: str = Form(None),
    bottom_id: str = Form(None),
    onepiece_id: str = Form(None),
):
    """
    Try-on a complete outfit (top + bottom or one-piece).
    Chains multiple FASHN calls: first top, then bottom on the result.
    """
    if onepiece_id:
        # Single garment (dress/onepiece)
        if onepiece_id not in CATALOG:
            raise HTTPException(status_code=404, detail="One-piece not found")
        garment = CATALOG[onepiece_id]
        model_path = await save_upload(model_image, prefix="model_")
        model_b64 = image_to_base64_url(model_path)
        garment_b64 = image_to_base64_url(garment["image_path"])

        result = await fashn_run(
            model_name="tryon-v1.6",
            inputs={
                "model_image": model_b64,
                "garment_image": garment_b64,
                "category": "one-pieces",
                "mode": "balanced",
                "garment_photo_type": "flat-lay"
            }
        )
        output_urls = result.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="No output generated")

        result_url = output_urls[0]
        result_id = uuid.uuid4().hex[:8]
        result_path = RESULTS_DIR / f"outfit_{result_id}.png"
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(result_url)
            result_path.write_bytes(img_resp.content)

        return {
            "success": True,
            "result_image": f"/static/results/outfit_{result_id}.png",
            "fashn_url": result_url,
            "garments_used": [garment]
        }

    if not top_id and not bottom_id:
        raise HTTPException(status_code=400, detail="Select at least one garment")

    model_path = await save_upload(model_image, prefix="model_")
    current_image_b64 = image_to_base64_url(model_path)
    current_fashn_url = None
    garments_used = []

    # Step 1: Apply TOP first
    if top_id:
        if top_id not in CATALOG:
            raise HTTPException(status_code=404, detail="Top not found")
        top = CATALOG[top_id]
        top_b64 = image_to_base64_url(top["image_path"])

        result_top = await fashn_run(
            model_name="tryon-v1.6",
            inputs={
                "model_image": current_image_b64,
                "garment_image": top_b64,
                "category": "tops",
                "mode": "balanced",
                "garment_photo_type": "flat-lay"
            }
        )
        output_urls = result_top.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating top try-on")

        current_fashn_url = output_urls[0]
        garments_used.append(top)

        # Download intermediate result for next step
        intermediate_path = RESULTS_DIR / f"intermediate_{uuid.uuid4().hex[:8]}.png"
        async with httpx.AsyncClient() as client:
            img_resp = await client.get(current_fashn_url)
            intermediate_path.write_bytes(img_resp.content)
        current_image_b64 = image_to_base64_url(str(intermediate_path))

    # Step 2: Apply BOTTOM on result of top
    if bottom_id:
        if bottom_id not in CATALOG:
            raise HTTPException(status_code=404, detail="Bottom not found")
        bottom = CATALOG[bottom_id]
        bottom_b64 = image_to_base64_url(bottom["image_path"])

        result_bottom = await fashn_run(
            model_name="tryon-v1.6",
            inputs={
                "model_image": current_image_b64,
                "garment_image": bottom_b64,
                "category": "bottoms",
                "mode": "balanced",
                "garment_photo_type": "flat-lay"
            }
        )
        output_urls = result_bottom.get("output", [])
        if not output_urls:
            raise HTTPException(status_code=500, detail="Failed generating bottom try-on")

        current_fashn_url = output_urls[0]
        garments_used.append(bottom)

    # Save final result
    result_id = uuid.uuid4().hex[:8]
    result_path = RESULTS_DIR / f"outfit_{result_id}.png"
    async with httpx.AsyncClient() as client:
        img_resp = await client.get(current_fashn_url)
        result_path.write_bytes(img_resp.content)

    return {
        "success": True,
        "result_image": f"/static/results/outfit_{result_id}.png",
        "fashn_url": current_fashn_url,
        "garments_used": garments_used
    }


@app.post("/api/generate-video")
async def generate_video(
    image_url: str = Form(...)
):
    """
    Generate a fashion video (runway/desfile) from a try-on result image.
    """
    result = await fashn_run(
        model_name="image-to-video",
        inputs={
            "image": image_url
        },
        timeout=180  # Videos take longer
    )
    
    output = result.get("output", [])
    if not output:
        raise HTTPException(status_code=500, detail="No video generated")
    
    return {
        "success": True,
        "video_url": output[0] if isinstance(output, list) else output
    }


@app.get("/api/credits")
async def check_credits():
    """Check FASHN API credit balance."""
    headers = {"Authorization": f"Bearer {FASHN_API_KEY}"}
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{FASHN_BASE_URL}/credits", headers=headers)
        if resp.status_code == 200:
            return resp.json()
        return {"error": "Could not fetch credits", "status": resp.status_code}


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
