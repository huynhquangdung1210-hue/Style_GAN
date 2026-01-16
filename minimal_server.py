#!/usr/bin/env python3
"""
Minimal working style transfer server
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import uuid
import io
import base64
import asyncio
from typing import Optional
from PIL import Image

# Simple in-memory job storage
jobs_storage = {}

app = FastAPI(title="Style Transfer API", version="1.0.0")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class StyleTransferRequest(BaseModel):
    subject_image: str  # base64
    style_image: str    # base64
    style_strength: float = 0.8
    quality: str = "medium"
    num_inference_steps: int = 20

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "total_jobs": len(jobs_storage)
    }

@app.get("/")
async def root():
    return {"message": "Style Transfer API", "docs": "/docs"}

@app.post("/v1/generate")
async def generate(request: StyleTransferRequest):
    job_id = str(uuid.uuid4())
    
    # Validate base64 images
    try:
        subject_data = base64.b64decode(request.subject_image)
        style_data = base64.b64decode(request.style_image)
        
        # Validate as images
        subject_img = Image.open(io.BytesIO(subject_data))
        style_img = Image.open(io.BytesIO(style_data))
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    # Store job
    jobs_storage[job_id] = {
        "id": job_id,
        "status": "pending", 
        "progress": 0.0,
        "subject_image": request.subject_image,
        "style_image": request.style_image,
        "style_strength": request.style_strength,
        "quality": request.quality,
        "num_inference_steps": request.num_inference_steps,
        "result_path": None,
        "error": None
    }
    
    # Start processing in background
    asyncio.create_task(process_job(job_id))
    
    return {"job_id": job_id, "status": "accepted"}

@app.get("/v1/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job.get("error")
    }

@app.get("/v1/results/{job_id}")
async def get_result(job_id: str):
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_storage[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result not ready")
    
    # Return actual style transfer result if available
    try:
        if "result_data" in job:
            # Use the processed result
            image_data = base64.b64decode(job["result_data"])
        else:
            # Fallback to original subject image
            image_data = base64.b64decode(job["subject_image"])
            
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"attachment; filename=result_{job_id}.jpg"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error retrieving result")

async def process_job(job_id: str):
    """Process a style transfer job with a lightweight, CPU-only blend."""
    if job_id not in jobs_storage:
        return
    
    job = jobs_storage[job_id]
    
    try:
        # Update status
        job["status"] = "processing"
        job["progress"] = 0.1

        # Load images
        subject_data = base64.b64decode(job["subject_image"])
        style_data = base64.b64decode(job["style_image"])

        subject_img = Image.open(io.BytesIO(subject_data)).convert("RGB")
        style_img = Image.open(io.BytesIO(style_data)).convert("RGB")

        job["progress"] = 0.4

        # Lightweight CPU-only "style transfer": resize + blend
        style_resized = style_img.resize(subject_img.size, Image.BICUBIC)
        strength = max(0.0, min(1.0, float(job["style_strength"])))
        result_img = Image.blend(subject_img, style_resized, strength)

        job["progress"] = 0.9

        # Save result as base64 for now
        buffer = io.BytesIO()
        result_img.save(buffer, format="JPEG", quality=90)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()

        job["progress"] = 1.0
        job["status"] = "completed"
        job["result_path"] = f"result_{job_id}.jpg"
        job["result_data"] = result_base64  # Store result data

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"Style transfer error: {e}")  # Debug output

if __name__ == "__main__":
    print("🚀 Starting minimal style transfer server on port 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
