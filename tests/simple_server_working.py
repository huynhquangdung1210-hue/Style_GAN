"""
Simple FastAPI server with basic job tracking (without RQ for now)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import asyncio
import uuid
import logging
import io
import time
from enum import Enum
from datetime import datetime

from utils.config import get_settings
from utils.storage_simple import get_storage_client, StorageError
from utils.image_processing import get_image_processor, QualitySetting

logger = logging.getLogger(__name__)

# Simple in-memory job storage for demonstration
jobs_storage: Dict[str, Dict[str, Any]] = {}

class Quality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

class StyleTransferRequest(BaseModel):
    subject_image: str  # base64 encoded
    style_image: str    # base64 encoded
    quality: Optional[Quality] = Quality.MEDIUM
    style_strength: Optional[float] = 0.8
    num_inference_steps: Optional[int] = 20

app = FastAPI(
    title="Style Transfer API",
    description="Neural Style Transfer API with MinIO storage",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "message": "Style Transfer API is running",
        "total_jobs": len(jobs_storage)
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Style Transfer API", "docs": "/docs"}

@app.post("/v1/generate")
async def generate_style_transfer(request: StyleTransferRequest):
    """Generate a style transfer image."""
    try:
        import base64
        
        # Validate base64 images first
        image_processor = get_image_processor()
        
        try:
            subject_data = base64.b64decode(request.subject_image)
            style_data = base64.b64decode(request.style_image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Validate images
        subject_validation = await image_processor.validate_image_data(subject_data)
        if not subject_validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid subject image: {subject_validation['error']}")
        
        style_validation = await image_processor.validate_image_data(style_data)
        if not style_validation["valid"]:
            raise HTTPException(status_code=400, detail=f"Invalid style image: {style_validation['error']}")
        
        # Create job
        job_id = str(uuid.uuid4())
        
        # Store job in memory
        jobs_storage[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "progress": 0.0,
            "quality": request.quality.value,
            "style_strength": request.style_strength,
            "num_inference_steps": request.num_inference_steps,
            "subject_image_data": request.subject_image,
            "style_image_data": request.style_image
        }
        
        # Start processing in background
        asyncio.create_task(process_job_async(job_id))
        
        # Estimate processing time based on quality
        time_estimates = {"low": 15, "medium": 30, "high": 60, "ultra": 120}
        estimated_time = time_estimates.get(request.quality.value, 30)
        
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": "pending",
                "message": "Style transfer job queued for processing",
                "estimated_time": estimated_time,
                "quality": request.quality.value,
                "style_strength": request.style_strength,
                "subject_image_info": {
                    "format": subject_validation["format"],
                    "size": subject_validation["size"],
                    "file_size": subject_validation["file_size"]
                },
                "style_image_info": {
                    "format": style_validation["format"],
                    "size": style_validation["size"],
                    "file_size": style_validation["file_size"]
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

async def process_job_async(job_id: str):
    """Process a job asynchronously."""
    try:
        # Update job status
        if job_id not in jobs_storage:
            return
        
        jobs_storage[job_id]["status"] = "processing"
        jobs_storage[job_id]["started_at"] = datetime.utcnow().isoformat()
        jobs_storage[job_id]["progress"] = 0.1
        
        # Get job data
        job_data = jobs_storage[job_id]
        image_processor = get_image_processor()
        storage_client = get_storage_client()
        
        # Load images
        subject_image = await image_processor.load_image_from_base64(
            job_data["subject_image_data"]
        )
        style_image = await image_processor.load_image_from_base64(
            job_data["style_image_data"]
        )
        
        jobs_storage[job_id]["progress"] = 0.3
        
        # Preprocess images
        quality_setting = QualitySetting(job_data["quality"])
        subject_processed = await image_processor.preprocess_image(
            subject_image, quality_setting
        )
        style_processed = await image_processor.preprocess_image(
            style_image, quality_setting
        )
        
        jobs_storage[job_id]["progress"] = 0.5
        
        # Load ML model and process
        from models.style_transfer import get_model
        model = get_model("neural")  # Use real neural style transfer
        await model.load_model()
        
        jobs_storage[job_id]["progress"] = 0.6
        
        # Perform style transfer
        result_image = await model.transfer_style(
            content_image=subject_processed,
            style_image=style_processed,
            style_strength=job_data["style_strength"],
            num_inference_steps=job_data["num_inference_steps"],
            quality=job_data["quality"]
        )
        
        jobs_storage[job_id]["progress"] = 0.8
        
        # Post-process and save result
        from utils.image_processing import ImageFormat
        result_bytes = await image_processor.postprocess_image(
            result_image, 
            ImageFormat.JPEG, 
            quality_setting
        )
        
        # Upload to storage
        result_key = f"jobs/{job_id}/result.jpg"
        result_url = await storage_client.upload_bytes(result_bytes, result_key)
        
        # Create thumbnail
        thumbnail = await image_processor.create_thumbnail(result_image, (256, 256))
        thumbnail_bytes = await image_processor.postprocess_image(
            thumbnail, ImageFormat.JPEG, QualitySetting.MEDIUM
        )
        
        thumbnail_key = f"jobs/{job_id}/thumbnail.jpg"
        await storage_client.upload_bytes(thumbnail_bytes, thumbnail_key)
        
        # Update job as completed
        jobs_storage[job_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result_url": result_url,
            "progress": 1.0
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        if job_id in jobs_storage:
            jobs_storage[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow().isoformat(),
                "error": str(e)
            })

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a style transfer job."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id].copy()
    # Remove sensitive data
    job_data.pop("subject_image_data", None)
    job_data.pop("style_image_data", None)
    
    return job_data

@app.get("/v1/results/{job_id}")
async def get_result_image(job_id: str):
    """Get the result image for a completed job."""
    try:
        if job_id not in jobs_storage:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_data = jobs_storage[job_id]
        if job_data["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed (status: {job_data['status']})")
        
        if "result_url" not in job_data:
            raise HTTPException(status_code=404, detail="Result not available")
        
        # Get image from storage
        storage_client = get_storage_client()
        result_key = f"jobs/{job_id}/result.jpg"
        
        try:
            image_data = await storage_client.download_bytes(result_key)
            
            return StreamingResponse(
                io.BytesIO(image_data),
                media_type="image/jpeg",
                headers={"Content-Disposition": f"inline; filename=style_transfer_{job_id}.jpg"}
            )
        except StorageError as e:
            logger.error(f"Storage error retrieving result {job_id}: {e}")
            raise HTTPException(status_code=404, detail="Result image not found in storage")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result image {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/thumbnails/{job_id}")
async def get_thumbnail_image(job_id: str):
    """Get the thumbnail image for a completed job."""
    if job_id not in jobs_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs_storage[job_id]
    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job_data['status']})")
    
    # Get thumbnail from storage
    storage_client = get_storage_client()
    thumbnail_key = f"jobs/{job_id}/thumbnail.jpg"
    
    try:
        image_data = await storage_client.download_bytes(thumbnail_key)
        
        return StreamingResponse(
            io.BytesIO(image_data),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=thumbnail_{job_id}.jpg"}
        )
    except StorageError:
        raise HTTPException(status_code=404, detail="Thumbnail not found in storage")

@app.get("/v1/queue/stats")
async def get_queue_stats():
    """Get queue statistics."""
    status_counts = {}
    for job in jobs_storage.values():
        status = job["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "total_jobs": len(jobs_storage),
        "status_counts": status_counts,
        "queue_length": status_counts.get("pending", 0) + status_counts.get("processing", 0)
    }

if __name__ == "__main__":
    print("🚀 Starting Style Transfer API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )