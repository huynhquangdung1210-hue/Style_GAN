"""
Simple FastAPI server for testing the style transfer system
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import asyncio
import uuid
import logging
import io

from utils.config import get_settings
from utils.storage_simple import get_storage_client, StorageError
from utils.job_queue import get_job_queue, StyleTransferJob, JobPriority
from utils.image_processing import get_image_processor

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Style Transfer API",
    description="Neural Style Transfer API with MinIO storage",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    status: str
    version: str
    storage_type: str
    storage_healthy: bool

from enum import Enum

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        settings = get_settings()
        storage_client = get_storage_client()
        
        # Test storage connectivity
        storage_healthy = True
        try:
            # Simple test - check if we can access storage client
            _ = storage_client.settings
        except Exception:
            storage_healthy = False
        
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            storage_type=settings.storage.storage_type,
            storage_healthy=storage_healthy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
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
        job = StyleTransferJob(
            job_id=job_id,
            subject_image_data=request.subject_image,
            style_image_data=request.style_image,
            style_strength=request.style_strength,
            num_inference_steps=request.num_inference_steps,
            quality=request.quality.value,
            priority=JobPriority.NORMAL
        )
        
        # Enqueue job
        job_queue = get_job_queue()
        await job_queue.enqueue_job(job)
        
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
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a style transfer job."""
    try:
        job_queue = get_job_queue()
        job = await job_queue.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        response_data = {
            "job_id": job.job_id,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "quality": job.quality,
            "style_strength": job.style_strength,
            "num_inference_steps": job.num_inference_steps
        }
        
        # Add result URL if completed
        if job.result_url:
            response_data["result_url"] = job.result_url
        
        # Add error if failed
        if job.error_message:
            response_data["error"] = job.error_message
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/results/{job_id}")
async def get_result_image(job_id: str):
    """Get the result image for a completed job."""
    try:
        job_queue = get_job_queue()
        job = await job_queue.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status.value != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status.value})")
        
        if not job.result_url:
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
        except StorageError:
            raise HTTPException(status_code=404, detail="Result image not found in storage")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get result image {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/thumbnails/{job_id}")
async def get_thumbnail_image(job_id: str):
    """Get the thumbnail image for a completed job."""
    try:
        job_queue = get_job_queue()
        job = await job_queue.get_job_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.status.value != "completed":
            raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status.value})")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get thumbnail {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    try:
        job_queue = get_job_queue()
        success = await job_queue.cancel_job(job_id)
        
        if not success:
            job = await job_queue.get_job_status(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            else:
                raise HTTPException(status_code=400, detail=f"Cannot cancel job in status: {job.status.value}")
        
        return {"message": f"Job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/v1/queue/stats")
async def get_queue_stats():
    """Get queue statistics."""
    try:
        job_queue = get_job_queue()
        stats = await job_queue.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    print("🚀 Starting Style Transfer API server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )