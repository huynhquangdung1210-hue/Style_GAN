"""
Production FastAPI Service for Style Transfer

High-performance, async API service with comprehensive validation,
error handling, monitoring, and scalability features.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import uuid
import asyncio
import aioredis
import structlog
import time
from datetime import datetime, timedelta
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import boto3
from botocore.exceptions import ClientError
from PIL import Image
import io
import torch

from ..models import create_model, StyleTransferConfig, compute_image_hash, validate_model_inputs
from ..services.job_manager import JobManager, JobStatus
from ..preprocessing.image_processor import ImageProcessor
from ..postprocessing.safety_filter import SafetyFilter
from ..utils.config import get_settings
from ..utils.auth import verify_token
from ..utils.monitoring import setup_logging, metrics

# Setup structured logging
logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('style_transfer_requests_total', 'Total requests', ['endpoint', 'method'])
REQUEST_DURATION = Histogram('style_transfer_request_duration_seconds', 'Request duration')
INFERENCE_DURATION = Histogram('style_transfer_inference_duration_seconds', 'Inference duration', ['preset'])
ACTIVE_JOBS = Gauge('style_transfer_active_jobs', 'Number of active jobs')
GPU_MEMORY_USAGE = Gauge('style_transfer_gpu_memory_gb', 'GPU memory usage in GB')

# FastAPI app
app = FastAPI(
    title="Style Transfer API",
    description="Production-ready neural style transfer service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()
settings = get_settings()

# Global instances
model_instance = None
job_manager = None
image_processor = None
safety_filter = None
redis_client = None


# Pydantic models
class StyleTransferRequest(BaseModel):
    """Request model for style transfer."""
    
    subject_image_uri: str = Field(..., description="S3 URI or URL to subject image")
    style_image_uri: str = Field(..., description="S3 URI or URL to style image")
    preset: str = Field("balanced", description="Quality preset: fast, balanced, high-quality")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="Guidance scale override")
    num_inference_steps: Optional[int] = Field(None, ge=10, le=100, description="Inference steps override")
    resolution: List[int] = Field([1024, 1024], description="Output resolution [width, height]")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    
    @validator('preset')
    def validate_preset(cls, v):
        valid_presets = ["fast", "balanced", "high-quality"]
        if v not in valid_presets:
            raise ValueError(f"Invalid preset. Choose from: {valid_presets}")
        return v
    
    @validator('resolution')
    def validate_resolution(cls, v):
        if len(v) != 2:
            raise ValueError("Resolution must be [width, height]")
        if v[0] < 256 or v[1] < 256 or v[0] > 2048 or v[1] > 2048:
            raise ValueError("Resolution must be between 256 and 2048 pixels")
        return v


class StyleTransferResponse(BaseModel):
    """Response model for style transfer."""
    
    job_id: str
    status: str
    message: str
    estimated_time_seconds: Optional[float] = None
    result_uri: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    
    job_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    result_uri: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    timestamp: datetime
    version: str
    gpu_available: bool
    gpu_memory_gb: Optional[float] = None
    active_jobs: int
    model_loaded: bool


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global model_instance, job_manager, image_processor, safety_filter, redis_client
    
    logger.info("Starting Style Transfer API service...")
    
    # Initialize Redis
    redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize job manager
    job_manager = JobManager(redis_client)
    
    # Initialize image processor
    image_processor = ImageProcessor()
    
    # Initialize safety filter
    safety_filter = SafetyFilter()
    
    # Initialize and warm up model
    config = StyleTransferConfig(
        use_fp16=settings.use_fp16,
        max_batch_size=settings.max_batch_size,
        warmup_on_init=True
    )
    model_instance = create_model(config)
    
    logger.info("Style Transfer API service started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global redis_client
    
    logger.info("Shutting down Style Transfer API service...")
    
    if redis_client:
        await redis_client.close()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Style Transfer API service shut down complete")


# Dependency injection
async def get_redis() -> aioredis.Redis:
    """Get Redis client."""
    return redis_client


def get_model():
    """Get model instance."""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_instance


def get_job_manager():
    """Get job manager instance."""
    if job_manager is None:
        raise HTTPException(status_code=503, detail="Job manager not initialized")
    return job_manager


def get_image_processor():
    """Get image processor instance.""" 
    if image_processor is None:
        raise HTTPException(status_code=503, detail="Image processor not initialized")
    return image_processor


def get_safety_filter():
    """Get safety filter instance."""
    if safety_filter is None:
        raise HTTPException(status_code=503, detail="Safety filter not initialized")
    return safety_filter


# Authentication dependency
async def verify_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify authentication token."""
    try:
        payload = verify_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    
    REQUEST_COUNT.labels(endpoint="health", method="GET").inc()
    
    gpu_available = torch.cuda.is_available()
    gpu_memory = None
    
    if gpu_available and model_instance:
        memory_stats = model_instance.get_memory_usage()
        gpu_memory = memory_stats.get("allocated_gb", 0)
        GPU_MEMORY_USAGE.set(gpu_memory)
    
    active_jobs_count = 0
    if job_manager:
        active_jobs_count = await job_manager.get_active_job_count()
        ACTIVE_JOBS.set(active_jobs_count)
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        gpu_available=gpu_available,
        gpu_memory_gb=gpu_memory,
        active_jobs=active_jobs_count,
        model_loaded=model_instance is not None
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


@app.post("/v1/generate", response_model=StyleTransferResponse)
async def generate_style_transfer(
    request: StyleTransferRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(verify_auth),
    job_mgr: JobManager = Depends(get_job_manager)
):
    """Submit a style transfer job."""
    
    REQUEST_COUNT.labels(endpoint="generate", method="POST").inc()
    
    with REQUEST_DURATION.time():
        try:
            # Generate unique job ID
            job_id = str(uuid.uuid4())
            
            # Estimate processing time
            estimated_time = estimate_processing_time(
                request.preset,
                request.resolution,
                settings.device_type
            )
            
            # Create job
            job_data = {
                "job_id": job_id,
                "user_id": user["sub"],
                "request": request.dict(),
                "status": JobStatus.QUEUED,
                "created_at": datetime.utcnow().isoformat(),
                "estimated_time": estimated_time
            }
            
            # Store job in Redis
            await job_mgr.create_job(job_id, job_data)
            
            # Queue background processing
            background_tasks.add_task(
                process_style_transfer_job,
                job_id,
                request,
                user["sub"]
            )
            
            logger.info("Created style transfer job", job_id=job_id, user_id=user["sub"])
            
            return StyleTransferResponse(
                job_id=job_id,
                status="queued",
                message="Job queued successfully",
                estimated_time_seconds=estimated_time,
                created_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to create job", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to create job")


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user: dict = Depends(verify_auth),
    job_mgr: JobManager = Depends(get_job_manager)
):
    """Get job status and results."""
    
    REQUEST_COUNT.labels(endpoint="job_status", method="GET").inc()
    
    try:
        job_data = await job_mgr.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check ownership
        if job_data.get("user_id") != user["sub"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        processing_time = None
        if job_data.get("completed_at") and job_data.get("created_at"):
            created = datetime.fromisoformat(job_data["created_at"])
            completed = datetime.fromisoformat(job_data["completed_at"])
            processing_time = (completed - created).total_seconds()
        
        return JobStatusResponse(
            job_id=job_id,
            status=job_data.get("status", "unknown"),
            progress=job_data.get("progress", 0.0),
            result_uri=job_data.get("result_uri"),
            error_message=job_data.get("error_message"),
            created_at=datetime.fromisoformat(job_data["created_at"]),
            completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
            processing_time_seconds=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job status", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get job status")


@app.delete("/v1/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    user: dict = Depends(verify_auth),
    job_mgr: JobManager = Depends(get_job_manager)
):
    """Cancel a running job."""
    
    REQUEST_COUNT.labels(endpoint="cancel_job", method="DELETE").inc()
    
    try:
        job_data = await job_mgr.get_job(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Check ownership
        if job_data.get("user_id") != user["sub"]:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if job can be cancelled
        if job_data.get("status") in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(status_code=400, detail="Job cannot be cancelled")
        
        # Cancel job
        await job_mgr.update_job_status(job_id, JobStatus.CANCELLED)
        
        logger.info("Cancelled job", job_id=job_id, user_id=user["sub"])
        
        return {"message": "Job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to cancel job")


# Background processing function
async def process_style_transfer_job(job_id: str, request: StyleTransferRequest, user_id: str):
    """Process style transfer job in background."""
    
    logger.info("Starting job processing", job_id=job_id)
    
    try:
        # Update job status to processing
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        await job_manager.update_job_progress(job_id, 0.1)
        
        # Download and preprocess images
        logger.info("Downloading images", job_id=job_id)
        subject_image = await image_processor.download_and_preprocess(
            request.subject_image_uri,
            target_size=tuple(request.resolution)
        )
        
        await job_manager.update_job_progress(job_id, 0.3)
        
        style_image = await image_processor.download_and_preprocess(
            request.style_image_uri,
            target_size=tuple(request.resolution)
        )
        
        await job_manager.update_job_progress(job_id, 0.5)
        
        # Safety filtering
        logger.info("Running safety checks", job_id=job_id)
        
        subject_safe = await safety_filter.is_safe(subject_image)
        style_safe = await safety_filter.is_safe(style_image)
        
        if not subject_safe or not style_safe:
            raise ValueError("Input images failed safety checks")
        
        await job_manager.update_job_progress(job_id, 0.6)
        
        # Run style transfer
        logger.info("Running style transfer", job_id=job_id, preset=request.preset)
        
        with INFERENCE_DURATION.labels(preset=request.preset).time():
            
            # Convert to tensors
            subject_tensor = torch.from_numpy(subject_image).permute(2, 0, 1).unsqueeze(0)
            style_tensor = torch.from_numpy(style_image).permute(2, 0, 1).unsqueeze(0)
            
            # Normalize to [-1, 1]
            subject_tensor = subject_tensor * 2.0 - 1.0
            style_tensor = style_tensor * 2.0 - 1.0
            
            # Generate style cache key
            style_cache_key = compute_image_hash(style_tensor)
            
            # Override parameters if provided
            kwargs = {}
            if request.guidance_scale:
                kwargs["guidance_scale"] = request.guidance_scale
            if request.num_inference_steps:
                kwargs["num_inference_steps"] = request.num_inference_steps
            
            # Run inference
            result_tensor = model_instance.transfer_style(
                subject_tensor,
                style_tensor,
                preset=request.preset,
                seed=request.seed,
                style_cache_key=style_cache_key,
                **kwargs
            )
        
        await job_manager.update_job_progress(job_id, 0.8)
        
        # Post-process result
        logger.info("Post-processing result", job_id=job_id)
        
        result_image = await image_processor.postprocess(result_tensor[0])
        
        # Final safety check
        if not await safety_filter.is_safe(result_image):
            raise ValueError("Generated image failed safety checks")
        
        await job_manager.update_job_progress(job_id, 0.9)
        
        # Upload result to S3
        logger.info("Uploading result", job_id=job_id)
        
        result_uri = await upload_result_to_s3(result_image, job_id, user_id)
        
        # Mark job as completed
        await job_manager.complete_job(job_id, result_uri)
        
        # Send webhook notification if provided
        if request.callback_url:
            await send_webhook_notification(request.callback_url, job_id, "completed", result_uri)
        
        logger.info("Job completed successfully", job_id=job_id, result_uri=result_uri)
        
    except Exception as e:
        logger.error("Job processing failed", job_id=job_id, error=str(e))
        
        # Mark job as failed
        await job_manager.fail_job(job_id, str(e))
        
        # Send webhook notification if provided
        if request.callback_url:
            await send_webhook_notification(request.callback_url, job_id, "failed", None, str(e))


def estimate_processing_time(preset: str, resolution: List[int], device: str) -> float:
    """Estimate job processing time."""
    from ..models.utils import estimate_inference_time
    
    return estimate_inference_time(
        batch_size=1,
        resolution=tuple(resolution),
        preset=preset,
        device=device
    )


async def upload_result_to_s3(image: any, job_id: str, user_id: str) -> str:
    """Upload result image to S3 and return URI."""
    
    s3_client = boto3.client(
        's3',
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region
    )
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    Image.fromarray((image * 255).astype('uint8')).save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Upload to S3
    key = f"results/{user_id}/{job_id}.png"
    
    try:
        s3_client.put_object(
            Bucket=settings.s3_bucket,
            Key=key,
            Body=img_bytes,
            ContentType='image/png'
        )
        
        return f"s3://{settings.s3_bucket}/{key}"
        
    except ClientError as e:
        logger.error("Failed to upload to S3", job_id=job_id, error=str(e))
        raise


async def send_webhook_notification(url: str, job_id: str, status: str, result_uri: str = None, error: str = None):
    """Send webhook notification."""
    
    import aiohttp
    
    payload = {
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if result_uri:
        payload["result_uri"] = result_uri
    if error:
        payload["error"] = error
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.info("Webhook notification sent", job_id=job_id, url=url)
                else:
                    logger.warning("Webhook notification failed", job_id=job_id, status=response.status)
    except Exception as e:
        logger.error("Webhook notification error", job_id=job_id, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)