"""
Background worker for processing style transfer jobs.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.style_transfer import get_model, MockStyleTransfer
from utils.image_processing import get_image_processor, QualitySetting
from utils.job_queue import get_job_queue, JobStatus
from utils.storage_simple import get_storage_client, StorageError

logger = logging.getLogger(__name__)

async def process_style_transfer(job_id: str) -> Dict[str, Any]:
    """
    Process a style transfer job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    job_queue = get_job_queue()
    image_processor = get_image_processor()
    storage_client = get_storage_client()
    
    try:
        logger.info(f"Starting style transfer job: {job_id}")
        
        # Get job metadata and images
        job = await job_queue.get_job_status(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        images_data = await job_queue.get_job_images(job_id)
        if not images_data:
            raise ValueError(f"Job images not found: {job_id}")
        
        # Update job status
        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job.progress = 0.1
        await job_queue._store_job_metadata(job)
        
        # Load images from base64
        logger.info(f"Loading images for job {job_id}")
        subject_image = await image_processor.load_image_from_base64(
            images_data["subject_image"]
        )
        style_image = await image_processor.load_image_from_base64(
            images_data["style_image"]
        )
        
        job.progress = 0.3
        await job_queue._store_job_metadata(job)
        
        # Preprocess images
        logger.info(f"Preprocessing images for job {job_id}")
        quality_setting = QualitySetting(job.quality.lower())
        
        subject_processed = await image_processor.preprocess_image(
            subject_image, quality_setting
        )
        style_processed = await image_processor.preprocess_image(
            style_image, quality_setting
        )
        
        job.progress = 0.5
        await job_queue._store_job_metadata(job)
        
        # Get and load ML model
        logger.info(f"Loading style transfer model for job {job_id}")
        # Use mock model for now - can be changed to "neural" for real processing
        model = get_model("mock")  
        await model.load_model()
        
        job.progress = 0.6
        await job_queue._store_job_metadata(job)
        
        # Perform style transfer
        logger.info(f"Performing style transfer for job {job_id}")
        result_image = await model.transfer_style(
            content_image=subject_processed,
            style_image=style_processed,
            style_strength=job.style_strength,
            num_inference_steps=job.num_inference_steps,
            quality=job.quality
        )
        
        job.progress = 0.8
        await job_queue._store_job_metadata(job)
        
        # Post-process and encode result
        logger.info(f"Post-processing result for job {job_id}")
        from utils.image_processing import ImageFormat
        result_bytes = await image_processor.postprocess_image(
            result_image, 
            ImageFormat.JPEG, 
            quality_setting
        )
        
        # Upload result to storage
        logger.info(f"Uploading result for job {job_id}")
        result_key = f"jobs/{job_id}/result.jpg"
        result_url = await storage_client.upload_bytes(result_bytes, result_key)
        
        job.progress = 0.95
        await job_queue._store_job_metadata(job)
        
        # Create thumbnail
        logger.info(f"Creating thumbnail for job {job_id}")
        thumbnail = await image_processor.create_thumbnail(result_image, (256, 256))
        thumbnail_bytes = await image_processor.postprocess_image(
            thumbnail, ImageFormat.JPEG, QualitySetting.MEDIUM
        )
        
        thumbnail_key = f"jobs/{job_id}/thumbnail.jpg"
        thumbnail_url = await storage_client.upload_bytes(thumbnail_bytes, thumbnail_key)
        
        # Update job as completed
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.result_url = result_url
        job.progress = 1.0
        await job_queue._store_job_metadata(job)
        
        processing_time = time.time() - start_time
        logger.info(f"Style transfer job {job_id} completed in {processing_time:.2f}s")
        
        return {
            "success": True,
            "job_id": job_id,
            "result_url": result_url,
            "thumbnail_url": thumbnail_url,
            "processing_time": processing_time,
            "model_info": model.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Style transfer job {job_id} failed: {e}")
        
        # Update job as failed
        try:
            job = await job_queue.get_job_status(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error_message = str(e)
                await job_queue._store_job_metadata(job)
        except Exception:
            pass  # Don't fail on metadata update
        
        return {
            "success": False,
            "job_id": job_id,
            "error": str(e),
            "processing_time": time.time() - start_time
        }