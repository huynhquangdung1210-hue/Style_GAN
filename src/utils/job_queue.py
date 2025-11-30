"""
Job queue system for async style transfer processing.
Uses Redis for job storage and RQ for task management.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, Callable, List
import redis
from rq import Queue, Worker
from rq.job import Job
from rq.exceptions import NoSuchJobError, WorkerException

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobPriority(Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class StyleTransferJob:
    """Style transfer job data structure."""
    
    def __init__(
        self,
        job_id: str,
        subject_image_data: str,
        style_image_data: str,
        style_strength: float = 0.8,
        num_inference_steps: int = 20,
        quality: str = "medium",
        priority: JobPriority = JobPriority.NORMAL,
        user_id: Optional[str] = None,
        callback_url: Optional[str] = None
    ):
        self.job_id = job_id
        self.subject_image_data = subject_image_data
        self.style_image_data = style_image_data
        self.style_strength = style_strength
        self.num_inference_steps = num_inference_steps
        self.quality = quality
        self.priority = priority
        self.user_id = user_id
        self.callback_url = callback_url
        
        self.status = JobStatus.PENDING
        self.created_at = datetime.utcnow()
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.result_url = None
        self.progress = 0.0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "priority": self.priority.value,
            "user_id": self.user_id,
            "callback_url": self.callback_url,
            "style_strength": self.style_strength,
            "num_inference_steps": self.num_inference_steps,
            "quality": self.quality,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "result_url": self.result_url,
            "progress": self.progress
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleTransferJob":
        """Create job from dictionary."""
        job = cls(
            job_id=data["job_id"],
            subject_image_data="",  # Image data not stored in metadata
            style_image_data="",
            style_strength=data.get("style_strength", 0.8),
            num_inference_steps=data.get("num_inference_steps", 20),
            quality=data.get("quality", "medium"),
            priority=JobPriority(data.get("priority", JobPriority.NORMAL.value)),
            user_id=data.get("user_id"),
            callback_url=data.get("callback_url")
        )
        
        job.status = JobStatus(data["status"])
        job.created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
        job.started_at = datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
        job.completed_at = datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None
        job.error_message = data.get("error_message")
        job.result_url = data.get("result_url")
        job.progress = data.get("progress", 0.0)
        
        return job

class JobQueue:
    """Redis-based job queue for style transfer tasks."""
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        queue_name: str = "style_transfer",
        job_timeout: int = 3600  # 1 hour
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.queue_name = queue_name
        self.job_timeout = job_timeout
        
        # Initialize Redis connection
        self.redis_conn = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False  # Keep binary data as bytes
        )
        
        # Initialize RQ queue
        self.queue = Queue(
            name=queue_name,
            connection=self.redis_conn,
            default_timeout=job_timeout
        )
        
        logger.info(f"Job queue initialized: {queue_name}")
    
    async def enqueue_job(self, job: StyleTransferJob) -> str:
        """
        Enqueue a style transfer job.
        
        Args:
            job: StyleTransferJob instance
            
        Returns:
            Job ID
        """
        try:
            # Store job metadata in Redis
            await self._store_job_metadata(job)
            
            # Store image data separately with compression
            await self._store_job_images(job)
            
            # Enqueue the job with RQ
            rq_job = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.queue.enqueue(
                    "src.workers.style_transfer_worker.process_style_transfer",
                    job.job_id,
                    job_timeout=self.job_timeout,
                    job_id=job.job_id
                )
            )
            
            logger.info(f"Job enqueued: {job.job_id}")
            return job.job_id
            
        except Exception as e:
            logger.error(f"Failed to enqueue job {job.job_id}: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[StyleTransferJob]:
        """
        Get job status and metadata.
        
        Args:
            job_id: Job identifier
            
        Returns:
            StyleTransferJob instance or None if not found
        """
        try:
            # Get job metadata from Redis
            job_data = await self._get_job_metadata(job_id)
            if not job_data:
                return None
            
            # Get RQ job status
            try:
                rq_job = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Job.fetch(job_id, connection=self.redis_conn)
                )
                
                # Update status based on RQ job
                if rq_job.is_queued:
                    job_data["status"] = JobStatus.PENDING.value
                elif rq_job.is_started:
                    job_data["status"] = JobStatus.PROCESSING.value
                elif rq_job.is_finished:
                    job_data["status"] = JobStatus.COMPLETED.value
                elif rq_job.is_failed:
                    job_data["status"] = JobStatus.FAILED.value
                    job_data["error_message"] = str(rq_job.exc_info) if rq_job.exc_info else "Unknown error"
            except NoSuchJobError:
                # Job not in RQ, check if it's completed in our metadata
                if job_data.get("status") not in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    job_data["status"] = JobStatus.FAILED.value
                    job_data["error_message"] = "Job not found in queue"
            
            job = StyleTransferJob.from_dict(job_data)
            await self._store_job_metadata(job)  # Update metadata
            
            return job
            
        except Exception as e:
            logger.error(f"Failed to get job status {job_id}: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or processing job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if cancelled successfully
        """
        try:
            # Get current job
            job = await self.get_job_status(job_id)
            if not job:
                return False
            
            # Can only cancel pending or processing jobs
            if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
                logger.warning(f"Cannot cancel job in status: {job.status}")
                return False
            
            # Cancel RQ job
            try:
                rq_job = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: Job.fetch(job_id, connection=self.redis_conn)
                )
                await asyncio.get_event_loop().run_in_executor(
                    None, rq_job.cancel
                )
            except NoSuchJobError:
                pass  # Job already removed from queue
            
            # Update job status
            job.status = JobStatus.CANCELLED
            job.completed_at = datetime.utcnow()
            await self._store_job_metadata(job)
            
            logger.info(f"Job cancelled: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def cleanup_old_jobs(self, max_age_days: int = 7) -> int:
        """
        Clean up old completed/failed jobs.
        
        Args:
            max_age_days: Maximum age in days for jobs to keep
            
        Returns:
            Number of jobs cleaned up
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            
            # Get all job keys
            job_keys = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis_conn.keys("job:*:metadata")
            )
            
            cleaned_count = 0
            for key in job_keys:
                job_id = key.decode().split(":")[1]
                job = await self.get_job_status(job_id)
                
                if (job and 
                    job.completed_at and 
                    job.completed_at < cutoff_date and
                    job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]):
                    
                    await self._delete_job_data(job_id)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old jobs")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        try:
            # Get RQ queue stats
            queue_length = await asyncio.get_event_loop().run_in_executor(
                None, lambda: len(self.queue)
            )
            
            # Count jobs by status
            job_keys = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis_conn.keys("job:*:metadata")
            )
            
            status_counts = {status.value: 0 for status in JobStatus}
            
            for key in job_keys:
                try:
                    job_data = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: json.loads(self.redis_conn.get(key))
                    )
                    status = job_data.get("status", "unknown")
                    if status in status_counts:
                        status_counts[status] += 1
                except Exception:
                    continue
            
            return {
                "queue_length": queue_length,
                "total_jobs": len(job_keys),
                "status_counts": status_counts,
                "queue_name": self.queue_name,
                "redis_host": self.redis_host,
                "redis_port": self.redis_port
            }
            
        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {}
    
    async def _store_job_metadata(self, job: StyleTransferJob) -> None:
        """Store job metadata in Redis."""
        key = f"job:{job.job_id}:metadata"
        data = json.dumps(job.to_dict())
        
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.redis_conn.setex(key, 604800, data)  # 7 days TTL
        )
    
    async def _get_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata from Redis."""
        key = f"job:{job_id}:metadata"
        
        data = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.redis_conn.get(key)
        )
        
        if data:
            return json.loads(data)
        return None
    
    async def _store_job_images(self, job: StyleTransferJob) -> None:
        """Store job image data in Redis with compression."""
        import gzip
        
        # Compress and store subject image
        subject_key = f"job:{job.job_id}:subject_image"
        subject_compressed = gzip.compress(job.subject_image_data.encode())
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.redis_conn.setex(subject_key, 604800, subject_compressed)
        )
        
        # Compress and store style image
        style_key = f"job:{job.job_id}:style_image"
        style_compressed = gzip.compress(job.style_image_data.encode())
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.redis_conn.setex(style_key, 604800, style_compressed)
        )
    
    async def get_job_images(self, job_id: str) -> Optional[Dict[str, str]]:
        """Get job image data from Redis."""
        import gzip
        
        try:
            subject_key = f"job:{job_id}:subject_image"
            style_key = f"job:{job_id}:style_image"
            
            # Get compressed data
            subject_compressed = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis_conn.get(subject_key)
            )
            style_compressed = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.redis_conn.get(style_key)
            )
            
            if not subject_compressed or not style_compressed:
                return None
            
            # Decompress
            subject_data = gzip.decompress(subject_compressed).decode()
            style_data = gzip.decompress(style_compressed).decode()
            
            return {
                "subject_image": subject_data,
                "style_image": style_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get job images {job_id}: {e}")
            return None
    
    async def _delete_job_data(self, job_id: str) -> None:
        """Delete all job data from Redis."""
        keys = [
            f"job:{job_id}:metadata",
            f"job:{job_id}:subject_image", 
            f"job:{job_id}:style_image"
        ]
        
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.redis_conn.delete(*keys)
        )

# Global queue instance
_job_queue = None

def get_job_queue() -> JobQueue:
    """Get global job queue instance."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue()
    return _job_queue