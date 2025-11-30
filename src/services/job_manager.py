"""
Job Management System

Redis-based job queue and state management for async style transfer processing.
Supports job lifecycle management, progress tracking, and failure recovery.
"""

import aioredis
import json
import asyncio
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timedelta
import structlog
from dataclasses import dataclass, asdict
import uuid

logger = structlog.get_logger()


class JobStatus(Enum):
    """Job status enumeration."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """Job information container."""
    job_id: str
    user_id: str
    status: JobStatus
    progress: float = 0.0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result_uri: Optional[str] = None
    error_message: Optional[str] = None
    request_data: Optional[Dict] = None
    estimated_time: Optional[float] = None
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, JobStatus):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'JobInfo':
        """Create JobInfo from dictionary."""
        # Convert ISO strings back to datetime objects
        datetime_fields = ['created_at', 'started_at', 'completed_at']
        for field in datetime_fields:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert status back to enum
        if data.get('status'):
            data['status'] = JobStatus(data['status'])
        
        return cls(**data)


class JobManager:
    """
    Redis-based job management system.
    
    Handles job lifecycle, queuing, status updates, and cleanup.
    Supports job prioritization, retries, and batch operations.
    """
    
    def __init__(
        self,
        redis_client: aioredis.Redis,
        job_ttl: int = 3600 * 24,  # 24 hours
        cleanup_interval: int = 300,  # 5 minutes
        max_retries: int = 3
    ):
        self.redis = redis_client
        self.job_ttl = job_ttl
        self.cleanup_interval = cleanup_interval
        self.max_retries = max_retries
        
        # Redis key patterns
        self.job_key_pattern = "job:{job_id}"
        self.queue_key = "job_queue"
        self.processing_key = "processing_jobs"
        self.user_jobs_key = "user_jobs:{user_id}"
        self.status_key = "job_status:{status}"
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def create_job(self, job_id: str, job_data: Dict) -> JobInfo:
        """
        Create a new job.
        
        Args:
            job_id: Unique job identifier
            job_data: Job data dictionary
            
        Returns:
            JobInfo object
        """
        
        job_info = JobInfo(
            job_id=job_id,
            user_id=job_data.get("user_id"),
            status=JobStatus.QUEUED,
            request_data=job_data.get("request"),
            estimated_time=job_data.get("estimated_time")
        )
        
        # Store job data
        await self.redis.hset(
            self.job_key_pattern.format(job_id=job_id),
            mapping=job_info.to_dict()
        )
        
        # Set TTL
        await self.redis.expire(
            self.job_key_pattern.format(job_id=job_id),
            self.job_ttl
        )
        
        # Add to queue
        await self.redis.lpush(self.queue_key, job_id)
        
        # Track by user
        if job_info.user_id:
            await self.redis.sadd(
                self.user_jobs_key.format(user_id=job_info.user_id),
                job_id
            )
        
        # Track by status
        await self.redis.sadd(
            self.status_key.format(status=JobStatus.QUEUED.value),
            job_id
        )
        
        logger.info("Created job", job_id=job_id, user_id=job_info.user_id)
        return job_info
    
    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job information."""
        
        job_data = await self.redis.hgetall(
            self.job_key_pattern.format(job_id=job_id)
        )
        
        if not job_data:
            return None
        
        return JobInfo.from_dict(job_data)
    
    async def update_job_status(self, job_id: str, status: JobStatus) -> bool:
        """Update job status."""
        
        job_info = await self.get_job(job_id)
        if not job_info:
            return False
        
        old_status = job_info.status
        
        # Update status and timestamps
        updates = {"status": status.value}
        
        if status == JobStatus.PROCESSING and old_status == JobStatus.QUEUED:
            updates["started_at"] = datetime.utcnow().isoformat()
            # Move from queue to processing
            await self.redis.lrem(self.queue_key, 1, job_id)
            await self.redis.sadd(self.processing_key, job_id)
        
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            updates["completed_at"] = datetime.utcnow().isoformat()
            if status == JobStatus.COMPLETED:
                updates["progress"] = 1.0
            # Remove from processing
            await self.redis.srem(self.processing_key, job_id)
        
        # Update job data
        await self.redis.hset(
            self.job_key_pattern.format(job_id=job_id),
            mapping=updates
        )
        
        # Update status tracking sets
        await self.redis.srem(
            self.status_key.format(status=old_status.value),
            job_id
        )
        await self.redis.sadd(
            self.status_key.format(status=status.value),
            job_id
        )
        
        logger.info("Updated job status", job_id=job_id, old_status=old_status.value, new_status=status.value)
        return True
    
    async def update_job_progress(self, job_id: str, progress: float) -> bool:
        """Update job progress (0.0 to 1.0)."""
        
        await self.redis.hset(
            self.job_key_pattern.format(job_id=job_id),
            "progress",
            progress
        )
        
        return True
    
    async def complete_job(self, job_id: str, result_uri: str) -> bool:
        """Mark job as completed with result URI."""
        
        await self.redis.hset(
            self.job_key_pattern.format(job_id=job_id),
            mapping={
                "result_uri": result_uri,
                "progress": 1.0
            }
        )
        
        return await self.update_job_status(job_id, JobStatus.COMPLETED)
    
    async def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark job as failed with error message."""
        
        job_info = await self.get_job(job_id)
        if not job_info:
            return False
        
        # Increment retry count
        retry_count = job_info.retry_count + 1
        
        await self.redis.hset(
            self.job_key_pattern.format(job_id=job_id),
            mapping={
                "error_message": error_message,
                "retry_count": retry_count
            }
        )
        
        # Check if should retry
        if retry_count < self.max_retries:
            logger.info("Retrying job", job_id=job_id, retry_count=retry_count)
            # Re-queue for retry
            await self.redis.lpush(self.queue_key, job_id)
            return await self.update_job_status(job_id, JobStatus.QUEUED)
        else:
            logger.info("Job failed permanently", job_id=job_id, error=error_message)
            return await self.update_job_status(job_id, JobStatus.FAILED)
    
    async def get_next_job(self) -> Optional[str]:
        """Get next job from queue."""
        
        job_id = await self.redis.brpop(self.queue_key, timeout=1)
        if job_id:
            return job_id[1]  # brpop returns (key, value)
        return None
    
    async def get_user_jobs(self, user_id: str, limit: int = 100) -> List[JobInfo]:
        """Get jobs for a specific user."""
        
        job_ids = await self.redis.smembers(
            self.user_jobs_key.format(user_id=user_id)
        )
        
        jobs = []
        for job_id in list(job_ids)[:limit]:
            job_info = await self.get_job(job_id)
            if job_info:
                jobs.append(job_info)
        
        # Sort by created_at descending
        jobs.sort(key=lambda x: x.created_at, reverse=True)
        return jobs
    
    async def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> List[JobInfo]:
        """Get jobs by status."""
        
        job_ids = await self.redis.smembers(
            self.status_key.format(status=status.value)
        )
        
        jobs = []
        for job_id in list(job_ids)[:limit]:
            job_info = await self.get_job(job_id)
            if job_info:
                jobs.append(job_info)
        
        return jobs
    
    async def get_active_job_count(self) -> int:
        """Get count of active jobs (queued + processing)."""
        
        queued_count = await self.redis.llen(self.queue_key)
        processing_count = await self.redis.scard(self.processing_key)
        
        return queued_count + processing_count
    
    async def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        
        stats = {}
        
        for status in JobStatus:
            count = await self.redis.scard(
                self.status_key.format(status=status.value)
            )
            stats[status.value] = count
        
        stats["queue_length"] = await self.redis.llen(self.queue_key)
        stats["processing_count"] = await self.redis.scard(self.processing_key)
        
        return stats
    
    async def cleanup_old_jobs(self):
        """Clean up old completed/failed jobs."""
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.job_ttl)
        cutoff_str = cutoff_time.isoformat()
        
        cleanup_statuses = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
        cleaned_count = 0
        
        for status in cleanup_statuses:
            job_ids = await self.redis.smembers(
                self.status_key.format(status=status.value)
            )
            
            for job_id in job_ids:
                job_info = await self.get_job(job_id)
                if job_info and job_info.completed_at and job_info.completed_at.isoformat() < cutoff_str:
                    await self.delete_job(job_id)
                    cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info("Cleaned up old jobs", count=cleaned_count)
        
        return cleaned_count
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job completely."""
        
        job_info = await self.get_job(job_id)
        if not job_info:
            return False
        
        # Remove from all tracking structures
        await self.redis.delete(self.job_key_pattern.format(job_id=job_id))
        await self.redis.lrem(self.queue_key, 0, job_id)
        await self.redis.srem(self.processing_key, job_id)
        
        if job_info.user_id:
            await self.redis.srem(
                self.user_jobs_key.format(user_id=job_info.user_id),
                job_id
            )
        
        await self.redis.srem(
            self.status_key.format(status=job_info.status.value),
            job_id
        )
        
        logger.info("Deleted job", job_id=job_id)
        return True
    
    async def recover_stuck_jobs(self):
        """Recover jobs that are stuck in processing state."""
        
        processing_jobs = await self.redis.smembers(self.processing_key)
        recovery_cutoff = datetime.utcnow() - timedelta(minutes=30)  # 30 minute timeout
        
        recovered_count = 0
        
        for job_id in processing_jobs:
            job_info = await self.get_job(job_id)
            if job_info and job_info.started_at and job_info.started_at < recovery_cutoff:
                logger.warning("Recovering stuck job", job_id=job_id)
                await self.fail_job(job_id, "Job timeout - recovered by cleanup")
                recovered_count += 1
        
        if recovered_count > 0:
            logger.info("Recovered stuck jobs", count=recovered_count)
        
        return recovered_count
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_old_jobs()
                    await self.recover_stuck_jobs()
                except Exception as e:
                    logger.error("Cleanup task error", error=str(e))
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def shutdown(self):
        """Shutdown job manager and cleanup tasks."""
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Job manager shutdown complete")


class JobWorker:
    """
    Job worker for processing style transfer jobs.
    
    Runs in separate process/container to handle job execution.
    """
    
    def __init__(self, job_manager: JobManager, worker_id: str = None):
        self.job_manager = job_manager
        self.worker_id = worker_id or str(uuid.uuid4())
        self.running = False
    
    async def start(self):
        """Start the job worker."""
        
        self.running = True
        logger.info("Starting job worker", worker_id=self.worker_id)
        
        while self.running:
            try:
                # Get next job
                job_id = await self.job_manager.get_next_job()
                
                if job_id:
                    await self.process_job(job_id)
                else:
                    # No jobs available, short sleep
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error("Worker error", worker_id=self.worker_id, error=str(e))
                await asyncio.sleep(5)  # Back off on error
    
    async def process_job(self, job_id: str):
        """Process a single job."""
        
        logger.info("Processing job", worker_id=self.worker_id, job_id=job_id)
        
        try:
            # Mark as processing
            await self.job_manager.update_job_status(job_id, JobStatus.PROCESSING)
            
            # Get job details
            job_info = await self.job_manager.get_job(job_id)
            if not job_info:
                logger.error("Job not found", job_id=job_id)
                return
            
            # Process the job (implement your style transfer logic here)
            await self._execute_style_transfer(job_info)
            
        except Exception as e:
            logger.error("Job processing failed", job_id=job_id, error=str(e))
            await self.job_manager.fail_job(job_id, str(e))
    
    async def _execute_style_transfer(self, job_info: JobInfo):
        """Execute the actual style transfer (to be implemented)."""
        
        # This would contain the actual style transfer logic
        # For now, simulate processing time
        await asyncio.sleep(2)
        
        # Mark as completed
        result_uri = f"s3://bucket/results/{job_info.job_id}.png"
        await self.job_manager.complete_job(job_info.job_id, result_uri)
    
    def stop(self):
        """Stop the worker."""
        
        self.running = False
        logger.info("Stopping job worker", worker_id=self.worker_id)