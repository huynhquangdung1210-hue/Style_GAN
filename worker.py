"""
RQ Worker script for processing style transfer jobs in the background.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import redis
from rq import Worker, Queue
from rq.worker import WorkerStatus

from utils.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('worker.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Start RQ worker for style transfer processing."""
    
    # Get Redis connection settings
    settings = get_settings()
    
    # Create Redis connection
    redis_conn = redis.Redis(
        host=settings.redis.host,
        port=settings.redis.port,
        db=settings.redis.db,
        password=settings.redis.password if hasattr(settings.redis, 'password') else None
    )
    
    # Test Redis connection
    try:
        redis_conn.ping()
        logger.info("✅ Connected to Redis successfully")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Redis: {e}")
        sys.exit(1)
    
    # Create queue
    queue = Queue('style_transfer', connection=redis_conn)
    
    logger.info(f"🚀 Starting RQ Worker for queue: style_transfer")
    logger.info(f"📊 Current queue length: {len(queue)}")
    
    # Start worker
    worker = Worker([queue], name="style_transfer_worker", connection=redis_conn)
    logger.info(f"🔄 Worker started with name: {worker.name}")
    
    try:
        worker.work(
            with_scheduler=True,
            logging_level='INFO'
        )
    except KeyboardInterrupt:
        logger.info("👋 Worker stopped by user")
    except Exception as e:
        logger.error(f"💥 Worker error: {e}")
        raise

if __name__ == "__main__":
    main()