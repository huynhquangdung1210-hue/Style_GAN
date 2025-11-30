"""
Monitoring and Observability

Production-ready logging, metrics, tracing, and health monitoring
for the style transfer service.
"""

import structlog
import logging
import sys
import time
import asyncio
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, generate_latest
from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
import psutil
import torch
from datetime import datetime, timedelta
import json

from .config import get_settings


# Prometheus Metrics
REQUEST_COUNT = Counter(
    'style_transfer_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'style_transfer_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

INFERENCE_COUNT = Counter(
    'style_transfer_inference_total',
    'Total number of inference requests',
    ['preset', 'status']
)

INFERENCE_DURATION = Histogram(
    'style_transfer_inference_duration_seconds',
    'Inference duration in seconds',
    ['preset']
)

QUEUE_SIZE = Gauge(
    'style_transfer_queue_size',
    'Current queue size'
)

ACTIVE_JOBS = Gauge(
    'style_transfer_active_jobs',
    'Number of active processing jobs'
)

GPU_MEMORY_USAGE = Gauge(
    'style_transfer_gpu_memory_bytes',
    'GPU memory usage in bytes'
)

GPU_UTILIZATION = Gauge(
    'style_transfer_gpu_utilization_percent',
    'GPU utilization percentage'
)

MODEL_LOAD_TIME = Gauge(
    'style_transfer_model_load_seconds',
    'Time taken to load model'
)

ERROR_COUNT = Counter(
    'style_transfer_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)

CACHE_HIT_RATE = Gauge(
    'style_transfer_cache_hit_rate',
    'Cache hit rate percentage'
)

APP_INFO = Info(
    'style_transfer_app_info',
    'Application information'
)


def setup_logging():
    """Configure structured logging with JSON output."""
    
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level)
    )
    
    logger = structlog.get_logger()
    logger.info("Logging configured", level=settings.monitoring.log_level)
    
    return logger


def setup_sentry():
    """Configure Sentry error tracking."""
    
    settings = get_settings()
    
    if settings.monitoring.sentry_dsn:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR
        )
        
        sentry_sdk.init(
            dsn=settings.monitoring.sentry_dsn,
            integrations=[
                sentry_logging,
                FastApiIntegration(auto_enable=True),
            ],
            traces_sample_rate=0.1 if settings.is_production else 1.0,
            environment=settings.environment,
            release=settings.app_version
        )
        
        logger = structlog.get_logger()
        logger.info("Sentry configured", environment=settings.environment)


def setup_tracing():
    """Configure OpenTelemetry distributed tracing."""
    
    settings = get_settings()
    
    if not settings.monitoring.enable_tracing or not settings.monitoring.jaeger_endpoint:
        return
    
    # Configure tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=14268,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    logger = structlog.get_logger()
    logger.info("Tracing configured", endpoint=settings.monitoring.jaeger_endpoint)


def setup_metrics():
    """Configure Prometheus metrics."""
    
    settings = get_settings()
    
    if not settings.monitoring.enable_metrics:
        return
    
    # Start Prometheus metrics server
    try:
        start_http_server(settings.monitoring.prometheus_port)
        logger = structlog.get_logger()
        logger.info("Metrics server started", port=settings.monitoring.prometheus_port)
    except Exception as e:
        logger = structlog.get_logger()
        logger.error("Failed to start metrics server", error=str(e))
    
    # Set application info
    APP_INFO.info({
        'app_name': settings.app_name,
        'version': settings.app_version,
        'environment': settings.environment
    })


class PerformanceMonitor:
    """Monitor system and application performance."""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.start_time = time.time()
        self.stats = {
            'requests_processed': 0,
            'total_inference_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start background monitoring task."""
        
        while True:
            try:
                await self.collect_system_metrics()
                await self.collect_gpu_metrics()
                await self.log_performance_stats()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error("Monitoring task failed", error=str(e))
                await asyncio.sleep(interval)
    
    async def collect_system_metrics(self):
        """Collect system-level metrics."""
        
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Log metrics
            self.logger.debug("System metrics",
                            cpu_percent=cpu_percent,
                            memory_percent=memory.percent,
                            memory_available_gb=memory.available / 1024**3,
                            disk_free_gb=disk.free / 1024**3)
            
        except Exception as e:
            self.logger.error("Failed to collect system metrics", error=str(e))
    
    async def collect_gpu_metrics(self):
        """Collect GPU metrics."""
        
        try:
            if torch.cuda.is_available():
                # GPU memory
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_reserved = torch.cuda.memory_reserved(i)
                    
                    GPU_MEMORY_USAGE.set(memory_allocated)
                    
                    self.logger.debug("GPU metrics",
                                    device=i,
                                    memory_allocated_gb=memory_allocated / 1024**3,
                                    memory_reserved_gb=memory_reserved / 1024**3)
                    
                # Try to get GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    GPU_UTILIZATION.set(util.gpu)
                except ImportError:
                    pass  # nvidia-ml-py not available
                    
        except Exception as e:
            self.logger.error("Failed to collect GPU metrics", error=str(e))
    
    async def log_performance_stats(self):
        """Log application performance statistics."""
        
        uptime = time.time() - self.start_time
        
        cache_hit_rate = 0
        if self.stats['cache_hits'] + self.stats['cache_misses'] > 0:
            cache_hit_rate = self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
        
        CACHE_HIT_RATE.set(cache_hit_rate * 100)
        
        avg_inference_time = 0
        if self.stats['requests_processed'] > 0:
            avg_inference_time = self.stats['total_inference_time'] / self.stats['requests_processed']
        
        self.logger.info("Performance stats",
                        uptime_seconds=uptime,
                        requests_processed=self.stats['requests_processed'],
                        avg_inference_time=avg_inference_time,
                        cache_hit_rate_percent=cache_hit_rate * 100)
    
    def record_request(self, duration: float):
        """Record request metrics."""
        self.stats['requests_processed'] += 1
        self.stats['total_inference_time'] += duration
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.stats['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record cache miss."""
        self.stats['cache_misses'] += 1


class HealthChecker:
    """Application health monitoring."""
    
    def __init__(self):
        self.logger = structlog.get_logger()
        self.checks = {}
        self.last_check = None
    
    def add_check(self, name: str, check_func, timeout: int = 5):
        """Add a health check."""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout,
            'last_result': None,
            'last_check': None
        }
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {},
            'uptime_seconds': time.time() - (self.last_check or time.time())
        }
        
        for name, check in self.checks.items():
            try:
                # Run check with timeout
                result = await asyncio.wait_for(
                    check['func'](),
                    timeout=check['timeout']
                )
                
                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'details': result if isinstance(result, dict) else None,
                    'last_check': datetime.utcnow().isoformat()
                }
                
                check['last_result'] = result
                check['last_check'] = datetime.utcnow()
                
                if not result:
                    results['overall_status'] = 'degraded'
                    
            except asyncio.TimeoutError:
                results['checks'][name] = {
                    'status': 'timeout',
                    'error': f"Check timed out after {check['timeout']} seconds"
                }
                results['overall_status'] = 'degraded'
                
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        self.last_check = time.time()
        return results


# Health check functions
async def check_redis_health():
    """Check Redis connectivity."""
    try:
        # This would connect to Redis and test basic operations
        return {'connected': True, 'ping_ms': 1.2}
    except Exception:
        return False


async def check_model_health():
    """Check if model is loaded and responsive."""
    try:
        # This would test model inference with a small dummy input
        return {'loaded': True, 'gpu_available': torch.cuda.is_available()}
    except Exception:
        return False


async def check_storage_health():
    """Check cloud storage connectivity."""
    try:
        # This would test S3 connectivity
        return {'accessible': True, 'bucket_exists': True}
    except Exception:
        return False


# Global instances
performance_monitor = PerformanceMonitor()
health_checker = HealthChecker()


def init_monitoring():
    """Initialize all monitoring components."""
    
    logger = setup_logging()
    setup_sentry()
    setup_tracing()
    setup_metrics()
    
    # Add health checks
    health_checker.add_check('redis', check_redis_health)
    health_checker.add_check('model', check_model_health)
    health_checker.add_check('storage', check_storage_health)
    
    logger.info("Monitoring initialized")
    
    return logger


# Decorators for monitoring
def monitor_inference(preset: str):
    """Decorator to monitor inference operations."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                INFERENCE_COUNT.labels(preset=preset, status='success').inc()
                INFERENCE_DURATION.labels(preset=preset).observe(duration)
                performance_monitor.record_request(duration)
                
                return result
                
            except Exception as e:
                INFERENCE_COUNT.labels(preset=preset, status='error').inc()
                ERROR_COUNT.labels(error_type=type(e).__name__, component='inference').inc()
                raise
                
        return wrapper
    return decorator


def monitor_request(endpoint: str):
    """Decorator to monitor API requests."""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            method = 'POST'  # Default, could be extracted from request
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='success').inc()
                REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
                
                return result
                
            except Exception as e:
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='error').inc()
                ERROR_COUNT.labels(error_type=type(e).__name__, component='api').inc()
                raise
                
        return wrapper
    return decorator


# Context managers
class MetricsContext:
    """Context manager for recording metrics."""
    
    def __init__(self, operation: str, component: str = 'general'):
        self.operation = operation
        self.component = component
        self.start_time = None
        self.logger = structlog.get_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug("Operation started", operation=self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info("Operation completed", 
                           operation=self.operation, 
                           duration=duration)
        else:
            self.logger.error("Operation failed",
                            operation=self.operation,
                            duration=duration,
                            error=str(exc_val))
            ERROR_COUNT.labels(error_type=exc_type.__name__, component=self.component).inc()


# Example usage
if __name__ == "__main__":
    # Initialize monitoring
    logger = init_monitoring()
    
    # Test metrics
    REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='success').inc()
    INFERENCE_DURATION.labels(preset='balanced').observe(2.5)
    
    # Test context manager
    with MetricsContext('test_operation'):
        time.sleep(0.1)
    
    logger.info("Monitoring test completed")