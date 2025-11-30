"""
Configuration Management

Centralized configuration system using Pydantic settings with
environment variable support and validation.
"""

from pydantic import Field, validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    redis_url: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    
    redis_max_connections: int = Field(
        default=20,
        env="REDIS_MAX_CONNECTIONS"
    )
    
    redis_timeout: int = Field(
        default=30,
        env="REDIS_TIMEOUT"
    )


class StorageSettings(BaseSettings):
    """Storage configuration with support for MinIO, S3, and local storage."""
    
    # Storage backend: 'minio' (default), 's3' (AWS), or 'local'
    storage_type: str = Field(
        default="minio",
        env="STORAGE_TYPE"
    )
    
    # MinIO settings (default for self-hosted)
    minio_endpoint: str = Field(
        default="localhost:9000",
        env="MINIO_ENDPOINT"
    )
    
    minio_access_key: str = Field(
        default="minioadmin",
        env="MINIO_ACCESS_KEY"
    )
    
    minio_secret_key: str = Field(
        default="minioadmin",
        env="MINIO_SECRET_KEY"
    )
    
    minio_bucket: str = Field(
        default="style-transfer",
        env="MINIO_BUCKET"
    )
    
    minio_secure: bool = Field(
        default=False,  # HTTP for local development
        env="MINIO_SECURE"
    )
    
    # AWS S3 settings (optional)
    aws_region: str = Field(
        default="us-east-1",
        env="AWS_REGION"
    )
    
    aws_access_key_id: Optional[str] = Field(
        default=None,
        env="AWS_ACCESS_KEY_ID"
    )
    
    aws_secret_access_key: Optional[str] = Field(
        default=None,
        env="AWS_SECRET_ACCESS_KEY"
    )
    
    s3_bucket: Optional[str] = Field(
        default=None,
        env="S3_BUCKET"
    )
    
    s3_endpoint_url: Optional[str] = Field(
        default=None,
        env="S3_ENDPOINT_URL"
    )
    
    # Local storage fallback
    local_storage_path: str = Field(
        default="./storage",
        env="LOCAL_STORAGE_PATH"
    )
    
    @validator('storage_type')
    def validate_storage_type(cls, v):
        valid_types = ['minio', 's3', 'local']
        if v not in valid_types:
            raise ValueError(f"Storage type must be one of {valid_types}")
        return v


class ModelSettings(BaseSettings):
    """Model configuration."""
    
    clip_model_name: str = Field(
        default="openai/clip-vit-large-patch14",
        env="CLIP_MODEL_NAME"
    )
    
    diffusion_model_name: str = Field(
        default="stabilityai/stable-diffusion-xl-base-1.0",
        env="DIFFUSION_MODEL_NAME"
    )
    
    use_fp16: bool = Field(
        default=True,
        env="USE_FP16"
    )
    
    max_batch_size: int = Field(
        default=4,
        env="MAX_BATCH_SIZE"
    )
    
    cache_style_embeddings: bool = Field(
        default=True,
        env="CACHE_STYLE_EMBEDDINGS"
    )
    
    warmup_on_startup: bool = Field(
        default=True,
        env="WARMUP_ON_STARTUP"
    )
    
    device_type: str = Field(
        default="cuda",
        env="DEVICE_TYPE"
    )
    
    @validator('device_type')
    def validate_device(cls, v):
        if v not in ['cuda', 'cpu', 'mps']:
            raise ValueError("Device must be cuda, cpu, or mps")
        return v


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        env="JWT_SECRET_KEY"
    )
    
    jwt_algorithm: str = Field(
        default="HS256",
        env="JWT_ALGORITHM"
    )
    
    jwt_expiration_hours: int = Field(
        default=24,
        env="JWT_EXPIRATION_HOURS"
    )
    
    api_key_header: str = Field(
        default="X-API-Key",
        env="API_KEY_HEADER"
    )
    
    rate_limit_per_minute: int = Field(
        default=60,
        env="RATE_LIMIT_PER_MINUTE"
    )
    
    cors_origins: List[str] = Field(
        default=["*"],
        env="CORS_ORIGINS"
    )
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v


class ProcessingSettings(BaseSettings):
    """Image processing configuration."""
    
    max_file_size_mb: int = Field(
        default=50,
        env="MAX_FILE_SIZE_MB"
    )
    
    min_resolution: List[int] = Field(
        default=[256, 256],
        env="MIN_RESOLUTION"
    )
    
    max_resolution: List[int] = Field(
        default=[2048, 2048],
        env="MAX_RESOLUTION"
    )
    
    supported_formats: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"],
        env="SUPPORTED_FORMATS"
    )
    
    default_quality: str = Field(
        default="balanced",
        env="DEFAULT_QUALITY"
    )
    
    enable_nsfw_detection: bool = Field(
        default=True,
        env="ENABLE_NSFW_DETECTION"
    )
    
    nsfw_threshold: float = Field(
        default=0.7,
        env="NSFW_THRESHOLD"
    )
    
    @validator('supported_formats', pre=True)
    def parse_supported_formats(cls, v):
        if isinstance(v, str):
            return [fmt.strip().lower() for fmt in v.split(',')]
        return [fmt.lower() for fmt in v]


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    
    sentry_dsn: Optional[str] = Field(
        default=None,
        env="SENTRY_DSN"
    )
    
    prometheus_port: int = Field(
        default=9090,
        env="PROMETHEUS_PORT"
    )
    
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        env="JAEGER_ENDPOINT"
    )
    
    enable_metrics: bool = Field(
        default=True,
        env="ENABLE_METRICS"
    )
    
    enable_tracing: bool = Field(
        default=False,
        env="ENABLE_TRACING"
    )
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class AppSettings(BaseSettings):
    """Main application configuration."""
    
    # Basic app settings
    app_name: str = Field(
        default="Style Transfer API",
        env="APP_NAME"
    )
    
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION"
    )
    
    environment: str = Field(
        default="development",
        env="ENVIRONMENT"
    )
    
    debug: bool = Field(
        default=False,
        env="DEBUG"
    )
    
    # Server settings
    host: str = Field(
        default="0.0.0.0",
        env="HOST"
    )
    
    port: int = Field(
        default=8080,
        env="PORT"
    )
    
    workers: int = Field(
        default=1,
        env="WORKERS"
    )
    
    # Nested settings
    database: DatabaseSettings = DatabaseSettings()
    storage: StorageSettings = StorageSettings()
    model: ModelSettings = ModelSettings()
    security: SecuritySettings = SecuritySettings()
    processing: ProcessingSettings = ProcessingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get application settings (singleton pattern)."""
    global _settings
    
    if _settings is None:
        _settings = AppSettings()
    
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment."""
    global _settings
    _settings = AppSettings()
    return _settings


# Environment-specific configurations
def get_production_config() -> Dict[str, Any]:
    """Get production-specific configuration overrides."""
    return {
        "debug": False,
        "workers": 4,
        "monitoring": {
            "log_level": "INFO",
            "enable_metrics": True,
            "enable_tracing": True
        },
        "model": {
            "warmup_on_startup": True,
            "cache_style_embeddings": True
        },
        "security": {
            "rate_limit_per_minute": 30,
            "cors_origins": ["https://yourdomain.com"]
        }
    }


def get_development_config() -> Dict[str, Any]:
    """Get development-specific configuration overrides."""
    return {
        "debug": True,
        "workers": 1,
        "monitoring": {
            "log_level": "DEBUG",
            "enable_metrics": False,
            "enable_tracing": False
        },
        "model": {
            "warmup_on_startup": False,
            "cache_style_embeddings": False
        },
        "security": {
            "rate_limit_per_minute": 300,
            "cors_origins": ["*"]
        }
    }


def get_testing_config() -> Dict[str, Any]:
    """Get testing-specific configuration overrides."""
    return {
        "debug": True,
        "database": {
            "redis_url": "redis://localhost:6379/15"  # Use test database
        },
        "aws": {
            "s3_bucket": "test-bucket"
        },
        "model": {
            "warmup_on_startup": False,
            "cache_style_embeddings": False,
            "use_fp16": False
        },
        "processing": {
            "max_file_size_mb": 5,
            "enable_nsfw_detection": False
        }
    }


# Configuration validation
def validate_config(settings: AppSettings) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    
    warnings = []
    
    # Check required production settings
    if settings.is_production:
        if settings.security.jwt_secret_key == "your-secret-key-change-in-production":
            warnings.append("JWT secret key should be changed in production")
        
        if settings.storage.storage_type == "s3" and not settings.storage.aws_access_key_id:
            warnings.append("S3 storage configured but AWS credentials not provided")
        
        if settings.storage.storage_type == "minio" and settings.storage.minio_access_key == "minioadmin":
            warnings.append("MinIO using default credentials - change for production")
        
        if settings.security.cors_origins == ["*"]:
            warnings.append("CORS origins should be restricted in production")
    
    # Check model settings
    if settings.model.device_type == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        warnings.append("CUDA device specified but CUDA_VISIBLE_DEVICES not set")
    
    # Check storage settings
    if settings.storage.storage_type == "s3" and not settings.storage.s3_bucket:
        warnings.append("S3 storage selected but bucket not configured")
    elif settings.storage.storage_type == "minio" and not settings.storage.minio_bucket:
        warnings.append("MinIO storage selected but bucket not configured")
    
    return warnings


def create_config_file(env: str = "development") -> str:
    """Create a sample configuration file."""
    
    config_content = f"""# Style Transfer API Configuration
# Environment: {env}

# Application Settings
APP_NAME=Style Transfer API
APP_VERSION=1.0.0
ENVIRONMENT={env}
DEBUG={'true' if env == 'development' else 'false'}

# Server Settings
HOST=0.0.0.0
PORT=8080
WORKERS={'1' if env == 'development' else '4'}

# Database Settings
REDIS_URL=redis://localhost:6379
REDIS_MAX_CONNECTIONS=20

# Storage Settings (MinIO default)
STORAGE_TYPE=minio
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=style-transfer
MINIO_SECURE={'false' if env == 'development' else 'true'}

# AWS S3 Settings (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
S3_BUCKET=style-transfer-s3

# Local Storage Fallback
LOCAL_STORAGE_PATH=./storage

# Model Settings
CLIP_MODEL_NAME=openai/clip-vit-large-patch14
DIFFUSION_MODEL_NAME=stabilityai/stable-diffusion-xl-base-1.0
USE_FP16=true
MAX_BATCH_SIZE=4
DEVICE_TYPE=cuda

# Security Settings
JWT_SECRET_KEY=your-very-secure-secret-key-change-this
JWT_ALGORITHM=HS256
RATE_LIMIT_PER_MINUTE={'300' if env == 'development' else '30'}
CORS_ORIGINS={'*' if env == 'development' else 'https://yourdomain.com'}

# Processing Settings
MAX_FILE_SIZE_MB=50
MIN_RESOLUTION=256,256
MAX_RESOLUTION=2048,2048
SUPPORTED_FORMATS=jpg,jpeg,png,webp
ENABLE_NSFW_DETECTION=true
NSFW_THRESHOLD=0.7

# Monitoring Settings
LOG_LEVEL={'DEBUG' if env == 'development' else 'INFO'}
SENTRY_DSN=your_sentry_dsn
PROMETHEUS_PORT=9090
ENABLE_METRICS={'false' if env == 'development' else 'true'}
ENABLE_TRACING={'false' if env == 'development' else 'true'}
"""
    
    return config_content


# Example usage
if __name__ == "__main__":
    # Test configuration loading
    settings = get_settings()
    
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Redis: {settings.database.redis_url}")
    print(f"Model: {settings.model.diffusion_model_name}")
    
    # Validate configuration
    warnings = validate_config(settings)
    if warnings:
        print("Configuration warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Create sample config file
    config_content = create_config_file("production")
    print("\nSample production config:")
    print(config_content[:500] + "...")