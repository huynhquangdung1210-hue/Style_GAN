#!/bin/bash
set -e

# Docker entrypoint script for Style Transfer API
# Handles initialization, configuration, and graceful startup

echo "🚀 Starting Style Transfer API..."

# Function to wait for dependencies
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name at $host:$port..."
    
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $attempt -ge $max_attempts ]; then
            echo "❌ Failed to connect to $service_name after $max_attempts attempts"
            exit 1
        fi
        
        echo "   Attempt $attempt/$max_attempts..."
        sleep 2
        ((attempt++))
    done
    
    echo "✅ $service_name is ready"
}

# Function to download models if needed
download_models() {
    echo "📦 Checking for required models..."
    
    # Create models directory if it doesn't exist
    mkdir -p /app/models
    
    # Check if models are already present
    if [ -d "/app/models/clip" ] && [ -d "/app/models/diffusion" ]; then
        echo "✅ Models already present, skipping download"
        return
    fi
    
    echo "⬇️ Downloading required models (this may take a while)..."
    
    # Download models using Python script
    python3 -c "
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import StableDiffusionXLPipeline
import os

print('Downloading CLIP model...')
clip_model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')
clip_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
clip_model.save_pretrained('/app/models/clip')
clip_processor.save_pretrained('/app/models/clip')

print('Downloading Stable Diffusion XL model...')
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.save_pretrained('/app/models/diffusion')
print('✅ Models downloaded successfully')
"
    
    echo "✅ Model download completed"
}

# Function to run database migrations
run_migrations() {
    echo "🗄️ Running database migrations..."
    
    # Check Redis connectivity
    if [ -n "$REDIS_URL" ]; then
        redis_host=$(echo "$REDIS_URL" | sed -n 's|.*://\([^:]*\):.*|\1|p')
        redis_port=$(echo "$REDIS_URL" | sed -n 's|.*:\([0-9]*\).*|\1|p')
        
        if [ -n "$redis_host" ] && [ -n "$redis_port" ]; then
            wait_for_service "$redis_host" "$redis_port" "Redis"
        fi
    fi
    
    # Run any database setup scripts
    if [ -f "/app/scripts/setup_redis.py" ]; then
        echo "🔧 Setting up Redis schemas..."
        python3 /app/scripts/setup_redis.py
    fi
    
    echo "✅ Database setup completed"
}

# Function to warm up the model
warm_up_model() {
    if [ "$WARMUP_ON_STARTUP" = "true" ]; then
        echo "🔥 Warming up model..."
        python3 -c "
from src.models import create_model, StyleTransferConfig
import torch

config = StyleTransferConfig(
    warmup_on_init=True,
    use_fp16=True
)

try:
    model = create_model(config)
    print('✅ Model warmup completed')
except Exception as e:
    print(f'⚠️ Model warmup failed: {e}')
    # Don't fail startup if warmup fails
"
    else
        echo "⏭️ Skipping model warmup (disabled)"
    fi
}

# Function to validate configuration
validate_config() {
    echo "🔍 Validating configuration..."
    
    python3 -c "
from src.utils.config import get_settings, validate_config

settings = get_settings()
warnings = validate_config(settings)

if warnings:
    print('⚠️ Configuration warnings:')
    for warning in warnings:
        print(f'   - {warning}')
else:
    print('✅ Configuration validation passed')

print(f'Environment: {settings.environment}')
print(f'Debug mode: {settings.debug}')
print(f'GPU available: {torch.cuda.is_available() if \"torch\" in globals() else \"Unknown\"}')
"
}

# Function to check GPU availability
check_gpu() {
    echo "🖥️ Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "📊 GPU Status:"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits || true
    else
        echo "⚠️ nvidia-smi not found, GPU status unknown"
    fi
    
    python3 -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('Running on CPU')
"
}

# Function to setup monitoring
setup_monitoring() {
    echo "📊 Setting up monitoring..."
    
    # Ensure log directory exists
    mkdir -p /app/logs
    
    # Initialize monitoring components
    python3 -c "
from src.utils.monitoring import init_monitoring
try:
    logger = init_monitoring()
    logger.info('Monitoring initialized successfully')
except Exception as e:
    print(f'⚠️ Monitoring setup warning: {e}')
"
    
    echo "✅ Monitoring setup completed"
}

# Main startup sequence
main() {
    echo "🏁 Starting initialization sequence..."
    
    # Validate configuration
    validate_config
    
    # Check GPU
    check_gpu
    
    # Setup monitoring
    setup_monitoring
    
    # Wait for dependencies
    run_migrations
    
    # Download models if needed and enabled
    if [ "$DOWNLOAD_MODELS_ON_STARTUP" = "true" ]; then
        download_models
    fi
    
    # Warm up model
    warm_up_model
    
    echo "✅ Initialization completed successfully!"
    echo "🎯 Starting application with command: $*"
    
    # Execute the main command
    exec "$@"
}

# Handle signals for graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, cleaning up..."
    
    # Kill background processes
    jobs -p | xargs -r kill
    
    # Clear GPU memory
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
" 2>/dev/null || true
    
    echo "✨ Cleanup completed"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Check if we're in development mode
if [ "$ENVIRONMENT" = "development" ]; then
    echo "🔧 Development mode detected"
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # Enable debug logging
    export LOG_LEVEL="DEBUG"
    
    # Skip some initialization steps for faster startup
    export WARMUP_ON_STARTUP="false"
    export DOWNLOAD_MODELS_ON_STARTUP="false"
fi

# Install netcat if not present (for service waiting)
if ! command -v nc &> /dev/null; then
    echo "📦 Installing netcat for service checks..."
    apt-get update && apt-get install -y netcat-openbsd
fi

# Start main function
main "$@"