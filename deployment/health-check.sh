#!/bin/bash
# Health check script for Docker container
# Returns 0 for healthy, 1 for unhealthy

set -e

# Configuration
HEALTH_URL="http://localhost:8080/health"
TIMEOUT=30
MAX_RETRIES=3

# Function to check service health
check_health() {
    local attempt=1
    
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "Health check attempt $attempt/$MAX_RETRIES..."
        
        # Use curl to check the health endpoint
        if curl -f -s --max-time $TIMEOUT "$HEALTH_URL" > /dev/null 2>&1; then
            echo "✅ Service is healthy"
            return 0
        fi
        
        echo "❌ Health check failed (attempt $attempt)"
        
        if [ $attempt -lt $MAX_RETRIES ]; then
            sleep 2
        fi
        
        ((attempt++))
    done
    
    echo "🚨 Service is unhealthy after $MAX_RETRIES attempts"
    return 1
}

# Function to check basic process health
check_process() {
    if ! pgrep -f "uvicorn" > /dev/null; then
        echo "❌ Main process not running"
        return 1
    fi
    
    echo "✅ Main process is running"
    return 0
}

# Function to check GPU availability (if enabled)
check_gpu() {
    if [ "$USE_GPU" = "true" ] && command -v nvidia-smi &> /dev/null; then
        if ! nvidia-smi -q > /dev/null 2>&1; then
            echo "⚠️ GPU check failed"
            return 1
        fi
        echo "✅ GPU is accessible"
    fi
    return 0
}

# Main health check
main() {
    echo "🏥 Starting health check..."
    
    # Check if the main process is running
    if ! check_process; then
        exit 1
    fi
    
    # Check GPU if required
    if ! check_gpu; then
        exit 1
    fi
    
    # Check the health endpoint
    if ! check_health; then
        exit 1
    fi
    
    echo "🎉 All health checks passed"
    exit 0
}

# Run health check
main "$@"