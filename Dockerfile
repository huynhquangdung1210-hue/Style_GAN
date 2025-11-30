# Production-ready Dockerfile for Style Transfer API
# Multi-stage build for optimized production image

# ========================================
# Stage 1: Build dependencies
# ========================================
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (this takes the longest)
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ========================================
# Stage 2: Production runtime
# ========================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r styleapp && useradd -r -g styleapp styleapp

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copy startup scripts
COPY deployment/docker-entrypoint.sh ./
COPY deployment/health-check.sh ./

# Make scripts executable
RUN chmod +x docker-entrypoint.sh health-check.sh

# Create directories for logs, models, and temporary files
RUN mkdir -p /app/logs /app/models /app/tmp \
    && chown -R styleapp:styleapp /app

# Set up health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD ./health-check.sh

# Switch to non-root user
USER styleapp

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

# ========================================
# Labels for metadata
# ========================================
LABEL maintainer="Style Transfer Team"
LABEL version="1.0.0"
LABEL description="Production-ready neural style transfer API"
LABEL org.opencontainers.image.source="https://github.com/your-org/style-transfer"
LABEL org.opencontainers.image.documentation="https://docs.your-domain.com/style-transfer"
LABEL org.opencontainers.image.licenses="MIT"

# ========================================
# Build arguments for customization
# ========================================
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.version=$VERSION