# Production-Ready Neural Style Transfer API

A scalable, GPU-accelerated neural style transfer service built with PyTorch, FastAPI, and Kubernetes. This system combines CLIP vision encoders with Stable Diffusion XL to create high-quality artistic style transfers.

## 🌟 Features

- **🎨 Advanced Style Transfer**: CLIP + Stable Diffusion XL pipeline
- **⚡ GPU Acceleration**: CUDA support with mixed precision (FP16)
- **🚀 Async Processing**: Redis + Celery job queue
- **📊 Production Monitoring**: Prometheus + Grafana metrics
- **🗄️ Flexible Storage**: MinIO (default), AWS S3, or local filesystem
- **🔒 Enterprise Security**: JWT auth, NSFW detection, rate limiting
- **☸️ Kubernetes Ready**: Full K8s deployment with autoscaling
- **🐳 Docker Support**: Multi-stage builds with GPU support
- **🛡️ Safety Filters**: Content moderation and authenticity checks
- **📈 Scalable Architecture**: Horizontal scaling with load balancing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Redis Queue    │    │  GPU Workers    │
│   Web Server    │───▶│   Job Manager    │───▶│  Style Transfer │
│                 │    │                  │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Prometheus    │    │   Grafana        │    │  MinIO Storage  │
│   Metrics       │    │   Dashboard      │    │  (S3 Compatible)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Docker & Docker Compose
- Kubernetes cluster (for production)

### Local Development

1. **Clone and Setup**:
   ```bash
   git clone <your-repo>
   cd Style_GAN
   
   # Create virtual environment
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   # Copy and configure environment
   cp .env.development .env
   # Edit .env with your settings
   ```

3. **Start Services**:
   ```bash
   # Start Redis
   docker run -d --name redis -p 6379:6379 redis:alpine
   
   # Start the API server
   python src/main.py
   
   # Start worker (in another terminal)
   python src/worker.py
   ```

4. **Test the API**:
   ```bash
   cd tests
   python test_api.py
   ```

### Docker Deployment

1. **Build and Run**:
   ```bash
   # Development
   docker-compose up -d
   
   # Production
   docker-compose -f docker-compose.prod.yml up -d
   ```

2. **Access Services**:
   - API: http://localhost:8080
   - Monitoring: http://localhost:3000 (Grafana)
   - Metrics: http://localhost:9090 (Prometheus)

### Kubernetes Deployment

1. **Deploy to Cluster**:
   ```bash
   cd deployment/k8s
   ./deploy.sh
   ```

2. **Configure Environment**:
   ```bash
   # Update secrets
   kubectl create secret generic style-transfer-secrets \
     --from-literal=jwt-secret=your-secret \
     --from-literal=aws-access-key=your-key \
     --from-literal=aws-secret-key=your-secret
   ```

3. **Monitor Deployment**:
   ```bash
   kubectl get pods -n style-transfer
   kubectl logs -f deployment/style-transfer-api -n style-transfer
   ```

## 📚 API Documentation

### Generate Style Transfer

**POST** `/v1/generate`

```json
{
  "subject_image": "base64_encoded_image",
  "style_image": "base64_encoded_image",
  "quality": "high",
  "style_strength": 0.8,
  "num_inference_steps": 20,
  "callback_url": "https://your-webhook.com/callback"
}
```

**Response**:
```json
{
  "job_id": "uuid-job-id",
  "status": "pending",
  "estimated_time": 30
}
```

### Check Job Status

**GET** `/v1/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "uuid-job-id",
  "status": "completed",
  "result_url": "https://s3.../result.jpg",
  "progress": 100,
  "processing_time": 25.3
}
```

### Health Check

**GET** `/health`

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gpu_available": true,
  "queue_size": 3
}
```

## 🔧 Configuration

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `CLIP_MODEL_NAME` | CLIP model for encoding | `openai/clip-vit-large-patch14` |
| `DIFFUSION_MODEL_NAME` | Stable Diffusion model | `stabilityai/stable-diffusion-xl-base-1.0` |
| `USE_FP16` | Enable mixed precision | `true` |
| `MAX_BATCH_SIZE` | Maximum batch size | `4` |
| `DEVICE_TYPE` | Device type (cuda/cpu) | `cuda` |

### Security Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET_KEY` | JWT signing secret | Required |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | `30` |
| `ENABLE_NSFW_DETECTION` | Content filtering | `true` |
| `NSFW_THRESHOLD` | NSFW detection threshold | `0.7` |

### Performance Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKERS` | API worker count | `4` |
| `MAX_FILE_SIZE_MB` | Max upload size | `50` |
| `CACHE_STYLE_EMBEDDINGS` | Enable style caching | `true` |
| `WARMUP_ON_STARTUP` | Model warmup | `true` |

## 🧪 Testing

### Unit Tests

```bash
python -m pytest tests/unit/
```

### API Tests

```bash
cd tests
python test_api.py --url http://localhost:8080
```

### Load Testing

```bash
python load_test.py --concurrent-users 10 --duration 60
```

## 📊 Monitoring

### Metrics

The service exposes Prometheus metrics at `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `style_transfer_jobs_total` - Total style transfer jobs
- `style_transfer_duration_seconds` - Processing time
- `gpu_memory_usage_bytes` - GPU memory usage
- `queue_size` - Job queue size

### Grafana Dashboard

Import the pre-configured dashboard from `deployment/monitoring/grafana-dashboard.json`.

### Alerting

Configure alerts for:
- High error rates (>5%)
- Long processing times (>60s)
- GPU memory exhaustion
- Queue backup (>50 jobs)

## 🛠️ Development

### Project Structure

```
src/
├── api/                 # FastAPI application
├── models/              # ML models and inference
├── preprocessing/       # Image processing pipeline
├── postprocessing/      # Result enhancement and safety
├── services/            # Business logic services
└── utils/               # Utilities and configuration

deployment/
├── docker/              # Docker configurations
├── k8s/                 # Kubernetes manifests
└── monitoring/          # Monitoring configurations

tests/
├── unit/                # Unit tests
├── integration/         # Integration tests
└── test_images/         # Sample images for testing
```

### Adding Features

1. **New Endpoints**: Add routes in `src/api/routes/`
2. **Model Updates**: Modify `src/models/style_transfer.py`
3. **Processing Pipeline**: Update `src/services/job_manager.py`
4. **Configuration**: Add settings to `src/utils/config.py`

### Code Quality

```bash
# Linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/

# Type checking
mypy src/

# Security scan
bandit -r src/
```

## 🚀 Production Deployment

### Scaling Guidelines

- **CPU**: 2-4 cores per API worker
- **Memory**: 8GB+ for GPU workers
- **GPU**: V100/A100 recommended, 16GB+ VRAM
- **Storage**: Fast SSD for model caching

### Performance Tuning

1. **Model Optimization**:
   - Enable FP16 mixed precision
   - Use model compilation (torch.compile)
   - Implement model quantization for inference

2. **Caching Strategy**:
   - Cache style embeddings
   - Use CDN for result images
   - Implement Redis clustering

3. **Scaling Configuration**:
   ```yaml
   # HPA configuration
   minReplicas: 2
   maxReplicas: 20
   targetCPUUtilizationPercentage: 70
   targetMemoryUtilizationPercentage: 80
   ```

### Security Checklist

- [ ] JWT secrets configured
- [ ] HTTPS/TLS enabled
- [ ] Rate limiting configured
- [ ] NSFW detection enabled
- [ ] Input validation implemented
- [ ] CORS properly configured
- [ ] Secrets managed securely

## 🔍 Troubleshooting

### Common Issues

**GPU Out of Memory**:
```bash
# Reduce batch size
export MAX_BATCH_SIZE=2
# Enable memory fraction
export GPU_MEMORY_FRACTION=0.8
```

**Model Loading Fails**:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/
# Re-download models
python -c "from src.models.style_transfer import StyleTransferModel; StyleTransferModel()"
```

**Redis Connection Issues**:
```bash
# Check Redis connectivity
redis-cli ping
# Restart Redis
docker restart redis
```

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

Check logs:
```bash
# API logs
kubectl logs -f deployment/style-transfer-api -n style-transfer

# Worker logs
kubectl logs -f deployment/style-transfer-worker -n style-transfer
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stable Diffusion XL team for the base model
- OpenAI for CLIP vision encoders
- FastAPI and PyTorch communities
- Kubernetes and Prometheus projects

---

**Built with ❤️ for production-scale AI applications**