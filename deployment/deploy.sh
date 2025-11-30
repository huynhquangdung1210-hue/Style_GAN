#!/bin/bash
# Kubernetes deployment script for Style Transfer API
# Handles complete deployment, updates, and rollbacks

set -e

# Configuration
NAMESPACE="style-transfer"
APP_NAME="style-transfer"
IMAGE_TAG="${1:-latest}"
REGISTRY="${REGISTRY:-your-registry.com}"
CONTEXT="${CONTEXT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check context
    current_context=$(kubectl config current-context)
    log_info "Current context: $current_context"
    
    if [ "$current_context" != "$CONTEXT" ]; then
        log_warning "Current context ($current_context) doesn't match expected ($CONTEXT)"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace
create_namespace() {
    log_info "Creating namespace..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f deployment/k8s/namespace.yaml
        log_success "Namespace created"
    fi
}

# Deploy configuration
deploy_config() {
    log_info "Deploying configuration..."
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f deployment/k8s/configmap.yaml
    
    # Update image tag in deployment
    sed "s|your-registry/style-transfer:1.0.0|$REGISTRY/$APP_NAME:$IMAGE_TAG|g" \
        deployment/k8s/deployment.yaml | kubectl apply -f -
    
    log_success "Configuration deployed"
}

# Deploy Redis
deploy_redis() {
    log_info "Deploying Redis..."
    
    kubectl apply -f deployment/k8s/redis.yaml
    
    # Wait for Redis to be ready
    log_info "Waiting for Redis to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=redis -n $NAMESPACE --timeout=300s
    
    log_success "Redis deployed and ready"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    kubectl apply -f deployment/k8s/monitoring.yaml
    
    # Wait for monitoring to be ready
    log_info "Waiting for monitoring stack to be ready..."
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=prometheus -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=grafana -n $NAMESPACE --timeout=300s
    
    log_success "Monitoring stack deployed"
}

# Deploy main application
deploy_app() {
    log_info "Deploying Style Transfer API..."
    
    # Update image tag and apply deployment
    sed "s|your-registry/style-transfer:1.0.0|$REGISTRY/$APP_NAME:$IMAGE_TAG|g" \
        deployment/k8s/deployment.yaml | kubectl apply -f -
    
    # Wait for deployment rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/style-transfer-api -n $NAMESPACE --timeout=600s
    
    log_success "Application deployed"
}

# Deploy ingress
deploy_ingress() {
    log_info "Deploying ingress..."
    
    kubectl apply -f deployment/k8s/ingress.yaml
    
    log_success "Ingress deployed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check if all pods are ready
    kubectl get pods -n $NAMESPACE
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=api -n $NAMESPACE --timeout=300s
    
    # Test health endpoint
    api_service=$(kubectl get svc style-transfer-api-service -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
    if kubectl run health-check --rm -i --restart=Never --image=curlimages/curl -- curl -f "http://$api_service/health"; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
}

# Scale deployment
scale_deployment() {
    local replicas=${1:-3}
    log_info "Scaling deployment to $replicas replicas..."
    
    kubectl scale deployment style-transfer-api --replicas=$replicas -n $NAMESPACE
    kubectl rollout status deployment/style-transfer-api -n $NAMESPACE
    
    log_success "Deployment scaled to $replicas replicas"
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    kubectl rollout undo deployment/style-transfer-api -n $NAMESPACE
    kubectl rollout status deployment/style-transfer-api -n $NAMESPACE
    
    log_success "Deployment rolled back"
}

# Get deployment status
get_status() {
    log_info "Getting deployment status..."
    
    echo "Namespace: $NAMESPACE"
    echo
    
    echo "Deployments:"
    kubectl get deployments -n $NAMESPACE
    echo
    
    echo "Services:"
    kubectl get services -n $NAMESPACE
    echo
    
    echo "Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    echo
    
    echo "Ingress:"
    kubectl get ingress -n $NAMESPACE
    echo
    
    echo "PVCs:"
    kubectl get pvc -n $NAMESPACE
}

# Clean up deployment
cleanup() {
    log_warning "This will delete all resources in namespace $NAMESPACE"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleaning up deployment..."
        kubectl delete namespace $NAMESPACE --cascade=foreground
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Show logs
show_logs() {
    local component=${1:-api}
    local lines=${2:-100}
    
    log_info "Showing logs for $component (last $lines lines)..."
    
    case $component in
        api)
            kubectl logs -l app.kubernetes.io/component=api -n $NAMESPACE --tail=$lines -f
            ;;
        redis)
            kubectl logs -l app.kubernetes.io/component=redis -n $NAMESPACE --tail=$lines -f
            ;;
        prometheus)
            kubectl logs -l app.kubernetes.io/component=prometheus -n $NAMESPACE --tail=$lines -f
            ;;
        grafana)
            kubectl logs -l app.kubernetes.io/component=grafana -n $NAMESPACE --tail=$lines -f
            ;;
        *)
            log_error "Unknown component: $component"
            log_info "Available components: api, redis, prometheus, grafana"
            ;;
    esac
}

# Port forward for local access
port_forward() {
    local service=${1:-api}
    local local_port=${2:-8080}
    
    log_info "Setting up port forward for $service..."
    
    case $service in
        api)
            kubectl port-forward svc/style-transfer-api-service $local_port:80 -n $NAMESPACE
            ;;
        grafana)
            kubectl port-forward svc/grafana-service $local_port:3000 -n $NAMESPACE
            ;;
        prometheus)
            kubectl port-forward svc/prometheus-service $local_port:9090 -n $NAMESPACE
            ;;
        redis)
            kubectl port-forward svc/redis-service $local_port:6379 -n $NAMESPACE
            ;;
        *)
            log_error "Unknown service: $service"
            log_info "Available services: api, grafana, prometheus, redis"
            ;;
    esac
}

# Main deployment function
deploy() {
    log_info "Starting deployment of Style Transfer API..."
    log_info "Image: $REGISTRY/$APP_NAME:$IMAGE_TAG"
    log_info "Namespace: $NAMESPACE"
    
    check_prerequisites
    create_namespace
    deploy_config
    deploy_redis
    deploy_monitoring
    deploy_app
    deploy_ingress
    run_health_checks
    
    log_success "Deployment completed successfully!"
    
    echo
    log_info "Useful commands:"
    echo "  View status: $0 status"
    echo "  View logs: $0 logs api"
    echo "  Scale: $0 scale 5"
    echo "  Port forward: $0 port-forward api 8080"
    echo "  Access Grafana: $0 port-forward grafana 3000"
}

# Help function
show_help() {
    cat << EOF
Style Transfer API Kubernetes Deployment Script

Usage: $0 <command> [options]

Commands:
  deploy [tag]        Deploy the application (default tag: latest)
  status              Show deployment status
  scale <replicas>    Scale the deployment
  rollback            Rollback to previous version
  logs <component>    Show logs (api, redis, prometheus, grafana)
  port-forward <svc>  Setup port forwarding
  cleanup             Delete all resources
  help                Show this help

Environment Variables:
  REGISTRY           Docker registry (default: your-registry.com)
  CONTEXT            Kubernetes context (default: production)

Examples:
  $0 deploy v1.2.3
  $0 scale 5
  $0 logs api
  $0 port-forward grafana 3000

EOF
}

# Main script
main() {
    case "${1:-help}" in
        deploy)
            deploy
            ;;
        status)
            get_status
            ;;
        scale)
            scale_deployment "${2:-3}"
            ;;
        rollback)
            rollback_deployment
            ;;
        logs)
            show_logs "${2:-api}" "${3:-100}"
            ;;
        port-forward)
            port_forward "${2:-api}" "${3:-8080}"
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"