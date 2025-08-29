#!/bin/bash

# Nexus AI Framework - Docker Build Script
# Builds multiple Docker images for different deployment scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${DOCKER_REGISTRY:-"docker.io"}
NAMESPACE=${DOCKER_NAMESPACE:-"nexus-ai"}
VERSION=${VERSION:-"latest"}
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build Docker image
build_image() {
    local target=$1
    local tag=$2
    local description=$3
    
    print_info "Building $description image..."
    
    docker build \
        --target $target \
        --tag $tag \
        --tag ${REGISTRY}/${NAMESPACE}/nexus:${target}-${VERSION} \
        --build-arg BUILD_DATE=${BUILD_DATE} \
        --build-arg VCS_REF=${VCS_REF} \
        --build-arg VERSION=${VERSION} \
        --cache-from ${REGISTRY}/${NAMESPACE}/nexus:${target}-latest \
        --file Dockerfile \
        .
    
    if [ $? -eq 0 ]; then
        print_info "Successfully built $description image: $tag"
    else
        print_error "Failed to build $description image"
        exit 1
    fi
}

# Parse command line arguments
TARGET=${1:-"all"}
PUSH=${2:-"false"}

# Main build process
print_info "Starting Nexus AI Docker build process..."
print_info "Registry: ${REGISTRY}/${NAMESPACE}"
print_info "Version: ${VERSION}"
print_info "Git Ref: ${VCS_REF}"

case $TARGET in
    "production"|"prod")
        build_image "production" "${REGISTRY}/${NAMESPACE}/nexus:latest" "production"
        ;;
    
    "development"|"dev")
        build_image "development" "${REGISTRY}/${NAMESPACE}/nexus:dev" "development"
        ;;
    
    "gpu")
        build_image "gpu" "${REGISTRY}/${NAMESPACE}/nexus:gpu" "GPU-enabled"
        ;;
    
    "minimal"|"min")
        build_image "minimal" "${REGISTRY}/${NAMESPACE}/nexus:minimal" "minimal"
        ;;
    
    "cloud")
        build_image "cloud" "${REGISTRY}/${NAMESPACE}/nexus:cloud" "cloud-optimized"
        ;;
    
    "all")
        build_image "production" "${REGISTRY}/${NAMESPACE}/nexus:latest" "production"
        build_image "development" "${REGISTRY}/${NAMESPACE}/nexus:dev" "development"
        build_image "gpu" "${REGISTRY}/${NAMESPACE}/nexus:gpu" "GPU-enabled"
        build_image "minimal" "${REGISTRY}/${NAMESPACE}/nexus:minimal" "minimal"
        build_image "cloud" "${REGISTRY}/${NAMESPACE}/nexus:cloud" "cloud-optimized"
        ;;
    
    *)
        print_error "Invalid target: $TARGET"
        echo "Usage: $0 [production|development|gpu|minimal|cloud|all] [push]"
        exit 1
        ;;
esac

# Push images if requested
if [ "$PUSH" == "push" ]; then
    print_info "Pushing images to registry..."
    
    case $TARGET in
        "all")
            docker push ${REGISTRY}/${NAMESPACE}/nexus:latest
            docker push ${REGISTRY}/${NAMESPACE}/nexus:dev
            docker push ${REGISTRY}/${NAMESPACE}/nexus:gpu
            docker push ${REGISTRY}/${NAMESPACE}/nexus:minimal
            docker push ${REGISTRY}/${NAMESPACE}/nexus:cloud
            ;;
        *)
            docker push ${REGISTRY}/${NAMESPACE}/nexus:${TARGET}
            ;;
    esac
    
    print_info "Images pushed successfully"
fi

# Generate SBOM (Software Bill of Materials)
if command -v syft &> /dev/null; then
    print_info "Generating SBOM..."
    syft ${REGISTRY}/${NAMESPACE}/nexus:latest -o json > sbom.json
    print_info "SBOM generated: sbom.json"
fi

# Scan for vulnerabilities
if command -v grype &> /dev/null; then
    print_info "Scanning for vulnerabilities..."
    grype ${REGISTRY}/${NAMESPACE}/nexus:latest
fi

print_info "Docker build process completed successfully!"

# Print image sizes
echo ""
print_info "Image sizes:"
docker images | grep "${NAMESPACE}/nexus"