# Multi-stage Dockerfile for Nexus AI Framework
# Supports multiple deployment configurations

# Base stage with common dependencies
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    # X11 for GUI support
    xvfb \
    x11vnc \
    x11-utils \
    # Development tools
    git \
    wget \
    curl \
    build-essential \
    # ADB for Android support
    android-tools-adb \
    # Audio support
    libportaudio2 \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DISPLAY=:99

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# ============================================
# Development stage - includes all dev tools
# ============================================
FROM base as development

# Install all dependencies including dev tools
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r requirements-dev.txt

# Copy entire project
COPY . .

# Install Nexus in editable mode
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/recordings

# Expose ports
EXPOSE 8080 8081 8082 5900

# Start X virtual framebuffer and VNC
CMD Xvfb :99 -screen 0 1920x1080x24 & \
    x11vnc -display :99 -forever -nopw & \
    nexus gui

# ============================================
# Production stage - optimized for runtime
# ============================================
FROM base as production

# Install only production dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy only necessary files
COPY nexus/ /app/nexus/
COPY plugins/ /app/plugins/
COPY setup.py /app/
COPY README.md /app/
COPY LICENSE /app/

# Install Nexus
RUN pip install .

# Create non-root user
RUN useradd -m -u 1000 nexus && \
    chown -R nexus:nexus /app

USER nexus

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/recordings

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD nexus --version || exit 1

# Expose ports
EXPOSE 8080 8081 8082

# Default command
CMD ["nexus", "api", "--host", "0.0.0.0", "--port", "8080"]

# ============================================
# GPU stage - includes CUDA support
# ============================================
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    # X11 for GUI
    xvfb \
    x11vnc \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:99

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies with GPU support
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy project
COPY nexus/ /app/nexus/
COPY plugins/ /app/plugins/
COPY setup.py /app/

# Install Nexus
RUN pip install .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/recordings

# Expose ports
EXPOSE 8080 8081 8082 5900

# Start with GPU support
CMD Xvfb :99 -screen 0 1920x1080x24 & \
    nexus train --gpu

# ============================================
# Minimal stage - smallest possible image
# ============================================
FROM python:3.11-alpine as minimal

# Install minimal dependencies
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    g++ \
    jpeg-dev \
    zlib-dev \
    libffi-dev \
    cairo-dev \
    pango-dev \
    gdk-pixbuf-dev

WORKDIR /app

# Copy requirements (minimal subset)
COPY requirements-minimal.txt .

# Install minimal dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements-minimal.txt

# Copy core modules only
COPY nexus/core/ /app/nexus/core/
COPY nexus/api/ /app/nexus/api/
COPY nexus/agents/ /app/nexus/agents/
COPY nexus/__init__.py /app/nexus/
COPY setup.py /app/

# Install Nexus
RUN pip install .

# Create non-root user
RUN adduser -D -u 1000 nexus && \
    chown -R nexus:nexus /app

USER nexus

# Expose API port only
EXPOSE 8080

# Minimal command
CMD ["nexus", "api", "--host", "0.0.0.0"]

# ============================================
# Cloud stage - optimized for cloud deployment
# ============================================
FROM base as cloud

# Install cloud-specific dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install \
        boto3>=1.28.0 \
        google-cloud-storage>=2.10.0 \
        azure-storage-blob>=12.19.0 \
        kubernetes>=28.1.0

# Copy project
COPY . .

# Install Nexus
RUN pip install .

# Create non-root user with specific UID for cloud
RUN useradd -m -u 10001 nexus && \
    chown -R nexus:nexus /app

USER nexus

# Create directories with cloud-friendly paths
RUN mkdir -p /app/data /app/logs /app/models /app/cache

# Cloud health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose standard ports
EXPOSE 8080 8081 9090

# Cloud-optimized startup
CMD ["nexus", "api", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]