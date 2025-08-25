# Installation Guide

This guide will walk you through installing Nexus Game AI Framework on your system.

## Table of Contents
- [System Requirements](#system-requirements)
- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [Platform-Specific Setup](#platform-specific-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 or higher
- **RAM**: 8GB
- **Storage**: 5GB free space
- **Processor**: Dual-core 2.0GHz+

### Recommended Requirements
- **OS**: Windows 11, Ubuntu 22.04, macOS 12+
- **Python**: 3.10 or 3.11
- **RAM**: 16GB or more
- **Storage**: 20GB free space (for models and datasets)
- **Processor**: Quad-core 3.0GHz+
- **GPU**: NVIDIA GPU with CUDA 11.x support
- **VRAM**: 6GB or more

## Prerequisites

### 1. Python Installation

Verify Python is installed:
```bash
python --version
# or
python3 --version
```

If not installed, download from [python.org](https://python.org) or use your package manager.

### 2. Git Installation

```bash
git --version
```

If not installed:
- **Windows**: Download from [git-scm.com](https://git-scm.com)
- **Linux**: `sudo apt install git` (Ubuntu/Debian)
- **macOS**: `brew install git` (with Homebrew)

### 3. Virtual Environment (Recommended)

```bash
# Install virtualenv
pip install virtualenv

# Create virtual environment
python -m venv nexus-env

# Activate
# Windows:
nexus-env\Scripts\activate
# Linux/macOS:
source nexus-env/bin/activate
```

## Installation Methods

### Method 1: From GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/neophrythe/nexus-ai.git
cd nexus-ai

# Install in development mode
pip install -e .

# Or install with all features
pip install -e .[full]
```

### Method 2: Direct Installation

```bash
# Basic installation
pip install git+https://github.com/neophrythe/nexus-ai.git

# With all features
pip install "nexus-game-ai[full] @ git+https://github.com/neophrythe/nexus-ai.git"
```

### Method 3: From PyPI (Coming Soon)

```bash
# Basic installation
pip install nexus-game-ai

# With extras
pip install nexus-game-ai[full]
```

## Platform-Specific Setup

### Windows

#### 1. Install Visual C++ Redistributables
Download and install from [Microsoft](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

#### 2. Install Windows-specific dependencies
```bash
pip install -e .[windows]
```

#### 3. Enable Developer Mode (Optional)
For better performance with certain games:
1. Open Settings → Update & Security → For developers
2. Enable "Developer mode"

### Linux

#### 1. Install system dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-tk \
    xdotool \
    xvfb \
    scrot \
    tesseract-ocr

# Fedora
sudo dnf install -y \
    python3-devel \
    python3-tkinter \
    xdotool \
    xorg-x11-server-Xvfb \
    scrot \
    tesseract

# Arch
sudo pacman -S \
    python \
    tk \
    xdotool \
    xorg-server-xvfb \
    scrot \
    tesseract
```

#### 2. X11 Configuration
```bash
# Allow X11 connections
xhost +local:

# For headless systems
export DISPLAY=:0
```

### macOS

#### 1. Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### 2. Install dependencies
```bash
brew install python@3.11 tesseract

# For screen capture permissions
# System Preferences → Security & Privacy → Screen Recording
# Add Terminal/IDE to allowed apps
```

## GPU Support (Optional)

### NVIDIA CUDA Setup

#### 1. Install CUDA Toolkit
Download from [NVIDIA CUDA](https://developer.nvidia.com/cuda-downloads)

#### 2. Install cuDNN
Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

#### 3. Install PyTorch with CUDA
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### AMD ROCm Setup (Linux only)

```bash
# Install ROCm
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms

# Install PyTorch for ROCm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Feature-Specific Installations

### Computer Vision Models
```bash
# YOLOv8
pip install ultralytics

# Segment Anything Model
pip install segment-anything

# EasyOCR
pip install easyocr
```

### Reinforcement Learning
```bash
# Stable Baselines3
pip install stable-baselines3[extra]

# Gymnasium
pip install gymnasium[all]
```

### Experiment Tracking
```bash
# Weights & Biases
pip install wandb

# MLflow
pip install mlflow

# TensorBoard
pip install tensorboard
```

### Visual Debugger (Qt)
```bash
pip install PyQt5
```

## Verification

### 1. Verify Installation
```bash
# Check Nexus installation
nexus --version

# Check Python imports
python -c "import nexus; print(nexus.__version__)"
```

### 2. Run System Check
```bash
# Check system compatibility
nexus doctor

# Test frame capture
nexus test capture

# Test input control
nexus test input
```

### 3. Run Example
```bash
# Generate example game plugin
nexus generate game TestGame

# Run in debug mode
nexus debug --game=TestGame
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'nexus'
```bash
# Ensure installation completed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### Permission Denied (Linux/macOS)
```bash
# Add user to input group (Linux)
sudo usermod -a -G input $USER

# Logout and login again
```

#### Screen Capture Not Working (macOS)
1. System Preferences → Security & Privacy → Privacy
2. Select "Screen Recording"
3. Add your terminal or IDE
4. Restart the application

#### CUDA Not Detected
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Low FPS on Windows
```bash
# Try DXCam backend
nexus config set capture.backend dxcam

# Disable Windows Game Mode
# Settings → Gaming → Game Mode → Off
```

### Getting Help

If you encounter issues:

1. Check the [[Troubleshooting]] guide
2. Search [existing issues](https://github.com/neophrythe/nexus-ai/issues)
3. Join our [Discord](https://discord.gg/nexus) for community support
4. Create a [new issue](https://github.com/neophrythe/nexus-ai/issues/new) with:
   - System information (`nexus doctor` output)
   - Error messages
   - Steps to reproduce

## Next Steps

- Continue to [[Quick Start Guide]] to create your first agent
- Read about [[Configuration]] options
- Explore [[Game Plugins]] to integrate your game
- Learn about [[Agent Development]]

---

<p align="center">
  <a href="https://github.com/neophrythe/nexus-ai/wiki/Quick-Start-Guide">Next: Quick Start Guide →</a>
</p>