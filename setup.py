import sys
import platform
import subprocess
from setuptools import setup, find_packages
from pathlib import Path

# Platform detection
def is_wsl():
    """Check if running on Windows Subsystem for Linux."""
    if platform.system() != "Linux":
        return False
    
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

def is_wsl2():
    """Check if running on WSL2 specifically."""
    if not is_wsl():
        return False
    
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        kernel_version = result.stdout.strip()
        if 'microsoft' in kernel_version.lower():
            version_parts = kernel_version.split('.')
            if len(version_parts) >= 1:
                return int(version_parts[0]) >= 5
    except:
        pass
    return False

def get_platform_type():
    """Determine the platform type."""
    if platform.system() == "Windows":
        return "windows"
    elif is_wsl2():
        return "wsl2"
    elif is_wsl():
        return "wsl1"
    elif platform.system() == "Linux":
        return "linux"
    elif platform.system() == "Darwin":
        # macOS not supported due to lack of CUDA
        print("Warning: macOS detected but not supported (no CUDA support)")
        return "unsupported"
    else:
        return "unknown"

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

# Base requirements (cross-platform)
base_requirements = [
    "numpy>=1.19.2",
    "opencv-python>=4.5.3",
    "Pillow>=8.3.1",
    "scipy>=1.7.0",
    "scikit-learn>=0.24.2",
    "scikit-image>=0.18.3",
    "torch>=1.10.0",
    "torchvision>=0.11.0",
    "click>=8.0.1",
    "pyyaml>=5.4.1",
    "python-dotenv>=0.19.0",
    "structlog>=21.1.0",
    "colorama>=0.4.4",
    "tqdm>=4.62.0",
    "requests>=2.26.0",
    "mss>=6.1.0",
    "pyautogui>=0.9.53",
]

# Platform-specific requirements
platform_requirements = {
    "windows": [
        "pywin32>=301",
        "dxcam>=0.0.5",
        "keyboard>=0.13.5",
    ],
    "linux": [
        "python-xlib>=0.31",
        "evdev>=1.4.0",
        "pynput>=1.7.3",
    ],
    "wsl2": [
        # WSL2 - exclude evdev and other problematic packages
        "pynput>=1.7.3",
        # No evdev for WSL2
    ],
    "wsl1": [
        # WSL1 - very limited
        "pynput>=1.7.3",
    ],
    "macos": [
        "pyobjc>=7.3",
        "pynput>=1.7.3",
    ]
}

# Detect platform and get appropriate requirements
current_platform = get_platform_type()
install_requires = base_requirements.copy()

# Add platform-specific requirements
if current_platform in platform_requirements:
    install_requires.extend(platform_requirements[current_platform])
    
# Print platform information during setup
print("=" * 60)
print(f"Nexus AI Framework Setup")
print(f"Detected Platform: {current_platform.upper()}")
if current_platform == "wsl2":
    print("WSL2 Detected: Using WSL2-compatible dependencies")
    print("Note: evdev excluded (incompatible with WSL2)")
    print("Tip: Run ./scripts/setup_wsl2.sh for complete setup")
elif current_platform == "wsl1":
    print("WSL1 Detected: Limited functionality available")
print("=" * 60)

# Optional extras
extras_require = {
    "full": [
        "tensorflow>=2.6.0",
        "transformers>=4.20.0",
        "ultralytics>=8.0.0",
        "segment-anything>=1.0",
        "pytesseract>=0.3.8",
        "easyocr>=1.6.0",
        "wandb>=0.12.0",
        "mlflow>=1.20.0",
        "tensorboard>=2.7.0",
        "stable-baselines3>=1.3.0",
        "gym>=0.21.0",
        "PyQt5>=5.15.4",
        "flask>=2.0.1",
        "redis>=3.5.3",
        "sqlalchemy>=1.4.23",
    ],
    "dev": [
        "pytest>=6.2.4",
        "pytest-cov>=2.12.1",
        "pytest-asyncio>=0.15.1",
        "pytest-mock>=3.6.1",
        "black>=21.7b0",
        "flake8>=3.9.2",
        "mypy>=0.910",
        "pre-commit>=2.14.0",
        "ipython>=7.26.0",
        "jupyter>=1.0.0",
    ],
    "windows": platform_requirements.get("windows", []),
    "linux": platform_requirements.get("linux", []),
    "wsl2": platform_requirements.get("wsl2", []),
}

setup(
    name="nexus-game-ai",
    version="1.0.0",
    author="neophrythe",
    author_email="contact@digitalmanufacturinglabs.de",
    description="Next-generation Game AI Development Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neophrythe/nexus-ai",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "nexus=nexus.cli.main:main",
            "serpent=nexus.cli.main:serpent_compat",
            "nexus-detect-platform=scripts.detect_platform:print_platform_info",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    keywords="game ai reinforcement-learning computer-vision automation",
    include_package_data=True,
    package_data={
        "nexus": ["config/*.yaml", "assets/*"],
    },
)