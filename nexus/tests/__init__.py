"""Test suite for the Nexus framework"""

import sys
from pathlib import Path

# Add nexus to Python path for testing
nexus_root = Path(__file__).parent.parent
sys.path.insert(0, str(nexus_root))

# Test configuration
TEST_CONFIG = {
    "nexus": {
        "debug": True,
        "version": "0.1.0-test"
    },
    "capture": {
        "backend": "mss",  # Use MSS for testing (more compatible)
        "fps": 30,
        "buffer_size": 16
    },
    "vision": {
        "detection_model": "mock",
        "ocr_engine": "mock",
        "confidence_threshold": 0.5
    },
    "agents": {
        "default_type": "scripted",
        "batch_size": 8
    },
    "api": {
        "enabled": False  # Disable API during tests
    },
    "logging": {
        "level": "DEBUG",
        "console": True
    }
}