"""Pytest configuration and fixtures"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import os

from nexus.core import ConfigManager, get_logger
from nexus.tests import TEST_CONFIG

logger = get_logger("test")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TEST_CONFIG.copy()


@pytest.fixture
def config_manager(test_config):
    """Create a ConfigManager with test configuration"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(test_config, f)
        config_path = f.name
    
    try:
        config = ConfigManager(config_path, auto_reload=False)
        yield config
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    import numpy as np
    
    # Create a simple 100x100 RGB image
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return image


@pytest.fixture
def mock_frame():
    """Create a mock game frame"""
    from nexus.vision.frame_processing import GameFrame
    import numpy as np
    
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return GameFrame(image, frame_id=1, timestamp=None)


@pytest.fixture
def sample_plugin_manifest():
    """Sample plugin manifest for testing"""
    return {
        "name": "test_plugin",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "Test plugin for unit tests",
        "plugin_type": "game",
        "entry_point": "test_plugin:TestPlugin",
        "dependencies": [],
        "min_nexus_version": "0.1.0"
    }


@pytest.fixture
def plugin_directory(temp_dir, sample_plugin_manifest):
    """Create a test plugin directory structure"""
    plugin_dir = temp_dir / "test_plugin"
    plugin_dir.mkdir()
    
    # Create manifest file
    import yaml
    manifest_path = plugin_dir / "manifest.yaml"
    with open(manifest_path, 'w') as f:
        yaml.dump(sample_plugin_manifest, f)
    
    # Create plugin file
    plugin_file = plugin_dir / "test_plugin.py"
    plugin_code = '''
from nexus.core.base import BasePlugin

class TestPlugin(BasePlugin):
    async def initialize(self):
        self.initialized = True
    
    async def shutdown(self):
        self.initialized = False
    
    async def validate(self):
        return hasattr(self, 'initialized') and self.initialized
'''
    plugin_file.write_text(plugin_code)
    
    return plugin_dir


@pytest.fixture
def sample_training_data(temp_dir):
    """Create sample training data for context classification"""
    data_dir = temp_dir / "training_data"
    data_dir.mkdir()
    
    # Create class directories
    for class_name in ["menu", "gameplay", "inventory"]:
        class_dir = data_dir / class_name
        class_dir.mkdir()
        
        # Create sample images
        import numpy as np
        from PIL import Image
        
        for i in range(5):
            image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            pil_image = Image.fromarray(image)
            pil_image.save(class_dir / f"sample_{i}.png")
    
    return data_dir


@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests"""
    import structlog
    
    # Configure structlog for testing
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@pytest.fixture
def mock_capture_backend():
    """Mock capture backend for testing"""
    from nexus.capture.base import CaptureBackend
    from nexus.vision.frame_processing import GameFrame
    import numpy as np
    import time
    
    class MockCaptureBackend(CaptureBackend):
        def __init__(self):
            super().__init__()
            self.frame_count = 0
        
        async def initialize(self):
            self.initialized = True
            return True
        
        async def capture_frame(self, region=None):
            self.frame_count += 1
            # Create a simple test frame
            width, height = (640, 480) if not region else (region[2] - region[0], region[3] - region[1])
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            return GameFrame(image, frame_id=self.frame_count, timestamp=time.time())
        
        async def get_screen_info(self):
            return {
                "width": 1920,
                "height": 1080,
                "refresh_rate": 60,
                "primary": True
            }
        
        async def cleanup(self):
            self.initialized = False
    
    return MockCaptureBackend()


@pytest.mark.asyncio
async def pytest_configure(config):
    """Pytest configuration"""
    # Set test environment variable
    os.environ["NEXUS_TESTING"] = "1"


def pytest_unconfigure(config):
    """Cleanup after tests"""
    # Remove test environment variable
    os.environ.pop("NEXUS_TESTING", None)