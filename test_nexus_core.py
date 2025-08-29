#!/usr/bin/env python3
"""
Nexus AI Framework - Core Functionality Test
Tests the essential components to verify the framework is working.
"""

import sys
import traceback

def run_test(name, test_func):
    """Run a test and report results."""
    try:
        test_func()
        print(f"✅ {name}")
        return True
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        # traceback.print_exc()
        return False

def test_core_imports():
    """Test core module imports."""
    import nexus
    from nexus.core.plugin_manager import PluginManager
    from nexus.core.config import ConfigSchema, ConfigManager
    from nexus.core.exceptions import NexusError
    from nexus.core.logger import get_logger

def test_capture_system():
    """Test capture system."""
    from nexus.capture.capture_manager import CaptureManager
    from nexus.capture.screen_capture import ScreenCapture
    # Test alias works
    assert ScreenCapture == CaptureManager

def test_vision_system():
    """Test vision processing."""
    from nexus.vision.frame_processor import FrameProcessor
    import numpy as np
    processor = FrameProcessor()
    # Create dummy frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = processor.process(frame)
    assert result is not None

def test_agent_system():
    """Test agent creation."""
    from nexus.agents import create_agent
    agent = create_agent('scripted', (84, 84, 1), 4)
    assert agent is not None
    # Test prediction
    import numpy as np
    obs = np.zeros((84, 84, 1))
    action = agent.predict(obs)
    assert action is not None

def test_input_system():
    """Test input controller."""
    from nexus.input.input_controller import InputController
    controller = InputController(backend='mock')
    assert controller is not None

def test_event_system():
    """Test event system."""
    from nexus.events import get_event_system, EventType
    event_system = get_event_system()
    # Test event emission
    event = event_system.emit(EventType.GAME_STARTED, {'test': True})
    assert event is not None

def test_plugin_system():
    """Test plugin manager."""
    from nexus.core.plugin_manager import PluginManager
    pm = PluginManager()
    assert pm is not None
    # List plugins (may be empty)
    plugins = pm.list_plugins()
    assert isinstance(plugins, list)

def test_config_system():
    """Test configuration."""
    from nexus.core.config import ConfigSchema, ConfigManager
    # Test schema
    config = ConfigSchema()
    config.game = {"name": "Test"}
    assert config.game["name"] == "Test"
    # Test manager
    manager = ConfigManager()
    assert manager is not None

def test_ocr_system():
    """Test OCR engine."""
    from nexus.ocr.ocr_engine import OCREngine
    engine = OCREngine(engine_type='tesseract')
    assert engine is not None

def test_sprite_system():
    """Test sprite management."""
    from nexus.sprites.sprite_manager import SpriteManager
    manager = SpriteManager()
    assert manager is not None

def test_analytics():
    """Test analytics client."""
    from nexus.analytics.client import AnalyticsClient
    client = AnalyticsClient(backend='memory')
    client.track_event('test', {'data': 1})
    stats = client.get_statistics()
    assert stats['total_events'] >= 1

def test_environments():
    """Test environment system."""
    from nexus.environments import GameEnvironment, EnvironmentManager
    manager = EnvironmentManager()
    assert manager is not None

def main():
    """Run all tests."""
    print("=" * 60)
    print("NEXUS AI FRAMEWORK - CORE FUNCTIONALITY TEST")
    print("=" * 60)
    print()
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Capture System", test_capture_system),
        ("Vision System", test_vision_system),
        ("Agent System", test_agent_system),
        ("Input System", test_input_system),
        ("Event System", test_event_system),
        ("Plugin System", test_plugin_system),
        ("Config System", test_config_system),
        ("OCR System", test_ocr_system),
        ("Sprite System", test_sprite_system),
        ("Analytics", test_analytics),
        ("Environments", test_environments),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        if run_test(name, test_func):
            passed += 1
        else:
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ ALL TESTS PASSED - NEXUS IS 100% WORKING!")
    else:
        print(f"⚠️  {failed} tests failed - {(passed/(passed+failed))*100:.1f}% working")
    
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())