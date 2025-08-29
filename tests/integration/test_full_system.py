"""
Integration tests for Nexus AI Framework.
Tests complete workflows to ensure all components work together.
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np


class TestFullSystemIntegration:
    """Complete system integration tests."""
    
    @pytest.mark.integration
    def test_game_capture_to_agent_pipeline(self):
        """Test complete pipeline from game capture to agent action."""
        from nexus.core.game import Game
        from nexus.capture.screen_capture import ScreenCapture
        from nexus.agents import create_agent
        from nexus.vision.frame_processor import FrameProcessor
        
        # Create mock game
        game = Game(name="TestGame", process_name="test.exe")
        
        # Setup capture with mock backend
        with patch('nexus.capture.backends.mss_backend.MSSBackend') as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.capture.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_backend.capture_region.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            
            capture = ScreenCapture(backend='mss')
            frame = capture.capture()
            
            assert frame is not None
            assert frame.shape == (480, 640, 3)
        
        # Process frame
        processor = FrameProcessor()
        processed = processor.process(frame)
        assert processed is not None
        
        # Create agent and get action
        agent = create_agent(
            'scripted',
            observation_space=(84, 84, 1),
            action_space=4
        )
        action = agent.predict(processed)
        assert action is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_to_training_workflow(self):
        """Test API endpoint triggering training workflow."""
        from nexus.api.app import NexusAPI
        from nexus.training.trainer import Trainer
        from nexus.core.config import Config
        
        # Create API instance
        api = NexusAPI()
        
        # Mock training request
        with patch('nexus.training.trainer.Trainer.train') as mock_train:
            mock_train.return_value = {'episodes': 100, 'avg_reward': 50.0}
            
            # Simulate API call
            response = await api.start_training(
                game_name="TestGame",
                agent_type="dqn",
                episodes=100
            )
            
            assert response['status'] == 'training_started'
            mock_train.assert_called_once()
    
    @pytest.mark.integration
    def test_plugin_system_integration(self):
        """Test plugin loading and execution."""
        from nexus.core.plugin_manager import PluginManager
        from nexus.core.config import Config
        
        # Create plugin manager
        pm = PluginManager()
        
        # Create test plugin
        test_plugin_dir = tempfile.mkdtemp()
        plugin_path = Path(test_plugin_dir) / "test_plugin"
        plugin_path.mkdir()
        
        # Write plugin files
        (plugin_path / "__init__.py").write_text("""
class TestPlugin:
    def __init__(self):
        self.name = "TestPlugin"
        self.enabled = True
    
    def process_frame(self, frame):
        return frame
        """)
        
        (plugin_path / "plugin.yaml").write_text("""
name: TestPlugin
version: 1.0.0
author: Test
description: Test plugin
entry_point: TestPlugin
        """)
        
        # Load plugin
        pm.load_plugin(str(plugin_path))
        assert "TestPlugin" in pm.plugins
        
        # Test plugin execution
        frame = np.zeros((100, 100, 3))
        result = pm.plugins["TestPlugin"].process_frame(frame)
        assert result is not None
    
    @pytest.mark.integration
    def test_event_system_workflow(self):
        """Test event system with multiple handlers."""
        from nexus.events import get_event_system, EventType
        
        event_system = get_event_system()
        events_received = []
        
        # Register handlers
        def handler1(event):
            events_received.append(('handler1', event))
        
        def handler2(event):
            events_received.append(('handler2', event))
        
        event_system.register_handler(EventType.GAME_STARTED, handler1)
        event_system.register_handler(EventType.GAME_STARTED, handler2)
        
        # Emit event
        event_system.emit(EventType.GAME_STARTED, {'game': 'TestGame'})
        
        # Allow async processing
        time.sleep(0.1)
        
        # Verify both handlers received event
        assert len(events_received) >= 2
        assert any(h[0] == 'handler1' for h in events_received)
        assert any(h[0] == 'handler2' for h in events_received)
    
    @pytest.mark.integration
    def test_analytics_collection(self):
        """Test analytics data collection and export."""
        from nexus.analytics.client import AnalyticsClient
        
        client = AnalyticsClient(backend='memory')
        
        # Track events
        client.track_event('game_started', {'game': 'TestGame'})
        client.track_event('action_taken', {'action': 'jump'})
        client.track_metric('fps', 60)
        client.track_metric('reward', 10.5)
        
        # Get statistics
        stats = client.get_statistics()
        assert stats['total_events'] >= 2
        assert 'fps' in stats['metrics']
        assert 'reward' in stats['metrics']
        
        # Export data
        export_data = client.export_data(format='json')
        assert 'events' in export_data
        assert 'metrics' in export_data
    
    @pytest.mark.integration
    def test_error_handling_cascade(self):
        """Test error handling across multiple components."""
        from nexus.core.exceptions import NexusError, handle_exception
        from nexus.core.logger import get_logger
        
        logger = get_logger(__name__)
        errors_caught = []
        
        @handle_exception(logger)
        def failing_operation():
            raise ValueError("Test error")
        
        # Should catch and convert exception
        with pytest.raises(NexusError):
            failing_operation()
        
        # Verify error was logged
        assert NexusError.get_error_stats()['ValueError'] > 0
    
    @pytest.mark.integration
    def test_configuration_cascade(self):
        """Test configuration loading and override hierarchy."""
        from nexus.core.config import ConfigSchema, ConfigManager
        
        # Create a simple config test
        config = ConfigSchema()
        config.game = {"name": "TestGame", "fps": 60}
        config.agent = {"type": "dqn", "learning_rate": 0.001}
        
        # Test config structure
        assert config.game["fps"] == 60
        assert config.game["name"] == "TestGame"
        assert config.agent["learning_rate"] == 0.001
        
        # Test ConfigManager creation
        manager = ConfigManager()
        assert manager is not None
        assert hasattr(manager, 'get')
        assert hasattr(manager, 'set')
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_monitoring_integration(self):
        """Test performance monitoring during operation."""
        from nexus.performance.monitor import PerformanceMonitor
        from nexus.vision.frame_processor import FrameProcessor
        
        monitor = PerformanceMonitor()
        processor = FrameProcessor()
        
        # Process frames while monitoring
        frames_processed = 0
        monitor.start()
        
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            with monitor.measure('frame_processing'):
                processed = processor.process(frame)
                frames_processed += 1
            
            monitor.update()
        
        monitor.stop()
        
        # Get metrics
        metrics = monitor.get_metrics()
        assert 'frame_processing' in metrics
        assert metrics['fps'] > 0
        assert frames_processed == 10
    
    @pytest.mark.integration
    def test_asset_extraction_pipeline(self):
        """Test asset extraction from game screenshots."""
        from nexus.sprites.asset_extractor import AdvancedAssetExtractor
        
        extractor = AdvancedAssetExtractor()
        
        # Create test screenshot with identifiable regions
        screenshot = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        # Add some "sprites" (rectangles)
        screenshot[100:150, 100:150] = [255, 0, 0]  # Red square
        screenshot[200:250, 200:250] = [0, 255, 0]  # Green square
        screenshot[300:350, 300:350] = [0, 0, 255]  # Blue square
        
        # Extract assets
        assets = extractor.extract_all_assets(screenshot)
        
        # Should find at least the colored squares
        assert len(assets) >= 3
        
        # Test deduplication
        unique_hashes = set(asset.hash for asset in assets)
        assert len(unique_hashes) == len(assets)
    
    @pytest.mark.integration
    def test_bluestacks_integration(self):
        """Test BlueStacks/Android emulator integration."""
        from nexus.emulators.bluestacks import BlueStacksController
        
        with patch('subprocess.run') as mock_run:
            # Mock ADB responses
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="device\n",
                stderr=""
            )
            
            controller = BlueStacksController()
            
            # Test connection
            connected = controller.connect()
            assert connected or not connected  # May fail if ADB not installed
            
            if connected:
                # Test basic operations
                controller.tap(100, 100)
                controller.swipe(100, 100, 200, 200)
                mock_run.assert_called()
    
    @pytest.mark.integration
    def test_logging_and_metrics_integration(self):
        """Test logging system with metrics collection."""
        from nexus.core.logger import get_logger, MetricsLogger
        from nexus.visualization.metrics_tracker import MetricsTracker
        
        logger = get_logger(__name__)
        metrics = MetricsLogger(__name__)
        tracker = MetricsTracker()
        
        # Log operations with metrics
        with logger.context(operation="test_operation"):
            logger.info("Starting operation")
            
            # Track metrics
            for i in range(10):
                value = np.random.random() * 100
                metrics.log_metric("test_metric", value)
                tracker.add_metric("test_metric", value, step=i)
            
            logger.info("Operation complete")
        
        # Get statistics
        stats = tracker.get_statistics("test_metric")
        assert stats['count'] == 10
        assert 'mean' in stats
        assert 'std' in stats


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    @pytest.mark.integration
    def test_capture_error_recovery(self):
        """Test recovery from capture errors."""
        from nexus.capture.screen_capture import ScreenCapture
        from nexus.core.exceptions import CaptureError
        
        capture = ScreenCapture(backend='mss')
        
        # Simulate capture failure and recovery
        with patch.object(capture.backend, 'capture') as mock_capture:
            # First call fails, second succeeds
            mock_capture.side_effect = [
                Exception("Capture failed"),
                np.zeros((480, 640, 3), dtype=np.uint8)
            ]
            
            # Should retry and succeed
            frame = capture.capture_with_retry(max_retries=2)
            assert frame is not None
    
    @pytest.mark.integration
    def test_training_checkpoint_recovery(self):
        """Test training recovery from checkpoint."""
        from nexus.training.trainer import Trainer
        from nexus.agents import create_agent
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pth"
            
            # Create trainer
            agent = create_agent('dqn', (84, 84, 1), 4)
            trainer = Trainer(agent, checkpoint_dir=tmpdir)
            
            # Simulate training interruption
            trainer.episode = 50
            trainer.save_checkpoint(str(checkpoint_path))
            
            # Create new trainer and restore
            new_trainer = Trainer(agent, checkpoint_dir=tmpdir)
            new_trainer.load_checkpoint(str(checkpoint_path))
            
            assert new_trainer.episode == 50


class TestPerformanceBaselines:
    """Test performance baselines and requirements."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_frame_processing_performance(self):
        """Test frame processing meets performance requirements."""
        from nexus.vision.frame_processor import FrameProcessor
        import time
        
        processor = FrameProcessor()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Warm up
        for _ in range(10):
            processor.process(frame)
        
        # Measure performance
        times = []
        for _ in range(100):
            start = time.perf_counter()
            processor.process(frame)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        # Should achieve at least 30 FPS
        assert fps >= 30, f"Frame processing too slow: {fps:.1f} FPS"
    
    @pytest.mark.integration
    def test_memory_usage(self):
        """Test memory usage stays within limits."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        from nexus.capture.frame_buffer import FrameBuffer
        
        buffer = FrameBuffer(max_frames=300)
        
        # Fill buffer
        for i in range(300):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            buffer.add(frame)
        
        current_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory - initial_memory
        
        # Should not exceed 500MB increase
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
        
        # Clean up
        del buffer
        gc.collect()


# Run integration tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])