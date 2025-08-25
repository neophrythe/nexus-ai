"""
Unit tests for Performance Monitor Plugin.
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from plugins.performance_monitor.plugin import PerformanceMonitorPlugin
from plugins.performance_monitor.config import PerformanceConfig


class TestPerformanceConfig(unittest.TestCase):
    """Test PerformanceConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig()
    
    def test_default_values(self):
        """Test default configuration values."""
        self.assertTrue(self.config.enabled)
        self.assertEqual(self.config.monitor_interval, 0.1)
        self.assertTrue(self.config.show_overlay)
        self.assertEqual(self.config.overlay_position, 'top_left')
    
    def test_validation_valid(self):
        """Test validation with valid config."""
        errors = self.config.validate()
        self.assertEqual(len(errors), 0)
    
    def test_validation_invalid_interval(self):
        """Test validation with invalid monitor interval."""
        self.config.monitor_interval = -1
        errors = self.config.validate()
        self.assertIn("monitor_interval must be between 0 and 10 seconds", errors)
        
        self.config.monitor_interval = 11
        errors = self.config.validate()
        self.assertIn("monitor_interval must be between 0 and 10 seconds", errors)
    
    def test_validation_invalid_overlay(self):
        """Test validation with invalid overlay settings."""
        self.config.overlay_position = 'center'
        errors = self.config.validate()
        self.assertTrue(any('overlay_position' in e for e in errors))
        
        self.config.overlay_position = 'top_left'
        self.config.overlay_opacity = 1.5
        errors = self.config.validate()
        self.assertIn("overlay_opacity must be between 0 and 1", errors)
    
    def test_validation_invalid_thresholds(self):
        """Test validation with invalid thresholds."""
        self.config.thresholds['min_fps'] = 0
        errors = self.config.validate()
        self.assertIn("min_fps must be at least 1", errors)
        
        self.config.thresholds['min_fps'] = 30
        self.config.thresholds['max_cpu'] = 101
        errors = self.config.validate()
        self.assertIn("max_cpu must be between 0 and 100", errors)
    
    def test_save_load_config(self):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            # Modify config
            self.config.monitor_interval = 0.5
            self.config.show_overlay = False
            self.config.thresholds['min_fps'] = 45
            
            # Save
            self.config.to_file(config_path)
            
            # Load
            loaded_config = PerformanceConfig.from_file(config_path)
            
            # Verify
            self.assertEqual(loaded_config.monitor_interval, 0.5)
            self.assertFalse(loaded_config.show_overlay)
            self.assertEqual(loaded_config.thresholds['min_fps'], 45)
        finally:
            Path(config_path).unlink()
    
    def test_apply_profile(self):
        """Test applying configuration profiles."""
        # Test minimal profile
        self.config.apply_profile('minimal')
        self.assertFalse(self.config.monitor_gpu)
        self.assertFalse(self.config.show_overlay)
        
        # Test full profile
        self.config.apply_profile('full')
        self.assertTrue(self.config.monitor_gpu)
        self.assertTrue(self.config.monitor_temperature)
        self.assertTrue(self.config.cpu_per_core)
        
        # Test competitive profile
        self.config.apply_profile('competitive')
        self.assertEqual(self.config.thresholds['min_fps'], 60)
        self.assertEqual(self.config.target_fps, 144)
        self.assertTrue(self.config.auto_optimize)
    
    def test_invalid_profile(self):
        """Test applying invalid profile."""
        with self.assertRaises(ValueError):
            self.config.apply_profile('nonexistent')


class TestPerformanceMonitorPlugin(unittest.TestCase):
    """Test PerformanceMonitorPlugin class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = PerformanceMonitorPlugin()
        self.plugin.config = {
            'show_overlay': True,
            'target_colors': ['red']
        }
    
    def test_initialization(self):
        """Test plugin initialization."""
        self.assertEqual(self.plugin.name, "Performance Monitor")
        self.assertEqual(self.plugin.version, "1.0.0")
        self.assertIsNotNone(self.plugin.metrics_history)
        self.assertIsNotNone(self.plugin.thresholds)
    
    def test_on_load(self):
        """Test plugin loading."""
        with patch.object(self.plugin, 'start_monitoring') as mock_start:
            self.plugin.on_load()
            mock_start.assert_called_once()
    
    def test_on_unload(self):
        """Test plugin unloading."""
        with patch.object(self.plugin, 'stop_monitoring') as mock_stop:
            self.plugin.on_unload()
            mock_stop.assert_called_once()
    
    def test_on_frame(self):
        """Test frame processing."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process frame
        result = self.plugin.on_frame(frame)
        
        # Check result
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(self.plugin.frame_count, 1)
    
    def test_fps_calculation(self):
        """Test FPS calculation."""
        # Simulate frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process multiple frames
        for _ in range(10):
            self.plugin.on_frame(frame)
            time.sleep(0.01)  # Simulate 100 FPS
        
        # Force FPS update
        self.plugin.last_fps_update = time.time() - 2
        self.plugin.on_frame(frame)
        
        # Check FPS was calculated
        self.assertGreater(self.plugin.current_fps, 0)
        self.assertGreater(len(self.plugin.metrics_history['fps']), 0)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_monitor_loop(self, mock_memory, mock_cpu):
        """Test monitoring loop."""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(percent=60.0)
        
        # Start monitoring
        self.plugin.start_monitoring()
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitoring
        self.plugin.stop_monitoring()
        
        # Check metrics were collected
        self.assertGreater(len(self.plugin.metrics_history['cpu_percent']), 0)
        self.assertGreater(len(self.plugin.metrics_history['memory_percent']), 0)
    
    def test_check_performance_alerts(self):
        """Test performance alert detection."""
        # Set up alert callback
        alert_received = []
        self.plugin.add_alert_callback(lambda alert: alert_received.append(alert))
        
        # Simulate low FPS
        self.plugin.current_fps = 20
        self.plugin.thresholds['min_fps'] = 30
        
        # Check alerts
        self.plugin._check_performance_alerts()
        
        # Verify alert was triggered
        self.assertEqual(len(alert_received), 1)
        self.assertEqual(alert_received[0]['type'], 'low_fps')
    
    def test_metrics_summary(self):
        """Test metrics summary generation."""
        # Add test data
        self.plugin.metrics_history['fps'].extend([30, 60, 45, 50, 55])
        self.plugin.metrics_history['cpu_percent'].extend([40, 50, 60, 55, 45])
        
        # Get summary
        summary = self.plugin.get_metrics_summary()
        
        # Verify summary
        self.assertIn('fps', summary)
        self.assertIn('cpu_percent', summary)
        self.assertEqual(summary['fps']['current'], 55)
        self.assertEqual(summary['fps']['min'], 30)
        self.assertEqual(summary['fps']['max'], 60)
        self.assertAlmostEqual(summary['fps']['avg'], 48.0, places=1)
    
    def test_optimize_settings(self):
        """Test optimization suggestions."""
        # Simulate poor performance
        self.plugin.current_fps = 20
        self.plugin.metrics_history['cpu_percent'].extend([90] * 30)
        self.plugin.metrics_history['memory_percent'].extend([85] * 30)
        
        # Get optimization suggestions
        result = self.plugin.optimize_settings()
        
        # Verify suggestions
        self.assertIn('suggestions', result)
        self.assertGreater(len(result['suggestions']), 0)
        
        # Check for expected suggestions
        suggestion_types = [s['setting'] for s in result['suggestions']]
        self.assertIn('resolution', suggestion_types)
        self.assertIn('processing_threads', suggestion_types)
    
    def test_export_metrics(self):
        """Test metrics export."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name
        
        try:
            # Add test data
            self.plugin.metrics_history['fps'].extend([30, 60, 45])
            
            # Export metrics
            self.plugin.export_metrics(export_path)
            
            # Verify file was created
            self.assertTrue(Path(export_path).exists())
            
            # Load and verify content
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            self.assertIn('metrics', data)
            self.assertIn('thresholds', data)
            self.assertIn('system_info', data)
        finally:
            Path(export_path).unlink()
    
    @patch('cv2.putText')
    @patch('cv2.circle')
    def test_overlay_rendering(self, mock_circle, mock_puttext):
        """Test overlay rendering."""
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add test metrics
        self.plugin.current_fps = 60
        self.plugin.metrics_history['frame_time'].append(16.67)
        self.plugin.metrics_history['cpu_percent'].append(50)
        self.plugin.metrics_history['memory_percent'].append(60)
        
        # Render overlay
        result = self.plugin._add_performance_overlay(frame)
        
        # Verify overlay was rendered
        self.assertTrue(mock_puttext.called)
        self.assertGreater(mock_puttext.call_count, 3)  # At least FPS, CPU, MEM
    
    def test_gpu_detection(self):
        """Test GPU detection methods."""
        # Test detection returns boolean
        result = self.plugin._check_gpu_monitoring()
        self.assertIsInstance(result, bool)
        
        # If GPU detected, test usage reading
        if self.plugin.gpu_available:
            usage = self.plugin._get_gpu_usage()
            if usage is not None:
                self.assertGreaterEqual(usage, 0)
                self.assertLessEqual(usage, 100)
    
    def test_network_latency_measurement(self):
        """Test network latency measurement."""
        # Test measurement
        latency = self.plugin._measure_network_latency()
        
        # Verify result
        if latency is not None:
            self.assertGreater(latency, 0)
            self.assertLess(latency, 10000)  # Less than 10 seconds
    
    def test_disk_io_measurement(self):
        """Test disk I/O measurement."""
        # First call initializes
        io1 = self.plugin._get_disk_io_rate()
        
        # Wait a bit
        time.sleep(0.1)
        
        # Second call should return rate
        io2 = self.plugin._get_disk_io_rate()
        
        if io2 is not None:
            self.assertGreaterEqual(io2, 0)
    
    def test_threshold_setting(self):
        """Test threshold configuration."""
        # Set custom threshold
        self.plugin.set_threshold('min_fps', 60)
        self.plugin.set_threshold('max_cpu', 70)
        
        # Verify
        self.assertEqual(self.plugin.thresholds['min_fps'], 60)
        self.assertEqual(self.plugin.thresholds['max_cpu'], 70)
    
    def test_alert_callbacks(self):
        """Test alert callback system."""
        # Set up callbacks
        callback1_called = []
        callback2_called = []
        
        self.plugin.add_alert_callback(lambda a: callback1_called.append(a))
        self.plugin.add_alert_callback(lambda a: callback2_called.append(a))
        
        # Trigger alert
        test_alert = {
            'type': 'test',
            'message': 'Test alert',
            'severity': 'info'
        }
        self.plugin._trigger_alert(test_alert)
        
        # Verify both callbacks were called
        self.assertEqual(len(callback1_called), 1)
        self.assertEqual(len(callback2_called), 1)
        self.assertEqual(callback1_called[0], test_alert)


class TestPerformanceIntegration(unittest.TestCase):
    """Integration tests for Performance Monitor Plugin."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plugin = PerformanceMonitorPlugin()
        self.config = PerformanceConfig()
    
    def test_full_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        # Configure plugin
        self.config.monitor_interval = 0.05
        self.config.export_metrics = True
        self.config.alerts_enabled = True
        
        # Apply config
        self.plugin.config = self.config.__dict__
        
        # Start monitoring
        self.plugin.on_load()
        
        # Process frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(20):
            result = self.plugin.on_frame(frame)
            self.assertIsNotNone(result)
            time.sleep(0.05)
        
        # Get metrics
        summary = self.plugin.get_metrics_summary()
        self.assertIsNotNone(summary)
        
        # Stop monitoring
        self.plugin.on_unload()
    
    def test_config_validation_integration(self):
        """Test configuration validation integration."""
        # Create invalid config
        self.config.monitor_interval = -1
        self.config.thresholds['min_fps'] = 0
        
        # Validate
        errors = self.config.validate()
        
        # Should have errors
        self.assertGreater(len(errors), 0)
        
        # Fix config
        self.config.monitor_interval = 0.1
        self.config.thresholds['min_fps'] = 30
        
        # Validate again
        errors = self.config.validate()
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()