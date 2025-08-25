"""Tests for screen capture functionality"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, patch, AsyncMock

from nexus.capture.base import CaptureBackend, CaptureBackendType
from nexus.capture.capture_manager import CaptureManager
from nexus.capture.frame_grabber import FrameGrabber, RedisFrameBuffer
from nexus.vision.frame_processing import GameFrame


class TestCaptureBackend:
    """Test base capture backend functionality"""
    
    def test_capture_backend_creation(self):
        """Test creating capture backend"""
        backend = CaptureBackend()
        
        assert not backend.initialized
        assert backend.stats["frames_captured"] == 0
    
    @pytest.mark.asyncio
    async def test_mock_capture_backend(self, mock_capture_backend):
        """Test mock capture backend"""
        await mock_capture_backend.initialize()
        
        assert mock_capture_backend.initialized
        
        # Test frame capture
        frame = await mock_capture_backend.capture_frame()
        
        assert isinstance(frame, GameFrame)
        assert frame.shape[0] == 480  # Height
        assert frame.shape[1] == 640  # Width
        assert frame.shape[2] == 3    # RGB
        assert frame.frame_id == 1
    
    @pytest.mark.asyncio
    async def test_capture_with_region(self, mock_capture_backend):
        """Test capturing with specified region"""
        await mock_capture_backend.initialize()
        
        # Capture with region
        region = (100, 100, 300, 300)  # x, y, width, height
        frame = await mock_capture_backend.capture_frame(region)
        
        assert frame.shape[0] == 200  # Height (300-100)
        assert frame.shape[1] == 200  # Width (300-100)
    
    @pytest.mark.asyncio
    async def test_screen_info(self, mock_capture_backend):
        """Test getting screen information"""
        await mock_capture_backend.initialize()
        
        info = await mock_capture_backend.get_screen_info()
        
        assert "width" in info
        assert "height" in info
        assert info["width"] == 1920
        assert info["height"] == 1080


class TestCaptureManager:
    """Test capture manager functionality"""
    
    @pytest.mark.asyncio
    async def test_capture_manager_creation(self, config_manager):
        """Test creating capture manager"""
        manager = CaptureManager(
            backend_type=CaptureBackendType.MSS,
            device_idx=0,
            buffer_size=32
        )
        
        assert manager.backend_type == CaptureBackendType.MSS
        assert manager.device_idx == 0
        assert manager.buffer_size == 32
        assert not manager.initialized
    
    @pytest.mark.asyncio
    async def test_capture_manager_with_mock_backend(self, config_manager, mock_capture_backend):
        """Test capture manager with mock backend"""
        # Patch the backend creation to return our mock
        with patch('nexus.capture.capture_manager.CaptureManager._create_backend') as mock_create:
            mock_create.return_value = mock_capture_backend
            
            manager = CaptureManager(
                backend_type=CaptureBackendType.MSS,
                buffer_size=16
            )
            
            await manager.initialize()
            assert manager.initialized
            
            # Test frame capture
            frame = await manager.capture_frame()
            assert isinstance(frame, GameFrame)
            
            # Test stats
            stats = manager.get_stats()
            assert "frames_captured" in stats
            assert "avg_capture_time_ms" in stats
    
    @pytest.mark.asyncio
    async def test_capture_performance_monitoring(self, config_manager, mock_capture_backend):
        """Test capture performance monitoring"""
        with patch('nexus.capture.capture_manager.CaptureManager._create_backend') as mock_create:
            mock_create.return_value = mock_capture_backend
            
            manager = CaptureManager(
                backend_type=CaptureBackendType.MSS,
                buffer_size=16
            )
            await manager.initialize()
            
            # Capture multiple frames to generate stats
            for _ in range(10):
                await manager.capture_frame()
            
            stats = manager.get_stats()
            
            assert stats["frames_captured"] == 10
            assert stats["avg_capture_time_ms"] > 0
            assert "fps" in stats


class TestFrameGrabber:
    """Test frame grabber functionality"""
    
    @pytest.mark.asyncio
    async def test_frame_grabber_creation(self, config_manager, mock_capture_backend):
        """Test creating frame grabber"""
        grabber = FrameGrabber(
            backend=mock_capture_backend,
            fps=30,
            buffer_size=32
        )
        
        assert grabber.fps == 30
        assert grabber.buffer_size == 32
        assert not grabber.running
    
    @pytest.mark.asyncio
    async def test_frame_grabber_continuous_capture(self, config_manager, mock_capture_backend):
        """Test continuous frame capture"""
        await mock_capture_backend.initialize()
        
        grabber = FrameGrabber(
            backend=mock_capture_backend,
            fps=60,  # High FPS for faster testing
            buffer_size=16
        )
        
        # Start continuous capture
        await grabber.start()
        assert grabber.running
        
        # Wait for some frames to be captured
        await asyncio.sleep(0.1)
        
        # Get latest frame
        frame = await grabber.get_latest_frame()
        assert isinstance(frame, GameFrame)
        
        # Stop capture
        await grabber.stop()
        assert not grabber.running
    
    @pytest.mark.asyncio
    async def test_frame_buffer(self, config_manager, mock_capture_backend):
        """Test frame buffering"""
        await mock_capture_backend.initialize()
        
        grabber = FrameGrabber(
            backend=mock_capture_backend,
            fps=30,
            buffer_size=5  # Small buffer for testing
        )
        
        await grabber.start()
        
        # Wait for buffer to fill
        await asyncio.sleep(0.2)
        
        # Get buffered frames
        frames = await grabber.get_frame_history(count=3)
        assert len(frames) <= 3
        
        for frame in frames:
            assert isinstance(frame, GameFrame)
        
        await grabber.stop()


class TestRedisFrameBuffer:
    """Test Redis-based frame buffer"""
    
    def test_redis_buffer_creation(self):
        """Test creating Redis frame buffer"""
        buffer = RedisFrameBuffer(
            host="localhost",
            port=6379,
            db=0,
            max_size=100,
            compression=True
        )
        
        assert buffer.max_size == 100
        assert buffer.compression is True
    
    @pytest.mark.asyncio 
    async def test_redis_buffer_operations(self, mock_frame):
        """Test Redis buffer operations with mock Redis"""
        # Mock Redis client
        mock_redis = Mock()
        mock_redis.ping.return_value = True
        mock_redis.llen.return_value = 0
        mock_redis.lpush.return_value = 1
        mock_redis.lrange.return_value = []
        
        with patch('redis.Redis', return_value=mock_redis):
            buffer = RedisFrameBuffer(max_size=10)
            
            # Test connection
            is_connected = await buffer.is_connected()
            assert is_connected
            
            # Test pushing frame
            await buffer.push_frame(mock_frame)
            mock_redis.lpush.assert_called_once()
            
            # Test getting frame count
            count = await buffer.get_frame_count()
            assert count == 0  # Mock returns 0


class TestCaptureIntegration:
    """Test capture system integration"""
    
    @pytest.mark.asyncio
    async def test_capture_with_vision_pipeline(self, config_manager, mock_capture_backend):
        """Test capture integration with vision pipeline"""
        from nexus.vision.vision_pipeline import VisionPipeline
        
        await mock_capture_backend.initialize()
        
        # Create mock vision pipeline
        pipeline = Mock()
        pipeline.process_frame = AsyncMock(return_value={"detections": []})
        
        # Capture frame and process
        frame = await mock_capture_backend.capture_frame()
        result = await pipeline.process_frame(frame)
        
        assert "detections" in result
        pipeline.process_frame.assert_called_once_with(frame)
    
    @pytest.mark.asyncio
    async def test_capture_error_handling(self, config_manager):
        """Test capture error handling"""
        # Create backend that will fail
        class FailingBackend(CaptureBackend):
            async def initialize(self):
                self.initialized = True
                
            async def capture_frame(self, region=None):
                raise Exception("Capture failed")
        
        backend = FailingBackend()
        await backend.initialize()
        
        # Should handle the exception gracefully
        with pytest.raises(Exception):
            await backend.capture_frame()
    
    @pytest.mark.asyncio
    async def test_capture_memory_usage(self, config_manager, mock_capture_backend):
        """Test capture system memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        await mock_capture_backend.initialize()
        
        grabber = FrameGrabber(
            backend=mock_capture_backend,
            fps=60,
            buffer_size=100  # Large buffer
        )
        
        await grabber.start()
        
        # Capture for a short time
        await asyncio.sleep(0.5)
        
        await grabber.stop()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 200MB)
        assert memory_increase < 200 * 1024 * 1024


class TestCapturePerformance:
    """Test capture performance and optimization"""
    
    @pytest.mark.asyncio
    async def test_capture_fps_accuracy(self, config_manager, mock_capture_backend):
        """Test capture FPS accuracy"""
        await mock_capture_backend.initialize()
        
        target_fps = 30
        grabber = FrameGrabber(
            backend=mock_capture_backend,
            fps=target_fps,
            buffer_size=32
        )
        
        await grabber.start()
        
        # Measure actual FPS
        start_time = time.time()
        initial_count = mock_capture_backend.frame_count
        
        await asyncio.sleep(1.0)  # Capture for 1 second
        
        end_time = time.time()
        final_count = mock_capture_backend.frame_count
        
        actual_duration = end_time - start_time
        frames_captured = final_count - initial_count
        actual_fps = frames_captured / actual_duration
        
        await grabber.stop()
        
        # Allow some tolerance for timing variations
        assert abs(actual_fps - target_fps) < 5
    
    @pytest.mark.asyncio
    async def test_capture_latency(self, config_manager, mock_capture_backend):
        """Test capture latency"""
        await mock_capture_backend.initialize()
        
        # Measure single frame capture latency
        start_time = time.time()
        frame = await mock_capture_backend.capture_frame()
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Mock backend should be very fast
        assert latency_ms < 10  # Less than 10ms
        assert frame.capture_time_ms is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_capture(self, config_manager, mock_capture_backend):
        """Test concurrent capture operations"""
        await mock_capture_backend.initialize()
        
        # Start multiple concurrent capture operations
        tasks = []
        for _ in range(5):
            task = asyncio.create_task(mock_capture_backend.capture_frame())
            tasks.append(task)
        
        # Wait for all captures to complete
        frames = await asyncio.gather(*tasks)
        
        # All captures should succeed
        assert len(frames) == 5
        for frame in frames:
            assert isinstance(frame, GameFrame)
            assert frame.frame_id > 0


@pytest.mark.asyncio
class TestCaptureBackendSelection:
    """Test capture backend selection and switching"""
    
    async def test_backend_type_enum(self):
        """Test capture backend type enumeration"""
        assert CaptureBackendType.DXCAM.value == "dxcam"
        assert CaptureBackendType.MSS.value == "mss"
    
    async def test_backend_switching(self, config_manager):
        """Test switching between capture backends"""
        # This test would need real backends to be meaningful
        # For now, we'll test the enum and basic structure
        
        backend_types = [CaptureBackendType.MSS, CaptureBackendType.DXCAM]
        
        for backend_type in backend_types:
            manager = CaptureManager(
                backend_type=backend_type,
                buffer_size=16
            )
            
            assert manager.backend_type == backend_type