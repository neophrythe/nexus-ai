"""Advanced Frame Transformation Pipeline for Nexus Framework"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import structlog
from collections import deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

from nexus.vision.frame_processing import GameFrame

logger = structlog.get_logger()


class TransformType(Enum):
    """Types of frame transformations"""
    # Color space transformations
    RGB_TO_HSV = "rgb_to_hsv"
    RGB_TO_LAB = "rgb_to_lab"
    RGB_TO_YUV = "rgb_to_yuv"
    
    # Geometric transformations
    ROTATE = "rotate"
    SCALE = "scale"
    TRANSLATE = "translate"
    FLIP = "flip"
    PERSPECTIVE = "perspective"
    AFFINE = "affine"
    
    # Enhancement transformations
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE_SHIFT = "hue_shift"
    
    # Filter transformations
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    MEDIAN_BLUR = "median_blur"
    BILATERAL_FILTER = "bilateral_filter"
    
    # Edge and feature transformations
    SOBEL = "sobel"
    LAPLACIAN = "laplacian"
    CANNY = "canny"
    HARRIS_CORNERS = "harris_corners"
    
    # Advanced transformations
    SUPER_RESOLUTION = "super_resolution"
    DENOISE = "denoise"
    DEBLUR = "deblur"
    INPAINT = "inpaint"
    
    # Custom transformations
    CUSTOM = "custom"


@dataclass
class TransformStep:
    """Single transformation step in pipeline"""
    transform_type: TransformType
    parameters: Dict[str, Any]
    enabled: bool = True
    name: Optional[str] = None
    condition: Optional[Callable] = None
    
    def should_apply(self, frame: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Check if transform should be applied"""
        if not self.enabled:
            return False
        
        if self.condition:
            return self.condition(frame, metadata)
        
        return True


@dataclass
class TransformResult:
    """Result of frame transformation"""
    frame: np.ndarray
    transform_history: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class FrameTransformer:
    """Advanced frame transformation engine"""
    
    def __init__(self, max_threads: int = 4):
        """
        Initialize frame transformer
        
        Args:
            max_threads: Maximum threads for parallel processing
        """
        self.transform_registry: Dict[TransformType, Callable] = {}
        self.pipelines: Dict[str, List[TransformStep]] = {}
        self.cache: Dict[str, np.ndarray] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_threads)
        
        # Register built-in transforms
        self._register_builtin_transforms()
    
    def _register_builtin_transforms(self):
        """Register all built-in transformation functions"""
        
        # Color space transforms
        self.transform_registry[TransformType.RGB_TO_HSV] = self._transform_rgb_to_hsv
        self.transform_registry[TransformType.RGB_TO_LAB] = self._transform_rgb_to_lab
        self.transform_registry[TransformType.RGB_TO_YUV] = self._transform_rgb_to_yuv
        
        # Geometric transforms
        self.transform_registry[TransformType.ROTATE] = self._transform_rotate
        self.transform_registry[TransformType.SCALE] = self._transform_scale
        self.transform_registry[TransformType.TRANSLATE] = self._transform_translate
        self.transform_registry[TransformType.FLIP] = self._transform_flip
        self.transform_registry[TransformType.PERSPECTIVE] = self._transform_perspective
        self.transform_registry[TransformType.AFFINE] = self._transform_affine
        
        # Enhancement transforms
        self.transform_registry[TransformType.BRIGHTNESS] = self._transform_brightness
        self.transform_registry[TransformType.CONTRAST] = self._transform_contrast
        self.transform_registry[TransformType.SATURATION] = self._transform_saturation
        self.transform_registry[TransformType.HUE_SHIFT] = self._transform_hue_shift
        
        # Filter transforms
        self.transform_registry[TransformType.GAUSSIAN_BLUR] = self._transform_gaussian_blur
        self.transform_registry[TransformType.MOTION_BLUR] = self._transform_motion_blur
        self.transform_registry[TransformType.MEDIAN_BLUR] = self._transform_median_blur
        self.transform_registry[TransformType.BILATERAL_FILTER] = self._transform_bilateral_filter
        
        # Edge transforms
        self.transform_registry[TransformType.SOBEL] = self._transform_sobel
        self.transform_registry[TransformType.LAPLACIAN] = self._transform_laplacian
        self.transform_registry[TransformType.CANNY] = self._transform_canny
        self.transform_registry[TransformType.HARRIS_CORNERS] = self._transform_harris_corners
        
        # Advanced transforms
        self.transform_registry[TransformType.SUPER_RESOLUTION] = self._transform_super_resolution
        self.transform_registry[TransformType.DENOISE] = self._transform_denoise
        self.transform_registry[TransformType.DEBLUR] = self._transform_deblur
        self.transform_registry[TransformType.INPAINT] = self._transform_inpaint
    
    def register_custom_transform(self, name: str, transform_func: Callable):
        """Register a custom transformation function"""
        self.transform_registry[TransformType.CUSTOM] = transform_func
        logger.info(f"Registered custom transform: {name}")
    
    def create_pipeline(self, name: str, steps: List[TransformStep]) -> bool:
        """
        Create a transformation pipeline
        
        Args:
            name: Pipeline name
            steps: List of transformation steps
        
        Returns:
            True if pipeline created successfully
        """
        try:
            self.pipelines[name] = steps
            logger.info(f"Created pipeline '{name}' with {len(steps)} steps")
            return True
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            return False
    
    def apply_transform(self, frame: np.ndarray, transform_step: TransformStep) -> np.ndarray:
        """
        Apply a single transformation
        
        Args:
            frame: Input frame
            transform_step: Transformation to apply
        
        Returns:
            Transformed frame
        """
        if transform_step.transform_type not in self.transform_registry:
            logger.warning(f"Unknown transform type: {transform_step.transform_type}")
            return frame
        
        transform_func = self.transform_registry[transform_step.transform_type]
        
        try:
            return transform_func(frame, **transform_step.parameters)
        except Exception as e:
            logger.error(f"Transform failed: {e}")
            return frame
    
    def apply_pipeline(self, frame: np.ndarray, pipeline_name: str,
                      metadata: Optional[Dict[str, Any]] = None) -> TransformResult:
        """
        Apply a transformation pipeline
        
        Args:
            frame: Input frame
            pipeline_name: Name of pipeline to apply
            metadata: Optional metadata for conditional transforms
        
        Returns:
            Transformation result
        """
        if pipeline_name not in self.pipelines:
            return TransformResult(
                frame=frame,
                transform_history=[],
                processing_time_ms=0,
                metadata={},
                success=False,
                error=f"Pipeline '{pipeline_name}' not found"
            )
        
        import time
        start_time = time.time()
        
        pipeline = self.pipelines[pipeline_name]
        current_frame = frame.copy()
        transform_history = []
        metadata = metadata or {}
        
        try:
            for step in pipeline:
                if step.should_apply(current_frame, metadata):
                    current_frame = self.apply_transform(current_frame, step)
                    transform_history.append(step.name or str(step.transform_type.value))
            
            processing_time = (time.time() - start_time) * 1000
            
            return TransformResult(
                frame=current_frame,
                transform_history=transform_history,
                processing_time_ms=processing_time,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            
            return TransformResult(
                frame=current_frame,
                transform_history=transform_history,
                processing_time_ms=processing_time,
                metadata=metadata,
                success=False,
                error=str(e)
            )
    
    async def apply_pipeline_async(self, frame: np.ndarray, pipeline_name: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> TransformResult:
        """Async version of apply_pipeline"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.apply_pipeline,
            frame,
            pipeline_name,
            metadata
        )
    
    def batch_transform(self, frames: List[np.ndarray], 
                       transform_step: TransformStep,
                       parallel: bool = True) -> List[np.ndarray]:
        """
        Apply transformation to multiple frames
        
        Args:
            frames: List of frames
            transform_step: Transformation to apply
            parallel: Use parallel processing
        
        Returns:
            List of transformed frames
        """
        if parallel and len(frames) > 1:
            futures = []
            for frame in frames:
                future = self.executor.submit(self.apply_transform, frame, transform_step)
                futures.append(future)
            
            return [future.result() for future in futures]
        else:
            return [self.apply_transform(frame, transform_step) for frame in frames]
    
    # Color Space Transformations
    def _transform_rgb_to_hsv(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Convert RGB to HSV color space"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def _transform_rgb_to_lab(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Convert RGB to LAB color space"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    def _transform_rgb_to_yuv(self, frame: np.ndarray, **kwargs) -> np.ndarray:
        """Convert RGB to YUV color space"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    # Geometric Transformations
    def _transform_rotate(self, frame: np.ndarray, angle: float = 0, 
                         center: Optional[Tuple[int, int]] = None, **kwargs) -> np.ndarray:
        """Rotate frame"""
        h, w = frame.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h))
    
    def _transform_scale(self, frame: np.ndarray, scale_x: float = 1.0, 
                        scale_y: float = 1.0, **kwargs) -> np.ndarray:
        """Scale frame"""
        h, w = frame.shape[:2]
        new_w = int(w * scale_x)
        new_h = int(h * scale_y)
        return cv2.resize(frame, (new_w, new_h))
    
    def _transform_translate(self, frame: np.ndarray, dx: int = 0, 
                           dy: int = 0, **kwargs) -> np.ndarray:
        """Translate frame"""
        h, w = frame.shape[:2]
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, matrix, (w, h))
    
    def _transform_flip(self, frame: np.ndarray, direction: str = "horizontal", **kwargs) -> np.ndarray:
        """Flip frame"""
        if direction == "horizontal":
            return cv2.flip(frame, 1)
        elif direction == "vertical":
            return cv2.flip(frame, 0)
        elif direction == "both":
            return cv2.flip(frame, -1)
        return frame
    
    def _transform_perspective(self, frame: np.ndarray, 
                              src_points: np.ndarray,
                              dst_points: np.ndarray, **kwargs) -> np.ndarray:
        """Apply perspective transformation"""
        h, w = frame.shape[:2]
        matrix = cv2.getPerspectiveTransform(src_points.astype(np.float32), 
                                            dst_points.astype(np.float32))
        return cv2.warpPerspective(frame, matrix, (w, h))
    
    def _transform_affine(self, frame: np.ndarray, 
                         src_points: np.ndarray,
                         dst_points: np.ndarray, **kwargs) -> np.ndarray:
        """Apply affine transformation"""
        h, w = frame.shape[:2]
        matrix = cv2.getAffineTransform(src_points.astype(np.float32)[:3], 
                                       dst_points.astype(np.float32)[:3])
        return cv2.warpAffine(frame, matrix, (w, h))
    
    # Enhancement Transformations
    def _transform_brightness(self, frame: np.ndarray, value: float = 0, **kwargs) -> np.ndarray:
        """Adjust brightness"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _transform_contrast(self, frame: np.ndarray, alpha: float = 1.0, **kwargs) -> np.ndarray:
        """Adjust contrast"""
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    
    def _transform_saturation(self, frame: np.ndarray, value: float = 1.0, **kwargs) -> np.ndarray:
        """Adjust saturation"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * value, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def _transform_hue_shift(self, frame: np.ndarray, shift: int = 0, **kwargs) -> np.ndarray:
        """Shift hue values"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Filter Transformations
    def _transform_gaussian_blur(self, frame: np.ndarray, kernel_size: int = 5, 
                                sigma: float = 1.0, **kwargs) -> np.ndarray:
        """Apply Gaussian blur"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def _transform_motion_blur(self, frame: np.ndarray, size: int = 15, 
                              angle: float = 0, **kwargs) -> np.ndarray:
        """Apply motion blur"""
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel = kernel / size
        
        # Rotate kernel
        matrix = cv2.getRotationMatrix2D((size/2, size/2), angle, 1)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
        
        return cv2.filter2D(frame, -1, kernel)
    
    def _transform_median_blur(self, frame: np.ndarray, kernel_size: int = 5, **kwargs) -> np.ndarray:
        """Apply median blur"""
        return cv2.medianBlur(frame, kernel_size)
    
    def _transform_bilateral_filter(self, frame: np.ndarray, d: int = 9, 
                                   sigma_color: float = 75, 
                                   sigma_space: float = 75, **kwargs) -> np.ndarray:
        """Apply bilateral filter"""
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    # Edge Detection Transformations
    def _transform_sobel(self, frame: np.ndarray, dx: int = 1, dy: int = 0, 
                        kernel_size: int = 3, **kwargs) -> np.ndarray:
        """Apply Sobel edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=kernel_size)
        sobel = np.absolute(sobel)
        sobel = np.uint8(np.clip(sobel, 0, 255))
        
        if len(frame.shape) == 3:
            return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
        return sobel
    
    def _transform_laplacian(self, frame: np.ndarray, kernel_size: int = 3, **kwargs) -> np.ndarray:
        """Apply Laplacian edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        laplacian = np.absolute(laplacian)
        laplacian = np.uint8(np.clip(laplacian, 0, 255))
        
        if len(frame.shape) == 3:
            return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        return laplacian
    
    def _transform_canny(self, frame: np.ndarray, low_threshold: int = 50, 
                        high_threshold: int = 150, **kwargs) -> np.ndarray:
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        if len(frame.shape) == 3:
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges
    
    def _transform_harris_corners(self, frame: np.ndarray, block_size: int = 2, 
                                 ksize: int = 3, k: float = 0.04, **kwargs) -> np.ndarray:
        """Detect Harris corners"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        corners = cv2.cornerHarris(gray, block_size, ksize, k)
        
        # Dilate for marking corners
        corners = cv2.dilate(corners, None)
        
        # Create output image
        result = frame.copy()
        result[corners > 0.01 * corners.max()] = [0, 0, 255] if len(frame.shape) == 3 else 255
        
        return result
    
    # Advanced Transformations
    def _transform_super_resolution(self, frame: np.ndarray, scale: int = 2, **kwargs) -> np.ndarray:
        """Apply super-resolution (simple upscaling with enhancement)"""
        # Simple implementation - can be replaced with deep learning model
        upscaled = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Enhance details
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)
        
        return sharpened
    
    def _transform_denoise(self, frame: np.ndarray, h: float = 10, 
                          template_window_size: int = 7,
                          search_window_size: int = 21, **kwargs) -> np.ndarray:
        """Apply denoising"""
        if len(frame.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(frame, None, h, h, 
                                                  template_window_size, search_window_size)
        else:
            return cv2.fastNlMeansDenoising(frame, None, h, 
                                          template_window_size, search_window_size)
    
    def _transform_deblur(self, frame: np.ndarray, kernel_size: int = 5, **kwargs) -> np.ndarray:
        """Apply deblurring (simple unsharp mask)"""
        # Create blur for unsharp mask
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        
        return sharpened
    
    def _transform_inpaint(self, frame: np.ndarray, mask: np.ndarray, 
                          method: str = "telea", **kwargs) -> np.ndarray:
        """Apply inpainting to remove objects"""
        if method == "telea":
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)


class AdaptiveTransformer:
    """Adaptive frame transformer that adjusts based on content"""
    
    def __init__(self, transformer: FrameTransformer):
        """
        Initialize adaptive transformer
        
        Args:
            transformer: Base frame transformer
        """
        self.transformer = transformer
        self.history: deque = deque(maxlen=30)
        self.scene_detector = SceneDetector()
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frame characteristics
        
        Args:
            frame: Input frame
        
        Returns:
            Frame analysis results
        """
        analysis = {
            "brightness": self._calculate_brightness(frame),
            "contrast": self._calculate_contrast(frame),
            "sharpness": self._calculate_sharpness(frame),
            "noise_level": self._estimate_noise(frame),
            "motion_blur": self._detect_motion_blur(frame),
            "dominant_colors": self._get_dominant_colors(frame)
        }
        
        return analysis
    
    def select_transforms(self, analysis: Dict[str, Any]) -> List[TransformStep]:
        """
        Select appropriate transforms based on analysis
        
        Args:
            analysis: Frame analysis results
        
        Returns:
            List of recommended transform steps
        """
        transforms = []
        
        # Adjust brightness if needed
        if analysis["brightness"] < 50:
            transforms.append(TransformStep(
                TransformType.BRIGHTNESS,
                {"value": 30},
                name="Brighten Dark Frame"
            ))
        elif analysis["brightness"] > 200:
            transforms.append(TransformStep(
                TransformType.BRIGHTNESS,
                {"value": -30},
                name="Darken Bright Frame"
            ))
        
        # Adjust contrast if needed
        if analysis["contrast"] < 30:
            transforms.append(TransformStep(
                TransformType.CONTRAST,
                {"alpha": 1.5},
                name="Increase Contrast"
            ))
        
        # Denoise if needed
        if analysis["noise_level"] > 10:
            transforms.append(TransformStep(
                TransformType.DENOISE,
                {"h": 10},
                name="Remove Noise"
            ))
        
        # Deblur if motion blur detected
        if analysis["motion_blur"] > 0.5:
            transforms.append(TransformStep(
                TransformType.DEBLUR,
                {"kernel_size": 5},
                name="Remove Motion Blur"
            ))
        
        # Enhance if low sharpness
        if analysis["sharpness"] < 50:
            transforms.append(TransformStep(
                TransformType.GAUSSIAN_BLUR,
                {"kernel_size": 3, "sigma": 0.5},
                name="Slight Sharpen"
            ))
        
        return transforms
    
    def process_adaptive(self, frame: np.ndarray) -> TransformResult:
        """
        Process frame with adaptive transformations
        
        Args:
            frame: Input frame
        
        Returns:
            Transformation result
        """
        # Analyze frame
        analysis = self.analyze_frame(frame)
        
        # Select transforms
        transforms = self.select_transforms(analysis)
        
        # Create temporary pipeline
        pipeline_name = "adaptive_temp"
        self.transformer.create_pipeline(pipeline_name, transforms)
        
        # Apply pipeline
        result = self.transformer.apply_pipeline(frame, pipeline_name, {"analysis": analysis})
        
        # Store in history
        self.history.append({
            "analysis": analysis,
            "transforms": transforms,
            "result": result
        })
        
        return result
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.mean(gray))
    
    def _calculate_contrast(self, frame: np.ndarray) -> float:
        """Calculate contrast (standard deviation)"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        return float(np.std(gray))
    
    def _calculate_sharpness(self, frame: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    def _estimate_noise(self, frame: np.ndarray) -> float:
        """Estimate noise level"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Use median absolute deviation
        median = np.median(gray)
        mad = np.median(np.abs(gray - median))
        return float(mad)
    
    def _detect_motion_blur(self, frame: np.ndarray) -> float:
        """Detect motion blur level"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Use FFT to detect motion blur
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Check for directional patterns in frequency domain
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Sample lines through center
        horizontal = magnitude[center_h, :]
        vertical = magnitude[:, center_w]
        
        # Motion blur shows as streaks in frequency domain
        h_var = np.var(horizontal)
        v_var = np.var(vertical)
        
        motion_indicator = abs(h_var - v_var) / (h_var + v_var + 1e-10)
        return float(motion_indicator)
    
    def _get_dominant_colors(self, frame: np.ndarray, n_colors: int = 3) -> List[Tuple[int, int, int]]:
        """Get dominant colors in frame"""
        # Resize for speed
        small = cv2.resize(frame, (50, 50))
        pixels = small.reshape(-1, 3)
        
        # Use k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        return [tuple(color) for color in colors]


class SceneDetector:
    """Detect scene changes and transitions"""
    
    def __init__(self, threshold: float = 0.3):
        """
        Initialize scene detector
        
        Args:
            threshold: Scene change threshold
        """
        self.threshold = threshold
        self.previous_frame = None
        self.scene_history = deque(maxlen=100)
    
    def detect_scene_change(self, frame: np.ndarray) -> bool:
        """
        Detect if scene has changed
        
        Args:
            frame: Current frame
        
        Returns:
            True if scene changed
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            return False
        
        # Calculate similarity
        similarity = self._calculate_similarity(self.previous_frame, frame)
        
        scene_changed = similarity < self.threshold
        
        self.scene_history.append({
            "similarity": similarity,
            "changed": scene_changed
        })
        
        self.previous_frame = frame
        
        return scene_changed
    
    def _calculate_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate frame similarity using histogram correlation"""
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2
        
        # Calculate histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        # Normalize
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(correlation)