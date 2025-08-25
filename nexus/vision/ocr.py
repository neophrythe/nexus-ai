import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import cv2
import asyncio
import structlog

logger = structlog.get_logger()


@dataclass
class TextDetection:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    language: str = "en"
    metadata: Dict[str, Any] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "language": self.language,
            "center": self.center,
            "metadata": self.metadata or {}
        }


class OCREngine:
    
    def __init__(self,
                 engine: str = "easyocr",
                 languages: List[str] = None,
                 gpu: bool = True):
        
        self.engine_type = engine
        self.languages = languages or ["en"]
        self.use_gpu = gpu and torch.cuda.is_available() if 'torch' in globals() else False
        
        self.engine = None
        self._initialized = False
        
        logger.info(f"OCR Engine created - Type: {engine}, Languages: {self.languages}")
    
    async def initialize(self) -> None:
        """Initialize the OCR engine"""
        try:
            if self.engine_type == "easyocr":
                import easyocr
                self.engine = easyocr.Reader(
                    self.languages,
                    gpu=self.use_gpu,
                    verbose=False
                )
            elif self.engine_type == "rapidocr":
                from rapidocr_onnxruntime import RapidOCR
                self.engine = RapidOCR()
            else:
                logger.warning(f"Unknown OCR engine: {self.engine_type}, using mock")
                
            self._initialized = True
            logger.info(f"OCR Engine initialized: {self.engine_type}")
            
        except ImportError as e:
            logger.warning(f"OCR engine {self.engine_type} not available: {e}")
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            raise
    
    async def detect_text(self, 
                         frame: np.ndarray,
                         region: Optional[Tuple[int, int, int, int]] = None) -> List[TextDetection]:
        """Detect text in frame or specific region"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Crop to region if specified
            if region:
                x1, y1, x2, y2 = region
                frame = frame[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
            else:
                offset_x, offset_y = 0, 0
            
            if self.engine is None:
                return self._mock_detect_text(frame, offset_x, offset_y)
            
            # Run OCR
            if self.engine_type == "easyocr":
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.engine.readtext(frame)
                )
                
                detections = []
                for bbox, text, confidence in results:
                    # Convert bbox format
                    x1 = int(min(bbox[0][0], bbox[3][0])) + offset_x
                    y1 = int(min(bbox[0][1], bbox[1][1])) + offset_y
                    x2 = int(max(bbox[1][0], bbox[2][0])) + offset_x
                    y2 = int(max(bbox[2][1], bbox[3][1])) + offset_y
                    
                    detection = TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        language=self.languages[0],
                        metadata={"engine": "easyocr"}
                    )
                    detections.append(detection)
                
                return detections
                
            elif self.engine_type == "rapidocr":
                results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.engine(frame)
                )
                
                if results is None or results[0] is None:
                    return []
                
                detections = []
                for item in results[0]:
                    bbox, text, confidence = item
                    x1, y1 = int(bbox[0][0]) + offset_x, int(bbox[0][1]) + offset_y
                    x2, y2 = int(bbox[2][0]) + offset_x, int(bbox[2][1]) + offset_y
                    
                    detection = TextDetection(
                        text=text,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        language=self.languages[0],
                        metadata={"engine": "rapidocr"}
                    )
                    detections.append(detection)
                
                return detections
            
            return []
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []
    
    def _mock_detect_text(self, frame: np.ndarray, offset_x: int, offset_y: int) -> List[TextDetection]:
        """Mock text detection for testing"""
        mock_texts = [
            "Score: 1000",
            "Lives: 3",
            "Level 1",
            "Game Over",
            "Press Start"
        ]
        
        h, w = frame.shape[:2]
        detections = []
        num_texts = np.random.randint(0, 3)
        
        for i in range(num_texts):
            x1 = np.random.randint(0, max(1, w - 100))
            y1 = np.random.randint(0, max(1, h - 30))
            x2 = x1 + np.random.randint(80, min(150, w - x1))
            y2 = y1 + 30
            
            detection = TextDetection(
                text=np.random.choice(mock_texts),
                confidence=np.random.uniform(0.8, 0.99),
                bbox=(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                language="en",
                metadata={"mock": True}
            )
            detections.append(detection)
        
        return detections
    
    async def detect_text_batch(self, frames: List[np.ndarray]) -> List[List[TextDetection]]:
        """Detect text in multiple frames"""
        if not self._initialized:
            await self.initialize()
        
        tasks = [self.detect_text(frame) for frame in frames]
        results = await asyncio.gather(*tasks)
        return results
    
    def preprocess_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better OCR accuracy"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Dilation and erosion
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def draw_text_detections(self, frame: np.ndarray, detections: List[TextDetection]) -> np.ndarray:
        """Draw text detection boxes on frame"""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text label
            label = f"{det.text} ({det.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame_copy,
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame_copy, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 0, 0), 1)
        
        return frame_copy
    
    def filter_text(self,
                   detections: List[TextDetection],
                   min_confidence: float = 0.5,
                   min_length: int = 1,
                   contains: Optional[str] = None) -> List[TextDetection]:
        """Filter text detections based on criteria"""
        filtered = detections
        
        # Filter by confidence
        filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        # Filter by text length
        filtered = [d for d in filtered if len(d.text) >= min_length]
        
        # Filter by substring
        if contains:
            filtered = [d for d in filtered if contains.lower() in d.text.lower()]
        
        return filtered
    
    def find_text(self, 
                 detections: List[TextDetection],
                 target: str,
                 exact: bool = False) -> Optional[TextDetection]:
        """Find specific text in detections"""
        for det in detections:
            if exact:
                if det.text == target:
                    return det
            else:
                if target.lower() in det.text.lower():
                    return det
        return None
    
    def extract_numbers(self, detections: List[TextDetection]) -> List[Tuple[TextDetection, float]]:
        """Extract numeric values from text detections"""
        import re
        results = []
        
        for det in detections:
            # Find all numbers in text
            numbers = re.findall(r'[-+]?\d*\.?\d+', det.text)
            for num_str in numbers:
                try:
                    value = float(num_str)
                    results.append((det, value))
                except ValueError:
                    continue
        
        return results