import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import cv2
import torch
import asyncio
from datetime import datetime
import structlog

logger = structlog.get_logger()


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    metadata: Dict[str, Any] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "class_id": self.class_id,
            "center": self.center,
            "area": self.area,
            "metadata": self.metadata or {}
        }


class ObjectDetector:
    
    def __init__(self, 
                 model_name: str = "yolov8n.pt",
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model = None
        self.class_names = []
        self._initialized = False
        
        logger.info(f"ObjectDetector created - Model: {model_name}, Device: {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the detection model"""
        try:
            from ultralytics import YOLO
            
            # Load model
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            
            # Get class names
            self.class_names = self.model.names
            
            # Warm up model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            await self.detect(dummy_img)
            
            self._initialized = True
            logger.info(f"ObjectDetector initialized with {len(self.class_names)} classes")
            
        except ImportError:
            logger.warning("Ultralytics not installed, using mock detector")
            self._initialized = True
            self.class_names = ["person", "car", "bicycle", "dog", "cat"]
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")
            raise
    
    async def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in a frame"""
        if not self._initialized:
            await self.initialize()
        
        try:
            if self.model is None:
                # Mock detection for testing
                return self._mock_detect(frame)
            
            # Run inference
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False
                )
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names[class_id]
                        
                        detection = Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            class_id=class_id,
                            metadata={"model": self.model_name}
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:
        """Mock detection for testing without model"""
        h, w = frame.shape[:2]
        
        # Generate random detections
        detections = []
        num_objects = np.random.randint(0, 3)
        
        for i in range(num_objects):
            x1 = np.random.randint(0, w - 100)
            y1 = np.random.randint(0, h - 100)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(50, 150)
            
            detection = Detection(
                class_name=np.random.choice(self.class_names),
                confidence=np.random.uniform(0.5, 0.99),
                bbox=(x1, y1, min(x2, w), min(y2, h)),
                class_id=i,
                metadata={"mock": True}
            )
            detections.append(detection)
        
        return detections
    
    async def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Detect objects in multiple frames"""
        if not self._initialized:
            await self.initialize()
        
        tasks = [self.detect(frame) for frame in frames]
        results = await asyncio.gather(*tasks)
        return results
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detection boxes on frame"""
        frame_copy = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self._get_color_for_class(det.class_id)
            
            # Draw bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(frame_copy, 
                         (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1),
                         color, -1)
            
            # Draw label text
            cv2.putText(frame_copy, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
        return colors[class_id % len(colors)]
    
    def filter_detections(self, 
                         detections: List[Detection],
                         classes: Optional[List[str]] = None,
                         min_confidence: Optional[float] = None,
                         min_area: Optional[int] = None) -> List[Detection]:
        """Filter detections based on criteria"""
        filtered = detections
        
        if classes:
            filtered = [d for d in filtered if d.class_name in classes]
        
        if min_confidence:
            filtered = [d for d in filtered if d.confidence >= min_confidence]
        
        if min_area:
            filtered = [d for d in filtered if d.area >= min_area]
        
        return filtered
    
    async def train(self, 
                   dataset_path: str,
                   epochs: int = 100,
                   batch_size: int = 16,
                   img_size: int = 640) -> Dict[str, Any]:
        """Train the model on custom dataset"""
        if not self.model:
            logger.error("Model not initialized for training")
            return {"error": "Model not initialized"}
        
        try:
            # Train model
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.train(
                    data=dataset_path,
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=self.device
                )
            )
            
            return {
                "status": "success",
                "metrics": results.results_dict
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"error": str(e)}
    
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load a trained model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model.to(self.device)
            self.class_names = self.model.names
            self._initialized = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise