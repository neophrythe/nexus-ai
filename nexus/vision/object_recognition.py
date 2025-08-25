"""Advanced Object Recognition System for Nexus Framework"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import structlog
from enum import Enum

logger = structlog.get_logger()


class DetectorType(Enum):
    """Available object detection models"""
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    DETECTRON2 = "detectron2"
    SAM = "sam"  # Segment Anything Model
    GROUNDING_DINO = "grounding_dino"
    LUMINOTH = "luminoth"  # Legacy SerpentAI compatibility


@dataclass
class Detection:
    """Single object detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None
    track_id: Optional[int] = None


@dataclass
class RecognitionResult:
    """Complete recognition result for a frame"""
    detections: List[Detection]
    frame_shape: Tuple[int, int]
    processing_time_ms: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseObjectDetector:
    """Base class for object detectors"""
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize detector
        
        Args:
            model_path: Path to model weights
            device: Device to use (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = []
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in image
        
        Args:
            image: Input image
        
        Returns:
            List of detections
        """
        # Base implementation - preprocess, run model, postprocess
        preprocessed = self.preprocess(image)
        
        if not hasattr(self, 'model') or self.model is None:
            logger.warning("No model loaded for detection")
            return []
        
        # Run inference if model has predict method
        if hasattr(self.model, 'predict'):
            outputs = self.model.predict(preprocessed)
            return self.postprocess(outputs, image.shape[:2])
        
        return []
    
    def load_model(self):
        """Load model weights"""
        if self.model_path and Path(self.model_path).exists():
            logger.info(f"Loading model from {self.model_path}")
            # Model loading logic should be implemented in subclasses
            self.model = None  # Placeholder - subclasses should set this
        else:
            logger.warning("No valid model path provided")
            self.model = None
    
    def preprocess(self, image: np.ndarray) -> Any:
        """Preprocess image for model"""
        return image
    
    def postprocess(self, outputs: Any, image_shape: Tuple[int, int]) -> List[Detection]:
        """Postprocess model outputs"""
        # Base implementation - should be overridden in subclasses
        detections = []
        
        # Generic postprocessing for common output formats
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            for i, output in enumerate(outputs):
                if hasattr(output, 'shape') and len(output.shape) >= 2:
                    # Assume bbox format: [x1, y1, x2, y2, conf, class_id]
                    if output.shape[-1] >= 4:
                        detection = Detection(
                            bbox=tuple(map(int, output[:4])),
                            confidence=float(output[4]) if len(output) > 4 else 1.0,
                            class_id=int(output[5]) if len(output) > 5 else 0,
                            class_name="unknown"
                        )
                        detections.append(detection)
        
        return detections


class YOLOv8Detector(BaseObjectDetector):
    """YOLOv8 object detector with modern features"""
    
    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 model_size: str = "n"):  # n, s, m, l, x
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to model weights
            device: Device to use
            confidence_threshold: Confidence threshold
            model_size: Model size variant
        """
        super().__init__(model_path, device, confidence_threshold)
        self.model_size = model_size
        self.load_model()
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Load pretrained model
                self.model = YOLO(f'yolov8{self.model_size}.pt')
            
            # Get class names
            self.class_names = self.model.names
            
            logger.info(f"Loaded YOLOv8{self.model_size} model")
            
        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects using YOLOv8"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold, device=self.device)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    detection = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.class_names[class_id]
                    )
                    
                    detections.append(detection)
        
        return detections
    
    def detect_and_track(self, image: np.ndarray) -> List[Detection]:
        """Detect and track objects"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Run tracking
        results = self.model.track(image, persist=True, conf=self.confidence_threshold)
        
        detections = []
        
        for r in results:
            boxes = r.boxes
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get track ID if available
                    track_id = None
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                    
                    detection = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=self.class_names[class_id],
                        track_id=track_id
                    )
                    
                    detections.append(detection)
        
        return detections


class SAMDetector(BaseObjectDetector):
    """Segment Anything Model for advanced segmentation"""
    
    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 model_type: str = "vit_h"):  # vit_h, vit_l, vit_b
        """
        Initialize SAM detector
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            model_type: SAM model type
        """
        super().__init__(model_path, device, 0.0)  # SAM doesn't use confidence threshold
        self.model_type = model_type
        self.predictor = None
        self.load_model()
    
    def load_model(self):
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
            
            # Model URLs
            model_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
            
            if self.model_path and Path(self.model_path).exists():
                checkpoint_path = self.model_path
            else:
                # Download model if needed
                import requests
                checkpoint_path = f"sam_{self.model_type}.pth"
                
                if not Path(checkpoint_path).exists():
                    logger.info(f"Downloading SAM {self.model_type} model...")
                    url = model_urls[self.model_type]
                    response = requests.get(url, stream=True)
                    
                    with open(checkpoint_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
            
            # Load model
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            
            self.predictor = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            
            logger.info(f"Loaded SAM {self.model_type} model")
            
        except ImportError:
            logger.error("Segment Anything not installed. Run: pip install segment-anything")
            raise
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Generate masks for all objects"""
        if self.mask_generator is None:
            raise ValueError("Model not loaded")
        
        # Generate masks
        masks = self.mask_generator.generate(image)
        
        detections = []
        
        for i, mask_data in enumerate(masks):
            # Get bounding box from mask
            segmentation = mask_data['segmentation']
            bbox = mask_data['bbox']  # x, y, w, h
            
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            detection = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=mask_data['predicted_iou'],
                class_id=0,  # SAM doesn't provide classes
                class_name="object",
                mask=segmentation
            )
            
            detections.append(detection)
        
        return detections
    
    def detect_with_prompts(self, image: np.ndarray, 
                          point_prompts: Optional[List[Tuple[int, int]]] = None,
                          box_prompts: Optional[List[Tuple[int, int, int, int]]] = None) -> List[Detection]:
        """Detect objects with point or box prompts"""
        if self.predictor is None:
            raise ValueError("Model not loaded")
        
        self.predictor.set_image(image)
        
        detections = []
        
        # Process point prompts
        if point_prompts:
            for point in point_prompts:
                masks, scores, _ = self.predictor.predict(
                    point_coords=np.array([point]),
                    point_labels=np.array([1]),  # 1 for foreground
                    multimask_output=True
                )
                
                # Use best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                
                # Get bbox from mask
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    x1, y1 = x_coords.min(), y_coords.min()
                    x2, y2 = x_coords.max(), y_coords.max()
                    
                    detection = Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(scores[best_idx]),
                        class_id=0,
                        class_name="object",
                        mask=mask
                    )
                    detections.append(detection)
        
        # Process box prompts
        if box_prompts:
            for box in box_prompts:
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=True
                )
                
                # Use best mask
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                
                detection = Detection(
                    bbox=box,
                    confidence=float(scores[best_idx]),
                    class_id=0,
                    class_name="object",
                    mask=mask
                )
                detections.append(detection)
        
        return detections


class GroundingDINODetector(BaseObjectDetector):
    """Grounding DINO for open-vocabulary object detection"""
    
    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.35):
        """
        Initialize Grounding DINO detector
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            confidence_threshold: Confidence threshold
        """
        super().__init__(model_path, device, confidence_threshold)
        self.load_model()
    
    def load_model(self):
        """Load Grounding DINO model"""
        try:
            # Note: Grounding DINO requires specific installation
            # This is a placeholder for the actual implementation
            logger.info("Grounding DINO detector initialized (placeholder)")
            
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
    
    def detect_with_text(self, image: np.ndarray, text_prompt: str) -> List[Detection]:
        """
        Detect objects described by text prompt
        
        Args:
            image: Input image
            text_prompt: Text description of objects to detect
        
        Returns:
            List of detections
        """
        # Placeholder implementation
        logger.warning("Grounding DINO detection not fully implemented")
        return []


class LuminothDetector(BaseObjectDetector):
    """Luminoth detector for SerpentAI compatibility"""
    
    def __init__(self, model_path: Optional[str] = None,
                 device: Optional[str] = None,
                 confidence_threshold: float = 0.5):
        """
        Initialize Luminoth detector
        
        Args:
            model_path: Path to model checkpoint
            device: Device to use
            confidence_threshold: Confidence threshold
        """
        super().__init__(model_path, device, confidence_threshold)
        self.config_path = None
        self.load_model()
    
    def load_model(self):
        """Load Luminoth model"""
        try:
            # Luminoth compatibility layer
            logger.info("Luminoth detector initialized (compatibility mode)")
            
            # Try to use Detectron2 as modern replacement
            from detectron2 import model_zoo
            from detectron2.engine import DefaultPredictor
            from detectron2.config import get_cfg
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.DEVICE = self.device
            
            self.predictor = DefaultPredictor(cfg)
            
            # COCO classes
            from detectron2.data import MetadataCatalog
            self.class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            
            logger.info("Using Detectron2 as Luminoth replacement")
            
        except ImportError:
            logger.warning("Detectron2 not available for Luminoth compatibility")
            self.predictor = None
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect objects using Luminoth/Detectron2"""
        if self.predictor is None:
            return []
        
        outputs = self.predictor(image)
        
        detections = []
        instances = outputs["instances"].to("cpu")
        
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.numpy()[0]
            x1, y1, x2, y2 = bbox
            
            detection = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(instances.scores[i]),
                class_id=int(instances.pred_classes[i]),
                class_name=self.class_names[instances.pred_classes[i]]
            )
            
            detections.append(detection)
        
        return detections


class ObjectRecognitionSystem:
    """Unified object recognition system"""
    
    def __init__(self):
        """Initialize recognition system"""
        self.detectors: Dict[DetectorType, BaseObjectDetector] = {}
        self.active_detector: Optional[DetectorType] = None
        self.tracker = None
        
        # Initialize available detectors
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize available object detectors"""
        # Try to initialize each detector
        try:
            self.detectors[DetectorType.YOLOV8] = YOLOv8Detector()
            self.active_detector = DetectorType.YOLOV8
            logger.info("YOLOv8 detector available")
        except Exception as e:
            logger.warning(f"YOLOv8 not available: {e}")
        
        try:
            self.detectors[DetectorType.SAM] = SAMDetector()
            logger.info("SAM detector available")
        except Exception as e:
            logger.warning(f"SAM not available: {e}")
        
        try:
            self.detectors[DetectorType.LUMINOTH] = LuminothDetector()
            logger.info("Luminoth compatibility detector available")
        except Exception as e:
            logger.warning(f"Luminoth compatibility not available: {e}")
    
    def detect(self, image: np.ndarray, 
              detector_type: Optional[DetectorType] = None) -> RecognitionResult:
        """
        Detect objects in image
        
        Args:
            image: Input image
            detector_type: Specific detector to use
        
        Returns:
            Recognition result
        """
        import time
        start_time = time.time()
        
        # Select detector
        detector_type = detector_type or self.active_detector
        
        if not detector_type or detector_type not in self.detectors:
            return RecognitionResult(
                detections=[],
                frame_shape=image.shape[:2],
                processing_time_ms=0,
                model_name="none",
                metadata={"error": "No detector available"}
            )
        
        detector = self.detectors[detector_type]
        
        # Run detection
        detections = detector.detect(image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return RecognitionResult(
            detections=detections,
            frame_shape=image.shape[:2],
            processing_time_ms=processing_time,
            model_name=detector_type.value,
            metadata={"num_detections": len(detections)}
        )
    
    def detect_and_segment(self, image: np.ndarray) -> RecognitionResult:
        """
        Detect objects and generate segmentation masks
        
        Args:
            image: Input image
        
        Returns:
            Recognition result with masks
        """
        # Use SAM if available, otherwise YOLOv8-seg
        if DetectorType.SAM in self.detectors:
            return self.detect(image, DetectorType.SAM)
        
        # Fallback to regular detection
        return self.detect(image)
    
    def track_objects(self, image: np.ndarray) -> RecognitionResult:
        """
        Track objects across frames
        
        Args:
            image: Current frame
        
        Returns:
            Recognition result with track IDs
        """
        import time
        start_time = time.time()
        
        # Use YOLOv8 tracking if available
        if DetectorType.YOLOV8 in self.detectors:
            detector = self.detectors[DetectorType.YOLOV8]
            detections = detector.detect_and_track(image)
            
            processing_time = (time.time() - start_time) * 1000
            
            return RecognitionResult(
                detections=detections,
                frame_shape=image.shape[:2],
                processing_time_ms=processing_time,
                model_name="yolov8-track",
                metadata={"tracking": True}
            )
        
        # Fallback to regular detection
        return self.detect(image)
    
    def visualize_detections(self, image: np.ndarray, 
                            result: RecognitionResult,
                            show_labels: bool = True,
                            show_confidence: bool = True) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Original image
            result: Recognition result
            show_labels: Show class labels
            show_confidence: Show confidence scores
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for detection in result.detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Draw bounding box
            color = self._get_color_for_class(detection.class_id)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw mask if available
            if detection.mask is not None:
                mask_color = (*color, 128)  # Semi-transparent
                mask_overlay = np.zeros_like(annotated)
                mask_overlay[detection.mask > 0] = color
                annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)
            
            # Draw label
            if show_labels or show_confidence:
                label_parts = []
                
                if show_labels:
                    label_parts.append(detection.class_name)
                
                if show_confidence:
                    label_parts.append(f"{detection.confidence:.2f}")
                
                if detection.track_id is not None:
                    label_parts.append(f"ID:{detection.track_id}")
                
                label = " ".join(label_parts)
                
                # Draw label background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                cv2.rectangle(annotated, 
                            (x1, y1 - text_height - 4),
                            (x1 + text_width, y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated, label,
                          (x1, y1 - 2),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID"""
        np.random.seed(class_id)
        color = np.random.randint(0, 255, 3)
        return tuple(map(int, color))