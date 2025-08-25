import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import asyncio
import cv2
import structlog

from nexus.vision.detector import ObjectDetector, Detection
from nexus.vision.ocr import OCREngine, TextDetection
from nexus.vision.template_matcher import TemplateMatcher, TemplateMatch

logger = structlog.get_logger()


@dataclass
class VisionResult:
    """Combined result from vision pipeline"""
    frame_id: int
    objects: List[Detection]
    text: List[TextDetection]
    templates: List[TemplateMatch]
    metadata: Dict[str, Any]
    processing_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "text": [txt.to_dict() for txt in self.text],
            "templates": [tpl.to_dict() for tpl in self.templates],
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms
        }
    
    def get_objects_by_class(self, class_name: str) -> List[Detection]:
        return [obj for obj in self.objects if obj.class_name == class_name]
    
    def get_text_containing(self, substring: str) -> List[TextDetection]:
        return [txt for txt in self.text if substring.lower() in txt.text.lower()]
    
    def get_template_matches(self, template_name: str) -> List[TemplateMatch]:
        return [tpl for tpl in self.templates if tpl.template_name == template_name]


class VisionPipeline:
    """Complete vision processing pipeline"""
    
    def __init__(self,
                 enable_detection: bool = True,
                 enable_ocr: bool = True,
                 enable_templates: bool = True,
                 detection_model: str = "yolov8n.pt",
                 ocr_engine: str = "easyocr",
                 templates_dir: Optional[str] = None):
        
        self.enable_detection = enable_detection
        self.enable_ocr = enable_ocr
        self.enable_templates = enable_templates
        
        # Initialize components
        self.detector = ObjectDetector(detection_model) if enable_detection else None
        self.ocr = OCREngine(ocr_engine) if enable_ocr else None
        self.template_matcher = TemplateMatcher(templates_dir) if enable_templates else None
        
        # Processing options
        self.preprocessors: List[Callable] = []
        self.postprocessors: List[Callable] = []
        
        self._initialized = False
        self.frame_count = 0
    
    async def initialize(self) -> None:
        """Initialize all vision components"""
        tasks = []
        
        if self.detector:
            tasks.append(self.detector.initialize())
        if self.ocr:
            tasks.append(self.ocr.initialize())
        if self.template_matcher:
            tasks.append(self.template_matcher.initialize())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        self._initialized = True
        logger.info("Vision pipeline initialized")
    
    async def process(self,
                     frame: np.ndarray,
                     detect_objects: bool = True,
                     detect_text: bool = True,
                     match_templates: Optional[List[str]] = None,
                     regions: Optional[Dict[str, Tuple[int, int, int, int]]] = None) -> VisionResult:
        """Process a frame through the vision pipeline"""
        
        if not self._initialized:
            await self.initialize()
        
        import time
        start_time = time.perf_counter()
        
        self.frame_count += 1
        
        # Apply preprocessors
        processed_frame = frame
        for preprocessor in self.preprocessors:
            processed_frame = preprocessor(processed_frame)
        
        # Parallel processing
        tasks = []
        task_names = []
        
        # Object detection
        if detect_objects and self.detector:
            if regions and "objects" in regions:
                region = regions["objects"]
                roi = frame[region[1]:region[3], region[0]:region[2]]
                tasks.append(self.detector.detect(roi))
            else:
                tasks.append(self.detector.detect(processed_frame))
            task_names.append("objects")
        
        # OCR
        if detect_text and self.ocr:
            if regions and "text" in regions:
                tasks.append(self.ocr.detect_text(processed_frame, regions["text"]))
            else:
                tasks.append(self.ocr.detect_text(processed_frame))
            task_names.append("text")
        
        # Template matching
        if match_templates and self.template_matcher:
            for template_name in match_templates:
                tasks.append(self.template_matcher.find_template(processed_frame, template_name))
                task_names.append(f"template_{template_name}")
        
        # Execute all tasks in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
        
        # Parse results
        objects = []
        text = []
        templates = []
        
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Vision task {name} failed: {result}")
                continue
            
            if name == "objects" and result:
                objects = result
                # Adjust coordinates if region was used
                if regions and "objects" in regions:
                    offset_x, offset_y = regions["objects"][0], regions["objects"][1]
                    for obj in objects:
                        x1, y1, x2, y2 = obj.bbox
                        obj.bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            
            elif name == "text" and result:
                text = result
            
            elif name.startswith("template_") and result:
                templates.append(result)
        
        # Apply postprocessors
        for postprocessor in self.postprocessors:
            objects, text, templates = postprocessor(objects, text, templates)
        
        processing_time = (time.perf_counter() - start_time) * 1000
        
        return VisionResult(
            frame_id=self.frame_count,
            objects=objects,
            text=text,
            templates=templates,
            metadata={
                "shape": frame.shape,
                "detection_enabled": detect_objects,
                "ocr_enabled": detect_text,
                "templates_searched": match_templates or []
            },
            processing_time_ms=processing_time
        )
    
    def add_preprocessor(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        """Add a preprocessing function"""
        self.preprocessors.append(func)
    
    def add_postprocessor(self, func: Callable) -> None:
        """Add a postprocessing function"""
        self.postprocessors.append(func)
    
    def create_annotated_frame(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """Create an annotated frame with all detections"""
        annotated = frame.copy()
        
        # Draw object detections
        if self.detector and result.objects:
            annotated = self.detector.draw_detections(annotated, result.objects)
        
        # Draw text detections
        if self.ocr and result.text:
            annotated = self.ocr.draw_text_detections(annotated, result.text)
        
        # Draw template matches
        if self.template_matcher and result.templates:
            annotated = self.template_matcher.draw_matches(annotated, result.templates)
        
        # Add info overlay
        info_text = [
            f"Frame: {result.frame_id}",
            f"Objects: {len(result.objects)}",
            f"Text: {len(result.text)}",
            f"Templates: {len(result.templates)}",
            f"Time: {result.processing_time_ms:.1f}ms"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return annotated
    
    async def track_objects(self, 
                           frames: List[np.ndarray],
                           class_filter: Optional[List[str]] = None) -> List[List[Detection]]:
        """Track objects across multiple frames"""
        all_detections = []
        
        for frame in frames:
            result = await self.process(frame, detect_objects=True, detect_text=False)
            
            if class_filter:
                filtered = [d for d in result.objects if d.class_name in class_filter]
                all_detections.append(filtered)
            else:
                all_detections.append(result.objects)
        
        # Simple tracking by proximity
        # More sophisticated tracking algorithms can be implemented here
        
        return all_detections
    
    async def find_ui_element(self,
                             frame: np.ndarray,
                             element_type: str,
                             identifier: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Find a specific UI element"""
        
        # Process frame
        result = await self.process(
            frame,
            detect_objects=True,
            detect_text=True,
            match_templates=[element_type] if self.template_matcher and element_type in self.template_matcher.template_cache else None
        )
        
        # Search in different detection types
        if element_type == "button" and identifier:
            # Look for text containing identifier
            for text_det in result.text:
                if identifier.lower() in text_det.text.lower():
                    return {
                        "type": "button",
                        "text": text_det.text,
                        "location": text_det.center,
                        "bbox": text_det.bbox
                    }
        
        elif element_type == "icon":
            # Look for template match
            matches = result.get_template_matches(identifier or element_type)
            if matches:
                best_match = max(matches, key=lambda m: m.confidence)
                return {
                    "type": "icon",
                    "name": best_match.template_name,
                    "location": best_match.center,
                    "bbox": best_match.bbox,
                    "confidence": best_match.confidence
                }
        
        # Generic object detection
        objects = result.get_objects_by_class(element_type)
        if objects:
            best_obj = max(objects, key=lambda o: o.confidence)
            return {
                "type": element_type,
                "class": best_obj.class_name,
                "location": best_obj.center,
                "bbox": best_obj.bbox,
                "confidence": best_obj.confidence
            }
        
        return None