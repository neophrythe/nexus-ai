import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import asyncio
import structlog

logger = structlog.get_logger()


@dataclass
class TemplateMatch:
    """Result of template matching"""
    template_name: str
    confidence: float
    location: Tuple[int, int]  # Top-left corner
    size: Tuple[int, int]  # Width, height
    center: Tuple[int, int]
    method: str
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        x, y = self.location
        w, h = self.size
        return (x, y, x + w, y + h)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "template": self.template_name,
            "confidence": self.confidence,
            "location": self.location,
            "size": self.size,
            "center": self.center,
            "bbox": self.bbox,
            "method": self.method
        }


class TemplateMatcher:
    """Template matching for finding UI elements and sprites"""
    
    METHODS = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED
    }
    
    def __init__(self, templates_dir: Optional[Path] = None, cache_templates: bool = True):
        self.templates_dir = Path(templates_dir) if templates_dir else Path("templates")
        self.cache_templates = cache_templates
        self.template_cache: Dict[str, np.ndarray] = {}
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize template matcher"""
        if self.templates_dir.exists():
            await self.load_templates()
        self._initialized = True
        logger.info(f"Template matcher initialized with {len(self.template_cache)} templates")
    
    async def load_templates(self) -> None:
        """Load all templates from directory"""
        template_files = list(self.templates_dir.glob("*.png")) + \
                        list(self.templates_dir.glob("*.jpg"))
        
        for template_file in template_files:
            template_name = template_file.stem
            template_img = cv2.imread(str(template_file))
            
            if template_img is not None:
                self.template_cache[template_name] = template_img
                logger.debug(f"Loaded template: {template_name}")
    
    def add_template(self, name: str, template: np.ndarray) -> None:
        """Add a template to the cache"""
        self.template_cache[name] = template
    
    async def find_template(self,
                           frame: np.ndarray,
                           template_name: str,
                           threshold: float = 0.8,
                           method: str = "TM_CCOEFF_NORMED",
                           scale_range: Optional[Tuple[float, float]] = None) -> Optional[TemplateMatch]:
        """Find a single template in frame"""
        if not self._initialized:
            await self.initialize()
        
        if template_name not in self.template_cache:
            logger.warning(f"Template {template_name} not found")
            return None
        
        template = self.template_cache[template_name]
        
        if scale_range:
            # Multi-scale template matching
            return await self._find_template_multiscale(
                frame, template, template_name, threshold, method, scale_range
            )
        else:
            # Single scale matching
            return await self._find_template_single(
                frame, template, template_name, threshold, method
            )
    
    async def _find_template_single(self,
                                   frame: np.ndarray,
                                   template: np.ndarray,
                                   template_name: str,
                                   threshold: float,
                                   method: str) -> Optional[TemplateMatch]:
        """Find template at single scale"""
        try:
            # Get method
            cv_method = self.METHODS.get(method, cv2.TM_CCOEFF_NORMED)
            
            # Perform template matching
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                cv2.matchTemplate,
                frame, template, cv_method
            )
            
            # Find best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Get match value and location based on method
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_val = 1 - min_val if cv_method == cv2.TM_SQDIFF_NORMED else -min_val
                match_loc = min_loc
            else:
                match_val = max_val
                match_loc = max_loc
            
            # Check threshold
            if match_val < threshold:
                return None
            
            h, w = template.shape[:2]
            x, y = match_loc
            
            return TemplateMatch(
                template_name=template_name,
                confidence=float(match_val),
                location=(x, y),
                size=(w, h),
                center=(x + w // 2, y + h // 2),
                method=method
            )
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return None
    
    async def _find_template_multiscale(self,
                                       frame: np.ndarray,
                                       template: np.ndarray,
                                       template_name: str,
                                       threshold: float,
                                       method: str,
                                       scale_range: Tuple[float, float]) -> Optional[TemplateMatch]:
        """Find template at multiple scales"""
        best_match = None
        best_confidence = 0
        
        min_scale, max_scale = scale_range
        scales = np.linspace(min_scale, max_scale, 20)
        
        for scale in scales:
            # Resize template
            scaled_template = cv2.resize(
                template,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            )
            
            # Skip if template is larger than frame
            if scaled_template.shape[0] > frame.shape[0] or scaled_template.shape[1] > frame.shape[1]:
                continue
            
            # Find match at this scale
            match = await self._find_template_single(
                frame, scaled_template, template_name, threshold, method
            )
            
            if match and match.confidence > best_confidence:
                best_match = match
                best_confidence = match.confidence
        
        return best_match
    
    async def find_all_templates(self,
                                frame: np.ndarray,
                                template_name: str,
                                threshold: float = 0.8,
                                method: str = "TM_CCOEFF_NORMED",
                                max_matches: int = 10) -> List[TemplateMatch]:
        """Find all occurrences of a template"""
        if not self._initialized:
            await self.initialize()
        
        if template_name not in self.template_cache:
            logger.warning(f"Template {template_name} not found")
            return []
        
        template = self.template_cache[template_name]
        cv_method = self.METHODS.get(method, cv2.TM_CCOEFF_NORMED)
        
        try:
            # Perform template matching
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                cv2.matchTemplate,
                frame, template, cv_method
            )
            
            h, w = template.shape[:2]
            matches = []
            
            # Find all locations above threshold
            if cv_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                locations = np.where(result <= (1 - threshold))
            else:
                locations = np.where(result >= threshold)
            
            # Convert to list of matches
            for pt in zip(*locations[::-1]):
                x, y = pt
                confidence = float(result[y, x])
                
                if cv_method == cv2.TM_SQDIFF_NORMED:
                    confidence = 1 - confidence
                
                match = TemplateMatch(
                    template_name=template_name,
                    confidence=confidence,
                    location=(x, y),
                    size=(w, h),
                    center=(x + w // 2, y + h // 2),
                    method=method
                )
                matches.append(match)
            
            # Sort by confidence and limit
            matches.sort(key=lambda m: m.confidence, reverse=True)
            
            # Non-maximum suppression to remove overlapping matches
            filtered_matches = []
            for match in matches[:max_matches]:
                overlap = False
                for existing in filtered_matches:
                    if self._is_overlapping(match.bbox, existing.bbox, 0.5):
                        overlap = True
                        break
                if not overlap:
                    filtered_matches.append(match)
            
            return filtered_matches
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return []
    
    def _is_overlapping(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int],
                       threshold: float = 0.5) -> bool:
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return False
        
        intersection = (x_max - x_min) * (y_max - y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        
        # IoU calculation
        iou = intersection / (area1 + area2 - intersection)
        
        return iou > threshold
    
    def draw_matches(self, frame: np.ndarray, matches: List[TemplateMatch]) -> np.ndarray:
        """Draw template matches on frame"""
        frame_copy = frame.copy()
        
        for match in matches:
            x1, y1, x2, y2 = match.bbox
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(frame_copy, match.center, 3, (0, 0, 255), -1)
            
            # Draw label
            label = f"{match.template_name}: {match.confidence:.2f}"
            cv2.putText(frame_copy, label,
                       (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
        
        return frame_copy
    
    async def wait_for_template(self,
                               capture_func: callable,
                               template_name: str,
                               timeout: float = 30,
                               check_interval: float = 0.5,
                               threshold: float = 0.8) -> Optional[TemplateMatch]:
        """Wait for a template to appear"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            frame = await capture_func()
            
            if frame is not None:
                match = await self.find_template(frame, template_name, threshold)
                if match:
                    return match
            
            await asyncio.sleep(check_interval)
        
        return None