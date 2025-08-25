"""OCR utilities adapted from SerpentAI with modern OCR engines"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import structlog

logger = structlog.get_logger()

# Try to import various OCR libraries
OCR_AVAILABLE = {
    "easyocr": False,
    "rapidocr": False,
    "tesseract": False
}

try:
    import easyocr
    OCR_AVAILABLE["easyocr"] = True
except ImportError:
    logger.debug("EasyOCR not available")

try:
    from rapidocr_onnxruntime import RapidOCR
    OCR_AVAILABLE["rapidocr"] = True
except ImportError:
    logger.debug("RapidOCR not available")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE["tesseract"] = True
except ImportError:
    logger.debug("Tesseract/pytesseract not available")


class OCREngine:
    """Unified OCR engine with multiple backends"""
    
    def __init__(self, backend: str = "auto"):
        """
        Initialize OCR engine.
        
        Args:
            backend: OCR backend to use ("easyocr", "rapidocr", "tesseract", "auto")
        """
        self.backend = self._select_backend(backend)
        self.engine = None
        self._initialize_engine()
    
    def _select_backend(self, backend: str) -> str:
        """Select the best available OCR backend"""
        if backend == "auto":
            # Priority order: EasyOCR > RapidOCR > Tesseract
            if OCR_AVAILABLE["easyocr"]:
                return "easyocr"
            elif OCR_AVAILABLE["rapidocr"]:
                return "rapidocr"
            elif OCR_AVAILABLE["tesseract"]:
                return "tesseract"
            else:
                logger.warning("No OCR backend available")
                return "none"
        else:
            if not OCR_AVAILABLE.get(backend, False):
                logger.warning(f"Requested backend '{backend}' not available")
                return "none"
            return backend
    
    def _initialize_engine(self):
        """Initialize the selected OCR engine"""
        if self.backend == "easyocr":
            self.engine = easyocr.Reader(['en'], gpu=True)
            logger.info("Initialized EasyOCR engine")
        elif self.backend == "rapidocr":
            self.engine = RapidOCR()
            logger.info("Initialized RapidOCR engine")
        elif self.backend == "tesseract":
            # Check if tesseract is available
            try:
                pytesseract.get_tesseract_version()
                self.engine = "tesseract"
                logger.info("Initialized Tesseract engine")
            except:
                logger.error("Tesseract not found in PATH")
                self.backend = "none"
        else:
            logger.warning("No OCR engine initialized")
    
    def read_text(self, image: np.ndarray, **kwargs) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """
        Read text from image.
        
        Args:
            image: Input image
            **kwargs: Backend-specific parameters
        
        Returns:
            List of (text, bbox, confidence) tuples
        """
        if self.backend == "none":
            return []
        
        results = []
        
        if self.backend == "easyocr":
            detections = self.engine.readtext(image, **kwargs)
            for detection in detections:
                bbox_points, text, confidence = detection
                # Convert points to x1, y1, x2, y2
                x1 = int(min(p[0] for p in bbox_points))
                y1 = int(min(p[1] for p in bbox_points))
                x2 = int(max(p[0] for p in bbox_points))
                y2 = int(max(p[1] for p in bbox_points))
                results.append((text, (x1, y1, x2, y2), confidence))
        
        elif self.backend == "rapidocr":
            detection_result, _ = self.engine(image)
            if detection_result:
                for line in detection_result:
                    bbox, text, confidence = line
                    x1 = int(min(p[0] for p in bbox))
                    y1 = int(min(p[1] for p in bbox))
                    x2 = int(max(p[0] for p in bbox))
                    y2 = int(max(p[1] for p in bbox))
                    results.append((text, (x1, y1, x2, y2), confidence))
        
        elif self.backend == "tesseract":
            # Convert to PIL for tesseract
            pil_image = Image.fromarray(image)
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    confidence = data['conf'][i] / 100.0
                    results.append((data['text'][i], (x, y, x + w, y + h), confidence))
        
        return results


def extract_text_regions(image: np.ndarray,
                         gradient_size: int = 3,
                         closing_size: int = 10,
                         minimum_area: int = 100,
                         minimum_aspect_ratio: float = 2.0) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Extract potential text regions from image using morphological operations.
    
    Args:
        image: Input image
        gradient_size: Size of gradient filter
        closing_size: Size of morphological closing
        minimum_area: Minimum area for text region
        minimum_aspect_ratio: Minimum width/height ratio
    
    Returns:
        Tuple of (region images, region bounding boxes)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gradient_size, gradient_size))
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
    # Threshold using Otsu's method
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to connect text regions
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_size, 1))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, closing_kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    region_images = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        # Filter regions
        if area >= minimum_area and aspect_ratio >= minimum_aspect_ratio and h >= 8:
            text_regions.append((x, y, x + w, y + h))
            region_images.append(gray[y:y+h, x:x+w])
    
    return region_images, text_regions


def preprocess_for_ocr(image: np.ndarray,
                       scale: int = 2,
                       denoise: bool = True,
                       binarize: bool = True,
                       invert_if_needed: bool = True) -> np.ndarray:
    """
    Preprocess image for better OCR results.
    
    Args:
        image: Input image
        scale: Scaling factor
        denoise: Apply denoising
        binarize: Convert to binary image
        invert_if_needed: Invert if more black than white pixels
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        processed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        processed = image.copy()
    
    # Scale up for better OCR
    if scale > 1:
        processed = cv2.resize(processed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    if denoise:
        processed = cv2.fastNlMeansDenoising(processed)
    
    # Binarize
    if binarize:
        # Adaptive threshold for better results with varying lighting
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if needed (OCR typically expects dark text on light background)
        if invert_if_needed:
            black_pixels = np.sum(processed == 0)
            white_pixels = np.sum(processed == 255)
            
            if black_pixels > white_pixels:
                processed = cv2.bitwise_not(processed)
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    return processed


def locate_text(query_string: str, image: np.ndarray,
               ocr_engine: Optional[OCREngine] = None,
               fuzziness: int = 2,
               preprocess: bool = True) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate text string in image.
    
    Args:
        query_string: Text to search for
        image: Image to search in
        ocr_engine: OCR engine to use (creates new if None)
        fuzziness: Maximum edit distance for fuzzy matching
        preprocess: Whether to preprocess image
    
    Returns:
        Bounding box (x1, y1, x2, y2) if found, None otherwise
    """
    if ocr_engine is None:
        ocr_engine = OCREngine()
    
    # Preprocess if requested
    if preprocess:
        processed_image = preprocess_for_ocr(image)
    else:
        processed_image = image
    
    # Read text
    detections = ocr_engine.read_text(processed_image)
    
    # Search for exact match first
    for text, bbox, confidence in detections:
        if query_string.lower() in text.lower():
            return bbox
    
    # Fuzzy matching if no exact match
    if fuzziness > 0:
        from difflib import SequenceMatcher
        
        best_match = None
        best_score = 0
        
        for text, bbox, confidence in detections:
            # Calculate similarity
            score = SequenceMatcher(None, query_string.lower(), text.lower()).ratio()
            
            if score > best_score and score > (1.0 - fuzziness / len(query_string)):
                best_score = score
                best_match = bbox
        
        if best_match:
            return best_match
    
    return None


def read_text_from_region(image: np.ndarray,
                         region: Tuple[int, int, int, int],
                         ocr_engine: Optional[OCREngine] = None,
                         preprocess: bool = True) -> str:
    """
    Read text from specific region of image.
    
    Args:
        image: Source image
        region: Region to read from (x1, y1, x2, y2)
        ocr_engine: OCR engine to use
        preprocess: Whether to preprocess region
    
    Returns:
        Extracted text
    """
    if ocr_engine is None:
        ocr_engine = OCREngine()
    
    # Extract region
    x1, y1, x2, y2 = region
    region_image = image[y1:y2, x1:x2]
    
    # Preprocess if requested
    if preprocess:
        region_image = preprocess_for_ocr(region_image)
    
    # Read text
    detections = ocr_engine.read_text(region_image)
    
    # Combine all detected text
    texts = [text for text, _, _ in detections]
    return " ".join(texts)


def create_text_mask(image: np.ndarray,
                    ocr_engine: Optional[OCREngine] = None) -> np.ndarray:
    """
    Create a mask highlighting text regions.
    
    Args:
        image: Input image
        ocr_engine: OCR engine to use
    
    Returns:
        Binary mask with text regions marked as 255
    """
    if ocr_engine is None:
        ocr_engine = OCREngine()
    
    # Create empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Detect text
    detections = ocr_engine.read_text(image)
    
    # Fill text regions
    for text, bbox, confidence in detections:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    
    return mask


def enhance_text_regions(image: np.ndarray) -> np.ndarray:
    """
    Enhance text regions in image for better visibility.
    
    Args:
        image: Input image
    
    Returns:
        Enhanced image
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        is_color = True
    else:
        gray = image.copy()
        is_color = False
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to color if needed
    if is_color:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced


def extract_text_metadata(image: np.ndarray,
                         ocr_engine: Optional[OCREngine] = None) -> Dict[str, Any]:
    """
    Extract comprehensive text metadata from image.
    
    Args:
        image: Input image
        ocr_engine: OCR engine to use
    
    Returns:
        Dictionary with text metadata
    """
    if ocr_engine is None:
        ocr_engine = OCREngine()
    
    detections = ocr_engine.read_text(image)
    
    metadata = {
        "total_text_regions": len(detections),
        "texts": [],
        "average_confidence": 0,
        "text_coverage": 0  # Percentage of image covered by text
    }
    
    if detections:
        total_confidence = 0
        text_area = 0
        image_area = image.shape[0] * image.shape[1]
        
        for text, bbox, confidence in detections:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            
            metadata["texts"].append({
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "area": area
            })
            
            total_confidence += confidence
            text_area += area
        
        metadata["average_confidence"] = total_confidence / len(detections)
        metadata["text_coverage"] = (text_area / image_area) * 100
    
    return metadata