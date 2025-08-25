"""OCR (Optical Character Recognition) Engine for Nexus Framework"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2
import pytesseract
from PIL import Image
import easyocr
import structlog
import platform

logger = structlog.get_logger()


class OCREngine(Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"


class TextOrientation(Enum):
    """Text orientation"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    MIXED = "mixed"


@dataclass
class OCRResult:
    """OCR detection result"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    language: str
    metadata: Dict[str, Any] = None
    
    @property
    def x(self) -> int:
        return self.bbox[0]
    
    @property
    def y(self) -> int:
        return self.bbox[1]
    
    @property
    def width(self) -> int:
        return self.bbox[2]
    
    @property
    def height(self) -> int:
        return self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class TesseractOCR:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, tesseract_path: Optional[str] = None, 
                 tessdata_dir: Optional[str] = None):
        """
        Initialize Tesseract OCR
        
        Args:
            tesseract_path: Path to tesseract executable
            tessdata_dir: Path to tessdata directory
        """
        self.tesseract_path = tesseract_path or self._find_tesseract()
        self.tessdata_dir = tessdata_dir
        
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        
        if not self.is_available():
            logger.warning("Tesseract not found. OCR functionality will be limited.")
    
    def _find_tesseract(self) -> Optional[str]:
        """Find Tesseract executable"""
        # Check common locations
        if platform.system() == "Windows":
            # Check portable version in tools directory
            portable_paths = [
                Path.cwd() / "tools" / "tesseract" / "tesseract.exe",
                Path(__file__).parent.parent.parent / "tools" / "tesseract" / "tesseract.exe",
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe")
            ]
            
            for path in portable_paths:
                if path.exists():
                    return str(path)
        
        # Check if tesseract is in PATH
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd:
            return tesseract_cmd
        
        return None
    
    def is_available(self) -> bool:
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def detect_text(self, image: Union[np.ndarray, Image.Image], 
                   language: str = "eng",
                   config: str = "") -> List[OCRResult]:
        """
        Detect text in image
        
        Args:
            image: Input image
            language: Language code (e.g., 'eng', 'chi_sim')
            config: Additional Tesseract config
        
        Returns:
            List of OCR results
        """
        if not self.is_available():
            return []
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            # Get detailed data
            data = pytesseract.image_to_data(
                image, 
                lang=language,
                config=config,
                output_type=pytesseract.Output.DICT
            )
            
            results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:
                    confidence = data['conf'][i]
                    if confidence > 0:  # Filter out low confidence
                        x = data['left'][i]
                        y = data['top'][i]
                        w = data['width'][i]
                        h = data['height'][i]
                        
                        results.append(OCRResult(
                            text=text,
                            confidence=confidence / 100.0,
                            bbox=(x, y, w, h),
                            language=language,
                            metadata={'level': data['level'][i]}
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return []
    
    def get_text(self, image: Union[np.ndarray, Image.Image],
                language: str = "eng") -> str:
        """
        Get plain text from image
        
        Args:
            image: Input image
            language: Language code
        
        Returns:
            Extracted text
        """
        if not self.is_available():
            return ""
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        try:
            return pytesseract.image_to_string(image, lang=language)
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return ""


class EasyOCREngine:
    """EasyOCR engine wrapper"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = True):
        """
        Initialize EasyOCR
        
        Args:
            languages: List of language codes
            gpu: Use GPU acceleration
        """
        self.languages = languages or ['en']
        self.gpu = gpu and self._check_gpu()
        self.reader = None
        
        try:
            import easyocr
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu)
            logger.info(f"EasyOCR initialized with languages: {self.languages}")
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        return self.reader is not None
    
    def detect_text(self, image: Union[np.ndarray, Image.Image]) -> List[OCRResult]:
        """
        Detect text in image
        
        Args:
            image: Input image
        
        Returns:
            List of OCR results
        """
        if not self.is_available():
            return []
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        try:
            # Run detection
            results = self.reader.readtext(image)
            
            ocr_results = []
            for (bbox, text, confidence) in results:
                # Convert bbox format
                x_min = min(point[0] for point in bbox)
                y_min = min(point[1] for point in bbox)
                x_max = max(point[0] for point in bbox)
                y_max = max(point[1] for point in bbox)
                
                ocr_results.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                    language=self.languages[0],
                    metadata={'bbox_points': bbox}
                ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return []


class OCRService:
    """Unified OCR service with multiple engine support"""
    
    def __init__(self, default_engine: OCREngine = OCREngine.TESSERACT,
                 enable_preprocessing: bool = True):
        """
        Initialize OCR service
        
        Args:
            default_engine: Default OCR engine to use
            enable_preprocessing: Enable image preprocessing
        """
        self.default_engine = default_engine
        self.enable_preprocessing = enable_preprocessing
        
        # Initialize engines
        self.engines = {}
        self._init_tesseract()
        self._init_easyocr()
        
        logger.info(f"OCR Service initialized with default engine: {default_engine.value}")
    
    def _init_tesseract(self):
        """Initialize Tesseract engine"""
        self.engines[OCREngine.TESSERACT] = TesseractOCR()
    
    def _init_easyocr(self):
        """Initialize EasyOCR engine"""
        self.engines[OCREngine.EASYOCR] = EasyOCREngine()
    
    def detect_text(self, image: Union[np.ndarray, Image.Image, str],
                   engine: Optional[OCREngine] = None,
                   preprocess: Optional[bool] = None,
                   region: Optional[Tuple[int, int, int, int]] = None,
                   **kwargs) -> List[OCRResult]:
        """
        Detect text in image
        
        Args:
            image: Input image (array, PIL Image, or path)
            engine: OCR engine to use
            preprocess: Override preprocessing setting
            region: Region of interest (x, y, width, height)
            **kwargs: Additional engine-specific arguments
        
        Returns:
            List of OCR results
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract region if specified
        if region:
            if isinstance(image, np.ndarray):
                x, y, w, h = region
                image = image[y:y+h, x:x+w]
            elif isinstance(image, Image.Image):
                x, y, w, h = region
                image = image.crop((x, y, x+w, y+h))
        
        # Preprocess image
        if preprocess or (preprocess is None and self.enable_preprocessing):
            image = self.preprocess_image(image)
        
        # Select engine
        engine = engine or self.default_engine
        
        if engine not in self.engines:
            logger.error(f"Engine {engine} not available")
            return []
        
        # Run OCR
        ocr_engine = self.engines[engine]
        
        if not ocr_engine.is_available():
            # Fallback to another engine
            for fallback_engine in self.engines.values():
                if fallback_engine.is_available():
                    logger.warning(f"Falling back to available OCR engine")
                    return fallback_engine.detect_text(image, **kwargs)
            
            logger.error("No OCR engine available")
            return []
        
        return ocr_engine.detect_text(image, **kwargs)
    
    def get_text(self, image: Union[np.ndarray, Image.Image, str],
                engine: Optional[OCREngine] = None,
                **kwargs) -> str:
        """
        Get plain text from image
        
        Args:
            image: Input image
            engine: OCR engine to use
            **kwargs: Additional arguments
        
        Returns:
            Extracted text
        """
        results = self.detect_text(image, engine, **kwargs)
        return " ".join(r.text for r in results)
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Preprocess image for better OCR
        
        Args:
            image: Input image
        
        Returns:
            Preprocessed image
        """
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if colored
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh)
        
        # Deskew if needed
        deskewed = self._deskew(denoised)
        
        return deskewed
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew image"""
        # Find contours
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) < 4:
            return image
        
        # Calculate angle
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate image
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def find_text(self, image: Union[np.ndarray, Image.Image],
                 target_text: str,
                 case_sensitive: bool = False,
                 partial_match: bool = True) -> List[OCRResult]:
        """
        Find specific text in image
        
        Args:
            image: Input image
            target_text: Text to find
            case_sensitive: Case sensitive matching
            partial_match: Allow partial matches
        
        Returns:
            List of matching OCR results
        """
        results = self.detect_text(image)
        
        matches = []
        target = target_text if case_sensitive else target_text.lower()
        
        for result in results:
            text = result.text if case_sensitive else result.text.lower()
            
            if partial_match:
                if target in text:
                    matches.append(result)
            else:
                if text == target:
                    matches.append(result)
        
        return matches
    
    def read_number(self, image: Union[np.ndarray, Image.Image],
                   region: Optional[Tuple[int, int, int, int]] = None) -> Optional[float]:
        """
        Read number from image
        
        Args:
            image: Input image
            region: Region of interest
        
        Returns:
            Extracted number or None
        """
        # Use whitelist for digits
        text = self.get_text(image, region=region, config="--psm 7 -c tessedit_char_whitelist=0123456789.-")
        
        # Clean and parse
        text = text.strip().replace(" ", "")
        
        try:
            if "." in text:
                return float(text)
            else:
                return int(text)
        except ValueError:
            return None
    
    def is_engine_available(self, engine: OCREngine) -> bool:
        """Check if specific engine is available"""
        if engine in self.engines:
            return self.engines[engine].is_available()
        return False


# Global OCR service instance
_ocr_service = None


def get_ocr_service() -> OCRService:
    """Get global OCR service instance"""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service


# Convenience functions
def detect_text(image: Union[np.ndarray, Image.Image, str], **kwargs) -> List[OCRResult]:
    """Detect text in image"""
    return get_ocr_service().detect_text(image, **kwargs)


def get_text(image: Union[np.ndarray, Image.Image, str], **kwargs) -> str:
    """Get text from image"""
    return get_ocr_service().get_text(image, **kwargs)


def find_text(image: Union[np.ndarray, Image.Image, str], target: str, **kwargs) -> List[OCRResult]:
    """Find specific text in image"""
    return get_ocr_service().find_text(image, target, **kwargs)


def read_number(image: Union[np.ndarray, Image.Image, str], **kwargs) -> Optional[float]:
    """Read number from image"""
    return get_ocr_service().read_number(image, **kwargs)