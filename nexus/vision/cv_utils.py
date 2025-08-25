"""Computer vision utilities adapted from SerpentAI with modern improvements"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Union
from pathlib import Path
import structlog

logger = structlog.get_logger()


def extract_region_from_image(image: np.ndarray, 
                             region: Union[Tuple[int, int, int, int], 
                                         Tuple[int, int, int, int, str]]) -> np.ndarray:
    """
    Extract a region from an image using bounding box coordinates.
    
    Args:
        image: Source image as numpy array
        region: Bounding box as (y1, x1, y2, x2) or (y1, x1, y2, x2, name)
    
    Returns:
        Extracted region as numpy array
    """
    if len(region) == 5:
        # Named region
        y1, x1, y2, x2, _ = region
    else:
        y1, x1, y2, x2 = region
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    
    return image[y1:y2, x1:x2]


def isolate_sprite(image_paths: Union[List[str], List[Path], str, Path],
                  output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
    """
    Isolate a sprite by finding consistent pixels across multiple images.
    
    Args:
        image_paths: List of image paths or directory containing images
        output_path: Optional path to save the isolated sprite
    
    Returns:
        Isolated sprite with alpha channel
    """
    # Handle directory input
    if isinstance(image_paths, (str, Path)):
        path = Path(image_paths)
        if path.is_dir():
            image_paths = list(path.glob("*.png")) + list(path.glob("*.jpg"))
        else:
            image_paths = [path]
    
    result_image = None
    
    for image_path in image_paths:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        
        # Add alpha channel if not present
        if image.shape[2] == 3:
            alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
            image = np.concatenate((image, alpha), axis=2)
        
        if result_image is None:
            result_image = image.copy()
        else:
            # Compare pixels and make non-matching pixels transparent
            height, width = image.shape[:2]
            
            # Vectorized comparison for speed
            mask = np.all(image[:, :, :3] == result_image[:, :, :3], axis=2)
            result_image[:, :, 3] = np.where(mask, result_image[:, :, 3], 0)
    
    if output_path:
        cv2.imwrite(str(output_path), result_image)
    
    return result_image


def normalize(value: Union[float, np.ndarray], 
              source_min: float, source_max: float,
              target_min: float = 0, target_max: float = 1) -> Union[float, np.ndarray]:
    """
    Normalize a value from source range to target range.
    
    Args:
        value: Value or array to normalize
        source_min: Minimum of source range
        source_max: Maximum of source range
        target_min: Minimum of target range (default: 0)
        target_max: Maximum of target range (default: 1)
    
    Returns:
        Normalized value or array
    """
    # Avoid division by zero
    if source_max == source_min:
        return target_min
    
    return ((value - source_min) * (target_max - target_min) / 
            (source_max - source_min)) + target_min


def calculate_ssim(image1: np.ndarray, image2: np.ndarray,
                   multichannel: bool = True) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        image1: First image
        image2: Second image
        multichannel: Whether images have multiple channels
    
    Returns:
        SSIM score between 0 and 1
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure images have same shape
    if image1.shape != image2.shape:
        logger.warning(f"Images have different shapes: {image1.shape} vs {image2.shape}")
        return 0.0
    
    # Convert to grayscale if needed for single channel comparison
    if not multichannel and len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    return ssim(image1, image2, multichannel=multichannel)


def locate_string(image: np.ndarray, text: str,
                 font: int = cv2.FONT_HERSHEY_SIMPLEX,
                 font_size: float = 1.0) -> Optional[Tuple[int, int, int, int]]:
    """
    Locate text string in an image using template matching.
    
    Args:
        image: Image to search in
        text: Text to find
        font: OpenCV font type
        font_size: Font size for rendering
    
    Returns:
        Bounding box (x1, y1, x2, y2) if found, None otherwise
    """
    # Create template with text
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_size, 1)
    
    # Create blank template
    template = np.zeros((text_height + baseline + 10, text_width + 10, 3), dtype=np.uint8)
    
    # Draw text on template
    cv2.putText(template, text, (5, text_height + 5), font, font_size, (255, 255, 255), 1)
    
    # Convert to grayscale for matching
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Template matching
    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val > 0.8:  # Threshold for match
        x, y = max_loc
        return (x, y, x + text_width + 10, y + text_height + baseline + 10)
    
    return None


def multi_scale_template_match(image: np.ndarray, template: np.ndarray,
                               scales: List[float] = None,
                               threshold: float = 0.8) -> List[Tuple[int, int, int, int, float]]:
    """
    Perform template matching at multiple scales.
    
    Args:
        image: Image to search in
        template: Template to find
        scales: List of scales to try (default: [0.5, 0.75, 1.0, 1.25, 1.5])
        threshold: Confidence threshold
    
    Returns:
        List of (x1, y1, x2, y2, confidence) tuples
    """
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    matches = []
    
    for scale in scales:
        # Resize template
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        
        # Skip if template is larger than image
        if (scaled_template.shape[0] > image.shape[0] or 
            scaled_template.shape[1] > image.shape[1]):
            continue
        
        # Match template
        result = cv2.matchTemplate(image, scaled_template, cv2.TM_CCOEFF_NORMED)
        
        # Find all matches above threshold
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt
            w, h = scaled_template.shape[1], scaled_template.shape[0]
            confidence = float(result[y, x])
            matches.append((x, y, x + w, y + h, confidence))
    
    # Non-maximum suppression
    return non_max_suppression(matches)


def non_max_suppression(boxes: List[Tuple[int, int, int, int, float]],
                        overlap_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
    """
    Apply non-maximum suppression to remove overlapping bounding boxes.
    
    Args:
        boxes: List of (x1, y1, x2, y2, confidence) tuples
        overlap_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of boxes
    """
    if not boxes:
        return []
    
    # Sort by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep = []
    while boxes:
        # Take highest confidence box
        current = boxes.pop(0)
        keep.append(current)
        
        # Filter remaining boxes
        boxes = [box for box in boxes 
                if calculate_iou(current[:4], box[:4]) < overlap_threshold]
    
    return keep


def calculate_iou(box1: Tuple[int, int, int, int],
                  box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)
    
    Returns:
        IoU score between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max < x_min or y_max < y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_color_mask(image: np.ndarray,
                    lower_bound: Union[Tuple[int, int, int], np.ndarray],
                    upper_bound: Union[Tuple[int, int, int], np.ndarray],
                    color_space: str = "BGR") -> np.ndarray:
    """
    Apply color mask to isolate specific colors.
    
    Args:
        image: Input image
        lower_bound: Lower color bound
        upper_bound: Upper color bound
        color_space: Color space ("BGR", "HSV", "RGB")
    
    Returns:
        Masked image
    """
    # Convert color space if needed
    if color_space == "HSV":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "RGB":
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        converted = image
    
    # Create mask
    mask = cv2.inRange(converted, np.array(lower_bound), np.array(upper_bound))
    
    # Apply mask
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result


def detect_edges(image: np.ndarray,
                low_threshold: int = 50,
                high_threshold: int = 150) -> np.ndarray:
    """
    Detect edges using Canny edge detection.
    
    Args:
        image: Input image
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection
    
    Returns:
        Edge map
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    return edges


def find_contours(image: np.ndarray,
                 min_area: Optional[int] = None,
                 max_area: Optional[int] = None) -> List[np.ndarray]:
    """
    Find contours in an image with optional area filtering.
    
    Args:
        image: Binary or grayscale image
        min_area: Minimum contour area
        max_area: Maximum contour area
    
    Returns:
        List of contours
    """
    # Ensure binary image
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    filtered = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if min_area and area < min_area:
            continue
        if max_area and area > max_area:
            continue
        
        filtered.append(contour)
    
    return filtered


def histogram_matching(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of source image to reference image.
    
    Args:
        source: Source image to adjust
        reference: Reference image with target histogram
    
    Returns:
        Adjusted image
    """
    # Calculate histograms
    source_hist = cv2.calcHist([source], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    ref_hist = cv2.calcHist([reference], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    source_hist = cv2.normalize(source_hist, source_hist).flatten()
    ref_hist = cv2.normalize(ref_hist, ref_hist).flatten()
    
    # Calculate cumulative distribution functions
    source_cdf = source_hist.cumsum()
    ref_cdf = ref_hist.cumsum()
    
    # Create lookup table
    lookup = np.zeros(256)
    for i in range(256):
        diff = np.abs(source_cdf[i] - ref_cdf)
        lookup[i] = np.argmin(diff)
    
    # Apply lookup table
    result = cv2.LUT(source, lookup.astype(np.uint8))
    
    return result