"""Advanced CNN Context Classifiers for Nexus Framework"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import structlog
from dataclasses import dataclass
import json
import pickle

logger = structlog.get_logger()


@dataclass
class ClassificationResult:
    """Result from context classification"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: np.ndarray
    features: Optional[np.ndarray] = None


class BaseContextClassifier:
    """Base class for context classifiers"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int],
                 device: Optional[str] = None):
        """
        Initialize base classifier
        
        Args:
            num_classes: Number of context classes
            input_shape: Input image shape (C, H, W)
            device: Device to use (cuda/cpu)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.class_names = []
        
        self._setup_transform()
    
    def _setup_transform(self):
        """Setup image preprocessing transform"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            Preprocessed tensor
        """
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # Convert BGR to RGB if needed
        if image.shape[-1] == 3:
            image = image[..., ::-1]
        
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device)
    
    def predict(self, image: np.ndarray) -> ClassificationResult:
        """
        Predict context class for image
        
        Args:
            image: Input image
        
        Returns:
            Classification result
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = self.preprocess(image)
            outputs = self.model(input_tensor)
            
            if isinstance(outputs, tuple):
                logits, features = outputs
            else:
                logits = outputs
                features = None
            
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
            class_id = int(np.argmax(probabilities))
            confidence = float(probabilities[class_id])
            
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            
            return ClassificationResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                probabilities=probabilities,
                features=features.cpu().numpy()[0] if features is not None else None
            )
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Single training step
        
        Args:
            images: Batch of images
            labels: Batch of labels
        
        Returns:
            Loss value
        """
        self.model.train()
        
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        loss = F.cross_entropy(logits, labels)
        return loss.item()
    
    def save(self, path: str):
        """Save model to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.model and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        logger.info(f"Model loaded from {path}")


class CNNInceptionV3ContextClassifier(BaseContextClassifier):
    """InceptionV3-based context classifier"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (3, 299, 299),
                 device: Optional[str] = None, pretrained: bool = True):
        """
        Initialize InceptionV3 classifier
        
        Args:
            num_classes: Number of context classes
            input_shape: Input shape (must be 299x299 for Inception)
            device: Device to use
            pretrained: Use pretrained weights
        """
        super().__init__(num_classes, (3, 299, 299), device)
        
        # Load InceptionV3
        self.model = models.inception_v3(pretrained=pretrained, aux_logits=True)
        
        # Modify final layer for our classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Also modify auxiliary classifier
        num_aux_features = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_aux_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized InceptionV3 classifier with {num_classes} classes")
    
    def forward_with_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature extraction
        
        Args:
            x: Input tensor
        
        Returns:
            Logits and features
        """
        self.model.eval()
        
        # Get features from second-to-last layer
        features = []
        
        def hook(module, input, output):
            features.append(output)
        
        handle = self.model.avgpool.register_forward_hook(hook)
        
        with torch.no_grad():
            outputs = self.model(x)
            
        handle.remove()
        
        if self.model.training and self.model.aux_logits:
            logits = outputs.logits
        else:
            logits = outputs
        
        feature_vector = features[0].squeeze() if features else None
        
        return logits, feature_vector


class CNNXceptionContextClassifier(BaseContextClassifier):
    """Xception-based context classifier"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (3, 299, 299),
                 device: Optional[str] = None):
        """
        Initialize Xception classifier
        
        Args:
            num_classes: Number of context classes
            input_shape: Input shape
            device: Device to use
        """
        super().__init__(num_classes, (3, 299, 299), device)
        
        # Create Xception model
        self.model = XceptionModel(num_classes)
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized Xception classifier with {num_classes} classes")


class XceptionModel(nn.Module):
    """Xception architecture implementation"""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Separable convolutions
        self.block1 = SeparableConvBlock(64, 128, 2)
        self.block2 = SeparableConvBlock(128, 256, 2)
        self.block3 = SeparableConvBlock(256, 728, 2)
        
        # Middle flow (8 times)
        self.middle_blocks = nn.ModuleList([
            SeparableConvBlock(728, 728, 1) for _ in range(8)
        ])
        
        # Exit flow
        self.block4 = SeparableConvBlock(728, 1024, 2)
        
        self.conv3 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        
        self.conv4 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Entry flow
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        for block in self.middle_blocks:
            x = block(x)
        
        # Exit flow
        x = self.block4(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class SeparableConv2d(nn.Module):
    """Separable convolution block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SeparableConvBlock(nn.Module):
    """Separable convolution block with residual connection"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = SeparableConv2d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = SeparableConv2d(out_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = x + residual
        return F.relu(x)


class EfficientNetContextClassifier(BaseContextClassifier):
    """Modern EfficientNet-based classifier"""
    
    def __init__(self, num_classes: int, model_name: str = "efficientnet_b0",
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 device: Optional[str] = None, pretrained: bool = True):
        """
        Initialize EfficientNet classifier
        
        Args:
            num_classes: Number of classes
            model_name: EfficientNet variant (b0-b7)
            input_shape: Input shape
            device: Device to use
            pretrained: Use pretrained weights
        """
        super().__init__(num_classes, input_shape, device)
        
        # Load EfficientNet
        if model_name == "efficientnet_b0":
            self.model = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == "efficientnet_b1":
            self.model = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == "efficientnet_b2":
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == "efficientnet_b3":
            self.model = models.efficientnet_b3(pretrained=pretrained)
        elif model_name == "efficientnet_b4":
            self.model = models.efficientnet_b4(pretrained=pretrained)
        else:
            self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Modify classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized {model_name} classifier with {num_classes} classes")


class VisionTransformerContextClassifier(BaseContextClassifier):
    """Vision Transformer (ViT) based classifier - Modern approach"""
    
    def __init__(self, num_classes: int, model_name: str = "vit_b_16",
                 input_shape: Tuple[int, int, int] = (3, 224, 224),
                 device: Optional[str] = None, pretrained: bool = True):
        """
        Initialize Vision Transformer classifier
        
        Args:
            num_classes: Number of classes
            model_name: ViT variant
            input_shape: Input shape
            device: Device to use
            pretrained: Use pretrained weights
        """
        super().__init__(num_classes, input_shape, device)
        
        # Load Vision Transformer
        if model_name == "vit_b_16":
            self.model = models.vit_b_16(pretrained=pretrained)
        elif model_name == "vit_b_32":
            self.model = models.vit_b_32(pretrained=pretrained)
        elif model_name == "vit_l_16":
            self.model = models.vit_l_16(pretrained=pretrained)
        else:
            self.model = models.vit_b_16(pretrained=pretrained)
        
        # Modify head
        num_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(num_features, num_classes)
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Initialized {model_name} classifier with {num_classes} classes")


class ContextClassificationManager:
    """Manage multiple context classifiers"""
    
    def __init__(self):
        """Initialize manager"""
        self.classifiers: Dict[str, BaseContextClassifier] = {}
        self.active_classifier: Optional[str] = None
    
    def register_classifier(self, name: str, classifier: BaseContextClassifier):
        """Register a classifier"""
        self.classifiers[name] = classifier
        logger.info(f"Registered classifier: {name}")
    
    def set_active(self, name: str):
        """Set active classifier"""
        if name in self.classifiers:
            self.active_classifier = name
            logger.info(f"Active classifier set to: {name}")
        else:
            logger.warning(f"Classifier not found: {name}")
    
    def classify(self, image: np.ndarray, classifier_name: Optional[str] = None) -> ClassificationResult:
        """
        Classify image using specified or active classifier
        
        Args:
            image: Input image
            classifier_name: Specific classifier to use
        
        Returns:
            Classification result
        """
        name = classifier_name or self.active_classifier
        
        if not name or name not in self.classifiers:
            raise ValueError(f"Classifier not available: {name}")
        
        return self.classifiers[name].predict(image)
    
    def ensemble_classify(self, image: np.ndarray, 
                         classifiers: Optional[List[str]] = None) -> ClassificationResult:
        """
        Ensemble classification using multiple models
        
        Args:
            image: Input image
            classifiers: List of classifier names to use
        
        Returns:
            Ensemble classification result
        """
        if not classifiers:
            classifiers = list(self.classifiers.keys())
        
        all_probabilities = []
        
        for name in classifiers:
            if name in self.classifiers:
                result = self.classifiers[name].predict(image)
                all_probabilities.append(result.probabilities)
        
        if not all_probabilities:
            raise ValueError("No valid classifiers for ensemble")
        
        # Average probabilities
        ensemble_probs = np.mean(all_probabilities, axis=0)
        class_id = int(np.argmax(ensemble_probs))
        confidence = float(ensemble_probs[class_id])
        
        # Get class name from first classifier
        first_classifier = self.classifiers[classifiers[0]]
        class_name = first_classifier.class_names[class_id] if class_id < len(first_classifier.class_names) else f"Class_{class_id}"
        
        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            probabilities=ensemble_probs
        )