"""Keras Context Classifier with InceptionV3 and Xception - SerpentAI Compatible

Provides CNN-based context classification using pre-trained models.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import pickle
import time
from datetime import datetime
from collections import deque
import structlog

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import InceptionV3, Xception, ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.utils import to_categorical
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    
# PyTorch alternative
try:
    import torch
    import torch.nn as nn
    import torchvision.models as torch_models
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Dataset
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False

logger = structlog.get_logger()


class ContextClassifierBackend:
    """Base context classifier backend"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (299, 299, 3)):
        self.input_shape = input_shape
        self.model = None
        self.preprocessing_func = None
        
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model"""
        # Resize if needed
        if image.shape[:2] != self.input_shape[:2]:
            image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
            
        # Apply model-specific preprocessing
        if self.preprocessing_func:
            image = self.preprocessing_func(image)
            
        return image
        
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict context from image"""
        # Base implementation - should be overridden by specific backends
        logger.warning("ContextClassifierBackend.predict() called - should be overridden by specific backend")
        # Return uniform probability distribution
        return np.ones(10) / 10.0
        
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 10):
        """Train the model"""
        # Base implementation - should be overridden by specific backends
        logger.warning("ContextClassifierBackend.train() called - should be overridden by specific backend")
        logger.info(f"Mock training completed for {epochs} epochs")
        
    def save(self, path: str):
        """Save model"""
        # Base implementation - should be overridden by specific backends
        logger.warning("ContextClassifierBackend.save() called - should be overridden by specific backend")
        logger.info(f"Mock model saved to {path}")
        
    def load(self, path: str):
        """Load model"""
        # Base implementation - should be overridden by specific backends
        logger.warning("ContextClassifierBackend.load() called - should be overridden by specific backend")
        logger.info(f"Mock model loaded from {path}")


class KerasInceptionV3Backend(ContextClassifierBackend):
    """InceptionV3 backend using Keras"""
    
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (299, 299, 3)):
        super().__init__(input_shape)
        
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
            
        self.num_classes = num_classes
        self.preprocessing_func = tf.keras.applications.inception_v3.preprocess_input
        self._build_model()
        
    def _build_model(self):
        """Build InceptionV3 model"""
        # Load pre-trained InceptionV3
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict context"""
        processed = self.preprocess(image)
        processed = np.expand_dims(processed, axis=0)
        predictions = self.model.predict(processed, verbose=0)
        return predictions[0]
        
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 10, batch_size: int = 32):
        """Train the model"""
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        datagen.fit(X_train)
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            datagen.flow(X_train, y_train_cat, batch_size=batch_size),
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
        
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        
    def load(self, path: str):
        """Load model"""
        self.model = keras.models.load_model(path)


class KerasXceptionBackend(ContextClassifierBackend):
    """Xception backend using Keras"""
    
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (299, 299, 3)):
        super().__init__(input_shape)
        
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
            
        self.num_classes = num_classes
        self.preprocessing_func = tf.keras.applications.xception.preprocess_input
        self._build_model()
        
    def _build_model(self):
        """Build Xception model"""
        # Load pre-trained Xception
        base_model = Xception(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict context"""
        processed = self.preprocess(image)
        processed = np.expand_dims(processed, axis=0)
        predictions = self.model.predict(processed, verbose=0)
        return predictions[0]
        
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 10, batch_size: int = 32):
        """Train the model with fine-tuning"""
        # Initial training with frozen base
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_val_cat = to_categorical(y_val, self.num_classes)
        
        history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs // 2,
            batch_size=batch_size,
            verbose=1
        )
        
        # Unfreeze some layers for fine-tuning
        base_model = self.model.layers[1]  # The Xception model
        base_model.trainable = True
        
        # Freeze all layers except last 30
        for layer in base_model.layers[:-30]:
            layer.trainable = False
            
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        # Continue training with fine-tuning
        history_fine = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs // 2,
            batch_size=batch_size,
            verbose=1
        )
        
        return history, history_fine
        
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        
    def load(self, path: str):
        """Load model"""
        self.model = keras.models.load_model(path)


class PyTorchContextBackend(ContextClassifierBackend):
    """PyTorch backend for context classification"""
    
    def __init__(self, model_name: str = "resnet50", num_classes: int = 10,
                 input_shape: Tuple[int, int, int] = (224, 224, 3)):
        super().__init__(input_shape)
        
        if not HAS_PYTORCH:
            raise ImportError("PyTorch not installed")
            
        self.num_classes = num_classes
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._build_model()
        self._setup_preprocessing()
        
    def _build_model(self):
        """Build PyTorch model"""
        # Load pre-trained model
        if self.model_name == "resnet50":
            self.model = torch_models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "inception_v3":
            self.model = torch_models.inception_v3(pretrained=True)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        elif self.model_name == "efficientnet":
            self.model = torch_models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, self.num_classes)
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        self.model = self.model.to(self.device)
        
    def _setup_preprocessing(self):
        """Setup preprocessing transform"""
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_shape[0], self.input_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict context"""
        self.model.eval()
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()[0]
        
    def save(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        }, path)
        
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_name = checkpoint['model_name']
        self.num_classes = checkpoint['num_classes']
        self.input_shape = checkpoint['input_shape']


class ContextClassifier:
    """Main context classifier - SerpentAI compatible interface"""
    
    def __init__(self, backend: str = "inception_v3", num_classes: int = 10,
                 input_shape: Tuple[int, int, int] = (299, 299, 3)):
        """
        Initialize context classifier
        
        Args:
            backend: Model backend ("inception_v3", "xception", "resnet50", etc.)
            num_classes: Number of context classes
            input_shape: Input image shape
        """
        self.backend_name = backend
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Class mappings
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Prediction cache
        self.cache = {}
        self.cache_size = 100
        
        # Initialize backend
        self._init_backend()
        
        # Metrics
        self.prediction_times = deque(maxlen=100)
        self.accuracy_history = []
        
    def _init_backend(self):
        """Initialize the backend model"""
        if self.backend_name == "inception_v3":
            if HAS_TENSORFLOW:
                self.backend = KerasInceptionV3Backend(self.num_classes, self.input_shape)
            elif HAS_PYTORCH:
                self.backend = PyTorchContextBackend("inception_v3", self.num_classes, self.input_shape)
            else:
                raise ImportError("No ML backend available")
                
        elif self.backend_name == "xception":
            if HAS_TENSORFLOW:
                self.backend = KerasXceptionBackend(self.num_classes, self.input_shape)
            else:
                raise ImportError("Xception requires TensorFlow")
                
        elif self.backend_name in ["resnet50", "efficientnet"]:
            if HAS_PYTORCH:
                self.backend = PyTorchContextBackend(self.backend_name, self.num_classes, self.input_shape)
            else:
                raise ImportError(f"{self.backend_name} requires PyTorch")
        else:
            raise ValueError(f"Unknown backend: {self.backend_name}")
            
        logger.info(f"Initialized {self.backend_name} context classifier")
        
    def predict(self, image: np.ndarray, use_cache: bool = True) -> Dict[str, Any]:
        """
        Predict context from image
        
        Args:
            image: Input image
            use_cache: Whether to use prediction cache
            
        Returns:
            Dictionary with prediction results
        """
        # Check cache
        if use_cache:
            image_hash = self._hash_image(image)
            if image_hash in self.cache:
                return self.cache[image_hash]
                
        # Time prediction
        start_time = time.time()
        
        # Get predictions
        probabilities = self.backend.predict(image)
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        
        result = {
            'predicted_class': self._get_class_name(top_indices[0]),
            'predicted_index': int(top_indices[0]),
            'confidence': float(probabilities[top_indices[0]]),
            'probabilities': probabilities.tolist(),
            'top_5': [
                {
                    'class': self._get_class_name(idx),
                    'index': int(idx),
                    'confidence': float(probabilities[idx])
                }
                for idx in top_indices
            ],
            'prediction_time': time.time() - start_time
        }
        
        # Update cache
        if use_cache:
            self.cache[image_hash] = result
            if len(self.cache) > self.cache_size:
                # Remove oldest entries
                oldest = list(self.cache.keys())[:len(self.cache) - self.cache_size]
                for key in oldest:
                    del self.cache[key]
                    
        # Track metrics
        self.prediction_times.append(result['prediction_time'])
        
        return result
        
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict context for multiple images"""
        results = []
        for image in images:
            results.append(self.predict(image))
        return results
        
    def train(self, dataset_path: str, epochs: int = 10, batch_size: int = 32,
              validation_split: float = 0.2):
        """
        Train the context classifier
        
        Args:
            dataset_path: Path to training dataset
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation data split
        """
        # Load dataset
        X_train, y_train, X_val, y_val, class_names = self._load_dataset(
            dataset_path, validation_split
        )
        
        # Update class mappings
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}
        
        # Train model
        logger.info(f"Training {self.backend_name} on {len(X_train)} samples")
        history = self.backend.train(X_train, y_train, X_val, y_val, epochs, batch_size)
        
        # Evaluate
        val_predictions = [self.predict(img)['predicted_index'] for img in X_val]
        accuracy = np.mean(np.array(val_predictions) == y_val)
        self.accuracy_history.append(accuracy)
        
        logger.info(f"Training complete. Validation accuracy: {accuracy:.3f}")
        
        return history
        
    def _load_dataset(self, dataset_path: str, validation_split: float):
        """Load dataset from directory structure"""
        dataset_path = Path(dataset_path)
        
        X, y = [], []
        class_names = []
        
        # Load images from class directories
        for class_dir in sorted(dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            for image_path in class_dir.glob("*.png"):
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    X.append(image)
                    y.append(class_idx)
                    
        X = np.array(X)
        y = np.array(y)
        
        # Split into train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        X_train = X[n_val:]
        y_train = y[n_val:]
        X_val = X[:n_val]
        y_val = y[:n_val]
        
        return X_train, y_train, X_val, y_val, class_names
        
    def _get_class_name(self, idx: int) -> str:
        """Get class name from index"""
        if idx in self.idx_to_class:
            return self.idx_to_class[idx]
        return f"class_{idx}"
        
    def _hash_image(self, image: np.ndarray) -> str:
        """Generate hash for image caching"""
        # Simple perceptual hash
        small = cv2.resize(image, (8, 8))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) if len(small.shape) == 3 else small
        avg = gray.mean()
        binary = (gray > avg).flatten()
        hash_val = 0
        for bit in binary:
            hash_val = (hash_val << 1) | int(bit)
        return hex(hash_val)
        
    def save(self, path: str):
        """Save classifier"""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.backend.save(str(save_path / "model"))
        
        # Save metadata
        metadata = {
            'backend': self.backend_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'class_names': self.class_names,
            'class_to_idx': self.class_to_idx,
            'accuracy_history': self.accuracy_history
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Classifier saved to {save_path}")
        
    def load(self, path: str):
        """Load classifier"""
        load_path = Path(path)
        
        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        self.backend_name = metadata['backend']
        self.num_classes = metadata['num_classes']
        self.input_shape = tuple(metadata['input_shape'])
        self.class_names = metadata['class_names']
        self.class_to_idx = metadata['class_to_idx']
        self.idx_to_class = {int(k): v for k, v in metadata.get('idx_to_class', {}).items()}
        if not self.idx_to_class:
            self.idx_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.accuracy_history = metadata.get('accuracy_history', [])
        
        # Reinitialize backend
        self._init_backend()
        
        # Load model
        self.backend.load(str(load_path / "model"))
        
        logger.info(f"Classifier loaded from {load_path}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get classifier metrics"""
        return {
            'backend': self.backend_name,
            'num_classes': self.num_classes,
            'avg_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0,
            'cache_size': len(self.cache),
            'accuracy_history': self.accuracy_history,
            'last_accuracy': self.accuracy_history[-1] if self.accuracy_history else None
        }


# SerpentAI compatibility functions
def train_context_classifier(game_name: str, model: str = "inception_v3",
                            dataset_path: Optional[str] = None):
    """Train context classifier for a game - SerpentAI compatible"""
    if dataset_path is None:
        dataset_path = Path.home() / ".nexus" / "datasets" / game_name / "context"
        
    classifier = ContextClassifier(backend=model)
    classifier.train(str(dataset_path))
    
    # Save to standard location
    save_path = Path.home() / ".nexus" / "models" / game_name / "context_classifier"
    classifier.save(str(save_path))
    
    return classifier


def load_context_classifier(game_name: str) -> ContextClassifier:
    """Load context classifier for a game - SerpentAI compatible"""
    load_path = Path.home() / ".nexus" / "models" / game_name / "context_classifier"
    
    classifier = ContextClassifier()
    classifier.load(str(load_path))
    
    return classifier