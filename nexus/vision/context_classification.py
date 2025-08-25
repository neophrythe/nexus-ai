"""Context classification system for game state recognition - adapted from SerpentAI"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import json
import asyncio
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()


@dataclass
class ContextClass:
    """Definition of a game context/state"""
    name: str
    id: int
    description: str = ""
    parent: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class GameContextDataset(Dataset):
    """Dataset for game context classification"""
    
    def __init__(self, data_dir: Path, transform=None, mode: str = "train"):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing class folders
            transform: Image transformations
            mode: "train" or "val"
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_samples()
    
    def _load_samples(self):
        """Load all samples from directory structure"""
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        for idx, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            # Load images from class directory
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), idx))
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), idx))
        
        logger.info(f"Loaded {len(self.samples)} samples from {len(class_dirs)} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx


class ContextClassifier:
    """Base context classifier using modern PyTorch"""
    
    def __init__(self, num_classes: int, input_shape: Tuple[int, int, int] = (3, 224, 224),
                 model_type: str = "resnet50", device: Optional[str] = None):
        """
        Initialize context classifier.
        
        Args:
            num_classes: Number of context classes
            input_shape: Input image shape (C, H, W)
            model_type: Type of model ("resnet50", "efficientnet", "vit", etc.)
            device: Device to use (auto-detect if None)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_type = model_type
        
        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # Training components
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Data transformations
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        logger.info(f"Context classifier initialized: {model_type} with {num_classes} classes on {self.device}")
    
    def _build_model(self) -> nn.Module:
        """Build the classification model"""
        if self.model_type == "resnet50":
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_type == "resnet101":
            model = models.resnet101(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        elif self.model_type == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        elif self.model_type == "efficientnet_b4":
            model = models.efficientnet_b4(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        
        elif self.model_type == "vit":
            model = models.vit_b_16(pretrained=True)
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)
        
        elif self.model_type == "inception_v3":
            model = models.inception_v3(pretrained=True, aux_logits=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            num_aux_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_aux_features, self.num_classes)
        
        else:
            # Default to ResNet50
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        
        return model
    
    def _get_train_transform(self):
        """Get training data transformations"""
        return transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transform(self):
        """Get validation data transformations"""
        return transforms.Compose([
            transforms.Resize((self.input_shape[1], self.input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train(self, train_dir: Path, val_dir: Optional[Path] = None,
             epochs: int = 10, batch_size: int = 32, learning_rate: float = 1e-3,
             save_best: bool = True, save_path: Optional[Path] = None):
        """
        Train the context classifier.
        
        Args:
            train_dir: Training data directory
            val_dir: Validation data directory
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_best: Save best model based on validation loss
            save_path: Path to save best model
        """
        # Create datasets
        train_dataset = GameContextDataset(train_dir, transform=self.train_transform, mode="train")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        val_loader = None
        if val_dir:
            val_dataset = GameContextDataset(val_dir, transform=self.val_transform, mode="val")
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Handle Inception v3 auxiliary outputs
                if self.model_type == "inception_v3" and self.model.training:
                    outputs, aux_outputs = self.model(images)
                    loss1 = self.criterion(outputs, labels)
                    loss2 = self.criterion(aux_outputs, labels)
                    loss = loss1 + 0.4 * loss2
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100.0 * train_correct / train_total
            
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Validation phase
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += predicted.eq(labels).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = 100.0 * val_correct / val_total
                
                self.val_losses.append(avg_val_loss)
                self.val_accuracies.append(val_accuracy)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}% - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                
                # Save best model
                if save_best and save_path and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    self.save_model(save_path)
                    logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            
            scheduler.step()
    
    def predict(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Tuple[int, float, Dict[int, float]]:
        """
        Predict context class for an image.
        
        Args:
            image: Input image (numpy array, tensor, or PIL Image)
        
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        self.model.eval()
        
        # Preprocess image
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Apply transformations
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        # Get all class probabilities
        all_probs = {i: prob.item() for i, prob in enumerate(probabilities[0])}
        
        return predicted.item(), confidence.item(), all_probs
    
    async def predict_async(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> Tuple[int, float, Dict[int, float]]:
        """Async version of predict"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, image)
    
    def save_model(self, path: Path):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'model_type': self.model_type,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        
        logger.info(f"Model loaded from {path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'best_val_accuracy': max(self.val_accuracies) if self.val_accuracies else None
        }


class GameStateClassifier(ContextClassifier):
    """Specialized classifier for game states"""
    
    def __init__(self, game_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_name = game_name
        self.state_definitions: Dict[int, ContextClass] = {}
        self.transition_matrix: Optional[np.ndarray] = None
    
    def register_state(self, state: ContextClass):
        """Register a game state"""
        self.state_definitions[state.id] = state
        logger.info(f"Registered state: {state.name} (ID: {state.id})")
    
    def predict_with_context(self, image: Union[np.ndarray, torch.Tensor, Image.Image],
                            previous_state: Optional[int] = None) -> Tuple[int, float, str]:
        """
        Predict game state with context awareness.
        
        Args:
            image: Current frame
            previous_state: Previous state ID for transition probability
        
        Returns:
            Tuple of (state_id, confidence, state_name)
        """
        # Get base prediction
        predicted_class, confidence, all_probs = self.predict(image)
        
        # Apply transition probability if available
        if previous_state is not None and self.transition_matrix is not None:
            transition_probs = self.transition_matrix[previous_state]
            
            # Combine with predicted probabilities
            for class_id, trans_prob in enumerate(transition_probs):
                all_probs[class_id] = all_probs.get(class_id, 0) * 0.7 + trans_prob * 0.3
            
            # Get new prediction
            predicted_class = max(all_probs, key=all_probs.get)
            confidence = all_probs[predicted_class]
        
        # Get state name
        state = self.state_definitions.get(predicted_class)
        state_name = state.name if state else f"Unknown_{predicted_class}"
        
        return predicted_class, confidence, state_name
    
    def learn_transitions(self, state_sequences: List[List[int]]):
        """
        Learn state transition probabilities from sequences.
        
        Args:
            state_sequences: List of state ID sequences
        """
        # Initialize transition matrix
        n_states = self.num_classes
        self.transition_matrix = np.zeros((n_states, n_states))
        
        # Count transitions
        for sequence in state_sequences:
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                self.transition_matrix[from_state, to_state] += 1
        
        # Normalize to probabilities
        for i in range(n_states):
            row_sum = self.transition_matrix[i].sum()
            if row_sum > 0:
                self.transition_matrix[i] /= row_sum
            else:
                # Uniform distribution if no transitions observed
                self.transition_matrix[i] = 1.0 / n_states
        
        logger.info("Learned state transition probabilities")