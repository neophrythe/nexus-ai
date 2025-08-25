import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
from datetime import datetime
import structlog
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

logger = structlog.get_logger()


class GameDataset(Dataset):
    """Dataset for game frames and annotations"""
    
    def __init__(self, 
                 data_dir: Path,
                 transform=None,
                 mode: str = "train"):
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.mode = mode
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.samples = self._load_samples()
        
        logger.info(f"Loaded {len(self.samples)} samples for {mode}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        metadata_file = self.data_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples"""
        samples = []
        
        # Load annotations
        annotations_file = self.data_dir / f"{self.mode}_annotations.json"
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                sample = {
                    "image_path": self.data_dir / "images" / ann["image"],
                    "label": ann.get("label"),
                    "bbox": ann.get("bbox"),
                    "action": ann.get("action"),
                    "reward": ann.get("reward", 0),
                    "metadata": ann.get("metadata", {})
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample["image_path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Prepare label
        label = sample["label"] if sample["label"] is not None else 0
        
        return image, {
            "label": label,
            "bbox": sample["bbox"],
            "action": sample["action"],
            "reward": sample["reward"]
        }


class DatasetManager:
    """Manages datasets for training and evaluation"""
    
    def __init__(self, base_dir: Path = Path("datasets")):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.datasets: Dict[str, GameDataset] = {}
        self.current_recording = None
        
    def create_dataset(self, name: str, game: str, description: str = "") -> Path:
        """Create a new dataset"""
        dataset_dir = self.base_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (dataset_dir / "images").mkdir(exist_ok=True)
        (dataset_dir / "labels").mkdir(exist_ok=True)
        (dataset_dir / "recordings").mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "name": name,
            "game": game,
            "description": description,
            "created": datetime.now().isoformat(),
            "version": "1.0",
            "stats": {
                "total_frames": 0,
                "total_annotations": 0,
                "classes": []
            }
        }
        
        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created dataset: {name}")
        return dataset_dir
    
    def load_dataset(self, name: str, mode: str = "train") -> GameDataset:
        """Load a dataset"""
        dataset_dir = self.base_dir / name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {name} not found")
        
        dataset = GameDataset(dataset_dir, mode=mode)
        self.datasets[name] = dataset
        
        return dataset
    
    def start_recording(self, dataset_name: str) -> None:
        """Start recording game data"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} not found")
        
        self.current_recording = {
            "dataset": dataset_name,
            "start_time": datetime.now(),
            "frames": [],
            "annotations": []
        }
        
        logger.info(f"Started recording for dataset: {dataset_name}")
    
    def add_frame(self, 
                 frame: np.ndarray,
                 annotations: Optional[Dict[str, Any]] = None,
                 save_image: bool = True) -> None:
        """Add a frame to the current recording"""
        
        if not self.current_recording:
            logger.warning("No active recording")
            return
        
        dataset_name = self.current_recording["dataset"]
        dataset_dir = self.base_dir / dataset_name
        
        # Generate frame ID
        frame_id = f"{dataset_name}_{len(self.current_recording['frames']):06d}"
        timestamp = datetime.now()
        
        # Save image if requested
        image_filename = None
        if save_image:
            image_filename = f"{frame_id}.jpg"
            image_path = dataset_dir / "images" / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Create frame record
        frame_record = {
            "id": frame_id,
            "timestamp": timestamp.isoformat(),
            "image": image_filename,
            "annotations": annotations or {}
        }
        
        self.current_recording["frames"].append(frame_record)
        
        # Add to annotations if provided
        if annotations:
            self.current_recording["annotations"].append({
                "image": image_filename,
                **annotations
            })
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and save data"""
        
        if not self.current_recording:
            logger.warning("No active recording")
            return {}
        
        dataset_name = self.current_recording["dataset"]
        dataset_dir = self.base_dir / dataset_name
        
        # Save recording
        recording_file = dataset_dir / "recordings" / f"recording_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(recording_file, 'w') as f:
            json.dump({
                "start_time": self.current_recording["start_time"].isoformat(),
                "end_time": datetime.now().isoformat(),
                "frames": self.current_recording["frames"],
                "total_frames": len(self.current_recording["frames"])
            }, f, indent=2)
        
        # Save annotations
        if self.current_recording["annotations"]:
            ann_file = dataset_dir / "train_annotations.json"
            
            # Load existing annotations
            existing_annotations = []
            if ann_file.exists():
                with open(ann_file, 'r') as f:
                    existing_annotations = json.load(f)
            
            # Append new annotations
            existing_annotations.extend(self.current_recording["annotations"])
            
            # Save updated annotations
            with open(ann_file, 'w') as f:
                json.dump(existing_annotations, f, indent=2)
        
        # Update metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata["stats"]["total_frames"] += len(self.current_recording["frames"])
        metadata["stats"]["total_annotations"] += len(self.current_recording["annotations"])
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        stats = {
            "dataset": dataset_name,
            "frames_recorded": len(self.current_recording["frames"]),
            "annotations_added": len(self.current_recording["annotations"]),
            "duration": (datetime.now() - self.current_recording["start_time"]).total_seconds()
        }
        
        self.current_recording = None
        logger.info(f"Stopped recording: {stats}")
        
        return stats
    
    def export_to_yolo(self, dataset_name: str, output_dir: Path) -> None:
        """Export dataset to YOLO format"""
        dataset_dir = self.base_dir / dataset_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create YOLO directory structure
        (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        ann_file = dataset_dir / "train_annotations.json"
        if not ann_file.exists():
            logger.warning(f"No annotations found for {dataset_name}")
            return
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        # Split into train/val (80/20)
        split_idx = int(len(annotations) * 0.8)
        train_anns = annotations[:split_idx]
        val_anns = annotations[split_idx:]
        
        # Process annotations
        for split, anns in [("train", train_anns), ("val", val_anns)]:
            for ann in anns:
                if "bbox" not in ann or not ann["bbox"]:
                    continue
                
                # Copy image
                src_image = dataset_dir / "images" / ann["image"]
                dst_image = output_dir / "images" / split / ann["image"]
                
                if src_image.exists():
                    import shutil
                    shutil.copy2(src_image, dst_image)
                    
                    # Get image dimensions
                    img = cv2.imread(str(src_image))
                    h, w = img.shape[:2]
                    
                    # Convert bbox to YOLO format
                    x1, y1, x2, y2 = ann["bbox"]
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    # Write label file
                    label_file = output_dir / "labels" / split / ann["image"].replace(".jpg", ".txt")
                    with open(label_file, 'w') as f:
                        class_id = ann.get("label", 0)
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        # Create data.yaml
        data_yaml = {
            "path": str(output_dir),
            "train": "images/train",
            "val": "images/val",
            "nc": len(set(ann.get("label", 0) for ann in annotations)),
            "names": list(set(ann.get("class_name", "object") for ann in annotations))
        }
        
        with open(output_dir / "data.yaml", 'w') as f:
            yaml.dump(data_yaml, f)
        
        logger.info(f"Exported {dataset_name} to YOLO format at {output_dir}")
    
    def get_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset statistics"""
        dataset_dir = self.base_dir / dataset_name
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset {dataset_name} not found")
        
        # Load metadata
        with open(dataset_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Count files
        num_images = len(list((dataset_dir / "images").glob("*.jpg")))
        num_recordings = len(list((dataset_dir / "recordings").glob("*.json")))
        
        # Load annotations
        ann_file = dataset_dir / "train_annotations.json"
        num_annotations = 0
        classes = set()
        
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
                num_annotations = len(annotations)
                classes = set(ann.get("class_name", "unknown") for ann in annotations)
        
        return {
            "name": dataset_name,
            "created": metadata["created"],
            "total_images": num_images,
            "total_annotations": num_annotations,
            "total_recordings": num_recordings,
            "classes": list(classes),
            "metadata": metadata
        }
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets"""
        datasets = []
        
        for dataset_dir in self.base_dir.iterdir():
            if dataset_dir.is_dir() and (dataset_dir / "metadata.json").exists():
                try:
                    stats = self.get_statistics(dataset_dir.name)
                    datasets.append(stats)
                except Exception as e:
                    logger.error(f"Failed to get stats for {dataset_dir.name}: {e}")
        
        return datasets