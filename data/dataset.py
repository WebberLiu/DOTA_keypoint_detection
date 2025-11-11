"""
Dataset implementation for DOTA keypoint detection.

This module provides the DOTAKeypointDataset class which handles:
- Loading images and annotations
- Generating keypoint heatmaps from bounding box centers
- Data splitting and batching
- Integration with PyTorch DataLoader
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import cv2

from config import DATA_CONFIG, TRAIN_CONFIG, MODEL_CONFIG
from .dota_parser import DOTAParser


def custom_collate_fn(batch):
    """
    Custom collate function to handle batches with variable-length metadata.
    
    This function properly batches images and heatmaps while keeping metadata
    as a list (since keypoints can have different lengths per image).
    """
    images = torch.stack([item[0] for item in batch])
    heatmaps = torch.stack([item[1] for item in batch])
    metadata = [item[2] for item in batch]  # Keep as list, don't try to batch
    
    return images, heatmaps, metadata


class DOTAKeypointDataset(Dataset):
    """
    PyTorch Dataset for DOTA keypoint detection.
    
    This dataset converts bounding box annotations to keypoint heatmaps,
    where each keypoint represents the center of a bounding box.
    
    Args:
        data_dir: Directory containing images and annotations
        transform: Optional transforms to apply to images
        split: Dataset split ('train', 'val', or 'test')
        generate_synthetic: If True, generates synthetic data for demonstration
    """
    
    def __init__(
        self,
        data_dir: str,
        transform=None,
        split: str = "train",
        generate_synthetic: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.split = split
        self.image_size = DATA_CONFIG["image_size"]
        self.heatmap_size = MODEL_CONFIG["heatmap_size"]
        self.heatmap_sigma = MODEL_CONFIG["heatmap_sigma"]
        
        # Load or generate dataset
        if generate_synthetic or not self._check_data_exists():
            print(f"Generating synthetic {split} data...")
            self.samples = self._generate_synthetic_data()
        else:
            print(f"Loading {split} data from {data_dir}...")
            self.samples = self._load_data()
    
    def _check_data_exists(self) -> bool:
        """Check if actual DOTA data exists in the data directory."""
        images_dir = self.data_dir / "images"
        # DOTA uses 'labelTxt' directory for annotations
        labels_dir = self.data_dir / "labelTxt"
        return images_dir.exists() and labels_dir.exists()
    
    def _load_data(self) -> List[Dict]:
        """
        Load actual DOTA dataset annotations using DOTA parser.
        Automatically splits data based on the specified split ratio.
        """
        parser = DOTAParser(str(self.data_dir))
        
        # Get image-annotation pairs (respecting max_samples config)
        max_samples = DATA_CONFIG.get("max_samples", None)
        categories = DATA_CONFIG.get("dota_categories", None)
        all_pairs = parser.get_image_annotation_pairs(
            max_samples=max_samples,
            categories=categories
        )
        
        if len(all_pairs) == 0:
            print(f"WARNING: No DOTA data found in {self.data_dir}")
            print("Falling back to synthetic data generation...")
            return self._generate_synthetic_data()
        
        print(f"Found {len(all_pairs)} total images from DOTA dataset")
        if max_samples:
            print(f"  (limited to {max_samples} images as per config)")
        if categories:
            print(f"  (filtered by categories: {categories})")
        
        # Shuffle with fixed seed for reproducibility
        np.random.seed(DATA_CONFIG["random_seed"])
        np.random.shuffle(all_pairs)
        
        # Split based on split type
        n_total = len(all_pairs)
        n_train = int(n_total * DATA_CONFIG["train_ratio"])
        n_val = int(n_total * DATA_CONFIG["val_ratio"])
        
        if self.split == "train":
            pairs = all_pairs[:n_train]
        elif self.split == "val":
            pairs = all_pairs[n_train:n_train + n_val]
        else:  # test
            pairs = all_pairs[n_train + n_val:]
        
        # Load samples using parser
        samples = []
        for img_path, ann_path in pairs:
            sample = parser.load_sample(img_path, ann_path)
            samples.append(sample)
        
        print(f"Loaded {len(samples)} {self.split} samples from DOTA dataset")
        return samples
    
    def _extract_keypoints_from_boxes(self, bboxes: List[Dict]) -> np.ndarray:
        """
        Extract center keypoints from bounding boxes.
        
        Args:
            bboxes: List of bounding boxes with format [x1, y1, x2, y2]
        
        Returns:
            Array of keypoints with shape (N, 2) containing (x, y) coordinates
        """
        keypoints = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["coords"]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            keypoints.append([center_x, center_y])
        return np.array(keypoints, dtype=np.float32)
    
    def _generate_synthetic_data(self) -> List[Dict]:
        """
        Generate synthetic data for demonstration purposes.
        Creates random images with object-like blobs and their center keypoints.
        """
        num_samples = DATA_CONFIG["num_synthetic_samples"]
        samples = []
        
        # Adjust samples based on split
        split_ratios = {
            "train": DATA_CONFIG["train_ratio"],
            "val": DATA_CONFIG["val_ratio"],
            "test": DATA_CONFIG["test_ratio"]
        }
        
        num_samples = int(num_samples * split_ratios[self.split])
        
        for i in range(num_samples):
            # Create synthetic image with random objects
            image, keypoints = self._create_synthetic_sample(i)
            
            # Save synthetic image temporarily
            sample_dir = self.data_dir / "synthetic" / self.split
            sample_dir.mkdir(parents=True, exist_ok=True)
            
            image_path = sample_dir / f"sample_{i:04d}.png"
            Image.fromarray(image).save(image_path)
            
            samples.append({
                "image_path": str(image_path),
                "keypoints": keypoints
            })
        
        return samples
    
    def _create_synthetic_sample(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a single synthetic sample with random objects and keypoints.
        
        Returns:
            image: RGB image array
            keypoints: Array of keypoint coordinates
        """
        np.random.seed(seed + hash(self.split) % 10000)
        
        # Create blank image
        image = np.ones((*self.image_size, 3), dtype=np.uint8) * 200
        
        # Add random objects
        num_objects = np.random.randint(1, DATA_CONFIG["max_objects_per_image"] + 1)
        keypoints = []
        
        for _ in range(num_objects):
            # Random object center
            center_x = np.random.randint(50, self.image_size[1] - 50)
            center_y = np.random.randint(50, self.image_size[0] - 50)
            
            # Random object size
            width = np.random.randint(20, 80)
            height = np.random.randint(20, 80)
            
            # Draw object (ellipse)
            color = tuple(np.random.randint(0, 200, 3).tolist())
            cv2.ellipse(
                image,
                (center_x, center_y),
                (width // 2, height // 2),
                angle=np.random.randint(0, 180),
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=-1
            )
            
            keypoints.append([center_x, center_y])
        
        return image, np.array(keypoints, dtype=np.float32)
    
    def _generate_heatmap(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Generate Gaussian heatmap from keypoints.
        
        Args:
            keypoints: Array of keypoint coordinates (N, 2)
        
        Returns:
            Heatmap array of shape (H, W) with Gaussian peaks at keypoint locations
        """
        heatmap = np.zeros(self.heatmap_size, dtype=np.float32)
        
        # Scale factor from image size to heatmap size
        scale_x = self.heatmap_size[1] / self.image_size[1]
        scale_y = self.heatmap_size[0] / self.image_size[0]
        
        for kp in keypoints:
            x, y = kp
            
            # Skip invalid keypoints (negative or out of bounds)
            if x < 0 or y < 0 or x >= self.image_size[1] or y >= self.image_size[0]:
                continue
            
            # Scale keypoint to heatmap size
            x_hm = int(x * scale_x)
            y_hm = int(y * scale_y)
            
            # Ensure heatmap coordinates are within bounds
            if 0 <= x_hm < self.heatmap_size[1] and 0 <= y_hm < self.heatmap_size[0]:
                # Generate Gaussian around keypoint
                self._add_gaussian_to_heatmap(heatmap, x_hm, y_hm)
        
        return heatmap
    
    def _add_gaussian_to_heatmap(self, heatmap: np.ndarray, x: int, y: int):
        """Add a Gaussian peak to the heatmap at specified location."""
        sigma = self.heatmap_sigma
        size = int(6 * sigma + 1)  # Gaussian kernel size (ensure integer)
        
        # Generate Gaussian kernel
        x_grid = np.arange(0, size, 1, float)
        y_grid = x_grid[:, np.newaxis]
        x0, y0 = size // 2, size // 2
        gaussian = np.exp(-((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
        
        # Determine placement bounds
        h, w = heatmap.shape
        left = int(x - size // 2)
        right = int(x + size // 2 + 1)
        top = int(y - size // 2)
        bottom = int(y + size // 2 + 1)
        
        # Handle boundary conditions (ensure all are integers)
        g_left = int(max(0, -left))
        g_right = int(min(size, w - left))
        g_top = int(max(0, -top))
        g_bottom = int(min(size, h - top))
        
        img_left = int(max(0, left))
        img_right = int(min(w, right))
        img_top = int(max(0, top))
        img_bottom = int(min(h, bottom))
        
        # Check if the region is valid (has non-zero size in both dimensions)
        if img_bottom > img_top and img_right > img_left:
            # Also check Gaussian region is valid
            if g_bottom > g_top and g_right > g_left:
                # Add Gaussian to heatmap (use maximum value if overlapping)
                heatmap[img_top:img_bottom, img_left:img_right] = np.maximum(
                    heatmap[img_top:img_bottom, img_left:img_right],
                    gaussian[g_top:g_bottom, g_left:g_right]
                )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor
            heatmap: Target keypoint heatmap
            metadata: Dictionary with additional information
        """
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = image.resize(self.image_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Get keypoints and generate heatmap
        keypoints = sample["keypoints"]
        heatmap = self._generate_heatmap(keypoints)
        
        # Apply transforms if provided
        if self.transform:
            # Note: For production, use transforms that handle both image and keypoints
            image_array = self.transform(image_array)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # Add channel dim
        
        metadata = {
            "image_path": sample["image_path"],
            "num_keypoints": len(keypoints),
            "keypoints": keypoints.tolist() if isinstance(keypoints, np.ndarray) else keypoints,
        }
        
        return image_tensor, heatmap_tensor, metadata


def create_dataloaders(
    data_dir: str,
    train_transform=None,
    val_transform=None,
    generate_synthetic: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory for dataset
        train_transform: Transforms for training data
        val_transform: Transforms for validation/test data
        generate_synthetic: Whether to generate synthetic data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DOTAKeypointDataset(
        data_dir=data_dir,
        transform=train_transform,
        split="train",
        generate_synthetic=generate_synthetic
    )
    
    val_dataset = DOTAKeypointDataset(
        data_dir=data_dir,
        transform=val_transform,
        split="val",
        generate_synthetic=generate_synthetic
    )
    
    test_dataset = DOTAKeypointDataset(
        data_dir=data_dir,
        transform=val_transform,
        split="test",
        generate_synthetic=generate_synthetic
    )
    
    # Configure dataloader settings based on CUDA availability
    use_cuda = torch.cuda.is_available()
    num_workers = TRAIN_CONFIG["num_workers"] if use_cuda else 0
    pin_memory = use_cuda
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader, test_loader

