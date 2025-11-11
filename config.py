"""
Configuration file for DOTA keypoint detection project.
This centralizes all hyperparameters and settings for reproducibility.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
# DATA_DIR = PROJECT_ROOT / "data_samples"
DATA_DIR = PROJECT_ROOT / "data_samples" / "dota"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data configuration
DATA_CONFIG = {
    "image_size": (512, 512),  # Target image size for training
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    # For synthetic data generation (when actual DOTA data is not available)
    "num_synthetic_samples": 500,
    "max_objects_per_image": 10,
    # For real DOTA dataset (set to None to use all available images)
    "max_samples": 200,  # Limit number of images (e.g., 200 for quick testing)
    "dota_categories": None,  # List of categories to use, None = all categories
}

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",  # Using lightweight model for demonstration
    "pretrained": True,
    "heatmap_size": (64, 64),  # Output heatmap resolution
    "heatmap_sigma": 2.0,  # Gaussian sigma for keypoint heatmaps
    "max_keypoints": 20,  # Maximum number of keypoints to detect per image
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 2,  # Small number for demonstration
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "step",
    "lr_step_size": 5,
    "lr_gamma": 0.1,
    "num_workers": 2,
    "device": "cuda",  # Will fall back to CPU if CUDA unavailable
    "save_every": 5,  # Save checkpoint every N epochs
    "log_every": 10,  # Log metrics every N batches
}

# Augmentation configuration
AUGMENTATION_CONFIG = {
    "horizontal_flip": 0.5,
    "vertical_flip": 0.5,
    "rotation_degrees": 15,
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
}

# Inference configuration
INFERENCE_CONFIG = {
    "batch_size": 16,
    "confidence_threshold": 0.3,  # Minimum confidence for keypoint detection
    "nms_threshold": 10,  # Non-maximum suppression distance (pixels)
    "device": "cuda",
}

# Evaluation metrics
METRICS = [
    "mse",  # Mean squared error for keypoint positions
    "mae",  # Mean absolute error
    "pck",  # Percentage of Correct Keypoints (within threshold)
]

# Evaluation thresholds for PCK (Percentage of Correct Keypoints)
PCK_THRESHOLDS = [5, 10, 15, 20]  # pixels

# Logging configuration
LOGGING_CONFIG = {
    "tensorboard": True,
    "log_dir": str(LOGS_DIR),
    "checkpoint_dir": str(CHECKPOINTS_DIR),
}

# Production/Deployment considerations
DEPLOYMENT_CONFIG = {
    "model_version": "v1.0",
    "monitoring_metrics": [
        "inference_time",
        "batch_processing_time",
        "memory_usage",
        "prediction_confidence_distribution",
    ],
    "scaling_strategy": "horizontal",  # How to scale for large volumes
    "max_batch_size": 32,
    "timeout_seconds": 30,
}

