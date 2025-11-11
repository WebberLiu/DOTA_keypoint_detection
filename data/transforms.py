"""
Data transformations and augmentation for DOTA keypoint detection.

This module provides transformation pipelines for training and validation,
including augmentations that preserve keypoint locations.
"""

import numpy as np
import torch
from typing import Callable, Optional
import cv2

from config import AUGMENTATION_CONFIG


class RandomHorizontalFlip:
    """Randomly flip image horizontally."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return np.fliplr(image).copy()
        return image


class RandomVerticalFlip:
    """Randomly flip image vertically."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return np.flipud(image).copy()
        return image


class ColorJitter:
    """Apply random color jittering to image."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert to uint8 for OpenCV operations
        if image.dtype == np.float32 or image.dtype == np.float64:
            image_uint8 = (image * 255).astype(np.uint8)
            need_convert_back = True
        else:
            image_uint8 = image
            need_convert_back = False
        
        # Convert to HSV for easier manipulation
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Adjust brightness (V channel)
        if self.brightness > 0:
            brightness_factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
        
        # Adjust saturation (S channel)
        if self.saturation > 0:
            saturation_factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
        
        # Adjust hue (H channel)
        if self.hue > 0:
            hue_shift = np.random.uniform(-self.hue, self.hue) * 180
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Convert back to RGB
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Apply contrast adjustment in RGB space
        if self.contrast > 0:
            contrast_factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            result = result.astype(np.float32)
            mean = result.mean()
            result = np.clip((result - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
        
        # Convert back to float if needed
        if need_convert_back:
            result = result.astype(np.float32) / 255.0
        
        return result


class Normalize:
    """Normalize image with ImageNet statistics."""
    
    def __init__(
        self,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225)
    ):
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Assume image is in [0, 1] range
        return (image - self.mean) / self.std


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            image = transform(image)
        return image


class GaussianNoise:
    """Add Gaussian noise to image."""
    
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self.std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0, 1)


def get_train_transforms() -> Callable:
    """
    Get training data transforms with augmentation.
    
    Returns:
        Composed transform function for training
    """
    transforms = [
        RandomHorizontalFlip(p=AUGMENTATION_CONFIG["horizontal_flip"]),
        RandomVerticalFlip(p=AUGMENTATION_CONFIG["vertical_flip"]),
        ColorJitter(
            brightness=AUGMENTATION_CONFIG["brightness"],
            contrast=AUGMENTATION_CONFIG["contrast"],
            saturation=AUGMENTATION_CONFIG["saturation"],
            hue=AUGMENTATION_CONFIG["hue"]
        ),
        GaussianNoise(std=0.01),
        Normalize(),
    ]
    
    return Compose(transforms)


def get_val_transforms() -> Callable:
    """
    Get validation/test data transforms (no augmentation).
    
    Returns:
        Composed transform function for validation/test
    """
    transforms = [
        Normalize(),
    ]
    
    return Compose(transforms)

