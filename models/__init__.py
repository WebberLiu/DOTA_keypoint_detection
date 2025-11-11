"""
Models module for DOTA keypoint detection.
Contains the neural network architecture for keypoint heatmap prediction.
"""

from .keypoint_detector import KeypointDetector, create_model

__all__ = ["KeypointDetector", "create_model"]

