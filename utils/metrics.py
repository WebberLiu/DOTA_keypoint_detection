"""
Metrics for evaluating keypoint detection performance.

This module provides various metrics for assessing the quality of
keypoint predictions, including:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Percentage of Correct Keypoints (PCK) at various thresholds
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment

from config import PCK_THRESHOLDS


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        point1: First point coordinates (x, y)
        point2: Second point coordinates (x, y)
    
    Returns:
        Euclidean distance
    """
    return np.sqrt(np.sum((point1 - point2) ** 2))


def match_keypoints(
    pred_keypoints: List[Tuple],
    gt_keypoints: np.ndarray
) -> List[Tuple[int, int, float]]:
    """
    Match predicted keypoints to ground truth using Hungarian algorithm.
    
    This ensures each prediction is matched to at most one ground truth
    and vice versa, minimizing total matching distance.
    
    Args:
        pred_keypoints: List of predicted (x, y, confidence) tuples
        gt_keypoints: Array of ground truth keypoints (N, 2)
    
    Returns:
        List of (pred_idx, gt_idx, distance) tuples for matched pairs
    """
    if len(pred_keypoints) == 0 or len(gt_keypoints) == 0:
        return []
    
    # Create cost matrix (distances between all pairs)
    n_pred = len(pred_keypoints)
    n_gt = len(gt_keypoints)
    cost_matrix = np.zeros((n_pred, n_gt))
    
    for i, (px, py, _) in enumerate(pred_keypoints):
        for j, (gx, gy) in enumerate(gt_keypoints):
            cost_matrix[i, j] = euclidean_distance(
                np.array([px, py]),
                np.array([gx, gy])
            )
    
    # Solve assignment problem
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Create matches list
    matches = []
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        distance = cost_matrix[pred_idx, gt_idx]
        matches.append((pred_idx, gt_idx, distance))
    
    return matches


def percentage_correct_keypoints(
    pred_keypoints: List[Tuple],
    gt_keypoints: np.ndarray,
    threshold: float,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (64, 64)
) -> float:
    """
    Compute Percentage of Correct Keypoints (PCK) metric.
    
    PCK measures the percentage of predicted keypoints that are within
    a certain threshold distance from the ground truth.
    
    Args:
        pred_keypoints: List of predicted (x, y, confidence) tuples in heatmap coordinates
        gt_keypoints: Array of ground truth keypoints (N, 2) in image coordinates
        threshold: Distance threshold in pixels (in image coordinates)
        image_size: Original image size
        heatmap_size: Heatmap size
    
    Returns:
        PCK value between 0 and 1
    """
    if len(gt_keypoints) == 0:
        return 0.0
    
    # Scale predicted keypoints from heatmap to image coordinates
    scale_x = image_size[1] / heatmap_size[1]
    scale_y = image_size[0] / heatmap_size[0]
    
    scaled_pred_keypoints = [
        (px * scale_x, py * scale_y, conf)
        for px, py, conf in pred_keypoints
    ]
    
    # Match predictions to ground truth
    matches = match_keypoints(scaled_pred_keypoints, gt_keypoints)
    
    # Count correct predictions (within threshold)
    correct = sum(1 for _, _, dist in matches if dist <= threshold)
    
    # PCK = correct predictions / total ground truth keypoints
    pck = correct / len(gt_keypoints)
    
    return pck


def mean_absolute_error(
    pred_keypoints: List[Tuple],
    gt_keypoints: np.ndarray,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (64, 64)
) -> float:
    """
    Compute Mean Absolute Error (MAE) for matched keypoints.
    
    Args:
        pred_keypoints: List of predicted (x, y, confidence) tuples
        gt_keypoints: Array of ground truth keypoints (N, 2)
        image_size: Original image size
        heatmap_size: Heatmap size
    
    Returns:
        MAE value (average distance in pixels)
    """
    if len(gt_keypoints) == 0:
        return 0.0
    
    # Scale predicted keypoints to image coordinates
    scale_x = image_size[1] / heatmap_size[1]
    scale_y = image_size[0] / heatmap_size[0]
    
    scaled_pred_keypoints = [
        (px * scale_x, py * scale_y, conf)
        for px, py, conf in pred_keypoints
    ]
    
    # Match predictions to ground truth
    matches = match_keypoints(scaled_pred_keypoints, gt_keypoints)
    
    if len(matches) == 0:
        # If no matches, return large error based on unmatched GT points
        return float(image_size[0])  # Return image height as penalty
    
    # Compute average distance for matched pairs
    mae = np.mean([dist for _, _, dist in matches])
    
    return mae


def mean_squared_error(
    pred_keypoints: List[Tuple],
    gt_keypoints: np.ndarray,
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (64, 64)
) -> float:
    """
    Compute Mean Squared Error (MSE) for matched keypoints.
    
    Args:
        pred_keypoints: List of predicted (x, y, confidence) tuples
        gt_keypoints: Array of ground truth keypoints (N, 2)
        image_size: Original image size
        heatmap_size: Heatmap size
    
    Returns:
        MSE value (average squared distance)
    """
    if len(gt_keypoints) == 0:
        return 0.0
    
    # Scale predicted keypoints to image coordinates
    scale_x = image_size[1] / heatmap_size[1]
    scale_y = image_size[0] / heatmap_size[0]
    
    scaled_pred_keypoints = [
        (px * scale_x, py * scale_y, conf)
        for px, py, conf in pred_keypoints
    ]
    
    # Match predictions to ground truth
    matches = match_keypoints(scaled_pred_keypoints, gt_keypoints)
    
    if len(matches) == 0:
        return float(image_size[0] ** 2)  # Return squared image dimension as penalty
    
    # Compute average squared distance for matched pairs
    mse = np.mean([dist ** 2 for _, _, dist in matches])
    
    return mse


def compute_metrics(
    pred_keypoints_batch: List[List[Tuple]],
    gt_keypoints_batch: List[np.ndarray],
    image_size: Tuple[int, int] = (512, 512),
    heatmap_size: Tuple[int, int] = (64, 64)
) -> Dict[str, float]:
    """
    Compute all metrics for a batch of predictions.
    
    Args:
        pred_keypoints_batch: List of predicted keypoints for each image
        gt_keypoints_batch: List of ground truth keypoints for each image
        image_size: Original image size
        heatmap_size: Heatmap size
    
    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {
        "mae": [],
        "mse": [],
    }
    
    # Add PCK metrics for different thresholds
    for threshold in PCK_THRESHOLDS:
        metrics[f"pck@{threshold}"] = []
    
    # Compute metrics for each sample in batch
    for pred_kps, gt_kps in zip(pred_keypoints_batch, gt_keypoints_batch):
        # MAE and MSE
        metrics["mae"].append(mean_absolute_error(pred_kps, gt_kps, image_size, heatmap_size))
        metrics["mse"].append(mean_squared_error(pred_kps, gt_kps, image_size, heatmap_size))
        
        # PCK at different thresholds
        for threshold in PCK_THRESHOLDS:
            pck = percentage_correct_keypoints(pred_kps, gt_kps, threshold, image_size, heatmap_size)
            metrics[f"pck@{threshold}"].append(pck)
    
    # Average metrics across batch
    averaged_metrics = {}
    for key, values in metrics.items():
        if len(values) > 0:
            averaged_metrics[key] = np.mean(values)
        else:
            averaged_metrics[key] = 0.0
    
    # Add additional statistics
    averaged_metrics["num_samples"] = len(pred_keypoints_batch)
    averaged_metrics["avg_pred_keypoints"] = np.mean([len(kps) for kps in pred_keypoints_batch])
    averaged_metrics["avg_gt_keypoints"] = np.mean([len(kps) for kps in gt_keypoints_batch])
    
    return averaged_metrics


def heatmap_loss_to_metrics(loss_value: float) -> Dict[str, float]:
    """
    Convert heatmap loss to a metrics dictionary.
    
    Args:
        loss_value: Loss value from training
    
    Returns:
        Dictionary with loss metric
    """
    return {"loss": loss_value}

