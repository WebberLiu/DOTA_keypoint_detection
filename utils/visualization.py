"""
Visualization utilities for keypoint detection results.

This module provides functions to visualize:
- Predicted keypoints overlaid on images
- Training curves and metrics
- Comparison between predictions and ground truth
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2

from config import RESULTS_DIR


def visualize_predictions(
    image: np.ndarray,
    pred_keypoints: List[Tuple],
    gt_keypoints: Optional[np.ndarray] = None,
    heatmap_size: Tuple[int, int] = (64, 64),
    title: str = "Keypoint Predictions"
) -> Figure:
    """
    Visualize predicted and ground truth keypoints on an image.
    
    Args:
        image: RGB image array (H, W, 3)
        pred_keypoints: List of predicted (x, y, confidence) tuples in heatmap coords
        gt_keypoints: Optional array of ground truth keypoints (N, 2) in image coords
        heatmap_size: Size of heatmap for scaling predictions
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Display image
    if image.max() <= 1.0:
        image_display = (image * 255).astype(np.uint8)
    else:
        image_display = image.astype(np.uint8)
    
    ax.imshow(image_display)
    
    # Scale predictions to image coordinates
    img_h, img_w = image.shape[:2]
    scale_x = img_w / heatmap_size[1]
    scale_y = img_h / heatmap_size[0]
    
    # Plot predicted keypoints
    if pred_keypoints:
        pred_x = [kp[0] * scale_x for kp in pred_keypoints]
        pred_y = [kp[1] * scale_y for kp in pred_keypoints]
        pred_conf = [kp[2] for kp in pred_keypoints]
        
        # Filter out keypoints outside image bounds
        valid_preds = []
        for x, y, conf in zip(pred_x, pred_y, pred_conf):
            if 0 <= x < img_w and 0 <= y < img_h:
                valid_preds.append((x, y, conf))
        
        if valid_preds:
            valid_x, valid_y, valid_conf = zip(*valid_preds)
            # Color by confidence
            scatter = ax.scatter(
                valid_x, valid_y,
                c=valid_conf,
                cmap='hot',
                s=200,
                alpha=0.7,
                marker='x',
                linewidths=3,
                label=f'Predicted ({len(valid_preds)})',
                vmin=0,
                vmax=1
            )
            plt.colorbar(scatter, ax=ax, label='Confidence')
        else:
            ax.text(img_w/2, img_h/2, 'No valid predictions\n(Model may need training)', 
                   ha='center', va='center', fontsize=14, color='red',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot ground truth keypoints
    if gt_keypoints is not None and len(gt_keypoints) > 0:
        gt_x = gt_keypoints[:, 0]
        gt_y = gt_keypoints[:, 1]
        ax.scatter(
            gt_x, gt_y,
            c='lime',
            s=150,
            marker='o',
            linewidths=2,
            facecolors='none',
            edgecolors='lime',
            label=f'Ground Truth ({len(gt_keypoints)})'
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    if pred_keypoints or (gt_keypoints is not None and len(gt_keypoints) > 0):
        ax.legend(loc='upper right', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def visualize_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str = "Predicted Heatmap"
) -> Figure:
    """
    Visualize predicted heatmap overlaid on image.
    
    Args:
        image: RGB image array
        heatmap: Predicted heatmap (H, W)
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    if image.max() <= 1.0:
        image_display = (image * 255).astype(np.uint8)
    else:
        image_display = image.astype(np.uint8)
    
    axes[0].imshow(image_display)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Predicted Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(image_display, 0.6, heatmap_colored, 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    metrics_history: Dict[str, List[float]],
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training curves including losses and metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        metrics_history: Dictionary of metric names to values per epoch
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    num_metrics = len(metrics_history)
    num_plots = 1 + num_metrics  # Loss + other metrics
    
    # Create subplots
    fig, axes = plt.subplots(1, min(num_plots, 3), figsize=(6 * min(num_plots, 3), 5))
    if num_plots == 1:
        axes = [axes]
    elif num_plots == 2:
        axes = list(axes)
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot other metrics
    if num_metrics > 0 and len(axes) > 1:
        metric_names = list(metrics_history.keys())[:len(axes) - 1]
        
        for i, metric_name in enumerate(metric_names, start=1):
            values = metrics_history[metric_name]
            axes[i].plot(epochs[:len(values)], values, 'g-^', linewidth=2)
            axes[i].set_xlabel('Epoch', fontsize=12)
            axes[i].set_ylabel(metric_name.upper(), fontsize=12)
            axes[i].set_title(f'{metric_name.upper()} over Epochs', fontsize=13, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def save_prediction_visualization(
    image: np.ndarray,
    pred_heatmap: np.ndarray,
    pred_keypoints: List[Tuple],
    gt_keypoints: Optional[np.ndarray],
    save_path: str,
    heatmap_size: Tuple[int, int] = (64, 64)
):
    """
    Save a comprehensive visualization of predictions.
    
    Args:
        image: Input image
        pred_heatmap: Predicted heatmap
        pred_keypoints: Predicted keypoints
        gt_keypoints: Ground truth keypoints
        save_path: Path to save the visualization
        heatmap_size: Size of heatmap
    """
    fig = plt.figure(figsize=(20, 8))
    
    # Original image with keypoints
    ax1 = plt.subplot(1, 3, 1)
    if image.max() <= 1.0:
        image_display = (image * 255).astype(np.uint8)
    else:
        image_display = image.astype(np.uint8)
    
    ax1.imshow(image_display)
    
    # Scale and plot predictions
    img_h, img_w = image.shape[:2]
    scale_x = img_w / heatmap_size[1]
    scale_y = img_h / heatmap_size[0]
    
    if pred_keypoints:
        pred_x = [kp[0] * scale_x for kp in pred_keypoints]
        pred_y = [kp[1] * scale_y for kp in pred_keypoints]
        pred_conf = [kp[2] for kp in pred_keypoints]
        
        # Filter out keypoints that are outside image bounds (likely false detections)
        valid_preds = []
        for x, y, conf in zip(pred_x, pred_y, pred_conf):
            if 0 <= x < img_w and 0 <= y < img_h:
                valid_preds.append((x, y, conf))
        
        if valid_preds:
            valid_x, valid_y, valid_conf = zip(*valid_preds)
            ax1.scatter(valid_x, valid_y, c=valid_conf, cmap='hot', s=200, 
                       alpha=0.7, marker='x', linewidths=3, label=f'Predicted ({len(valid_preds)})')
        else:
            ax1.text(img_w/2, img_h/2, 'No valid predictions', 
                    ha='center', va='center', fontsize=16, color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if gt_keypoints is not None and len(gt_keypoints) > 0:
        ax1.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], 
                   c='lime', s=150, marker='o', linewidths=2,
                   facecolors='none', edgecolors='lime', label=f'Ground Truth ({len(gt_keypoints)})')
    
    ax1.set_title('Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.axis('off')
    
    # Heatmap
    ax2 = plt.subplot(1, 3, 2)
    im = ax2.imshow(pred_heatmap, cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Predicted Heatmap', fontsize=14, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    # Overlay
    ax3 = plt.subplot(1, 3, 3)
    heatmap_resized = cv2.resize(pred_heatmap, (img_w, img_h))
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_display, 0.6, heatmap_colored, 0.4, 0)
    ax3.imshow(overlay)
    ax3.set_title('Heatmap Overlay', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def create_metrics_summary_plot(
    metrics_dict: Dict[str, float],
    save_path: Optional[str] = None
) -> Figure:
    """
    Create a bar plot summarizing metrics.
    
    Args:
        metrics_dict: Dictionary of metric names and values
        save_path: Optional path to save the figure
    
    Returns:
        Matplotlib figure
    """
    # Filter out non-plottable metrics
    plottable_metrics = {
        k: v for k, v in metrics_dict.items()
        if isinstance(v, (int, float)) and k != "num_samples"
    }
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = list(plottable_metrics.keys())
    metrics_values = list(plottable_metrics.values())
    
    bars = ax.bar(range(len(metrics_names)), metrics_values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(metrics_names)))
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Evaluation Metrics Summary', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{value:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics summary saved to {save_path}")
    
    return fig

