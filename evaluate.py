"""
Evaluation script for DOTA keypoint detection model.

This script:
- Loads a trained model
- Evaluates on test set
- Computes comprehensive metrics
- Generates visualizations
- Saves results summary

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import random

import torch
import numpy as np
from tqdm import tqdm

from config import (
    DATA_CONFIG, MODEL_CONFIG, INFERENCE_CONFIG,
    DATA_DIR, RESULTS_DIR, CHECKPOINTS_DIR
)
from data import create_dataloaders, get_val_transforms
from models import create_model
from utils.metrics import compute_metrics
from utils.visualization import (
    save_prediction_visualization,
    create_metrics_summary_plot
)


class Evaluator:
    """
    Evaluator class for keypoint detection model.
    
    Handles:
    - Model evaluation on test set
    - Metrics computation
    - Results visualization
    - Report generation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        results_dir: str
    ):
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        self.model.eval()
    
    def evaluate(
        self,
        test_loader,
        save_visualizations: bool = True,
        num_visualizations: int = 10
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            save_visualizations: Whether to save prediction visualizations
            num_visualizations: Number of samples to visualize
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        total_inference_time = 0.0
        all_pred_keypoints = []
        all_gt_keypoints = []
        all_images = []
        all_pred_heatmaps = []
        
        with torch.no_grad():
            for batch_idx, (images, heatmaps, metadata) in enumerate(tqdm(test_loader, desc="Evaluating")):
                # Move to device
                images_gpu = images.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                pred_heatmaps = self.model(images_gpu)
                torch.cuda.synchronize() if self.device == "cuda" else None
                inference_time = time.time() - start_time
                total_inference_time += inference_time
                
                # Extract keypoints
                pred_keypoints = self.model.extract_keypoints(
                    pred_heatmaps,
                    threshold=INFERENCE_CONFIG["confidence_threshold"],
                    nms_threshold=INFERENCE_CONFIG["nms_threshold"]
                )
                
                # Collect results
                all_pred_keypoints.extend(pred_keypoints)
                all_gt_keypoints.extend([np.array(m["keypoints"]) for m in metadata])
                
                # Store some samples for visualization
                if save_visualizations and len(all_images) < num_visualizations:
                    for i in range(len(images)):
                        if len(all_images) < num_visualizations:
                            all_images.append(images[i].cpu().numpy())
                            all_pred_heatmaps.append(pred_heatmaps[i].cpu().numpy())
        
        # Compute metrics
        print("\nComputing metrics...")
        metrics = compute_metrics(
            all_pred_keypoints,
            all_gt_keypoints,
            image_size=DATA_CONFIG["image_size"],
            heatmap_size=MODEL_CONFIG["heatmap_size"]
        )
        
        # Add timing metrics
        metrics["total_samples"] = len(all_pred_keypoints)
        metrics["total_inference_time"] = total_inference_time
        metrics["avg_inference_time"] = total_inference_time / len(all_pred_keypoints)
        metrics["throughput"] = len(all_pred_keypoints) / total_inference_time
        
        # Print results
        self._print_results(metrics)
        
        # Save visualizations
        if save_visualizations:
            print(f"\nSaving {min(num_visualizations, len(all_images))} sample visualizations...")
            self._save_visualizations(
                all_images[:num_visualizations],
                all_pred_heatmaps[:num_visualizations],
                all_pred_keypoints[:num_visualizations],
                all_gt_keypoints[:num_visualizations]
            )
        
        # Save metrics summary
        self._save_results(metrics)
        
        return metrics
    
    def _print_results(self, metrics: Dict):
        """Print evaluation results in a formatted manner."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Performance metrics
        print("\n--- Detection Performance ---")
        print(f"  Mean Absolute Error (MAE):     {metrics.get('mae', 0):.2f} pixels")
        print(f"  Mean Squared Error (MSE):      {metrics.get('mse', 0):.2f}")
        print(f"  Root Mean Squared Error (RMSE): {np.sqrt(metrics.get('mse', 0)):.2f} pixels")
        
        # PCK metrics
        print("\n--- Percentage of Correct Keypoints (PCK) ---")
        for threshold in [5, 10, 15, 20]:
            key = f"pck@{threshold}"
            if key in metrics:
                print(f"  PCK@{threshold}px:  {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
        
        # Detection statistics
        print("\n--- Detection Statistics ---")
        print(f"  Total samples:                 {metrics.get('total_samples', 0)}")
        print(f"  Avg predicted keypoints/image: {metrics.get('avg_pred_keypoints', 0):.2f}")
        print(f"  Avg ground truth keypoints/image: {metrics.get('avg_gt_keypoints', 0):.2f}")
        
        # Timing metrics
        print("\n--- Performance Metrics ---")
        print(f"  Total inference time:          {metrics.get('total_inference_time', 0):.2f}s")
        print(f"  Avg inference time/image:      {metrics.get('avg_inference_time', 0)*1000:.2f}ms")
        print(f"  Throughput:                    {metrics.get('throughput', 0):.2f} images/sec")
        
        print("="*70 + "\n")
    
    def _save_visualizations(
        self,
        images: List[np.ndarray],
        heatmaps: List[np.ndarray],
        pred_keypoints: List[List],
        gt_keypoints: List[np.ndarray]
    ):
        """Save prediction visualizations."""
        viz_dir = self.results_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        for i, (img, heatmap, pred_kp, gt_kp) in enumerate(
            zip(images, heatmaps, pred_keypoints, gt_keypoints)
        ):
            # Denormalize image (undo ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_denorm = img * std + mean
            img_denorm = np.clip(img_denorm, 0, 1)
            img_denorm = img_denorm.transpose(1, 2, 0)  # CHW -> HWC
            
            # Save visualization
            save_path = viz_dir / f"prediction_{i+1:03d}.png"
            save_prediction_visualization(
                image=img_denorm,
                pred_heatmap=heatmap[0],
                pred_keypoints=pred_kp,
                gt_keypoints=gt_kp,
                save_path=str(save_path),
                heatmap_size=MODEL_CONFIG["heatmap_size"]
            )
        
        print(f"  Visualizations saved to: {viz_dir}")
    
    def _save_results(self, metrics: Dict):
        """Save evaluation results to files."""
        # Save metrics as JSON
        metrics_path = self.results_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)
        print(f"\n  Metrics saved to: {metrics_path}")
        
        # Create and save metrics summary plot
        plot_path = self.results_dir / "metrics_summary.png"
        create_metrics_summary_plot(metrics, save_path=str(plot_path))
        
        # Save text report
        report_path = self.results_dir / "evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write("DOTA KEYPOINT DETECTION - EVALUATION REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Model Configuration:\n")
            f.write(f"  Backbone: {MODEL_CONFIG['backbone']}\n")
            f.write(f"  Heatmap Size: {MODEL_CONFIG['heatmap_size']}\n")
            f.write(f"  Image Size: {DATA_CONFIG['image_size']}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  MAE: {metrics.get('mae', 0):.2f} pixels\n")
            f.write(f"  MSE: {metrics.get('mse', 0):.2f}\n")
            f.write(f"  RMSE: {np.sqrt(metrics.get('mse', 0)):.2f} pixels\n\n")
            
            f.write("PCK Metrics:\n")
            for threshold in [5, 10, 15, 20]:
                key = f"pck@{threshold}"
                if key in metrics:
                    f.write(f"  PCK@{threshold}px: {metrics[key]:.4f}\n")
            
            f.write(f"\nInference Performance:\n")
            f.write(f"  Avg time per image: {metrics.get('avg_inference_time', 0)*1000:.2f}ms\n")
            f.write(f"  Throughput: {metrics.get('throughput', 0):.2f} images/sec\n")
        
        print(f"  Report saved to: {report_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate DOTA keypoint detection model")
    parser.add_argument("--checkpoint", type=str,
                       default=str(CHECKPOINTS_DIR / "best_model.pth"),
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default=INFERENCE_CONFIG["device"],
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--visualizations", type=int, default=10,
                       help="Number of samples to visualize")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                       help="Directory to save results")
    args = parser.parse_args()
    
    # Set device
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data
    print("Loading test data...")
    _, _, test_loader = create_dataloaders(
        data_dir=str(DATA_DIR),
        train_transform=None,
        val_transform=get_val_transforms(),
        generate_synthetic=False  # Will auto-fallback to synthetic if DOTA not found
    )
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, _ = create_model(device=device)
    
    if Path(args.checkpoint).exists():
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best validation loss: {checkpoint.get('best_val_loss', 'unknown')}")
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}")
        print("Evaluating with randomly initialized model...")
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        device=device,
        results_dir=args.results_dir
    )
    
    # Evaluate
    metrics = evaluator.evaluate(
        test_loader=test_loader,
        save_visualizations=True,
        num_visualizations=args.visualizations
    )
    
    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()

