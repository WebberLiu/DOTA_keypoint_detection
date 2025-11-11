"""
Inference pipeline for DOTA keypoint detection.

This script provides a streamlined interface for running inference on
new batches of images. It handles:
- Image loading and preprocessing
- Batch inference
- Keypoint extraction
- Results output in organized format

Usage:
    python inference.py --input images/ --output predictions/ --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import glob

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

from config import (
    DATA_CONFIG, MODEL_CONFIG, INFERENCE_CONFIG,
    CHECKPOINTS_DIR
)
from models import create_model
from data.transforms import Normalize


class InferencePipeline:
    """
    Inference pipeline for keypoint detection.
    
    This class provides a complete inference pipeline that:
    1. Loads images from disk
    2. Preprocesses them (resize, normalize)
    3. Runs batch inference
    4. Extracts keypoints from heatmaps
    5. Saves results in structured format
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
        confidence_threshold: float = 0.3,
        nms_threshold: int = 10
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
            batch_size: Batch size for inference
            confidence_threshold: Minimum confidence for keypoint detection
            nms_threshold: NMS distance threshold
        """
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Image and heatmap sizes
        self.image_size = DATA_CONFIG["image_size"]
        self.heatmap_size = MODEL_CONFIG["heatmap_size"]
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model, _ = create_model(device=self.device)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model.eval()
        
        # Preprocessing
        self.normalize = Normalize()
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Dict]:
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (preprocessed tensor, metadata dict)
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        # Resize
        image = image.resize(self.image_size)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Normalize
        image_normalized = self.normalize(image_array)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)  # HWC -> CHW
        
        metadata = {
            "image_path": image_path,
            "original_size": original_size,
            "processed_size": self.image_size,
        }
        
        return image_tensor, metadata
    
    def preprocess_batch(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image paths
        
        Returns:
            Tuple of (batched tensor, list of metadata dicts)
        """
        images = []
        metadata_list = []
        
        for img_path in image_paths:
            img_tensor, metadata = self.preprocess_image(img_path)
            images.append(img_tensor)
            metadata_list.append(metadata)
        
        # Stack into batch
        batch_tensor = torch.stack(images, dim=0)
        
        return batch_tensor, metadata_list
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[List[Tuple]]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: Batch of preprocessed images (B, 3, H, W)
        
        Returns:
            Tuple of (heatmaps, keypoints)
        """
        with torch.no_grad():
            images = images.to(self.device)
            
            # Forward pass
            heatmaps = self.model(images)
            
            # Extract keypoints
            keypoints = self.model.extract_keypoints(
                heatmaps,
                threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold
            )
        
        return heatmaps, keypoints
    
    def scale_keypoints_to_original(
        self,
        keypoints: List[Tuple],
        original_size: Tuple[int, int]
    ) -> List[Dict]:
        """
        Scale keypoints from heatmap coordinates to original image coordinates.
        
        Args:
            keypoints: List of (x, y, confidence) tuples in heatmap coordinates
            original_size: Original image size (width, height)
        
        Returns:
            List of keypoint dictionaries with original coordinates
        """
        # Scale factors
        scale_x = original_size[0] / self.heatmap_size[1]
        scale_y = original_size[1] / self.heatmap_size[0]
        
        scaled_keypoints = []
        for x, y, conf in keypoints:
            scaled_keypoints.append({
                "x": float(x * scale_x),
                "y": float(y * scale_y),
                "confidence": float(conf)
            })
        
        return scaled_keypoints
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: List[str] = None
    ) -> List[Dict]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            image_extensions: List of image extensions to process
        
        Returns:
            List of prediction results
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        # Find all images
        input_path = Path(input_dir)
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(input_path / f"*{ext}")))
            image_paths.extend(glob.glob(str(input_path / f"*{ext.upper()}")))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {input_dir}")
        
        print(f"\nFound {len(image_paths)} images to process")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Process in batches
        all_results = []
        total_time = 0.0
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + self.batch_size]
            
            # Preprocess
            batch_images, batch_metadata = self.preprocess_batch(batch_paths)
            
            # Inference
            start_time = time.time()
            heatmaps, keypoints_batch = self.predict(batch_images)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Process results
            for img_path, metadata, keypoints, heatmap in zip(
                batch_paths, batch_metadata, keypoints_batch, heatmaps
            ):
                # Scale keypoints to original image size
                scaled_keypoints = self.scale_keypoints_to_original(
                    keypoints,
                    metadata["original_size"]
                )
                
                result = {
                    "image_path": img_path,
                    "image_name": Path(img_path).name,
                    "original_size": {
                        "width": metadata["original_size"][0],
                        "height": metadata["original_size"][1]
                    },
                    "num_keypoints": len(scaled_keypoints),
                    "keypoints": scaled_keypoints,
                    "inference_time": inference_time / len(batch_paths)
                }
                
                all_results.append(result)
        
        # Save results
        self._save_results(all_results, output_path, total_time)
        
        return all_results
    
    def _save_results(self, results: List[Dict], output_dir: Path, total_time: float):
        """Save inference results to files."""
        # Save all results as JSON
        results_json_path = output_dir / "predictions.json"
        with open(results_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to: {results_json_path}")
        
        # Save individual predictions
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for result in results:
            image_name = Path(result["image_name"]).stem
            pred_path = individual_dir / f"{image_name}_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"  Individual predictions saved to: {individual_dir}")
        
        # Save summary statistics
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write("INFERENCE SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Total inference time: {total_time:.2f}s\n")
            f.write(f"Average time per image: {total_time / len(results) * 1000:.2f}ms\n")
            f.write(f"Throughput: {len(results) / total_time:.2f} images/sec\n\n")
            
            f.write(f"Keypoint Statistics:\n")
            num_keypoints = [r["num_keypoints"] for r in results]
            f.write(f"  Average keypoints per image: {np.mean(num_keypoints):.2f}\n")
            f.write(f"  Min keypoints: {np.min(num_keypoints)}\n")
            f.write(f"  Max keypoints: {np.max(num_keypoints)}\n")
            f.write(f"  Std keypoints: {np.std(num_keypoints):.2f}\n")
        
        print(f"  Summary saved to: {summary_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Run inference on images")
    parser.add_argument("--input", type=str, required=True,
                       help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory for predictions")
    parser.add_argument("--checkpoint", type=str,
                       default=str(CHECKPOINTS_DIR / "best_model.pth"),
                       help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int,
                       default=INFERENCE_CONFIG["batch_size"],
                       help="Batch size for inference")
    parser.add_argument("--device", type=str,
                       default=INFERENCE_CONFIG["device"],
                       help="Device to use (cuda or cpu)")
    parser.add_argument("--confidence", type=float,
                       default=INFERENCE_CONFIG["confidence_threshold"],
                       help="Confidence threshold for keypoint detection")
    parser.add_argument("--nms", type=int,
                       default=INFERENCE_CONFIG["nms_threshold"],
                       help="NMS distance threshold")
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = InferencePipeline(
        model_path=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms
    )
    
    # Run inference
    results = pipeline.process_directory(
        input_dir=args.input,
        output_dir=args.output
    )
    
    print(f"\n{'='*60}")
    print(f"Inference completed successfully!")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

