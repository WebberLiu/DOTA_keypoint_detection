"""
Metaflow-based inference pipeline for DOTA keypoint detection.

This is a BONUS implementation using Metaflow for workflow orchestration,
providing better scalability, reproducibility, and deployment capabilities.

**IMPORTANT:** Metaflow has limited Windows support due to Unix-specific dependencies.
- On Windows: Use inference.py instead, or run this script in WSL/Docker
- On Linux/Mac: This script works out of the box

Features:
- Structured workflow with clear steps
- Built-in versioning and experiment tracking
- Easy deployment to production
- Scalable to large volumes of images

Usage:
    # Run locally (Linux/Mac or WSL)
    python inference_flow.py run --input-dir images/
    
    # Run with cloud deployment
    python inference_flow.py run --input-dir images/ --with batch
    
    # Windows alternative
    python inference.py --input images/ --output predictions/
"""

from metaflow import FlowSpec, step, Parameter, current, batch, resources
from pathlib import Path
from typing import List, Dict
import json
import time
import glob

import torch
import numpy as np
from PIL import Image

from config import (
    DATA_CONFIG, MODEL_CONFIG, INFERENCE_CONFIG,
    CHECKPOINTS_DIR
)
from models import create_model
from data.transforms import Normalize


class KeypointInferenceFlow(FlowSpec):
    """
    Metaflow workflow for keypoint detection inference.
    
    This flow implements a production-ready inference pipeline with:
    - Preprocessing step for batch image loading
    - Inference step with model predictions
    - Postprocessing step for result aggregation
    - Monitoring and logging throughout
    """
    
    # Flow parameters
    input_dir = Parameter(
        'input-dir',
        help='Directory containing input images',
        required=True
    )
    
    output_dir = Parameter(
        'output-dir',
        help='Directory to save predictions',
        default='predictions_metaflow'
    )
    
    checkpoint_path = Parameter(
        'checkpoint',
        help='Path to model checkpoint',
        default=str(CHECKPOINTS_DIR / "best_model.pth")
    )
    
    batch_size = Parameter(
        'batch-size',
        help='Batch size for inference',
        default=INFERENCE_CONFIG["batch_size"]
    )
    
    confidence_threshold = Parameter(
        'confidence',
        help='Confidence threshold for detections',
        default=INFERENCE_CONFIG["confidence_threshold"]
    )
    
    device = Parameter(
        'device',
        help='Device to use (cuda or cpu)',
        default='cuda'
    )
    
    @step
    def start(self):
        """
        Initialize the flow and validate inputs.
        """
        print(f"\n{'='*70}")
        print("KEYPOINT DETECTION INFERENCE FLOW")
        print(f"{'='*70}\n")
        
        print(f"Flow Configuration:")
        print(f"  Input directory: {self.input_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Device: {self.device}\n")
        
        # Validate inputs
        input_path = Path(self.input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        
        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Store metadata
        self.flow_start_time = time.time()
        self.run_id = current.run_id
        
        print(f"Flow initialized successfully (Run ID: {self.run_id})")
        
        self.next(self.load_images)
    
    @step
    def load_images(self):
        """
        Load and preprocess images for inference.
        """
        print("\n--- Step: Load Images ---")
        
        # Find all images
        input_path = Path(self.input_dir)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(str(input_path / f"*{ext}")))
            image_paths.extend(glob.glob(str(input_path / f"*{ext.upper()}")))
        
        if len(image_paths) == 0:
            raise ValueError(f"No images found in {self.input_dir}")
        
        self.image_paths = sorted(image_paths)
        self.num_images = len(self.image_paths)
        
        print(f"Found {self.num_images} images to process")
        
        # Store preprocessing config
        self.image_size = DATA_CONFIG["image_size"]
        self.heatmap_size = MODEL_CONFIG["heatmap_size"]
        
        self.next(self.run_inference)
    
    @step
    def run_inference(self):
        """
        Run model inference on all images.
        
        This step can be scaled to cloud resources using @batch decorator
        for processing large volumes of images.
        """
        print("\n--- Step: Run Inference ---")
        
        # Determine device
        device = self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Load model
        print(f"Loading model from {self.checkpoint_path}...")
        model, _ = create_model(device=device)
        checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("  Model loaded successfully")
        
        # Initialize preprocessing
        normalize = Normalize()
        
        # Process images in batches
        all_results = []
        total_inference_time = 0.0
        
        print(f"\nProcessing {self.num_images} images in batches of {self.batch_size}...")
        
        for i in range(0, self.num_images, self.batch_size):
            batch_paths = self.image_paths[i:i + self.batch_size]
            
            # Preprocess batch
            batch_images = []
            batch_metadata = []
            
            for img_path in batch_paths:
                # Load and preprocess
                image = Image.open(img_path).convert("RGB")
                original_size = image.size
                
                image = image.resize(self.image_size)
                image_array = np.array(image, dtype=np.float32) / 255.0
                image_normalized = normalize(image_array)
                image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
                
                batch_images.append(image_tensor)
                batch_metadata.append({
                    "image_path": img_path,
                    "original_size": original_size
                })
            
            # Stack batch
            batch_tensor = torch.stack(batch_images, dim=0).to(device)
            
            # Inference
            start_time = time.time()
            with torch.no_grad():
                heatmaps = model(batch_tensor)
                keypoints_batch = model.extract_keypoints(
                    heatmaps,
                    threshold=self.confidence_threshold,
                    nms_threshold=INFERENCE_CONFIG["nms_threshold"]
                )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            # Process results
            for img_path, metadata, keypoints in zip(batch_paths, batch_metadata, keypoints_batch):
                # Scale keypoints to original size
                scale_x = metadata["original_size"][0] / self.heatmap_size[1]
                scale_y = metadata["original_size"][1] / self.heatmap_size[0]
                
                scaled_keypoints = [
                    {
                        "x": float(x * scale_x),
                        "y": float(y * scale_y),
                        "confidence": float(conf)
                    }
                    for x, y, conf in keypoints
                ]
                
                result = {
                    "image_path": img_path,
                    "image_name": Path(img_path).name,
                    "original_size": {
                        "width": metadata["original_size"][0],
                        "height": metadata["original_size"][1]
                    },
                    "num_keypoints": len(scaled_keypoints),
                    "keypoints": scaled_keypoints
                }
                
                all_results.append(result)
            
            # Progress update
            processed = min(i + self.batch_size, self.num_images)
            print(f"  Processed {processed}/{self.num_images} images...")
        
        # Store results
        self.predictions = all_results
        self.total_inference_time = total_inference_time
        self.avg_inference_time = total_inference_time / self.num_images
        self.throughput = self.num_images / total_inference_time
        
        print(f"\nInference completed:")
        print(f"  Total time: {self.total_inference_time:.2f}s")
        print(f"  Avg time per image: {self.avg_inference_time * 1000:.2f}ms")
        print(f"  Throughput: {self.throughput:.2f} images/sec")
        
        self.next(self.save_results)
    
    @step
    def save_results(self):
        """
        Save prediction results and generate summary.
        """
        print("\n--- Step: Save Results ---")
        
        # Create output directory
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save all predictions
        predictions_file = output_path / "predictions.json"
        with open(predictions_file, 'w') as f:
            json.dump(self.predictions, f, indent=2)
        print(f"  Predictions saved to: {predictions_file}")
        
        # Save individual predictions
        individual_dir = output_path / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        for result in self.predictions:
            image_name = Path(result["image_name"]).stem
            pred_path = individual_dir / f"{image_name}_predictions.json"
            with open(pred_path, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"  Individual predictions saved to: {individual_dir}")
        
        # Generate summary statistics
        num_keypoints = [r["num_keypoints"] for r in self.predictions]
        
        summary = {
            "flow_metadata": {
                "run_id": self.run_id,
                "input_directory": self.input_dir,
                "output_directory": self.output_dir,
                "checkpoint": self.checkpoint_path,
                "total_runtime": time.time() - self.flow_start_time
            },
            "processing_stats": {
                "total_images": self.num_images,
                "total_inference_time": self.total_inference_time,
                "avg_time_per_image_ms": self.avg_inference_time * 1000,
                "throughput_images_per_sec": self.throughput
            },
            "detection_stats": {
                "avg_keypoints_per_image": float(np.mean(num_keypoints)),
                "min_keypoints": int(np.min(num_keypoints)),
                "max_keypoints": int(np.max(num_keypoints)),
                "std_keypoints": float(np.std(num_keypoints)),
                "total_keypoints": int(np.sum(num_keypoints))
            }
        }
        
        # Save summary as JSON
        summary_json_file = output_path / "summary.json"
        with open(summary_json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary as text
        summary_txt_file = output_path / "summary.txt"
        with open(summary_txt_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("KEYPOINT DETECTION INFERENCE SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("Flow Metadata:\n")
            f.write(f"  Run ID: {summary['flow_metadata']['run_id']}\n")
            f.write(f"  Input directory: {summary['flow_metadata']['input_directory']}\n")
            f.write(f"  Output directory: {summary['flow_metadata']['output_directory']}\n")
            f.write(f"  Total runtime: {summary['flow_metadata']['total_runtime']:.2f}s\n\n")
            
            f.write("Processing Statistics:\n")
            f.write(f"  Total images: {summary['processing_stats']['total_images']}\n")
            f.write(f"  Total inference time: {summary['processing_stats']['total_inference_time']:.2f}s\n")
            f.write(f"  Avg time per image: {summary['processing_stats']['avg_time_per_image_ms']:.2f}ms\n")
            f.write(f"  Throughput: {summary['processing_stats']['throughput_images_per_sec']:.2f} images/sec\n\n")
            
            f.write("Detection Statistics:\n")
            f.write(f"  Avg keypoints per image: {summary['detection_stats']['avg_keypoints_per_image']:.2f}\n")
            f.write(f"  Min keypoints: {summary['detection_stats']['min_keypoints']}\n")
            f.write(f"  Max keypoints: {summary['detection_stats']['max_keypoints']}\n")
            f.write(f"  Std keypoints: {summary['detection_stats']['std_keypoints']:.2f}\n")
            f.write(f"  Total keypoints detected: {summary['detection_stats']['total_keypoints']}\n")
        
        print(f"  Summary saved to: {summary_txt_file}")
        
        self.summary = summary
        
        self.next(self.end)
    
    @step
    def end(self):
        """
        Finalize the flow and print completion message.
        """
        print(f"\n{'='*70}")
        print("FLOW COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nRun ID: {self.run_id}")
        print(f"Processed: {self.num_images} images")
        print(f"Output: {self.output_dir}")
        print(f"Total Runtime: {time.time() - self.flow_start_time:.2f}s")
        print(f"\n{'='*70}\n")


if __name__ == '__main__':
    KeypointInferenceFlow()

