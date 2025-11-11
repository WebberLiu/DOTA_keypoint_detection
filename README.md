# DOTA Keypoint Detection

A complete machine learning pipeline for detecting keypoints (bounding box centers) in aerial imagery from the DOTA dataset. This project demonstrates a production-ready approach to training, evaluating, and deploying keypoint detection models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
  - [Metaflow Pipeline (Bonus)](#metaflow-pipeline-bonus)
- [Configuration](#configuration)
- [Results](#results)
- [Scalability & Production Considerations](#scalability--production-considerations)
- [Future Improvements](#future-improvements)

## 🎯 Overview

This project implements an end-to-end pipeline for keypoint detection on aerial images. Instead of traditional bounding box detection, we focus on detecting the center point of each object, which is useful for:

- Object localization in aerial imagery
- Counting objects in large-scale images
- Tracking object positions over time
- Distance calculation between objects

The pipeline uses a CNN-based encoder-decoder architecture that predicts heatmaps, where Gaussian peaks indicate keypoint locations.

## ✨ Features

- **Complete Training Pipeline**: Data loading, augmentation, training loop, validation, and checkpointing
- **Comprehensive Evaluation**: Multiple metrics (MAE, MSE, PCK@multiple thresholds)
- **Production-Ready Inference**: Batch processing with preprocessing and postprocessing
- **Metaflow Integration** (Bonus): Workflow orchestration for scalable, reproducible inference
- **Visualization Tools**: Training curves, prediction overlays, heatmap visualizations
- **Modular Design**: Clean, maintainable code with proper documentation
- **TensorBoard Logging**: Real-time training monitoring
- **Synthetic Data Generation**: Built-in synthetic data for demonstration

## 📁 Project Structure

```
DOTA_keypoint_detection/
├── README.md                    # This file
├── RESULTS.md                   # Results and discussion
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and hyperparameters
│
├── data/                        # Data pipeline
│   ├── __init__.py
│   ├── dataset.py              # Dataset class with synthetic data generation
│   └── transforms.py           # Data augmentation and preprocessing
│
├── models/                      # Model architecture
│   ├── __init__.py
│   └── keypoint_detector.py    # CNN-based keypoint detection model
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics (MAE, MSE, PCK)
│   └── visualization.py        # Plotting and visualization functions
│
├── train.py                     # Training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Standard inference pipeline
├── inference_flow.py            # Metaflow-based inference pipeline
│
├── checkpoints/                 # Saved model checkpoints (created at runtime)
├── logs/                        # TensorBoard logs (created at runtime)
├── results/                     # Evaluation results (created at runtime)
└── data_samples/                # Dataset directory (created at runtime)
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster training)
- 4GB+ RAM
- 2GB+ disk space

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DOTA_keypoint_detection
   ```

2. **Create a virtual environment**
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 🚀 Quick Start

### Run Complete Pipeline (Training → Evaluation → Inference)

```bash
# 1. Train the model (uses synthetic data by default)
python train.py --epochs 10

# 2. Evaluate on test set
python evaluate.py --checkpoint checkpoints/best_model.pth

# 3. Run inference on new images
python inference.py --input data_samples/synthetic/test/ --output predictions/
```

### 🎯 Using Real DOTA Dataset (Recommended)

To use the actual DOTA dataset instead of synthetic data:

**1. Download DOTA dataset:**
- Visit: https://captain-whu.github.io/DOTA/dataset.html
- Download DOTA-v1.0 (2,806 images, ~2.5GB)
- See [DOTA_SETUP.md](DOTA_SETUP.md) for detailed instructions

**2. Organize the data:**
```bash
mkdir -p data_samples/dota
# Copy images/ and labelTxt/ folders to data_samples/dota/
```

**3. Update config for small subset (recommended for testing):**
```python
# In config.py
DATA_CONFIG = {
    # ... other settings ...
    "max_samples": 200,  # Use only 200 images for quick testing
    # "max_samples": None,  # Use all images for full training
}

DATA_DIR = PROJECT_ROOT / "data_samples" / "dota"
```

**4. Train with DOTA data:**
```bash
python train.py --epochs 15
# The code automatically detects DOTA data and uses it!
```

📚 **Complete Guide:** See [DOTA_SETUP.md](DOTA_SETUP.md) for detailed instructions on downloading, setting up, and using the DOTA dataset.

## 📖 Usage

### Training

Train a keypoint detection model from scratch:

```bash
# Basic training
python train.py

# Custom configuration
python train.py --epochs 20 --batch-size 16 --lr 0.001 --device cuda

# Resume from checkpoint
python train.py --checkpoint checkpoints/checkpoint_epoch_5.pth
```

**What happens during training:**
- Synthetic data is generated (or real DOTA data is loaded if available)
- Model trains with data augmentation
- Validation runs after each epoch
- Metrics are logged to TensorBoard
- Best model is saved based on validation loss
- Training curves are saved as images

**Monitor training:**
```bash
tensorboard --logdir logs/
```

### Evaluation

Evaluate a trained model on the test set:

```bash
# Evaluate best model
python evaluate.py --checkpoint checkpoints/best_model.pth

# Evaluate specific checkpoint
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_10.pth

# Generate more visualizations
python evaluate.py --visualizations 20 --results-dir results/evaluation_v2
```

**Evaluation outputs:**
- `evaluation_metrics.json`: All metrics in JSON format
- `evaluation_report.txt`: Human-readable report
- `metrics_summary.png`: Bar plot of metrics
- `visualizations/`: Prediction overlays on images

**Metrics computed:**
- **MAE (Mean Absolute Error)**: Average pixel distance between predicted and ground truth keypoints
- **MSE (Mean Squared Error)**: Average squared distance
- **PCK@N** (Percentage of Correct Keypoints): Percentage of predictions within N pixels of ground truth
  - PCK@5, PCK@10, PCK@15, PCK@20
- **Inference time**: Time per image and throughput

### Inference

Run inference on new images:

```bash
# Basic inference
python inference.py --input path/to/images/ --output predictions/

# Custom configuration
python inference.py \
    --input images/ \
    --output predictions/ \
    --checkpoint checkpoints/best_model.pth \
    --batch-size 32 \
    --confidence 0.3 \
    --device cuda
```

**Inference outputs:**
- `predictions.json`: All predictions in one file
- `individual/`: Individual JSON files per image
- `summary.txt`: Statistics summary

**Prediction format:**
```json
{
  "image_name": "sample_001.png",
  "original_size": {"width": 1024, "height": 768},
  "num_keypoints": 5,
  "keypoints": [
    {"x": 123.45, "y": 234.56, "confidence": 0.92},
    {"x": 456.78, "y": 345.67, "confidence": 0.87}
  ],
  "inference_time": 0.023
}
```

### Metaflow Pipeline (Bonus)

For production-scale inference, use the Metaflow pipeline:

> **⚠️ Windows Note:** Metaflow has limited Windows support due to Unix-specific dependencies (`fcntl` module). 
> - **On Windows:** Use the standard `inference.py` (fully functional) or run Metaflow in WSL/Docker
> - **On Linux/Mac:** Metaflow works out of the box

```bash
# Run locally (Linux/Mac or WSL)
python inference_flow.py run --input-dir images/

# With custom parameters
python inference_flow.py run \
    --input-dir images/ \
    --output-dir predictions_flow/ \
    --checkpoint checkpoints/best_model.pth \
    --batch-size 32 \
    --confidence 0.3

# Show flow structure
python inference_flow.py show

# View past runs
python inference_flow.py list
```

**Windows Alternative:**
```bash
# Use standard inference script instead (works on all platforms)
python inference.py --input images/ --output predictions/
```

**Benefits of Metaflow:**
- **Versioning**: Every run is tracked with a unique ID
- **Reproducibility**: Full parameter and artifact tracking
- **Scalability**: Easy deployment to cloud (AWS Batch, Kubernetes)
- **Monitoring**: Built-in logging and metrics
- **Checkpointing**: Automatic state management

## ⚙️ Configuration

All configuration is centralized in `config.py`:

```python
# Data configuration
DATA_CONFIG = {
    "image_size": (512, 512),        # Input image size
    "train_ratio": 0.7,              # Train/val/test split
    "val_ratio": 0.15,
    "test_ratio": 0.15,
}

# Model configuration
MODEL_CONFIG = {
    "backbone": "resnet18",          # Encoder backbone
    "pretrained": True,              # Use ImageNet weights
    "heatmap_size": (64, 64),       # Output resolution
    "heatmap_sigma": 2.0,           # Gaussian sigma for keypoints
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}
```

Modify these values to experiment with different configurations.

## 📊 Results

See [RESULTS.md](RESULTS.md) for detailed results and discussion.

**Sample Results** (on synthetic data):
- **Training time**: ~5-10 minutes for 10 epochs (GPU)
- **MAE**: ~15-25 pixels (depends on object size)
- **PCK@10**: ~0.65-0.75
- **PCK@20**: ~0.85-0.95
- **Inference speed**: ~50-100 images/sec (GPU, batch size 16)

**Note**: These are demonstration results on synthetic data. Real DOTA dataset performance will vary.

## 🚀 Scalability & Production Considerations

This implementation is designed with scalability and production deployment in mind:

### Scalability Approaches

1. **Data Pipeline**
   - Efficient data loading with PyTorch DataLoader
   - Multi-worker data loading (configurable `num_workers`)
   - Memory-efficient batch processing
   - Streaming inference for large datasets

2. **Model Serving**
   - Batch inference support
   - GPU acceleration
   - Model quantization ready (can reduce size by 4x)
   - ONNX export capability for deployment

3. **Horizontal Scaling**
   - Metaflow integration for distributed processing
   - Easy deployment to AWS Batch or Kubernetes
   - Stateless inference pipeline (scales easily)
   - Parallel processing of image batches

4. **Large-Scale Processing**
   ```python
   # Process millions of images
   python inference_flow.py run \
       --input-dir s3://bucket/images/ \
       --output-dir s3://bucket/predictions/ \
       --with batch
   ```

### Production Deployment

**Recommended Architecture:**
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   S3/Azure  │────▶│  Inference  │────▶│  Database   │
│   Storage   │     │   Service   │     │  (Results)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ Monitoring  │
                    │  (Grafana)  │
                    └─────────────┘
```

**Monitoring Metrics** (already tracked in code):
- Inference time per image
- Batch processing time
- Memory usage
- Prediction confidence distribution
- Number of keypoints detected per image
- Error rates and anomalies

**Model Drift Detection**:
- Monitor confidence score distributions
- Track average number of keypoints over time
- Alert on significant deviations
- Periodic re-evaluation on holdout set

**Deployment Options**:
1. **REST API**: Wrap inference in Flask/FastAPI
2. **Batch Processing**: Use Metaflow + AWS Batch
3. **Real-time Stream**: Kafka + microservices
4. **Edge Deployment**: ONNX + TensorRT for edge devices

### Performance Optimization

For production, consider:
- **Model optimization**: Quantization (INT8), pruning
- **TensorRT**: 2-5x speedup on NVIDIA GPUs
- **ONNX Runtime**: Cross-platform optimization
- **Batch size tuning**: Find optimal batch size for throughput
- **Mixed precision**: FP16 for faster inference
- **Model ensembling**: Multiple models for higher accuracy

## 🔮 Future Improvements

Potential enhancements for this project:

### Model Architecture
- [ ] Experiment with larger backbones (ResNet50, EfficientNet)
- [ ] Add skip connections from encoder to decoder
- [ ] Multi-scale feature fusion
- [ ] Attention mechanisms for better feature selection
- [ ] Test lightweight models (MobileNet) for edge deployment

### Data & Training
- [ ] Real DOTA dataset integration with proper parsing
- [ ] Advanced augmentations (rotation, mosaic, mixup)
- [ ] Hard negative mining
- [ ] Class-specific keypoint detection
- [ ] Multi-task learning (detection + classification)

### Metrics & Evaluation
- [ ] Object Detection Accuracy (ODA) metric
- [ ] Per-class performance analysis
- [ ] Confusion matrices for different object types
- [ ] Cross-dataset evaluation
- [ ] Failure case analysis

### Deployment & Ops
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] A/B testing framework
- [ ] Model versioning and registry
- [ ] Automated retraining pipeline

### Features
- [ ] Web UI for interactive inference
- [ ] Real-time video processing
- [ ] Multi-GPU training
- [ ] Distributed training (DDP)
- [ ] Active learning loop
- [ ] Explainability visualizations (Grad-CAM)

