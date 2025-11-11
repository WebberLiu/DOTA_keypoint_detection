"""
Quick test script to verify installation and basic functionality.

This script:
1. Checks all dependencies are installed
2. Verifies CUDA availability
3. Creates a small synthetic dataset
4. Trains for 1 epoch
5. Runs inference on a test image
6. Reports success/failure

Usage:
    python quick_test.py
"""

import sys
import time
from pathlib import Path

def check_imports():
    """Check that all required packages are importable."""
    print("Checking imports...")
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "PIL",
        "matplotlib",
        "tqdm",
        "sklearn",
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages found\n")
    return True


def check_cuda():
    """Check CUDA availability."""
    import torch
    
    print("Checking CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"  ✓ Device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ CUDA not available, will use CPU")
    print()


def test_data_pipeline():
    """Test data loading and preprocessing."""
    print("Testing data pipeline...")
    try:
        from data import create_dataloaders, get_train_transforms, get_val_transforms
        from config import DATA_DIR
        
        # Create minimal dataset
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=str(DATA_DIR),
            train_transform=get_train_transforms(),
            val_transform=get_val_transforms(),
            generate_synthetic=True
        )
        
        # Test loading one batch
        images, heatmaps, metadata = next(iter(train_loader))
        
        print(f"  ✓ Created dataloaders")
        print(f"  ✓ Train batches: {len(train_loader)}")
        print(f"  ✓ Batch shape: {images.shape}")
        print(f"  ✓ Heatmap shape: {heatmaps.shape}")
        print("✅ Data pipeline working\n")
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline failed: {e}\n")
        return False


def test_model():
    """Test model creation and forward pass."""
    print("Testing model...")
    try:
        import torch
        from models import create_model
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, criterion = create_model(device=device)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Model created")
        print(f"  ✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  ✓ Forward pass: {dummy_input.shape} → {output.shape}")
        print("✅ Model working\n")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}\n")
        return False


def test_training():
    """Test one training iteration."""
    print("Testing training loop...")
    try:
        import torch
        from models import create_model
        from data import create_dataloaders, get_train_transforms, get_val_transforms
        from config import DATA_DIR
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, criterion = create_model(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create minimal dataset
        train_loader, _, _ = create_dataloaders(
            data_dir=str(DATA_DIR),
            train_transform=get_train_transforms(),
            val_transform=get_val_transforms(),
            generate_synthetic=True
        )
        
        # Train for one batch
        model.train()
        images, heatmaps, metadata = next(iter(train_loader))
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, heatmaps)
        loss.backward()
        optimizer.step()
        
        print(f"  ✓ Training iteration completed")
        print(f"  ✓ Loss: {loss.item():.4f}")
        print("✅ Training working\n")
        return True
        
    except Exception as e:
        print(f"❌ Training test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference pipeline."""
    print("Testing inference...")
    try:
        import torch
        import numpy as np
        from PIL import Image
        from models import create_model
        from data.transforms import Normalize
        from config import DATA_CONFIG, DATA_DIR
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = create_model(device=device)
        model.eval()
        
        # Create a test image
        test_image_path = DATA_DIR / "synthetic" / "train" / "sample_0000.png"
        if test_image_path.exists():
            image = Image.open(test_image_path).convert("RGB")
        else:
            # Create dummy image
            image = Image.new("RGB", (512, 512), color=(200, 200, 200))
        
        # Preprocess
        image = image.resize(DATA_CONFIG["image_size"])
        image_array = np.array(image, dtype=np.float32) / 255.0
        normalize = Normalize()
        image_normalized = normalize(image_array)
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            heatmap = model(image_tensor)
            keypoints = model.extract_keypoints(heatmap, threshold=0.3)
        
        print(f"  ✓ Inference completed")
        print(f"  ✓ Detected keypoints: {len(keypoints[0])}")
        print("✅ Inference working\n")
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("DOTA KEYPOINT DETECTION - QUICK TEST")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    # Run tests
    results = {
        "Imports": check_imports(),
    }
    
    if results["Imports"]:
        check_cuda()
        results["Data Pipeline"] = test_data_pipeline()
        results["Model"] = test_model()
        results["Training"] = test_training()
        results["Inference"] = test_inference()
    
    # Summary
    elapsed = time.time() - start_time
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nTime elapsed: {elapsed:.2f}s")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Train model: python train.py")
        print("  2. Evaluate: python evaluate.py --checkpoint checkpoints/best_model.pth")
        print("  3. Inference: python inference.py --input images/ --output predictions/")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before proceeding.")
        sys.exit(1)
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

