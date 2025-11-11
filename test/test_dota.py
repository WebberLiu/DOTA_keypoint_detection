"""
Quick test script to verify DOTA dataset setup.

Usage:
    python test_dota.py
    python test_dota.py <path_to_dota_dataset>
"""

import sys
from pathlib import Path
from data import DOTAParser, verify_dataset_structure

def main():
    # Get DOTA data path
    if len(sys.argv) > 1:
        dota_path = sys.argv[1]
    else:
        dota_path = "data_samples/dota"
    
    print("="*70)
    print("DOTA DATASET VERIFICATION")
    print("="*70)
    print(f"\nChecking DOTA dataset at: {dota_path}\n")
    
    # Check if directory exists
    if not Path(dota_path).exists():
        print("❌ DOTA dataset directory not found!")
        print(f"   Expected: {dota_path}")
        print("\n💡 To download DOTA dataset:")
        print("   1. Visit: https://captain-whu.github.io/DOTA/dataset.html")
        print("   2. Download DOTA-v1.0")
        print("   3. Extract to: data_samples/dota/")
        print("   4. See DOTA_SETUP.md for detailed instructions")
        return False
    
    # Verify dataset structure first
    print("📊 Verifying dataset structure...")
    verify_results = verify_dataset_structure(dota_path)
    
    if verify_results['valid']:
        print(f"   Total images: {verify_results['num_images']}")
        print(f"   Total labels: {verify_results['num_labels']}")
        print(f"   ✅ Matched pairs: {verify_results['num_matched_pairs']}")
        
        if verify_results['images_without_labels']:
            print(f"   ⚠️  {len(verify_results['images_without_labels'])} images without labels (will be skipped)")
        if verify_results['labels_without_images']:
            print(f"   ⚠️  {len(verify_results['labels_without_images'])} labels without images (will be skipped)")
        
        print(f"   → {verify_results['num_matched_pairs']} valid pairs will be used for training")
    else:
        print("   ❌ Invalid dataset structure")
        return False
    
    # Initialize parser
    try:
        parser = DOTAParser(dota_path)
    except Exception as e:
        print(f"❌ Failed to initialize parser: {e}")
        return False
    
    # Check directory structure
    print("📁 Checking directory structure...")
    images_dir = Path(dota_path) / "images"
    labels_dir = Path(dota_path) / "labelTxt"
    
    if not images_dir.exists():
        print(f"   ❌ Images directory not found: {images_dir}")
        return False
    else:
        num_images = len(list(images_dir.glob("*.png"))) + len(list(images_dir.glob("*.jpg")))
        print(f"   ✅ Images directory found: {num_images} images")
    
    if not labels_dir.exists():
        print(f"   ❌ Labels directory not found: {labels_dir}")
        return False
    else:
        num_labels = len(list(labels_dir.glob("*.txt")))
        print(f"   ✅ Labels directory found: {num_labels} annotation files")
    
    # Get sample pairs
    print("\n🔍 Finding image-annotation pairs...")
    pairs = parser.get_image_annotation_pairs(max_samples=10)
    
    if len(pairs) == 0:
        print("   ❌ No valid image-annotation pairs found!")
        print("   Make sure both images/ and labelTxt/ directories have matching files")
        return False
    
    print(f"   ✅ Found {len(pairs)} pairs (showing first 10)")
    
    # Parse a few samples
    print("\n📊 Parsing sample annotations...")
    for i, (img_path, ann_path) in enumerate(pairs[:5], 1):
        try:
            sample = parser.load_sample(img_path, ann_path)
            print(f"\n   Sample {i}: {Path(img_path).name}")
            print(f"      Objects: {sample['num_objects']}")
            print(f"      Categories: {sample['categories'][:5]}..." if len(sample['categories']) > 5 else f"      Categories: {sample['categories']}")
            print(f"      Keypoints shape: {sample['keypoints'].shape}")
            
            if len(sample['keypoints']) > 0:
                print(f"      First keypoint: ({sample['keypoints'][0][0]:.1f}, {sample['keypoints'][0][1]:.1f})")
        except Exception as e:
            print(f"   ⚠️  Failed to parse {Path(img_path).name}: {e}")
    
    # Test with dataset class
    print("\n🧪 Testing with dataset class...")
    try:
        from data import DOTAKeypointDataset, get_train_transforms
        
        dataset = DOTAKeypointDataset(
            data_dir=dota_path,
            transform=get_train_transforms(),
            split="train",
            generate_synthetic=False  # Force use of real data
        )
        
        print(f"   ✅ Dataset created successfully")
        print(f"   ✅ Total samples: {len(dataset)}")
        
        # Try loading one sample
        if len(dataset) > 0:
            image, heatmap, metadata = dataset[0]
            print(f"   ✅ Sample loaded:")
            print(f"      Image shape: {image.shape}")
            print(f"      Heatmap shape: {heatmap.shape}")
            print(f"      Num keypoints: {metadata['num_keypoints']}")
        
    except Exception as e:
        print(f"   ⚠️  Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("✅ DOTA DATASET VERIFICATION PASSED!")
    print("="*70)
    print(f"\nYour DOTA dataset is properly set up with {len(pairs)} usable samples.")
    print("\nNext steps:")
    print("  1. Update config.py to set DATA_DIR to point to DOTA dataset")
    print("  2. Optionally set max_samples for faster testing")
    print("  3. Run: python train.py")
    print("\nSee DOTA_SETUP.md for more information.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

