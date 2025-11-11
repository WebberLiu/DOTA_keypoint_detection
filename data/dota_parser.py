"""
DOTA dataset parser for loading annotations and extracting keypoints.

The DOTA dataset uses text files with oriented bounding box annotations.
Each annotation file contains lines with format:
x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

This module parses these annotations and extracts center keypoints.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DOTAParser:
    """
    Parser for DOTA dataset annotations.
    
    DOTA uses oriented bounding boxes (OBB) with 4 corner points.
    We extract the center of each bounding box as the keypoint.
    """
    
    # DOTA category names
    CATEGORIES = [
        'plane', 'ship', 'storage-tank', 'baseball-diamond',
        'tennis-court', 'basketball-court', 'ground-track-field',
        'harbor', 'bridge', 'large-vehicle', 'small-vehicle',
        'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool'
    ]
    
    def __init__(self, data_dir: str):
        """
        Initialize DOTA parser.
        
        Args:
            data_dir: Root directory containing DOTA dataset
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labelTxt"
        
    def parse_annotation_file(self, annotation_path: Path) -> List[Dict]:
        """
        Parse a single DOTA annotation file.
        
        Args:
            annotation_path: Path to .txt annotation file
        
        Returns:
            List of annotation dictionaries with bounding boxes and keypoints
        """
        annotations = []
        
        try:
            with open(annotation_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
            
            # Skip header lines (first 2 lines are metadata in DOTA)
            for line in lines[2:]:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 9:
                    continue
                
                try:
                    # Parse coordinates (4 corner points)
                    x1, y1 = float(parts[0]), float(parts[1])
                    x2, y2 = float(parts[2]), float(parts[3])
                    x3, y3 = float(parts[4]), float(parts[5])
                    x4, y4 = float(parts[6]), float(parts[7])
                    
                    # Category and difficulty
                    category = parts[8]
                    difficulty = int(parts[9]) if len(parts) > 9 else 0
                    
                    # Calculate center keypoint from 4 corners
                    center_x = (x1 + x2 + x3 + x4) / 4
                    center_y = (y1 + y2 + y3 + y4) / 4
                    
                    # Calculate bounding box for reference
                    min_x = min(x1, x2, x3, x4)
                    max_x = max(x1, x2, x3, x4)
                    min_y = min(y1, y2, y3, y4)
                    max_y = max(y1, y2, y3, y4)
                    
                    annotations.append({
                        "category": category,
                        "difficulty": difficulty,
                        "corners": [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                        "bbox": [min_x, min_y, max_x, max_y],
                        "center": [center_x, center_y],
                    })
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse line in {annotation_path}: {line}")
                    continue
        
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")
            return []
        
        return annotations
    
    def get_image_annotation_pairs(
        self,
        max_samples: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> List[Tuple[Path, Path]]:
        """
        Get pairs of image and annotation file paths.
        
        Args:
            max_samples: Maximum number of samples to return (for small subset)
            categories: List of categories to include (None = all categories)
        
        Returns:
            List of (image_path, annotation_path) tuples
        """
        if not self.images_dir.exists():
            logger.error(f"Images directory not found: {self.images_dir}")
            return []
        
        if not self.labels_dir.exists():
            logger.error(f"Labels directory not found: {self.labels_dir}")
            return []
        
        pairs = []
        missing_labels = []
        filtered_by_category = 0
        
        # Find all images
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        for ext in image_extensions:
            for img_path in self.images_dir.glob(f"*{ext}"):
                # Look for corresponding annotation file
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    missing_labels.append(img_path.name)
                    continue
                
                # If category filtering is enabled, check annotations
                if categories:
                    annotations = self.parse_annotation_file(label_path)
                    has_category = any(
                        ann["category"] in categories 
                        for ann in annotations
                    )
                    if not has_category:
                        filtered_by_category += 1
                        continue
                
                pairs.append((img_path, label_path))
                
                # Stop if we've reached max_samples
                if max_samples and len(pairs) >= max_samples:
                    break
            
            # Stop if we've reached max_samples
            if max_samples and len(pairs) >= max_samples:
                break
        
        # Log summary
        logger.info(f"Found {len(pairs)} valid image-annotation pairs")
        if missing_labels:
            logger.warning(f"  {len(missing_labels)} images without corresponding labels (skipped)")
            if len(missing_labels) <= 5:
                for name in missing_labels:
                    logger.warning(f"    - {name}")
        if filtered_by_category > 0:
            logger.info(f"  {filtered_by_category} images filtered out (no matching categories)")
        
        return pairs
    
    def load_sample(self, image_path: Path, annotation_path: Path) -> Dict:
        """
        Load a complete sample with image path and keypoints.
        
        Args:
            image_path: Path to image file
            annotation_path: Path to annotation file
        
        Returns:
            Dictionary with image_path and keypoints array
        """
        annotations = self.parse_annotation_file(annotation_path)
        
        # Extract keypoints (centers) from all annotations
        keypoints = np.array(
            [ann["center"] for ann in annotations],
            dtype=np.float32
        )
        
        # Also store categories for reference
        categories = [ann["category"] for ann in annotations]
        difficulties = [ann["difficulty"] for ann in annotations]
        
        return {
            "image_path": str(image_path),
            "keypoints": keypoints,
            "categories": categories,
            "difficulties": difficulties,
            "num_objects": len(annotations)
        }


def prepare_dota_subset(
    data_dir: str,
    output_dir: str,
    max_samples: int = 100,
    categories: Optional[List[str]] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Prepare a small subset of DOTA dataset with train/val/test splits.
    
    Args:
        data_dir: Root directory containing DOTA dataset
        output_dir: Output directory for processed subset
        max_samples: Maximum number of samples to use
        categories: List of categories to include (None = all)
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'val', 'test' sample lists
    """
    np.random.seed(random_seed)
    
    parser = DOTAParser(data_dir)
    
    # Get image-annotation pairs
    pairs = parser.get_image_annotation_pairs(max_samples, categories)
    
    if len(pairs) == 0:
        logger.error("No valid image-annotation pairs found!")
        return {"train": [], "val": [], "test": []}
    
    # Shuffle pairs
    np.random.shuffle(pairs)
    
    # Split into train/val/test
    n_samples = len(pairs)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]
    
    # Load samples
    splits = {
        "train": [parser.load_sample(img, ann) for img, ann in train_pairs],
        "val": [parser.load_sample(img, ann) for img, ann in val_pairs],
        "test": [parser.load_sample(img, ann) for img, ann in test_pairs],
    }
    
    # Save metadata
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "total_samples": n_samples,
        "train_samples": len(splits["train"]),
        "val_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "categories": categories if categories else "all",
        "random_seed": random_seed,
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Prepared DOTA subset:")
    logger.info(f"  Train: {metadata['train_samples']} samples")
    logger.info(f"  Val: {metadata['val_samples']} samples")
    logger.info(f"  Test: {metadata['test_samples']} samples")
    
    return splits


def verify_dataset_structure(data_dir: str) -> Dict:
    """
    Verify DOTA dataset structure and report statistics.
    
    Args:
        data_dir: Root directory containing DOTA dataset
    
    Returns:
        Dictionary with verification results
    """
    data_path = Path(data_dir)
    images_dir = data_path / "images"
    labels_dir = data_path / "labelTxt"
    
    results = {
        "valid": True,
        "images_dir_exists": images_dir.exists(),
        "labels_dir_exists": labels_dir.exists(),
        "num_images": 0,
        "num_labels": 0,
        "num_matched_pairs": 0,
        "images_without_labels": [],
        "labels_without_images": [],
    }
    
    if not results["images_dir_exists"]:
        results["valid"] = False
        return results
    
    if not results["labels_dir_exists"]:
        results["valid"] = False
        return results
    
    # Count images
    image_files = {}
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        for img_path in images_dir.glob(f"*{ext}"):
            image_files[img_path.stem] = img_path
    results["num_images"] = len(image_files)
    
    # Count labels
    label_files = {}
    for label_path in labels_dir.glob("*.txt"):
        label_files[label_path.stem] = label_path
    results["num_labels"] = len(label_files)
    
    # Find matches
    matched = set(image_files.keys()) & set(label_files.keys())
    results["num_matched_pairs"] = len(matched)
    
    # Find mismatches
    results["images_without_labels"] = sorted(set(image_files.keys()) - matched)
    results["labels_without_images"] = sorted(set(label_files.keys()) - matched)
    
    return results


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dota_parser.py <path_to_dota_dataset>")
        print("Example: python dota_parser.py ./DOTA-v1.0")
        print("\nVerify dataset structure:")
        print("  python dota_parser.py ./DOTA-v1.0 --verify")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    
    data_dir = sys.argv[1]
    
    # Check if verify mode
    if len(sys.argv) > 2 and sys.argv[2] == "--verify":
        print("\n" + "="*70)
        print("DOTA DATASET STRUCTURE VERIFICATION")
        print("="*70)
        
        results = verify_dataset_structure(data_dir)
        
        print(f"\nDirectory: {data_dir}")
        print(f"  Images directory exists: {'✓' if results['images_dir_exists'] else '✗'}")
        print(f"  Labels directory exists: {'✓' if results['labels_dir_exists'] else '✗'}")
        
        if results['valid']:
            print(f"\nFile counts:")
            print(f"  Total images: {results['num_images']}")
            print(f"  Total labels: {results['num_labels']}")
            print(f"  Matched pairs: {results['num_matched_pairs']}")
            
            if results['images_without_labels']:
                print(f"\n⚠ Images without labels: {len(results['images_without_labels'])}")
                for name in results['images_without_labels'][:5]:
                    print(f"    - {name}")
                if len(results['images_without_labels']) > 5:
                    print(f"    ... and {len(results['images_without_labels']) - 5} more")
            
            if results['labels_without_images']:
                print(f"\n⚠ Labels without images: {len(results['labels_without_images'])}")
                for name in results['labels_without_images'][:5]:
                    print(f"    - {name}")
                if len(results['labels_without_images']) > 5:
                    print(f"    ... and {len(results['labels_without_images']) - 5} more")
            
            if results['num_matched_pairs'] == results['num_images'] == results['num_labels']:
                print(f"\n✓ Perfect match! All images have corresponding labels.")
            else:
                print(f"\n✓ Found {results['num_matched_pairs']} valid pairs to use.")
                print(f"  (Images/labels without matches will be skipped)")
        
        sys.exit(0)
    
    # Normal mode - show samples
    parser = DOTAParser(data_dir)
    
    # Get a few samples
    pairs = parser.get_image_annotation_pairs(max_samples=5)
    
    print(f"\nFound {len(pairs)} samples")
    
    for img_path, ann_path in pairs[:3]:
        sample = parser.load_sample(img_path, ann_path)
        print(f"\nImage: {Path(img_path).name}")
        print(f"  Objects: {sample['num_objects']}")
        print(f"  Categories: {sample['categories']}")
        print(f"  Keypoints shape: {sample['keypoints'].shape}")

