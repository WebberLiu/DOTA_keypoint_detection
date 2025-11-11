# DOTA Dataset Setup Guide

This guide explains how to download and use the actual DOTA dataset with this keypoint detection pipeline.

## 📥 Downloading DOTA Dataset

### Option 1: Official DOTA Website (Recommended)

1. **Visit the DOTA website:**
   - Website: https://captain-whu.github.io/DOTA/dataset.html
   - Register for access if required

2. **Download DOTA-v1.0 or DOTA-v1.5:**
   - DOTA-v1.0: ~2.5GB (2,806 images)
   - DOTA-v1.5: ~3.5GB (additional images)
   - DOTA-v2.0: ~14GB (larger dataset)

3. **Download both:**
   - Images (PNG format)
   - Annotations (TXT format in `labelTxt` folder)

### Option 2: Academic Torrents / Mirrors

Alternative download sources (check for updated links):
- Academic torrents
- Mirror sites provided by DOTA authors

## 📁 Expected Directory Structure

After downloading and extracting, organize your data as follows:

```
DOTA-v1.0/
├── images/                    # Image files
│   ├── P0000.png
│   ├── P0001.png
│   ├── P0002.png
│   └── ...
└── labelTxt/                  # Annotation files
    ├── P0000.txt
    ├── P0001.txt
    ├── P0002.txt
    └── ...
```

### Annotation Format

Each `.txt` file contains annotations in the following format:
```
imagesource:GoogleEarth
gsd:0.15
x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
...
```

Where:
- `x1,y1,x2,y2,x3,y3,x4,y4`: Four corner points of oriented bounding box
- `category`: Object class (plane, ship, storage-tank, etc.)
- `difficulty`: 0 (easy) or 1 (hard)

## 🚀 Using DOTA Dataset with This Project

### Step 1: Place DOTA Data

Copy or move your DOTA dataset to the project:

```bash
# Option A: Copy to data_samples/dota
mkdir -p data_samples/dota
cp -r /path/to/DOTA-v1.0/images data_samples/dota/
cp -r /path/to/DOTA-v1.0/labelTxt data_samples/dota/

# Option B: Create symbolic link
ln -s /path/to/DOTA-v1.0 data_samples/dota
```

### Step 2: Verify Data Structure

Run the parser to verify your data:

```bash
python -m data.dota_parser data_samples/dota
```

Expected output:
```
Found 2806 samples

Image: P0000.png
  Objects: 12
  Categories: ['plane', 'plane', 'plane', ...]
  Keypoints shape: (12, 2)
...
```

### Step 3: Train with DOTA Data

Update your training command to use DOTA data:

```python
# In config.py, update DATA_DIR to point to DOTA
DATA_DIR = PROJECT_ROOT / "data_samples" / "dota"
```

Or specify when running:

```bash
# Train with full DOTA dataset
python train.py --epochs 20

# The code automatically detects DOTA data and uses it!
```

## 🎯 Using a Small Subset

For faster experimentation, use only a subset:

### Method 1: Quick Subset (Recommended)

Update `config.py`:

```python
DATA_CONFIG = {
    "image_size": (512, 512),
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_seed": 42,
    
    # NEW: Limit number of images to use
    "max_samples": 200,  # Use only 200 images instead of all 2806
}
```

### Method 2: Category-Specific Subset

To use only specific object categories (e.g., planes and ships):

```python
# In data/dota_parser.py, modify get_image_annotation_pairs call
pairs = parser.get_image_annotation_pairs(
    max_samples=200,
    categories=['plane', 'ship', 'large-vehicle']
)
```

### Method 3: Create Dedicated Subset Directory

```bash
# Manually copy a subset
mkdir -p data_samples/dota_subset/images
mkdir -p data_samples/dota_subset/labelTxt

# Copy first 100 images and their annotations
cd data_samples/dota/images
ls *.png | head -100 | xargs -I {} cp {} ../../dota_subset/images/

cd ../labelTxt
ls *.txt | head -100 | xargs -I {} cp {} ../../dota_subset/labelTxt/
```

Then update `DATA_DIR` to point to `dota_subset`.

## 📊 DOTA Categories

The DOTA dataset contains 15 object categories:

1. **plane** - Airplanes
2. **ship** - Ships and boats
3. **storage-tank** - Storage tanks
4. **baseball-diamond** - Baseball fields
5. **tennis-court** - Tennis courts
6. **basketball-court** - Basketball courts
7. **ground-track-field** - Running tracks
8. **harbor** - Harbor areas
9. **bridge** - Bridges
10. **large-vehicle** - Large vehicles
11. **small-vehicle** - Small vehicles
12. **helicopter** - Helicopters
13. **roundabout** - Roundabouts
14. **soccer-ball-field** - Soccer fields
15. **swimming-pool** - Swimming pools

## 🔍 Verifying Your Setup

Run this quick test to ensure everything works:

```bash
# Test the parser
python -c "
from data.dota_parser import DOTAParser
parser = DOTAParser('data_samples/dota')
pairs = parser.get_image_annotation_pairs(max_samples=5)
print(f'Found {len(pairs)} samples!')
for img, ann in pairs[:3]:
    sample = parser.load_sample(img, ann)
    print(f'{img.name}: {sample[\"num_objects\"]} objects')
"
```

## 🎓 Training Tips with DOTA

### Start Small, Scale Up

1. **Initial experiment** (200 images, 5 epochs):
   ```bash
   # Update config.py: max_samples=200, num_epochs=5
   python train.py
   # Training time: ~15-20 minutes on GPU
   ```

2. **Medium scale** (1000 images, 10 epochs):
   ```bash
   # Update config.py: max_samples=1000, num_epochs=10
   python train.py
   # Training time: ~1-2 hours on GPU
   ```

3. **Full dataset** (2806 images, 20 epochs):
   ```bash
   # Update config.py: max_samples=None, num_epochs=20
   python train.py
   # Training time: ~4-6 hours on GPU
   ```

### Recommended Settings for DOTA

```python
# config.py adjustments for DOTA
DATA_CONFIG = {
    "image_size": (800, 800),  # DOTA images are large, use 800x800
    "max_samples": 500,        # Start with 500 images
}

MODEL_CONFIG = {
    "backbone": "resnet34",    # Slightly larger model for complex scenes
    "heatmap_size": (100, 100), # Higher resolution for better localization
    "heatmap_sigma": 3.0,       # Larger sigma for larger objects
}

TRAIN_CONFIG = {
    "batch_size": 4,           # Reduce if GPU memory is limited
    "num_epochs": 15,
    "learning_rate": 5e-4,
}
```

## 🐛 Troubleshooting

### "No DOTA data found"
- Verify directory structure: `data_samples/dota/images/` and `data_samples/dota/labelTxt/`
- Check that images are `.png` files
- Check that annotations are `.txt` files

### "Failed to parse annotation file"
- DOTA annotations have UTF-8 BOM, handled by parser
- First 2 lines are headers, parser skips them
- Verify annotation format matches expected structure

### GPU Out of Memory
- Reduce `batch_size` in `config.py`
- Reduce `image_size` to (512, 512) or (600, 600)
- Use `resnet18` instead of larger backbones

### Slow Training
- Use smaller subset (max_samples=200)
- Reduce `num_workers` if on CPU
- Consider using mixed precision training (requires code modification)

## 📈 Expected Results with DOTA

With the actual DOTA dataset, you should see better results than synthetic data:

| Metric | Synthetic Data | DOTA (200 imgs) | DOTA (1000 imgs) | DOTA (Full) |
|--------|----------------|-----------------|-------------------|-------------|
| MAE    | 15-25 px       | 10-18 px        | 8-15 px          | 6-12 px     |
| PCK@10 | 0.70-0.75      | 0.75-0.82       | 0.80-0.88        | 0.85-0.92   |
| PCK@20 | 0.90-0.95      | 0.92-0.96       | 0.94-0.98        | 0.95-0.99   |

*Results vary based on object categories, image complexity, and training hyperparameters*

## 📝 Citation

If you use the DOTA dataset, please cite:

```bibtex
@article{xia2018dota,
  title={DOTA: A large-scale dataset for object detection in aerial images},
  author={Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
  journal={CVPR},
  year={2018}
}
```

## 🔗 Resources

- **DOTA Website**: https://captain-whu.github.io/DOTA/dataset.html
- **Paper**: https://arxiv.org/abs/1711.10398
- **GitHub**: https://github.com/CAPTAIN-WHU/DOTA_devkit

---

**Happy training with real data! 🎯**

