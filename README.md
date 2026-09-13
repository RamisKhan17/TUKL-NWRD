# NWRD — Rust Segmentation with DeepLabV3+

Binary semantic segmentation of **rust** vs **non-rust** regions on the **NWRD** (NUST Wheat Rust Dataset) dataset using a **DeepLabV3+ (ResNet-50)** encoder-decoder architecture.

---

## Project Structure

```
NWRD/
├── data/
│   ├── train/
│   │   ├── images/        # Raw training images (.jpg)
│   │   └── masks/         # Binary segmentation masks (.png)
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   ├── test/
│   │   ├── images/
│   │   └── masks/
│   └── dividedTrain/      # Output of divide.py (4 sub-splits)
│       ├── images/
│       └── masks/
├── patchedData/           # Output of patching.py (224×224 patches)
│   ├── dividedTrain/
│   ├── train/
│   ├── val/
│   └── test/
├── models/                # Saved model checkpoints (.pth.tar)
├── divide.py              # Split training set into 4 folds
├── patching.py            # Tile images/masks into 224×224 patches
├── preprocessing.py       # Rust/non-rust classification & cosaliency prep
├── predict.py             # Run inference + Grad-CAM visualisation
└── utils.py               # Checkpoint I/O, data loaders, accuracy helpers
```

> **Note:** `data/`, `patchedData/`, and `models/` are excluded. Download them yourself.

---

## Requirements

| Package                       | Purpose                    |
| ----------------------------- | -------------------------- |
| `torch` / `torchvision`       | Deep learning framework    |
| `segmentation-models-pytorch` | DeepLabV3+ implementation  |
| `albumentations`              | Image augmentation         |
| `patchify`                    | Tiling images into patches |
| `Pillow`                      | Image I/O                  |
| `numpy`                       | Array operations           |
| `matplotlib`                  | Visualisation              |
| `pytorch-grad-cam`            | Grad-CAM explainability    |

Install all dependencies:

```bash
pip install torch torchvision segmentation-models-pytorch albumentations patchify Pillow numpy matplotlib grad-cam
```

---

## Usage

### 1. Prepare Data

Place your raw images and binary masks under:

```
data/train/images/   ← .jpg files
data/train/masks/    ← .png files (same base name as images)
data/val/images/
data/val/masks/
data/test/images/
data/test/masks/
```

### 2. Divide Training Set into 4 Folds

```bash
python divide.py
```

Randomly splits `data/train/` into four balanced sub-splits saved to `data/dividedTrain/`.

### 3. Create 224×224 Patches

```bash
python patching.py
```

Pads images to a multiple of 224 and tiles them into non-overlapping 224×224 patches saved to `patchedData/`.  
_(Edit `image_dir` / `mask_dir` at the top of the script to switch between train folds.)_

### 4. Run Inference + Grad-CAM

```bash
python predict.py
```

Loads the trained checkpoint from `models/DeepLabV3+(Res50-bin-NWRD).pth.tar`, runs inference on a single test patch, prints IoU, and displays a 4-panel figure:

| Panel        | Content           |
| ------------ | ----------------- |
| Top-left     | Input image       |
| Top-right    | Ground truth mask |
| Bottom-left  | Predicted mask    |
| Bottom-right | Grad-CAM overlay  |

---

## Model

- **Architecture:** DeepLabV3+ with ResNet-50 encoder (ImageNet pre-trained)
- **Task:** Binary segmentation (rust = 1, background = 0)
- **Loss:** BCE + Dice (trained externally)
- **Checkpoint:** `models/DeepLabV3+(Res50-bin-NWRD).pth.tar`

---

## Metrics

Evaluation metrics reported during inference:

- **Pixel Accuracy** — fraction of correctly classified pixels
- **Dice Score** — harmonic mean of precision & recall over the positive class
- **IoU (Jaccard Index)** — intersection over union for the rust class

---

## Notes

- Images must be RGB (`.jpg`); masks must be grayscale (`.png`) with pixel values `0` (background) or `1`/`255` (rust).
- `predict.py` normalises masks so any value > 1 is clamped to 1.
- The patching step uses `patchify` with `step=224` (non-overlapping tiles).
