import math
import os
import numpy as np
from PIL import Image, ImageOps
from patchify import patchify

def padding(img: np.ndarray, tile_size=224, fill_value=0):
    if img.ndim not in [2, 3]:
        raise ValueError("Input must be a 2D (grayscale) or 3D (RGB) NumPy array.")
    height, width = img.shape[:2]
    pad_h = (math.ceil(height / tile_size) * tile_size) - height
    pad_w = (math.ceil(width / tile_size) * tile_size) - width
    if img.ndim == 2: 
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=fill_value)
    else:  
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=fill_value)
    return padded

image_dir = "data/dividedTrain/images/train4/"
mask_dir = "data/dividedTrain/masks/train4/"
output_img_dir = "patchedData/dividedTrain/images/train4/"
output_mask_dir = "patchedData/dividedTrain/masks/train4/"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

image_files = os.listdir(image_dir)

for fname in image_files:
    base_name = os.path.splitext(fname)[0]
    img_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))

    image = np.array(ImageOps.exif_transpose(Image.open(img_path).convert("RGB")))
    mask = np.array(ImageOps.exif_transpose(Image.open(mask_path).convert("L"))).astype(np.float32)

    image = padding(image)
    mask = padding(mask)

    patches_img = patchify(image, (224, 224, 3), step=224)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0]
            patch_pil = Image.fromarray(single_patch_img.astype(np.uint8))
            filename = f"{base_name}_{i}_{j}.jpg"
            patch_pil.save(os.path.join(output_img_dir, filename))

    mask = np.expand_dims(mask, axis=-1)
    patches_mask = patchify(mask, (224, 224, 1), step=224)

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = np.squeeze(patches_mask[i, j, 0], axis=-1)
            patch_pil = Image.fromarray((single_patch_mask).astype(np.uint8))
            filename = f"{base_name}_{i}_{j}.png"
            patch_pil.save(os.path.join(output_mask_dir, filename))