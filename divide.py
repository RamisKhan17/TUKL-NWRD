import os
import random
import shutil

image_dir = "data/train/images/"
mask_dir = "data/train/masks/"

output_base_img = "data/dividedTrain/images/"
output_base_mask = "data/dividedTrain/masks/"

for i in range(1, 5):
    os.makedirs(os.path.join(output_base_img, f"train{i}"), exist_ok=True)
    os.makedirs(os.path.join(output_base_mask, f"train{i}"), exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
random.shuffle(image_files)

split_size = len(image_files) // 4
splits = [image_files[i * split_size:(i + 1) * split_size] for i in range(3)]
splits.append(image_files[3 * split_size:])

for idx, file_list in enumerate(splits, start=1):
    for fname in file_list:
        base = os.path.splitext(fname)[0]
        mask_name = base + ".png"

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            shutil.copy(img_path, os.path.join(output_base_img, f"train{idx}", fname))
            shutil.copy(mask_path, os.path.join(output_base_mask, f"train{idx}", mask_name))