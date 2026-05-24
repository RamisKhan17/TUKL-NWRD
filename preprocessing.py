import glob
import os
import shutil

source = "data/train"
dest = "processed/train"
patch_size = 224
rust_threshold = 150
max_number_of_images_per_group = 12

patches_path = os.path.join(dest, "patches")
images_dir = os.path.join(patches_path, "images")
masks_dir = os.path.join(patches_path, "masks")

destination = os.path.join(dest, "RustNonRustSplit")
root = patches_path

rust_images_dir = os.path.join(destination, "rust", "images")
non_rust_images_dir = os.path.join(destination, "non_rust", "images")

rustClassificationDir = os.path.join(dest, "classification", "rust")
nonRustClassificationDir = os.path.join(dest, "classification", "non_rust")
os.makedirs(rustClassificationDir, exist_ok=True)
os.makedirs(nonRustClassificationDir, exist_ok=True)

shutil.copytree(rust_images_dir, rustClassificationDir, dirs_exist_ok=True)
shutil.copytree(non_rust_images_dir, nonRustClassificationDir, dirs_exist_ok=True)

def delete_extra_images(directory, target_count):
    image_files = glob.glob(os.path.join(directory, '*.JPG')) + glob.glob(os.path.join(directory, '*.jpeg')) + glob.glob(os.path.join(directory, '*.png'))
    
    if len(image_files) > target_count:
        num_to_delete = len(image_files) - target_count
        image_files.sort(key=os.path.getmtime)
        for i in range(num_to_delete):
            os.remove(image_files[i])
        print(f"{num_to_delete} images deleted.")
    elif len(image_files) < target_count:
        print("Warning: Number of images in directory is less than the target count.")

if len(os.listdir(rustClassificationDir)) < len(os.listdir(nonRustClassificationDir)):
    delete_extra_images(nonRustClassificationDir, len(os.listdir(rustClassificationDir)))
else:
    delete_extra_images(rustClassificationDir, len(os.listdir(nonRustClassificationDir)))

rust_dir = os.path.join(destination, "rust")
rustCosaliencyDir = os.path.join(dest, "cosaliency")
shutil.copytree(rust_dir, rustCosaliencyDir, dirs_exist_ok=True)

def split_images_into_folders(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            image_no = filename.split('_')[0]
            if not image_no.isdigit():
                image_no = filename.split('_')[1]
            destination_subdir = os.path.join(destination_dir, image_no)
            if not os.path.exists(destination_subdir):
                os.makedirs(destination_subdir)
            shutil.move(os.path.join(source_dir, filename), destination_subdir)

def organize_images(main_directory):
    if not os.path.exists(main_directory):
        print(f"The specified main directory '{main_directory}' does not exist.")
        return

    subdirectories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]

    for subdir in subdirectories:
        subdir_path = os.path.join(main_directory, subdir)
        images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        images_per_subdir = 12
        num_subdirectories = len(images) // images_per_subdir
        n = 0
        for i in range(num_subdirectories - 1):
            new_subdir_name = f"{subdir}_part{i + 1}"
            new_subdir_path = os.path.join(main_directory, new_subdir_name)
            os.makedirs(new_subdir_path)
            for j in range(images_per_subdir):
                old_image_path = os.path.join(subdir_path, images[n])
                new_image_path = os.path.join(new_subdir_path, images[n])
                shutil.move(old_image_path, new_image_path)
                n += 1

source_directory = os.path.join(source, "cosaliency", "images")
destination_directory = os.path.join(dest, "cosaliency", "images")
split_images_into_folders(source_directory, destination_directory)
organize_images(destination_directory)

source_directory = os.path.join(source, "cosaliency", "masks")
destination_directory = os.path.join(dest, "cosaliency", "masks")
split_images_into_folders(source_directory, destination_directory)
organize_images(destination_directory)