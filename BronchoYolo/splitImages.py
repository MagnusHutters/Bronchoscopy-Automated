import os
import random
import shutil
from pathlib import Path
import argparse

def split_dataset(dataset_folder, output_folder, val_percentage=0.2):
    # Define paths for the training and validation sets
    train_images_folder = os.path.join(output_folder, "train", "images")
    train_labels_folder = os.path.join(output_folder, "train", "labels")
    val_images_folder = os.path.join(output_folder, "val", "images")
    val_labels_folder = os.path.join(output_folder, "val", "labels")

    # Create directories for train and val splits
    Path(train_images_folder).mkdir(parents=True, exist_ok=True)
    Path(train_labels_folder).mkdir(parents=True, exist_ok=True)
    Path(val_images_folder).mkdir(parents=True, exist_ok=True)
    Path(val_labels_folder).mkdir(parents=True, exist_ok=True)

    # Empty the destination folders if they already exist
    for folder in [train_images_folder, train_labels_folder, val_images_folder, val_labels_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        Path(folder).mkdir(parents=True, exist_ok=True)

    # List all images in the dataset
    image_files = [f for f in os.listdir(os.path.join(dataset_folder, "images")) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Shuffle the images
    random.shuffle(image_files)

    # Calculate the number of images to be used for validation
    num_val_images = int(len(image_files) * val_percentage)

    # Separate images into validation and training sets
    val_images = image_files[:num_val_images]
    train_images = image_files[num_val_images:]

    def copy_files(image_list, src_folder, dest_images_folder, dest_labels_folder):
        for img_name in image_list:
            label_name = f"{Path(img_name).stem}.txt"

            # Copy image file
            shutil.copy(os.path.join(src_folder, "images", img_name), os.path.join(dest_images_folder, img_name))
            
            # Copy corresponding label file
            shutil.copy(os.path.join(src_folder, "labels", label_name), os.path.join(dest_labels_folder, label_name))

    # Copy validation images and labels
    copy_files(val_images, dataset_folder, val_images_folder, val_labels_folder)

    # Copy training images and labels
    copy_files(train_images, dataset_folder, train_images_folder, train_labels_folder)

    print(f"Dataset split completed. Validation set: {len(val_images)} images, Training set: {len(train_images)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument('dataset_folder', type=str, help='Path to the dataset folder containing images and labels folders.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where the split dataset will be saved.')
    parser.add_argument('--v', type=float, default=0.1, help='Percentage of the dataset to use for validation.')

    args = parser.parse_args()

    split_dataset(args.dataset_folder, args.output_folder, args.v)
