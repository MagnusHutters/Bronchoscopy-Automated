import os
import cv2
import torch
import shutil
from pathlib import Path
import argparse

import warnings

# Suppress all DeprecationWarning messages
warnings.filterwarnings('ignore', category=DeprecationWarning)

def run_inference_and_save(model_path, input_folder, output_folder):
    # Load the YOLO model using the specified trained weights
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

    # Create paths for images and labels subfolders
    images_output_folder = os.path.join(output_folder, "images")
    labels_output_folder = os.path.join(output_folder, "labels")
    Path(images_output_folder).mkdir(parents=True, exist_ok=True)
    Path(labels_output_folder).mkdir(parents=True, exist_ok=True)

    # Process each image in the input folder
    items = os.listdir(input_folder)
    for index, img_name in enumerate(items):
        img_path = os.path.join(input_folder, img_name)

        # Skip if the file is not an image
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        
        #print progress
        print(f"Processing image {index + 1}/{len(items)}: {img_name}")
        
        # Load the image
        img = cv2.imread(img_path)
        
        # Run YOLOv5 inference
        results = model(img)

        # Prepare the paths for output .txt and .jpg files
        output_txt_path = os.path.join(labels_output_folder, f"{Path(img_name).stem}.txt")
        output_img_path = os.path.join(images_output_folder, img_name)
        
        # move the image to the output folder
        shutil.move(img_path, output_img_path)
        
        # Open the .txt file in write mode to store annotations
        with open(output_txt_path, "w") as f:
            # Extract the results and write them into the .txt file
            for pred in results.pred[0]:
                xmin, ymin, xmax, ymax, conf, cls = pred

                # Prepare YOLO format (class_id, center_x, center_y, width, height)
                img_height, img_width, _ = img.shape
                center_x = (xmin + xmax) / 2 / img_width
                center_y = (ymin + ymax) / 2 / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height
                class_id = int(cls)

                # Prepare output annotation line
                annotation = f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"

                # Write the annotation to the file
                f.write(annotation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO inference and save results for training.")
    parser.add_argument('input_folder', type=str, help='Path to the folder containing input images.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where the dataset will be saved.')
    parser.add_argument('model_path', type=str, help='Path to the trained YOLO model weights file (e.g., best.pt).')

    args = parser.parse_args()

    run_inference_and_save(args.model_path, args.input_folder, args.output_folder)
