import os
import cv2
import torch
import zipfile
import tempfile
import argparse
from pathlib import Path

def extract_images(zip_path, extract_to):
    """Extracts images from the zip file to a temporary directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def run_inference_and_compile_video(model, input_folder, video_output_path, prefix, fps=30):
    """Runs inference on images with a specific prefix and compiles them into a video."""
    image_files = sorted([img for img in Path(input_folder).glob(f"{prefix}*.*") if img.is_file()])
    if not image_files:
        print("No images found with the specified prefix.")
        return

    # Initialize video writer
    frame = cv2.imread(str(image_files[0]))
    height, width, layers = frame.shape
    video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for img_path in image_files:
        img = cv2.imread(str(img_path))

        # Run YOLOv5 inference
        results = model(img)

        # Draw bounding boxes on the image
        for pred in results.pred[0]:
            xmin, ymin, xmax, ymax, conf, cls = pred

            # Draw the bounding box
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(img, f"{int(cls)} {conf:.2f}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the processed frame to the video
        video_writer.write(img)

    video_writer.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO model performance and compile results into a video.")
    parser.add_argument('model_path', type=str, help='Path to the trained YOLO model weights file (e.g., best.pt).')
    parser.add_argument('zip_path', type=str, help='Path to the zip file containing images.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where the video will be saved.')
    parser.add_argument('prefix', type=str, help='Prefix of the images to run inference on.')
    parser.add_argument('--video_name', type=str, default=None, help='Name of the output video file (without extension).')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output video.')

    args = parser.parse_args()

    # Determine the output video name
    if args.video_name is None:
        # Use the name of the zip file (without extension) as the default video name
        video_name = Path(args.zip_path).stem
    else:
        video_name = args.video_name
    
    # Ensure the video name ends with .mp4
    video_name += ".mp4"

    # Full path to the output video
    video_output_path = os.path.join(args.output_folder, video_name)

    # Create a temporary directory to extract images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract images from the zip file to the temporary directory
        extract_images(args.zip_path, temp_dir)

        # Load the YOLO model
        model = torch.hub.load('yolov5', 'custom', path=args.model_path, source='local')
        model.eval()

        # Run inference and compile video directly
        run_inference_and_compile_video(model, temp_dir, video_output_path, args.prefix, args.fps)
