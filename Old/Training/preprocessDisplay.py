import cv2
import os
import numpy as np
import re
from Training.ImageMod import preprocess_image

def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    image_paths.sort(key=lambda x: (os.path.dirname(x), int(re.search(r'frame(\d+)_', os.path.basename(x)).group(1))))

    return image_paths

def display_images(original, processed):
    scale_factor = 3  # Scaling factor to enlarge the images for better visibility

    #




    # Check if the original image is grayscale; convert to color
    if len(original.shape) == 2:  # Grayscale images have no color channels
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    # Check if the processed image is grayscale; convert to color
    if len(processed.shape) == 2:  # Grayscale images have no color channels
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    


    #check the format of the iamges and convert to 0-255 uint8 format if necessary
    if original.dtype != np.uint8:
        original = (255 * original).clip(0, 255).astype(np.uint8)
    if processed.dtype != np.uint8:
        processed = (255 * processed).clip(0, 255).astype(np.uint8)


    # Resize images for better viewing
    original_resized = cv2.resize(original, (128 * scale_factor, 128 * scale_factor), interpolation=cv2.INTER_NEAREST)
    processed_resized = cv2.resize(processed, (128 * scale_factor, 128 * scale_factor), interpolation=cv2.INTER_NEAREST)
    
    # Combine both images horizontally
    combined_image = np.hstack((original_resized, processed_resized))
    
    # Show the combined image
    cv2.imshow('Original (Left) vs. Processed (Right)', combined_image)




def main():
    directory = 'Training/Data/PathData'  # Change this to your images' directory
    images = find_images(directory)
    current_index = 0
    increment = 10
    scale_factor = 6
    print(f"Found {len(images)} images")

    while True:
        # Load the current image
        image_path = images[current_index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        # Preprocess and show the image
        processed_image = preprocess_image(image.copy())

        display_images(image, processed_image)

        # Keyboard controls
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):  # Move to the previous image
            current_index = (current_index - increment) % len(images)
        elif key == ord('d'):  # Move to the next image
            current_index = (current_index + increment) % len(images)
        print(f"Current index: {current_index}, Image path: {image_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()