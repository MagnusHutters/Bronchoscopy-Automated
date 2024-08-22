
import cv2
import glob
import argparse

import os
import cv2
import shutil
from pathlib import Path



class ImageLabelHandler:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        
        # Automatically determine the labels folder
        image_dir = Path(image_path).parent
        self.labels_folder = os.path.join(image_dir.parent, "labels")
        self.label_path = os.path.join(self.labels_folder, f"{Path(image_path).stem}.txt")
        self.labels = []
        hasLabel = False

        self.min_area = 16

        # Load labels if they exist
        if os.path.exists(self.label_path):
            hasLabel = True
            self._load_labels()

    def _load_labels(self):
        with open(self.label_path, "r") as file:
            for line in file:
                class_id, center_x, center_y, bbox_width, bbox_height = map(float, line.strip().split())
                img_height, img_width, _ = self.image.shape
                xmin = int((center_x - bbox_width / 2) * img_width)
                xmax = int((center_x + bbox_width / 2) * img_width)
                ymin = int((center_y - bbox_height / 2) * img_height)
                ymax = int((center_y + bbox_height / 2) * img_height)
                self.labels.append((xmin, ymin, xmax, ymax))

    def save(self, dest_images_folder, dest_labels_folder):
        dest_image_path = os.path.join(dest_images_folder, os.path.basename(self.image_path))
        dest_label_path = os.path.join(dest_labels_folder, f"{Path(self.image_path).stem}.txt")

        # Save image
        shutil.copyfile(self.image_path, dest_image_path)

        # Save labels
        with open(dest_label_path, "w") as file:
            for xmin, ymin, xmax, ymax in self.labels:

                #Normalize the coordinates
                img_height, img_width, _ = self.image.shape

                xmin = xmin / img_width
                xmax = xmax / img_width
                ymin = ymin / img_height
                ymax = ymax / img_height

                #limit the coordinates to the image
                xmin = min(max(xmin, 0), 1)
                xmax = min(max(xmax, 0), 1)
                ymin = min(max(ymin, 0), 1)
                ymax = min(max(ymax, 0), 1)

                center_x = (xmin + xmax) / 2
                center_y = (ymin + ymax) / 2
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin


                file.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

    def add_label(self, x1, y1, x2, y2):
    # Ensure that xmin, ymin are the top-left corner, and xmax, ymax are the bottom-right
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)
        
        # Calculate the area of the bounding box
        area = (xmax - xmin) * (ymax - ymin)
        
        # Check if the area meets the minimum area requirement
        if area >= self.min_area:
            # Append the correctly ordered coordinates as the new label
            self.labels.append((xmin, ymin, xmax, ymax))
        else:
            print(f"Label with area {area} is too small and will not be added.")

    def delete_label(self, x, y):
        label_idx = self.get_selected(x, y)
        if label_idx is not None:
            del self.labels[label_idx]

    def draw_labels(self, highlight_idx=None):
        for i, (xmin, ymin, xmax, ymax) in enumerate(self.labels):
            color = (0, 0, 255)
            cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), color, 1)

            if i == highlight_idx:
                overlay = self.image.copy()
                overlayColor = (122, 122, 255)
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), overlayColor, -1)
                cv2.addWeighted(overlay, 0.5, self.image, 0.5, 0, self.image)




    def reset_image(self, highlight_idx=None):
        self.image = cv2.imread(self.image_path)
        self.draw_labels(highlight_idx)

    def move(self, dest_images_folder, dest_labels_folder):
        dest_image_path = os.path.join(dest_images_folder, os.path.basename(self.image_path))
        dest_label_path = os.path.join(dest_labels_folder, f"{Path(self.image_path).stem}.txt")
        shutil.move(self.image_path, dest_image_path)
        
        #create label if 
        
        with open(dest_label_path, "w") as file:
            for xmin, ymin, xmax, ymax in self.labels:
                img_height, img_width, _ = self.image.shape
                center_x = (xmin + xmax) / 2 / img_width
                center_y = (ymin + ymax) / 2 / img_height
                bbox_width = (xmax - xmin) / img_width
                bbox_height = (ymax - ymin) / img_height
                file.write(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")



    def get_selected(self, x, y):
        """Returns the index of the best label that contains the point (x, y), preferring the smallest one."""
        best_idx = None
        best_area = float('inf')  # Initialize with infinity for comparison

        for i, (xmin, ymin, xmax, ymax) in enumerate(self.labels):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                area = (xmax - xmin) * (ymax - ymin)
                if area < best_area:
                    best_area = area
                    best_idx = i

        return best_idx

def main(source_folder, dest_folder):
    # Locate all images recursively in the source folder
    images = glob.glob(os.path.join(source_folder, "**", "*.*"), recursive=True)
    images = [img for img in images if img.endswith(('.jpg', '.jpeg', '.png'))]

    dest_images_folder = os.path.join(dest_folder, "images")
    dest_labels_folder = os.path.join(dest_folder, "labels")

    # Create destination folders if they don't exist
    Path(dest_images_folder).mkdir(parents=True, exist_ok=True)
    Path(dest_labels_folder).mkdir(parents=True, exist_ok=True)


    displayEnlargeFactor = 2


    current_index = 0
    num_images = len(images)

    drawing = False
    ix, iy = -1, -1
    highlight_idx = None



    def show_image(image):

        # Resize the image for better visualization
        newShape = (image.shape[1] * displayEnlargeFactor, image.shape[0] * displayEnlargeFactor)
        image = cv2.resize(image, newShape)


        cv2.imshow("Image", image)
        
    



    def on_mouse(event, x, y, flags, param):

        x = x // displayEnlargeFactor
        y = y // displayEnlargeFactor


        nonlocal drawing, ix, iy, highlight_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            
            #limit the rectangle to the image
            x = min(max(x, 0), handler.image.shape[1])
            y = min(max(y, 0), handler.image.shape[0])
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                img_copy = handler.image.copy()

                #limit the rectangle to the image
                x = min(max(x, 0), img_copy.shape[1])
                y = min(max(y, 0), img_copy.shape[0])
                

                cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)

                show_image(img_copy)
            else:
                # Highlight the label that the mouse is over
                highlight_idx = handler.get_selected(x, y)
                handler.reset_image(highlight_idx)
                
                show_image(handler.image)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            handler.add_label(ix, iy, x, y)
            handler.reset_image()
            #cv2.imshow("Image", handler.image)
            show_image(handler.image)
        elif event == cv2.EVENT_RBUTTONDOWN:
            handler.delete_label(x, y)
            handler.reset_image()
            show_image(handler.image)


    imageHandlers = []
    for img in images:
        imageHandlers.append(ImageLabelHandler(img))

    isNewImage = True

    while True:
        handler = imageHandlers[current_index]
        handler.reset_image()
        
        show_image(handler.image)
        cv2.setMouseCallback("Image", on_mouse)


        if isNewImage:
            print(f"Editing image {current_index + 1} of {num_images}: {handler.image_path}")
            isNewImage = False

        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'):  # Move to next image
            current_index = (current_index + 1) % num_images
            isNewImage = True
        elif key == ord('a'):  # Move to previous image
            current_index = (current_index - 1) % num_images
            isNewImage = True
        elif key == 13 or key == 27:  # Enter or Escape to save and exit
            for img_handler in imageHandlers:

                img_handler.move(dest_images_folder, dest_labels_folder)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotation correction tool.")
    parser.add_argument('source_folder', type=str, help='Path to the source folder containing images.')
    parser.add_argument('dest_folder', type=str, help='Path to the destination folder to save corrected images and labels.')

    args = parser.parse_args()
    main(args.source_folder, args.dest_folder)