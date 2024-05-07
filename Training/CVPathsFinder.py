import cv2
import os
import numpy as np
import re
#from Training.ImageMod import preprocess_image

def find_images(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    image_paths.sort(key=lambda x: (os.path.dirname(x), int(re.search(r'frame(\d+)_', os.path.basename(x)).group(1))))

    return image_paths




def preprocess_image_tresholding(original, target_size=(64, 64)):


    image = original.copy()
    # Step 1: Convert the image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize the image if it's not already the correct size
    if image.shape[0] != target_size[0] or image.shape[1] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply Gaussian Blur for noise reduction
    image = cv2.GaussianBlur(image, (3,3), 0)

    # Step 4: Ensure the image is in 0-255 uint8 format if not already
    if image.dtype != np.uint8:
        image = (255 * image).clip(0, 255).astype(np.uint8)
    
    # Step 5: Apply histogram equalization
    #clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    #image = clahe.apply(image)
    #image = cv2.equalizeHist(image)

    #image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 2)
    #image = cv2.GaussianBlur(image, (7, 7), 0)

    #mean_val = np.mean(image)
    
    # Apply a simple threshold based on the global mean




    


    #Initialize as black image
    #pastThreshold = np.zeros(image.shape, np.uint8)

    points = []

    count=0
    for i in range(0,80,6):


        if(len(points) >0):
            count+=1
        if count > 8:
            break

        _,threshold = cv2.threshold(image,i,255,cv2.THRESH_BINARY_INV)

        #closing then opening
        kernel = np.ones((3,3),np.uint8)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        #threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)


        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            
            doSkip = False
            #check if any of the already found points is in the contour, if so, skip
            for point in points:
                if cv2.pointPolygonTest(contour, point, False) > 0:
                    doSkip = True                                
            if not doSkip:
                area = cv2.contourArea(contour)

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                distFromCenter = np.sqrt((cX-target_size[0]/2)**2 + (cY-target_size[1]/2)**2)
                

                nomalizedDistFromCenter = distFromCenter / np.sqrt((target_size[0]/2)**2 + (target_size[1]/2)**2)
                invNormalizedDistFromCenter = 1-nomalizedDistFromCenter
                #print(f"Area: {area}")
                if area*((invNormalizedDistFromCenter**2)+0.1) > 10:

                    
                    

                    points.append((cX,cY))

    for point in points:
        cv2.circle(image, point, 5, 255, 1)

    display_images(image, threshold)


    # Combine the thresholded images into a single multi-channel image
    #image = np.stack((image1, image2, image3), axis=-1)
    
    
    

    # Step 6: Convert back to float32 from 0 to 1
    #image = image.astype('float32') / 255
    
    return image, points, target_size


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


def doCVPathFinding(image):

    #print(f"Image: {image*255}")
    img, rawPoints, targetSize = preprocess_image_tresholding(image)

    #normalize points from ranbge (0,0 - target size) to (-1,-1 - 1,1)
    points = np.zeros((len(rawPoints),3))
    for i in range(len(rawPoints)):
        x=rawPoints[i][0]
        y=rawPoints[i][1]
        #print(f"X: {x}, Y: {y}")
        newX = (x/targetSize[0])*2-1
        newY = (y/targetSize[1])*2-1
        #print(f"NewX: {newX}, NewY: {newY}")
        points[i] = [newX,newY,1]
        

    #print(f"Points: {points}")
    
    
    if len(points) > 4:#limit to 4 points
        points = points[0:4]
    elif len(points) < 4: #fill up to 4 points
        for i in range(len(points),4):
            #add to end of numpy array
            points = np.append(points,[[0,0,0]],axis=0)
    
    
    
    return points
    



def main():
    directory = 'Training/Data/PathData'  # Change this to your images' directory
    images = find_images(directory)
    current_index = 4790
    increment = 20
    scale_factor = 6
    #print(f"Found {len(images)} images")

    while True:
        # Load the current image
        image_path = images[current_index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image {image_path}")
            continue
        
        # Preprocess and show the image
        #print(image)
        processed_image, _, _ = preprocess_image_tresholding(image.copy())

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