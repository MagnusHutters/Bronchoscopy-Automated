

import cv2
import os
import numpy as np  # Make sure you have numpy installed

# Parameters

def generate_images(num_images, resolution, folder_name):



    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for i in range(0, num_images):
        # Create a blank image with a black background
        img = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Put the number on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(i), (50, resolution[1] // 2), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(img, (2, 2), (resolution[0]-2, resolution[1]-2), (255,255,255), thickness=2)
    
        
        # Save the image

        #name with zero filled to 4 digits
        name = f"frame_{i:04}.png"
        file_name = os.path.join(folder_name, name)
        cv2.imwrite(file_name, img)

    print(f'Generated {num_images} images in "{folder_name}"')


#call the function if main
if __name__ == '__main__':
    generate_images(100, (480, 480), 'trainging_data/unlabelled/episode1')
