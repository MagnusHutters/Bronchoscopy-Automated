



from Training.BasicPaths import *
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf


from Training.SETTINGS import *


def main():


    #Load validation data

    path="Training/Data/PathData"
    
    model = tf.keras.models.load_model("pathModelLabel.keras")
    index = 0

    #find input shape of the model
    input_shape = model.input_shape[1:]


    input_shape = (input_shape[0], input_shape[1])
    val_images, realImageSize = load_images(path, input_shape)
    val_labels = load_labels(path, realImageSize[0], realImageSize[1])

    print(val_images.shape)
    print(val_labels.shape)




    index = 0
    while(True):
        
        #Display result one image at a time
        val_image = val_images[index].copy()

        prediction = model.predict(np.array([val_image]))[0]

        
                
        #draw ground truth holes
        for hole in val_labels[index]:
            x = int((hole[0] + 1) / 2 * IMAGE_SIZE[0])
            y = int((hole[1] + 1) / 2 * IMAGE_SIZE[1])
            print(hole)

            exist = hole[2]
            if exist > 0.5:
                cv2.circle(val_image, (x, y), 8, (1, 0, 0), 2)




        # Draw the holes on the image

        i=1
        for hole in prediction:
            x = int((hole[0] + 1) / 2 * IMAGE_SIZE[0])
            y = int((hole[1] + 1) / 2 * IMAGE_SIZE[1])
            exist = hole[2]
            #draw holes with opacity based on existence

            #scale the existence to 0-255

            #print(hole)

            cv2.circle(val_image, (x, y), 8, (0, float(exist), 0), 2)

            #draw number of the hole
            cv2.putText(val_image, str(i), (x-8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0, float(exist)), 2)

            i+=1

        cv2.imshow("Image", val_image)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        #if d is pressed go to next image
        elif key == ord('d'):
            index += 1
            index %= len(val_images)
        #if a is pressed go to previous image  
        elif key == ord('a'):
            index -= 1
            index %= len(val_images)
        

                



        






if __name__ == '__main__':



    main()