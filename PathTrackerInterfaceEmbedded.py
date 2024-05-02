







import cv2
#from scipy.optimize import linear_sum_assignment
import numpy as np
#from tensorflow.keras.models import Model, load_model
#from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
#from tensorflow.keras.preprocessing.image import img_to_array
#import tensorflow as tf
import os

from TFLiteModel import TFLiteModel




from Training.PathTracker import PathTracker



                
class PathTrackerInterface:
    def __init__(self, modelPath, embedded=True):


        #Load validation data

        path="Training/Data/PathData"

        self.modelPath = modelPath
        #self.dataPath = dataPath
        
        self.model = TFLiteModel(modelPath)
        # Load the TFLite model and allocate tensors
        
        
        
        #self.model = tf.keras.models.load_model(self.modelPath)
        #index = 0

        
        self.realImageSize = None
        #val_labels = load_labels(path, self.realImageSize[0], self.realImageSize[1])


        #self.video_recorder = VideoRecorder("PathTracker", folder="Output", frame_size=(self.realImageSize[0], self.realImageSize[1]))

        self.pathTracker = PathTracker()

    
    

    def run(self):
        pass



    def predictAndTrack(self, val_image,displayImage):
        print("Predicting and tracking")
        
        displayImage = displayImage.copy()
        
        #ensure val_image matches input shape
        #input_shape=self.getInputShape()
        #if val_image.shape[0] != input_shape[0] or val_image.shape[1] != input_shape[1]:
        #    val_image = cv2.resize(val_image, (input_shape[0], input_shape[1]))
        

        if self.realImageSize is None:
            #set from displayImage
            self.realImageSize = (displayImage.shape[1], displayImage.shape[0])

        #Display result one image at a time
        
        #print(val_image)

        
        #val_image to numpy array
        
        val_image = np.array(val_image)

        prediction = self.model.predict(val_image)


        objects = self.pathTracker.update(prediction)



        #draw ground truth holes
            


        # Draw the holes on the image


        #draw predicted holes
        i=1
        for prediction in prediction:
            x = int((prediction[0] + 1) / 2 * self.realImageSize[0])
            y = int((prediction[1] + 1) / 2 * self.realImageSize[1])
            #draw holes with opacity based on existence
            existance = prediction[2]

            
            if existance > 0.5:
                cv2.circle(displayImage, (x, y), 10, (0, 0, float(existance)), 2)

        #draw tracked holes
        i=1
        for key, value in objects.items():
            x = int((value[0] + 1) / 2 * self.realImageSize[0])
            y = int((value[1] + 1) / 2 * self.realImageSize[1])
            #draw holes with opacity based on existence

            #scale the existence to 0-255


            cv2.circle(displayImage, (x, y), 8, (0, 1, 0), 2)

            #draw number of the hole
            cv2.putText(displayImage, str(key), (x-8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1,1,1), 2)

            i+=1



        doExit = False
        index=0

        #cv2.imshow("Image", displayImage)
        #key = cv2.waitKey(0)
        #if key == ord('q'):
        #    doExit = True
        #if d is pressed go to next image
        #elif key == ord('d'):
        #    index += 1
        #if a is pressed go to previous image  
        #elif key == ord('a'):
        #    index -= 1

        return index, doExit, objects







if __name__ == '__main__':

    


    main()