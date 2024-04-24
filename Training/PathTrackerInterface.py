







import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os





from Training.BasicPaths import *
from Training.VideoRecorder import VideoRecorder
from Training.SETTINGS import *
from Training.PathTracker import PathTracker






def main():
    pathInterface= PathTrackerInterface("Training/model.keras")
    input_shape = pathInterface.getInputShape()

    #val_images, realImageSize, originalImages = load_images("Training/Data/PathData", input_shape, saveOriginalImages=True)
    val_images, realImageSize, originalImages = load_images_single_episode("Training/Data/PathData/24-03-19-15-59-18_0", input_shape, saveOriginalImages=True)

    pathInterface.realImageSize = realImageSize

    index = 0
    while(True):

        val_image = val_images[index]

        displayImage = originalImages[index].copy()    
        newIndex, doExit, objects = pathInterface.predictAndTrack(val_image,displayImage)

        #Update index
        index = index+newIndex
        index = index % len(val_images) # make sure index is within bounds

        #print(f"Index: {index}")

        #Exit if doExit is True
        if doExit:
            break



        
        

                
class PathTrackerInterface:
    def __init__(self, modelPath):


        #Load validation data

        path="Training/Data/PathData"

        self.modelPath = modelPath
        #self.dataPath = dataPath
        
        self.model = tf.keras.models.load_model(self.modelPath)
        #index = 0

        
        self.realImageSize = None
        #val_labels = load_labels(path, self.realImageSize[0], self.realImageSize[1])


        #self.video_recorder = VideoRecorder("PathTracker", folder="Output", frame_size=(self.realImageSize[0], self.realImageSize[1]))

        self.pathTracker = PathTracker()

    def getInputShape(self):
        input_shape = self.model.input_shape[1:]
        return (input_shape[0], input_shape[1])
    

    def run(self):
        pass



    def predictAndTrack(self, val_image,displayImage):
        displayImage = displayImage.copy()
        
        #ensure val_image matches input shape
        input_shape=self.getInputShape()
        if val_image.shape[0] != input_shape[0] or val_image.shape[1] != input_shape[1]:
            val_image = cv2.resize(val_image, (input_shape[0], input_shape[1]))
        

        if self.realImageSize is None:
            #set from displayImage
            self.realImageSize = (displayImage.shape[1], displayImage.shape[0])

        #Display result one image at a time
        
        #print(val_image)

        prediction = self.model.predict(np.array([val_image]), verbose=0)[0]

        



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