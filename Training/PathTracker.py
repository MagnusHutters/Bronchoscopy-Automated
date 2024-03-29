







from BasicPaths import *
import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf





class PathTracker:


    def __init__(self):
        self.nextObjectID = 0
        self.objects = {}
        self.confidence = {}
        self.maxConfidence = 5



        self.maxDistance = 0.6
        self.smoothingFactor = 0.5

    def register(self, centroid, initialConfidence=1):
        self.objects[self.nextObjectID] = centroid
        self.confidence[self.nextObjectID] = initialConfidence
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.confidence[objectID]

    def update(self, detections): #detections is a list of points with likelihood [x, y, likelihood]
        np.random.shuffle(detections)
        rawPredictions = detections

        #randomize the order of the detections
        
        #remove detections whith low likelihood (less than 0.5)
        detections = [[detection[0], detection[1]] for detection in detections if detection[2] > 0.5]
        #remove likelihood from the detections
        

        #if there are no detections, take highest likelihood detection and use it as the only detection
        if len(detections) == 0:
            detection = max(rawPredictions, key=lambda x: x[2])
            detections = [[detection[0], detection[1]]]
            print("No detections")
            print(f"Using {detections[0]} as the only detection, detections is as follows: {detections}")

            
        inputCentroids = np.array(detections)
        if len(self.objects) == 0:
            for i in range(len(detections)):
                self.register(detections[i], self.maxConfidence)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            print(inputCentroids)
            print(objectCentroids)
            D = np.linalg.norm(np.array(objectCentroids) - inputCentroids[:, None], axis=2)

            print(D)
            rows, cols = linear_sum_assignment(D)

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance * (self.confidence[objectIDs[col]] / self.maxConfidence):
                    continue

                objectID = objectIDs[col]
                newCentroid = inputCentroids[row]
                currentCentroid = self.objects[objectID]

                # Apply smoothing by updating the tracker's position partway towards the new detection
                smoothedCentroid = (1 - self.smoothingFactor) * np.array(currentCentroid) + self.smoothingFactor * np.array(newCentroid)
                self.objects[objectID] = smoothedCentroid



                self.confidence[objectID] += 1
                if self.confidence[objectID] > self.maxConfidence:
                    self.confidence[objectID] = self.maxConfidence

                usedRows.add(row)
                usedCols.add(col)

            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            for col in unusedCols:
                self.confidence[objectIDs[col]] -= 1
                if self.confidence[objectIDs[col]] <=0: 
                    self.deregister(objectIDs[col])

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                self.register(inputCentroids[row])



        confidentObjects = {key: value for key, value in self.objects.items() if self.confidence[key] >= self.maxConfidence*0.5}
        #if there are no confident objects, return disct of juist most confident object
        if len(confidentObjects) == 0:
            maxConfidenceKey = max(self.confidence, key=self.confidence.get)
            print(f"No confident objects, returning only most confident object {maxConfidenceKey}")
            confidentObjects = {maxConfidenceKey: self.objects[maxConfidenceKey]}
        return confidentObjects



def main():


    #Load validation data

    path="Training/Data/PathData"
    
    model = tf.keras.models.load_model("Training/model.keras")
    index = 0

    #find input shape of the model
    input_shape = model.input_shape[1:]


    input_shape = (input_shape[0], input_shape[1])
    val_images, realImageSize = load_images(path, input_shape)
    val_labels = load_labels(path, realImageSize[0], realImageSize[1])


    pathTracker = PathTracker()


    index = 0
    while(True):
        
        #Display result one image at a time
        val_image = val_images[index].copy()

        prediction = model.predict(np.array([val_image]))[0]

        objects = pathTracker.update(prediction)
                


        # Draw the holes on the image


        #draw predicted holes
        i=1
        for prediction in prediction:
            x = int((prediction[0] + 1) / 2 * imagesSize[0])
            y = int((prediction[1] + 1) / 2 * imagesSize[1])
            #draw holes with opacity based on existence
            existance = prediction[2]

            
            cv2.circle(val_image, (x, y), 10, (float(existance), 0, 0), 2)

        #draw tracked holes
        i=1
        for key, value in objects.items():
            x = int((value[0] + 1) / 2 * imagesSize[0])
            y = int((value[1] + 1) / 2 * imagesSize[1])
            #draw holes with opacity based on existence

            #scale the existence to 0-255


            cv2.circle(val_image, (x, y), 8, (0, 1, 0), 2)

            #draw number of the hole
            cv2.putText(val_image, str(key), (x-8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1,1,1), 2)

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