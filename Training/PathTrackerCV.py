







import cv2
from scipy.optimize import linear_sum_assignment
import numpy as np

#import tensorflow as tf


#from Training.BasicPaths import *
from Training.VideoRecorder import VideoRecorder
from Training.SETTINGS import *
from Training.CVPathsFinder import doCVPathFinding



#from Training.BasicPaths import load_images



class PathTracker:


    def __init__(self):
        self.nextObjectID = 0
        self.objects = {}
        self.confidence = {}
        self.maxConfidence = 11



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
        
        detections = [[detection[0], detection[1]] for detection in detections if detection[2] > 0.5]
        np.random.shuffle(detections)
        inputCentroids = np.array(detections)

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())


        if len(detections) == 0:
            for objectID in objectIDs:
                self.confidence[objectID] -= 1
                if self.confidence[objectID] <=0: 
                    self.deregister(objectID)

        elif len(self.objects) == 0:
            
            for i in range(len(detections)):
                self.register(detections[i], self.maxConfidence)
        else:
            

            #print(inputCentroids)
            #print(objectCentroids)
            D = np.linalg.norm(np.array(objectCentroids) - inputCentroids[:, None], axis=2)

            #print(D)
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

        if len(self.objects) == 0:
            #print("No objects left, returning empty dict")
            return {}

        confidentObjects = {key: value for key, value in self.objects.items() if self.confidence[key] >= self.maxConfidence*0.5}
        #if there are no confident objects, return disct of juist most confident object
        if len(confidentObjects) == 0:
            maxConfidenceKey = max(self.confidence, key=self.confidence.get)
            #print(f"No confident objects, returning only most confident object {maxConfidenceKey}")
            confidentObjects = {maxConfidenceKey: self.objects[maxConfidenceKey]}
        return confidentObjects


'''
def main():


    #Load validation data

    path="Training/Data/PathData"
    
    #model = tf.keras.models.load_model("pathModelLabel.keras")
    index = 0

    #find input shape of the model
    #input_shape = model.input_shape[1:]


    #input_shape = (input_shape[0], input_shape[1])
    val_images, realImageSize, originalImages = load_images(path, input_shape, saveOriginalImages=True)
    
    
        
    #val_labels = load_labels(path, realImageSize[0], realImageSize[1])
    

    video_recorder = VideoRecorder("PathTracker", folder="Output", frame_size=(realImageSize[0], realImageSize[1]))

    pathTracker = PathTracker()

    index = 0
    while(True):
        
        #Display result one image at a time
        val_image = val_images[index]
        #print(val_image)

        
        print(val_image)
        prediction = doCVPathFinding(val_image)
        


        #print(prediction)
        objects = pathTracker.update(prediction)



        #draw ground truth holes
        displayImage = originalImages[index].copy()        


        # Draw the holes on the image


        #draw predicted holes
        i=1
        for prediction in prediction:
            x = int((prediction[0] + 1) / 2 * realImageSize[0])
            y = int((prediction[1] + 1) / 2 * realImageSize[1])
            #draw holes with opacity based on existence
            #existance = prediction[2]

            
            cv2.circle(displayImage, (x, y), 10, (0, 0, 1), 2)

        #draw tracked holes
        i=1
        for key, value in objects.items():
            x = int((value[0] + 1) / 2 * realImageSize[0])
            y = int((value[1] + 1) / 2 * realImageSize[1])
            #draw holes with opacity based on existence

            #scale the existence to 0-255


            cv2.circle(displayImage, (x, y), 8, (0, 1, 0), 2)

            #draw number of the hole
            cv2.putText(displayImage, str(key), (x-8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (1,1,1), 2)

            i+=1


        video_recorder.write(displayImage)
        cv2.imshow("Image", displayImage)
        key = cv2.waitKey(0)
        if key == ord('q'):
            video_recorder.release()
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
    '''