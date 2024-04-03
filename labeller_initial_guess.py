import cv2
import numpy as np
import os
import shutil
import TestImageGenerator as tig
import json
import time

import tensorflow as tf




class ImageLabeler:
    def __init__(self):
        self.unlabelledPath = "training_data/unlabelled"
        self.labelledPath = "training_data/labelled"
        self.inProgressPath = "training_data/inProgress"
        self.dataBasePath = "Database"
        self.episodePaths = []
        self.currentEpisodePath = None
        self.currentEpisodeName = None
        self.images = []
        self.dataPoints = []
        self.currentImageIndex = 3
        self.doExit = False
        self.imageRes=None
        self.leftMouseButtonDown = False
        self.rightMouseButtonDown = False
        self.dragedItem = None
        self.dragedIndex = None
        self.keys= (1,2,3,4)


        #load tensorflow model
        self.model = tf.keras.models.load_model("Training/model.keras")

    def initialize(self):
        # Clear the unlabelled folder
        for folder in os.listdir(self.unlabelledPath):
            shutil.rmtree(f"{self.unlabelledPath}/{folder}")

        #unzip all episode zip files from the database to the unlabelled folder
        for episode in os.listdir(self.dataBasePath):
            #if file is not a zip file, skip
            if not episode.endswith(".zip"):
                continue
            #get name of episode without extension
            episodeName = episode.split(".")[0]
            #unzip the episode to new folder in unlabelled
            shutil.unpack_archive(f"{self.dataBasePath}/{episode}", f"{self.unlabelledPath}/{episodeName}")

            #shutil.unpack_archive(f"{self.dataBasePath}/{episode}", self.unlabelledPath)        
        

        # Find all episodes in the unlabelled folder
        for episode in os.listdir(self.unlabelledPath):
            self.episodePaths.append(episode)

        self.next_episode()

    
    def next_episode(self):


        #save the current episode
        if self.currentEpisodePath is not None:
            # save the data points to a json file



            # fill in the missing data points
            for i in range(len(self.dataPoints)):
                
                for key in self.keys:
                    if key not in self.dataPoints[i]:
                        self.dataPoints[i][key] = {"x": 0, "y": 0, "existance": 0}
                    else:
                        self.dataPoints[i][key]["existance"] = 1

                    

            data = {"images": self.dataPoints}
            with open(f"{self.currentEpisodePath}/labels.json", 'w') as f:
                json.dump(data, f, indent=4)

            # Move the episode to the labelled folder
            os.rename(self.currentEpisodePath, f"{self.labelledPath}/{self.currentEpisodeName}")



        # Clear in progress folder of folders
        for folder in os.listdir(self.inProgressPath):
            shutil.rmtree(f"{self.inProgressPath}/{folder}")

        episodePath = self.episodePaths.pop(0)
        os.rename(f"{self.unlabelledPath}/{episodePath}", f"{self.inProgressPath}/{episodePath}")
        self.currentEpisodePath = f"{self.inProgressPath}/{episodePath}"
        self.currentEpisodeName = episodePath

        # Load the images from the episode


        self.images = []
        self.dataPoints = []


        images = os.listdir(self.currentEpisodePath)
        #extract number from name and sort by number
        #remove all files that are not images
        images = [image for image in images if image.endswith(".png") or image.endswith(".jpg")]


        images.sort(key=lambda x: int(x.split("_")[0].split("e")[-1]))


        for image in images:
            #extract number from name

            #print(image)

            #if image is jpg or png
            if image.endswith(".png") or image.endswith(".jpg"):

                

                image = cv2.imread(f"{self.currentEpisodePath}/{image}")

                modelInputShape = self.model.input_shape[1:3]
                imageScaled = cv2.resize(image, (modelInputShape[1], modelInputShape[0]))


                #predict the holes
                prediction = self.model.predict(np.array([imageScaled]))[0]

                #scale the prediction to the image size
                for i in range(len(prediction)):
                    prediction[i][0] = (prediction[i][0] + 1) / 2 * image.shape[1]
                    prediction[i][1] = (prediction[i][1] + 1) / 2 * image.shape[0]


                
                #add the prediction to the dataPoints
                self.dataPoints.append({i+1: {"x": int(prediction[i][0]), "y": int(prediction[i][1])} for i in range(4)}) 

                self.images.append(image)




        

        
        self.imageRes = self.images[0].shape[:2]

        self.currentImageIndex = 0



    def show_images(self):
        image1 = self.draw_image(self.currentImageIndex - 1)
        image2 = self.draw_image(self.currentImageIndex)
        # Stack the images horizontally
        stackedImages = np.hstack((image1, image2))
        cv2.imshow('frame', stackedImages)

    def draw_image(self, index):

        #print(f"Drawing image {index} of {len(self.images)}")

        if index < 0 or index >= len(self.images):
            dimensions = (self.imageRes[1], self.imageRes[0], 3)
            return np.zeros(dimensions, dtype=np.uint8)

        image = self.images[index].copy()
        # Draw the data points

        #if datapoints exist for the image
        if len(self.dataPoints) > index:

            imageDataPoints = self.dataPoints[index]
            for key, dataPoint in imageDataPoints.items():
                x = dataPoint["x"]
                y = dataPoint["y"]
                cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
                cv2.putText(image, str(key), (x-10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return image


    def imageClickEvent(self, event, x, y, flags, param, imageIndex):
    

        if imageIndex < 0 or imageIndex >= len(self.images):
            return



        #if right mouse is clicked, remove point if within n pixels, if not add point






        if event == cv2.EVENT_RBUTTONDOWN:
            #if within bounds of image¨
            if x >= 0 and x < self.imageRes[1] and y >= 0 and y < self.imageRes[0]:

                itemToDelete = None
                for key, dataPoint in self.dataPoints[imageIndex].items():
                    x0 = dataPoint["x"]
                    y0 = dataPoint["y"]
                    if abs(x - x0) < 60 and abs(y - y0) < 60:
                        itemToDelete = key
                if itemToDelete is not None:
                    del self.dataPoints[imageIndex][itemToDelete]
                else:       

                    for i in (1,2,3,4):
                        #if key does not exist in dataPoints
                        if i not in self.dataPoints[imageIndex]:
                            self.dataPoints[imageIndex][i] = {"x": x, "y": y}
                            break

        
        #drag point if left mouse is clicked
        
        if event == cv2.EVENT_LBUTTONDOWN:
            #if within bounds of image
            if x >= 0 and x < self.imageRes[1] and y >= 0 and y < self.imageRes[0]:

                for key, dataPoint in self.dataPoints[imageIndex].items():
                    x0 = dataPoint["x"]
                    y0 = dataPoint["y"]
                    if abs(x - x0) < 60 and abs(y - y0) < 60:
                        self.dragedItem = key
                        self.dragedIndex = imageIndex
                        
                        self.dataPoints[imageIndex][self.dragedItem] = {"x": x, "y": y}
                        break
        if event == cv2.EVENT_MOUSEMOVE:
            if self.leftMouseButtonDown and self.dragedItem is not None and self.dragedIndex == imageIndex:
                #cap x and y to image bounds
                xBounded = min(max(x, 0), self.imageRes[1])
                yBounded = min(max(y, 0), self.imageRes[0])
                self.dataPoints[imageIndex][self.dragedItem] = {"x": xBounded, "y": yBounded}

        if event == cv2.EVENT_LBUTTONUP:
            self.dragedItem = None
            self.dragedIndex = None


            




    def click_event(self, event, x, y, flags, param):
        

        if event == cv2.EVENT_LBUTTONDOWN:
            self.leftMouseButtonDown = True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.rightMouseButtonDown = True
        if event == cv2.EVENT_LBUTTONUP:
            self.leftMouseButtonDown = False
        if event == cv2.EVENT_RBUTTONUP:
            self.rightMouseButtonDown = False


        leftX = x
        leftY = y

        rightX = x - self.imageRes[1]
        rightY = y


        self.imageClickEvent(event, leftX, leftY, flags, param, self.currentImageIndex - 1)
        self.imageClickEvent(event, rightX, rightY, flags, param, self.currentImageIndex)


        print(f'x: {x}, y: {y}, left: {self.leftMouseButtonDown}, right: {self.rightMouseButtonDown}')

    def handle_key(self, key):
        #if key is not None (no key is pressed 
        if key == -1:
            return
        
        print(key)
        if key == 27:  # Escape key
            self.doExit = True
        elif key == 13:  # Enter key
            self.next_episode()

        elif key == 97:  # Left arrow key (A in WASD)
            self.currentImageIndex -= 1
            if self.currentImageIndex < 0:
                self.currentImageIndex = 0
            if self.dragedItem is not None:
                self.dragedIndex = self.currentImageIndex
                #move the draged item to the new image
                self.dataPoints[self.currentImageIndex][self.dragedItem] = self.dataPoints[self.currentImageIndex + 1][self.dragedItem].copy()
        elif key == 100:  # Right arrow key (D in WASD)
            self.currentImageIndex += 1
            if self.currentImageIndex >= len(self.images):
                self.currentImageIndex = len(self.images) - 1
            else:
                #if datapoints of new image is empty, copy from previous image
                if len(self.dataPoints[self.currentImageIndex]) == 0:
                    self.dataPoints[self.currentImageIndex] = self.dataPoints[self.currentImageIndex - 1].copy()
                
                #move the draged item to the new image
            if self.dragedItem is not None:
                self.dragedIndex = self.currentImageIndex
                #move the draged item to the new image
                self.dataPoints[self.currentImageIndex][self.dragedItem] = self.dataPoints[self.currentImageIndex - 1][self.dragedItem].copy()

        #if delete key is pressed, delete all datapoint after current image
        elif key == 8:
            for i in range(self.currentImageIndex+1, len(self.dataPoints)):
                self.dataPoints[i] = {}

            
        




    def main(self):
        self.initialize()

        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.click_event)

        while not self.doExit:
            start_time = time.time()  # Record start time

            key = cv2.waitKey(1)

            self.handle_key(key)
            self.show_images()

            end_time = time.time()
            elapsed_time = end_time - start_time

            minimum_time = 1 / 10  # 10 FPS

        cv2.destroyAllWindows()

if __name__ == "__main__":
    labeler = ImageLabeler()
    labeler.main()