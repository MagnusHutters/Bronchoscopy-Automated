

import os
import shutil
import tempfile
import json
import cv2
import numpy as np
import tensorflow as tf

from Training.PathTracker import PathTracker


from Training.SETTINGS import *


def doObjectTracking(predictions, imageSize=(256, 256)):
    pathTracker=PathTracker()
    objectsList = []

    for prediction in predictions:
        objects = pathTracker.update(prediction)
        objectsList.append(objects)
        #print(objects)
    return objectsList


def findChosenPath(objectsList):


    done = False


    currentPaths = list(objectsList[0].keys())
    #to list of keys
    currentIndex = 0

    doForwardPass = True


    #create of chosen paths of the same length as the objectsList
    chosenPaths = [0] * len(objectsList)

    while not done:


        if doForwardPass:
            currentIndex += 1
            if currentIndex >= len(objectsList):
                done = True
                for i in range(currentIndex-1, 0, -1):

                    key = currentPaths[0]
                    if key in objectsList[i].keys():
                        chosenPaths[i] = key
                    else:
                        break
            else:
                #check if all keys in the currentPaths are in the new frame
                for key in currentPaths:
                    if key not in objectsList[currentIndex].keys():
                        #remove the key from the currentPaths
                        currentPaths.remove(key)
                        
                        if len(currentPaths) == 0:
                            #for loop backwards until last key is no longer in frame
                            for i in range(currentIndex-1, 0, -1):
                                if key in objectsList[i].keys():
                                    chosenPaths[i] = key
                                else:
                                    break
                            if currentIndex >= len(objectsList) - 1:
                                done = True
                            else:
                                currentPaths = list(objectsList[currentIndex].keys())
    for i in range(len(chosenPaths)):
        #with zfill
        currentPathKeys = list(objectsList[i].keys())
        #print(f"Chosen: {chosenPaths[i]:02d}, {currentPathKeys}")


    return chosenPaths


def compilePaths(objectsList, chosenKeys):
    paths = []
    for i in range(len(objectsList)):
        
        path=[]
        n=0
        for key, value in objectsList[i].items():

            x = value[0]
            y = value[1]
            existance = 1
            chosen = 1 if key == chosenKeys[i] else 0
            path.append([x, y, existance, chosen])


            n+=1
            if n >= 4:
                break

        #fill the rest of the paths with zeros
        for j in range(n, 4):
            path.append([0, 0, 0, 0])

        paths.append(path)

    return paths












def prepEpisode(episodePath):
    imageSize = IMAGE_SIZE

    #create a temporary directory to extract the episode
    with tempfile.TemporaryDirectory() as tempDir:
        #extract the episode
        #print(f"Extracting {episodePath} to {tempDir}")
        print(f"Extracting {episodePath}")
        shutil.unpack_archive(episodePath, tempDir)

        #list the files in the directory
        
        jsonPath = os.path.join(tempDir, "data.json")
        with open(jsonPath, 'r') as f:
            data = json.load(f)



        images = [] #list of 256x256x3 images
        inputs = [] #list of 3 inputs, rotation, bend and extend
        states = [] #list of 2 states, currentRotation and currentBend
        paths = []  #list of 4 paths/holes, each with the format [x, y, existance, chosen] - with x and y normalized to the range [-1, 1]
        predictions = [] #list of 4 paths/holes, each with the format [x, y, existance] - with x and y normalized to the range [-1, 1]


        for frame in data["frames"]:
            frameData = data["frames"][frame]
            

            imagePath = frameData["imagePaths"][0].split("/")[-1]
            image = cv2.imread(os.path.join(tempDir, imagePath))
            image = cv2.resize(image, imageSize)
            image = image / 255.0
            
            images.append(image)
            newInput = [frameData["data"]["rotation"], frameData["data"]["bend"], frameData["data"]["extend"]]
            newState = [frameData["data"]["currentBend"], frameData["data"]["currentRot"]]

            inputs.append(newInput)
            states.append(newState)

        


        #load model and predict
        model = tf.keras.models.load_model("model.keras")
        

        for i in range(len(images)):
            prediction = model.predict(np.array([images[i]]))[0]
            #print(prediction)

            

            '''
            #Display result one image at a time
            val_image = images[i].copy()
            #draw predicted holes
            for hole in prediction:
                x = int((hole[0] + 1) / 2 * imageSize[0])
                y = int((hole[1] + 1) / 2 * imageSize[1])
                #draw holes with opacity based on existence
                existance = hole[2]
                cv2.circle(val_image, (x, y), 10, (float(existance), 0, 0), 2)
            cv2.imshow("image", val_image)
            cv2.waitKey(0)
            '''

            predictions.append(prediction)
        
        objectsList = doObjectTracking(predictions, imageSize)

        chosenKeys = findChosenPath(objectsList)

        paths = compilePaths(objectsList, chosenKeys)


        #to numpy arrays
        images = np.array(images)
        inputs = np.array(inputs)
        states = np.array(states)
        paths = np.array(paths)
        predictions = np.array(predictions)

        return images, inputs, states, paths, predictions







def saveEpisode(path, images, inputs, states, paths, predictions):
    
        #print(paths.tolist())
        #save the episode to the BronchoData folder

        #create new folder for the episode
        
        os.makedirs(path, exist_ok=True)

        #save the images

        data = {}
        data["frames"] = {}

        for i, image in enumerate(images):
            
            name = f"{i:04}.png"
            image_path = os.path.join(path, name)

            image = (image * 255).astype(np.uint8)

            cv2.imwrite(image_path, image)
            frameData = {}
            frameData["image"] = [name]
            frameData["inputs"] = inputs[i].tolist()
            frameData["states"] = states[i].tolist()
            frameData["paths"] = paths[i].tolist()
            frameData["predictions"] = predictions[i].tolist()
        
            data["frames"][i] = frameData
        
        #save the data.json file
        #print(data)
        json_path = os.path.join(path, "data.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)


def loadEpisodeFull(path):
    json_path = os.path.join(path, "data.json")

    with open(json_path, 'r') as f:
        data = json.load(f) 
    
    images = []
    inputs = []
    states = []
    paths = []
    predictions = []

    #print status
    print(f"Loading episode {path}")


    i=0
    for frame in data["frames"]:
        frameData = data["frames"][frame]
        image = cv2.imread(os.path.join(path, frameData["image"][0]))
        image = image / 255.0
        images.append(image)
        inputs.append(frameData["inputs"])
        state=frameData["states"]
        if state == []:
            state = [0, 0]
        states.append([state])
        paths.append(frameData["paths"])
        predictions.append(frameData["predictions"])


        i+=1

        #print status - with return to overwrite the line
        print(f"Loading frame {i}/{len(data['frames'])}", end="\r")



    #to numpy arrays
    images = np.array(images)
    inputs = np.array(inputs)
    states = np.array(states)
    paths = np.array(paths)
    predictions = np.array(predictions)

    print(f"Loaded {len(data['frames'])} frames")
    print(images.shape)
    

    return images, inputs, states, paths, predictions

def loadEpisode(path):

    json_path = os.path.join(path, "data.json")

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    inputs = []
    states = []
    paths = []
    predictions = []

    for frame in data["frames"]:
        frameData = data["frames"][frame]
        imagePath = os.path.join(path, frameData["image"][0])
        images.append(imagePath)
        inputs.append(frameData["inputs"])
        state=frameData["states"]
        if state == []:
            state = [0, 0]
        states.append([state])
        paths.append(frameData["paths"])
        predictions.append(frameData["predictions"])

    #to numpy arrays
    #images = np.array(images)
    #inputs = np.array(inputs)
    #states = np.array(states)
    #paths = np.array(paths)
    #predictions = np.array(predictions)
    

    return images, inputs, states, paths, predictions



    


if __name__ == "__main__":
    prepEpisode("..//Database//24-03-19-16-02-31_0.zip")