

import os
import shutil
import tempfile
import json
import cv2
import numpy as np
import tensorflow as tf

from Training.PathTracker import PathTracker


from Training.SETTINGS import *
from Training.ImageMod import preprocess_image
from Training.CVPathsFinder import doCVPathFinding

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


def correctPathForOS(mixed_path):
    # Replace backslashes with slashes to unify separators, then split
    path_parts = mixed_path.replace('\\', '/').split('/')
    
    # Use os.path.join to assemble the parts into a path with the correct separators
    corrected_path = os.path.join(*path_parts)
    
    return corrected_path

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
            
            
            
            image = preprocess_image(image)
            
            
            images.append(image)
            newInput = [frameData["data"]["rotation"], frameData["data"]["bend"], frameData["data"]["extend"]]
            #newState = [frameData["data"]["currentBend"], frameData["data"]["currentRot"]]
            newState = [0, 0]
            inputs.append(newInput)
            states.append(newState)

        


        #load model and predict
        #model = tf.keras.models.load_model("pathModel.keras")
        

        images_array = np.array(images)  # Convert list of images to a numpy array for efficient processing

        # Make predictions on all images at once
        
        #print shape
        print(images_array.shape)
        
        #wait for keypress
        
        
        #predictions = model.predict(images_array) 
        
        #create array of predictions
        
        for i in range(images_array.shape[0]):
            image = images_array[i]
            predictions.append(doCVPathFinding(image))
            
        
        
        

            
        
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
        
        
        image = preprocess_image(image)
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