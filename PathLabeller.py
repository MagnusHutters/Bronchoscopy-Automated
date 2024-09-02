




import numpy as np
from branchModelTracker import BranchModelTracker




def labelEpisode(episode, modelPath):
    

    #create branch tracker
    branchTracker = BranchModelTracker(modelPath)

    #track the branches of whole episode

    detections = []

    for i in range(len(episode)):

        frame = episode[i]

        #track the branches of the frame

        points, frameDetections = branchTracker.predict(frame.image)


        dictDetection = {}

        for frameDetection in frameDetections:

            dictDetection[frameDetection.id] = frameDetection