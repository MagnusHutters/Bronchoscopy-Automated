

import numpy as np
from branchModelTracker import BranchModelTracker, Detection
from DataHandling.Episode import EpisodeManager, Episode

import copy
from Input import Input
import random

import cv2

#from brochoRobVisual import brochoRobClass
from Timer import Timer




from VisualServoing import *


    








def labelFrame(episode, index, useGuess = False, doVisualize = True, maxDist = 5000):
    if doVisualize:
        frame = episode.get_frame(index, getImage = True)
    else:
        frame = episode.get_frame(index, getImage = True)
    
    action = ""
    limitCount = 0
    while action == "":
        if doVisualize:
            image = frame.image

            displayImage = image.copy()

        data = frame.data

        paths = data["paths"]


        state = frame.state

        rotationDegrees = state["rotationReal_deg"]
        bendDegrees = state["bendReal_deg"]
        extensionMM = state["extensionReal_mm"]

        if doVisualize:
            print(f"Rotation: {rotationDegrees}, bend: {bendDegrees}, extension: {extensionMM}")

        #bendDegreesToPixelsConstant = -2.2
        bendDegreesToPixelsConstant = 1.5

        bendOffset = int(bendDegreesToPixelsConstant*bendDegrees)

        currentJoints = [bendDegrees, rotationDegrees, extensionMM]


        #robVisual = brochoRobClass()

        





        if doVisualize:
            drawRotationAccess(displayImage, rotationDegrees, bendDegrees, bendMultiplier=bendDegreesToPixelsConstant)

        #draw line from center of image at angle of rotation

        

        #visualize paths on the frame using cv2

        keys = list(paths.keys())

        random.shuffle(keys)
        key1 = keys[0]
                
        pathData = paths[key1]



        imageSize = (400, 400)
        bbox = pathData["bbox"]
        targetCenter = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)



        #relativeChangeGuess, _ =robVisual.visualservoingcontrol(targetCenter, 0.005, imageSize, currentJoints)

        #find index of highest abs value in relativeChangeGuess
        #axis = np.argmax(np.abs(relativeChangeGuess))

        #guessInput = Input.fromInt(axis, relativeChangeGuess[axis])
        #print(f"Guess: {guessInput}")

        actionGuess = guessBranch(state, targetCenter, imageSize, currentJoints, doVisualize, maxDist, limitCount)

        if actionGuess == "" and useGuess:
            limitCount+=1
            continue


        
        #print(f"Guess1: bend: {guess1[0]:.2f}, rotation: {guess1[1]:.2f}, extension: {guess1[2]:.2f}")
        #print(f"Guess2: bend: {guess2[0]:.2f}, rotation: {guess2[1]:.2f}, extension: {guess2[2]:.2f}")






        if doVisualize:
            showBranch(displayImage, pathData, currentBend = bendDegrees, bendMultiplier = bendDegreesToPixelsConstant)

            


            cv2.imshow("Frame", displayImage)

            key = cv2.waitKey(0) & 0xFF


            psossibleActions = ["f", "b", "l", "r", "u", "d"]
            action = ""

            #if escape key is pressed, stop labelling
            if key == 27:
                return False, None
            #select action from 
            if key == ord("w"): #if w: u
                action = "u"
            elif key == ord("s"): #if s: d
                action = "d"
            elif key == ord("a"): #if a: l
                action = "l"
            elif key == ord("d"): #if d: r
                action = "r"
            elif key == ord("q"): #if q: b
                action = "b"
            elif key == ord("e"): #if e: f
                action = "f"
            #space is take guess
            elif key == 32:
                action = actionGuess
            #shift is backwards too
            elif key == 16:
                action = "b"

            #print(f"Action: {action}")
        elif useGuess:
            action = actionGuess
            break
        else:
            raise ValueError("Have to visualize or use guess")
        

    
    newFrame = copy.deepcopy(frame)


    action = Input.fromChar(action).toDict()
    newFrame.action = action

    #set chosen path
    newFrame.data["pathId"] = key1


    
    return True, newFrame










def main():


    episodeReader = EpisodeManager(mode = "Read", loadLocation = "DatabaseLabelled")

    episodeCreator = EpisodeManager(mode = "Recording", saveLocation = "DatabaseManualBendOffset", multiProcessing=True)
    episodeCreator.nextEpisode()

    doContinue = True
    episodeLenght = 1000
    
    
    lenght = 0
    
    epsiodesCreated = 0
    episodeNumber = 0
    numEpisodes = 50
    framesCreated =0

    

    actionCounts = {"f": 0, "b": 0, "l": 0, "r": 0, "u": 0, "d": 0}


    for i in range(10):
        episodeNumber = 0
        episodeIndexes = list(range(len(episodeReader)))

        random.shuffle(episodeIndexes)

        for episodeIndex in episodeIndexes:

            episode = episodeReader.nextEpisode(episodeIndex)

            print(f"Epsiode {episodeIndex}")

            
            randomFrames = sorted(random.sample(range(len(episode)), 100))


            for index in randomFrames:
                
                Timer.point("Start")
                curentEpisodeName = episode.name

                doContinue, frame = labelFrame(episode, index, useGuess = True, doVisualize = False, maxDist = 400)
                Timer.point("Labelled")
                if not doContinue:

                    break

                if frame is not None:

                    action = frame.action["char_value"]
                    actionCounts[action] += 1
                    frame.data["originalEpisodeName"] = curentEpisodeName
                    frame.data["origninalFrameIndex"] = index
                    episodeCreator.append(frame)
                    Timer.point("Saved to episode")
                    lenght += 1
                    framesCreated += 1

                    if lenght >= episodeLenght:
                        episodeCreator.nextEpisode()
                        lenght = 0

                    print(f"\rRun: {i} - Episode: {episodeNumber}/{len(episodeIndexes)} - Actions: {actionCounts}, Frames created: {framesCreated}, episodesCreated: {epsiodesCreated}", end="")
                Timer.point("End")
                report = Timer.reset()
                #print(report)
                
                    
            print("")
                    
            episodeNumber += 1
            if not doContinue:
                break

        if not doContinue:
            break



        






        #labelEpisode(episode, "C:/Users/magnu/OneDrive/Misc/BronchoYolo/yolov5/runs/train/branchTraining8-XL/weights/best.pt")


    episodeReader.endEpisode(discard=True)
    episodeCreator.endEpisode()
    




if __name__ == "__main__":
    main()