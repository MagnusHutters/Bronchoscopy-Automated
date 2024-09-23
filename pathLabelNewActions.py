

import numpy as np
from branchModelTracker import BranchModelTracker, Detection
from DataHandling.Episode import EpisodeManager, Episode

import copy
from Input import Input
import random

import cv2

#from brochoRobVisual import brochoRobClass
from Timer import Timer





def drawRotationAccess(image, rotationDegrees, bendDegrees, bendMultiplier = 1):

    rotationDegrees += 90
    rotationLimits = (-170, 170)

    actualImageCenter = (image.shape[1]//2, image.shape[0]//2)


    bendOffset = int(bendDegrees*bendMultiplier)
    imageCenter = (image.shape[1]//2, (image.shape[0]//2)-bendOffset)

    dist = 100
    axes = (dist, dist)

    angle = np.radians(rotationDegrees)

    startAngle = rotationLimits[0]-rotationDegrees
    endAngle = rotationLimits[1]-rotationDegrees

    startAngleRad = np.radians(startAngle)
    endAngleRad = np.radians(endAngle)

    color = (0, 255, 0)

    
    #==================Draw bending limits==================

    bendLimits = 170

    #bendLimitDown   = ((actualImageCenter[1]+bendLimits) + (imageCenter[1]+bendLimits))//2
    #bendLimitUp     = ((actualImageCenter[1]-bendLimits) + (imageCenter[1]-bendLimits))//2

    bendLimitDown = imageCenter[1]+int(bendLimits*bendMultiplier)
    bendLimitUp = imageCenter[1]-int(bendLimits*bendMultiplier)
    
    print(f"bendLimitDown: {bendLimitDown}, bendLimitUp: {bendLimitUp}")

    bendLimitDownPoint = (imageCenter[0], bendLimitDown)
    bendLimitUpPoint = (imageCenter[0], bendLimitUp)

    cv2.line(image, bendLimitDownPoint, (imageCenter[0], imageCenter[1]+125), (255, 0, 0), 2)
    cv2.line(image, (imageCenter[0], imageCenter[1]-125), (imageCenter[0], imageCenter[1]+125), (255, 0, 0), 1)
    cv2.line(image, bendLimitUpPoint, (imageCenter[0], imageCenter[1]-125), (255, 0, 0), 2)

    #draw caps
    cv2.line(image, (imageCenter[0]-5, bendLimitDown), (imageCenter[0]+5, bendLimitDown), (255, 0, 0), 2)
    cv2.line(image, (imageCenter[0]-5, bendLimitUp), (imageCenter[0]+5, bendLimitUp), (255, 0, 0), 2)


    #==================Draw rotation limits==================

    #5 pixels further in than the ellipse
    startAnglePoint1 = (int(imageCenter[0] + (dist-5)*np.cos(startAngleRad)), int(imageCenter[1] + (dist-5)*np.sin(startAngleRad)))
    endAnglePoint1 = (int(imageCenter[0] + (dist-5)*np.cos(endAngleRad)), int(imageCenter[1] + (dist-5)*np.sin(endAngleRad)))

    #5 pixels further out than the ellipse
    startAnglePoint2 = (int(imageCenter[0] + (dist+5)*np.cos(startAngleRad)), int(imageCenter[1] + (dist+5)*np.sin(startAngleRad)))
    endAnglePoint2 = (int(imageCenter[0] + (dist+5)*np.cos(endAngleRad)), int(imageCenter[1] + (dist+5)*np.sin(endAngleRad)))


    #draw line from center of image and straight up
    cv2.ellipse(image, imageCenter, axes, 0, startAngle, endAngle, color, 2)

    cv2.line(image, startAnglePoint1, startAnglePoint2, color, 2)
    cv2.line(image, endAnglePoint1, endAnglePoint2, color, 2)

    cv2.line(image, (imageCenter[0]+5, imageCenter[1]-100), (imageCenter[0], imageCenter[1]-120), color, 2)
    cv2.line(image, (imageCenter[0]-5, imageCenter[1]-100), (imageCenter[0], imageCenter[1]-120), color, 2)

    #cv2.line(image, imageCenter, (imageCenter[0], imageCenter[1]+int(bendDegrees)), (0,0,255), 2)


    #draw bending limit

    
    





def showBranch(image, pathData, currentBend = 0, bendMultiplier = 1):

    bendOffset = int(currentBend*bendMultiplier)

    
    #path = pathData["path"]
    bbox = pathData["bbox"]

    imageCenter = (image.shape[1]//2, image.shape[0]//2-bendOffset)
    actualImageCenter = (image.shape[1]//2, image.shape[0]//2)

    x1, y1, x2, y2 = bbox

    center = ((x1+x2)/2, (y1+y2)/2)

    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    #draw line from center of image to center of bbox

    distance = np.sqrt((center[0]-imageCenter[0])**2 + (center[1]-imageCenter[1])**2)

    cv2.line(image, imageCenter, (int(center[0]), int(center[1])), (0, 0, 255), 2)

    #draw line from center of image to opposite of bbox
    cv2.line(image, imageCenter, (int(2*imageCenter[0]-center[0]), int(2*imageCenter[1]-center[1])), (180, 0, 0), 1)


    #write distance on image

    

    cv2.putText(image, f"Distance: {int(distance)}", (int(imageCenter[0]-45), int(imageCenter[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    



    return image


def guessBranch(state, goal, imageSize, currentJoints, doVisualize = False, maxDist = 5000, limitCount = 0, bendMultiplier = 1):

    rotationDegrees = state["rotationReal_deg"]
    bendDegrees = state["bendReal_deg"]
    extensionMM = state["extensionReal_mm"]

    imageSize = np.array(imageSize)

    bendOffset = int(bendDegrees*bendMultiplier)

    rotationCenter = (imageSize[0]//2, imageSize[1]//2-bendOffset)
    actualImageCenter = (imageSize[0]//2, imageSize[1]//2)


    goal = np.array(goal)

    rotationLimits = np.array((-170, 170))
    rotationLimits = np.radians(rotationLimits)


    lowDist = 75
    highDist = 300

    goalVector = goal - rotationCenter #vector from approximate point of rotation to goal
    goalDist = np.linalg.norm(goalVector)

    centerGoalVector = goal - actualImageCenter #vector from center of image to goal
    centerGoalDist = np.linalg.norm(centerGoalVector)


    if goalDist > maxDist and limitCount < 10:
        return ""


    highRotationLimitVector = np.array([np.cos(rotationLimits[1]), np.sin(rotationLimits[1])])
    goalOppositeVector = -goalVector


    goalAngle = np.arctan2(goalVector[1], goalVector[0])+np.pi/2
    goalAngleOpposite = np.arctan2(goalOppositeVector[1], goalOppositeVector[0])+np.pi/2

    goalAnglesRelative = [goalAngle, goalAngleOpposite]
    goalAnglesRelative = [(angle+np.pi) % (2*np.pi) - np.pi for angle in goalAnglesRelative]

    goalAnglesAbs = [goalAngle+np.radians(rotationDegrees), goalAngleOpposite+np.radians(rotationDegrees)]
    #use circle angles from -pi to pi
    goalAnglesAbs = [(angle+np.pi) % (2*np.pi) - np.pi for angle in goalAnglesAbs]


    bendLimits = 170

    goalBendAngleAbs = [bendDegrees - (goalDist*bendMultiplier), bendDegrees + (goalDist*bendMultiplier)]

    #print bend stats
    if doVisualize:
        print(f"Goal: {goalDist}, bend: {bendDegrees}, goalBend: {goalBendAngleAbs}")

    #remove those that are outside of the rotation limits
    newGoalAnglesAbs = []
    newGoalAnglesRel = []
    for index in range(len(goalAnglesAbs)): #check both bend angles
        angle = goalAnglesAbs[index]
        if rotationLimits[0] <= angle <= rotationLimits[1] or rotationLimits[0] <= angle+2*np.pi <= rotationLimits[1] or rotationLimits[0] <= angle-2*np.pi <= rotationLimits[1]: #check if bend angle is within rotation limits
            if abs(goalBendAngleAbs[index]) < bendLimits+lowDist: #check if bend angle is within bending limits plus the radius of the center region

                newGoalAnglesAbs.append(angle)
                newGoalAnglesRel.append(goalAnglesRelative[index])
    goalAnglesAbs = newGoalAnglesAbs
    goalAnglesRel = newGoalAnglesRel



    bestScore = -999999
    rotationDirection = 0
    rotationDistance = 0



    

    if doVisualize:
        print("")

    for index in range(len(goalAnglesAbs)):
        angle = goalAnglesAbs[index]
        angleRel = goalAnglesRel[index]

        angleToNearestLimit = min(abs(angle-rotationLimits[0]), abs(angle-rotationLimits[1]))
        angleToCurrent = abs(angleRel)

        score = np.sqrt(angleToNearestLimit)-(angleToCurrent)

        if score > bestScore:
            bestScore = score
            rotationDirection = np.sign(angleRel)
            rotationDistance = np.abs(angleToCurrent)

        if doVisualize:
            dir = "R" if angleRel > 0 else "L"
            print(f"Direction: {dir}, absAngle: {np.degrees(angleRel):.2f}, relAngle: {np.degrees(angle):.2f}, score: {score:.2f}")

    bendDirection = np.sign(goalVector[1])
    bendDistance = np.abs(goalVector[1])


    action=""

    #check if goal is within center region
    #it is so i f its within:
        #a circle with radius lowDist from the center of the image
        #a circle with radius lowDist from the rotation center
        #a rectangle with the center of the image and the rotation center as the top and bottom. The width is twice lowDist centered around the center of the image


    #isWithinCenterRegion = (\
    #    goalDist < lowDist or \
    #    centerGoalDist < lowDist or \
    #    (actualImageCenter[0]-lowDist < goal[0] < actualImageCenter[0]+lowDist\
    #    and actualImageCenter[1] < goal[1] < rotationCenter[1]))
    



    smallestGoalDist = min(goalDist, centerGoalDist)

    if centerGoalDist < lowDist:
        action = "f"
    elif centerGoalDist > highDist:
        action = "b"
    elif np.degrees(rotationDistance) < 35 or np.degrees(rotationDistance) > (180-35):
        if bendDirection < 0 and abs(goalBendAngleAbs[0]) < bendLimits + lowDist: #if goal is within bending limits plus the radius of the center region
            action = "u"
        elif bendDirection > 0 and abs(goalBendAngleAbs[1]) < bendLimits + lowDist: #if goal is within bending limits plus the radius of the center region
            action = "d"
        else: #Bending is outside limits - go backwards
            action = "b"
    else:
        if rotationDirection < 0:
            action = "l"
        elif rotationDirection > 0:
            action = "r"
        else:
            action = "b" #if no rotation, go backwards - this will only happen if both goals are outside bending limits


    if doVisualize:
        
        print(f"Guess: {action}, distance: {goalDist}, rotation: {np.degrees(rotationDirection*rotationDistance):.2f}, bend: {np.degrees(bendDirection*bendDistance):.2f}")
    return action


    








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