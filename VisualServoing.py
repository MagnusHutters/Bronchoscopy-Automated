

import numpy as np






import cv2

#from brochoRobVisual import brochoRobClass
from Timer import Timer

def drawRotationAccess(image, rotationDegrees, bendDegrees, bendMultiplier = 1):

    rotationDegrees += 90
    rotationLimits = (-170, 170)

    actualImageCenter = (image.shape[1]//2, image.shape[0]//2)



    bendOffset = int(bendDegrees*bendMultiplier)
    #print(f"Rotation: {rotationDegrees}, bend: {bendDegrees}, bendOffset: {bendOffset}")
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
    bendLimitPixels = bendLimits*bendMultiplier #angle -> pixels

    #bendLimitDown   = ((actualImageCenter[1]+bendLimits) + (imageCenter[1]+bendLimits))//2
    #bendLimitUp     = ((actualImageCenter[1]-bendLimits) + (imageCenter[1]-bendLimits))//2

    bendLimitDown = imageCenter[1]+int(bendLimits*bendMultiplier) #angle -> pixels
    bendLimitUp = imageCenter[1]-int(bendLimits*bendMultiplier) #angle -> pixels
    
    #print(f"bendLimitDown: {bendLimitDown}, bendLimitUp: {bendLimitUp}")

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

    bendOffset = int(currentBend*bendMultiplier) #angle -> pixels

    
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





def doVisualServoing(image, state, goal, imageSize, doVisualize = False, maxDist = 5000, bendMultiplier = 1):
    if len(goal) == 0:
        return ""

    
    #print(f"Goal: {goal}")

    rotationDegrees = state["rotationReal_deg"]
    bendDegrees = state["bendReal_deg"]
    extensionMM = state["extensionReal_mm"]

    bbox = goal["bbox"]
    targetCenter = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    bendDegreesToPixelsConstant = bendMultiplier

    if doVisualize:
        drawImage = image.copy()

        drawRotationAccess(drawImage, rotationDegrees, bendDegrees, bendMultiplier=bendDegreesToPixelsConstant)

        showBranch(drawImage, goal, currentBend = bendDegrees, bendMultiplier = bendDegreesToPixelsConstant)

        #scale up image by factor of 2
        drawImage = cv2.resize(drawImage, (drawImage.shape[1]*2, drawImage.shape[0]*2))

        cv2.imshow("Visual servoing", drawImage)

        #cv2.waitKey(1)



    return guessBranch2(state, targetCenter, imageSize, doVisualize, maxDist, 10, bendMultiplier, bbox = bbox, doKeepScores=True)


accumulatedScore =  0

def guessBranch(state, goal, imageSize, doVisualize = False, maxDist = 5000, limitCount = 0, bendMultiplier = 1, bbox = None):

    global accumulatedScore

    rotationDegrees = state["rotationReal_deg"]
    bendDegrees = state["bendReal_deg"]
    extensionMM = state["extensionReal_mm"]

    imageSize = np.array(imageSize)

    bendOffset = int(bendDegrees*bendMultiplier) #angle -> pixels

    rotationCenter = (imageSize[0]//2, imageSize[1]//2-bendOffset)
    actualImageCenter = (imageSize[0]//2, imageSize[1]//2)


    goal = np.array(goal)

    




    rotationLimits = np.array((-170, 170))
    rotationLimits = np.radians(rotationLimits)


    lowDist = 50
    highDist = 300

    goalVector = goal - rotationCenter #vector from approximate point of rotation to goal
    goalDist = np.linalg.norm(goalVector)

    centerGoalVector = goal - actualImageCenter #vector from center of image to goal
    centerGoalDist = np.linalg.norm(centerGoalVector)

    if bbox is not None:

        minx, miny, maxx, maxy = bbox


        if minx < actualImageCenter[0] < maxx and miny < actualImageCenter[1] < maxy:
            centerGoalDist = 0
        else:


            dx = max(minx-actualImageCenter[0], 0, actualImageCenter[0]-maxx)
            dy = max(miny-actualImageCenter[1], 0, actualImageCenter[1]-maxy)

            centerGoalDist = np.sqrt(dx**2 + dy**2)


        #print(f"CenterGoalDist: {centerGoalDist}, bbox: {bbox}, imageCenter: {actualImageCenter}")
        


        





    #if goalDist > maxDist and limitCount < 10:
    #    return ""


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
    bendLimitPixels = bendLimits*bendMultiplier #angle -> pixels

    goalBendAngleAbs = [bendDegrees - (goalDist/bendMultiplier), bendDegrees + (goalDist/bendMultiplier)] #pixel -> angle

    print(f"Bend degrees: {bendDegrees}, goalDist: {goalDist}, goalBendAngleAbs: {goalBendAngleAbs}")

    #print bend stats
    if doVisualize:
        pass
        #print(f"Goal: {goalDist}, bend: {bendDegrees}, goalBend: {goalBendAngleAbs}")

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
        pass
        #print("")

    for index in range(len(goalAnglesAbs)):
        angle = goalAnglesAbs[index]
        angleRel = goalAnglesRel[index]

        angleToNearestLimit = min(abs(angle-rotationLimits[0]), abs(angle-rotationLimits[1]))
        angleToCurrent = abs(angleRel)

        score = np.sqrt(angleToNearestLimit)-(angleToCurrent)


        accScore = accumulatedScore if dir == "R" else -accumulatedScore
        combinedScore = score+accScore


        if combinedScore > bestScore:
            bestScore = combinedScore
            rotationDirection = np.sign(angleRel)
            rotationDistance = np.abs(angleToCurrent)

        if doVisualize:
            dir = "R" if angleRel > 0 else "L"

            
            #print(f"Direction: {dir}, absAngle: {np.degrees(angleRel):.2f}, relAngle: {np.degrees(angle):.2f}, combinedScore: {combinedScore:.2f}, score: {score:.2f}, accumulatedScore: {accScore:.2f}")

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
    print(f"GoalBendAngleAbs: {goalBendAngleAbs}")

    if centerGoalDist < lowDist:
        action = "f"

        if doVisualize:
            #explain why we are going forwards
            print(f"Action: Forward, centerGoalDist:{centerGoalDist:.1f} < lowDist:{lowDist:.1f}")
    elif centerGoalDist > highDist:
        action = "b"
        if doVisualize:
            print(f"Action: Backwards, centerGoalDist:{centerGoalDist:.1f} > highDist:{highDist:.1f}")

    elif np.degrees(rotationDistance) < 35 or np.degrees(rotationDistance) > (180-35):
        if bendDirection < 0 and abs(goalBendAngleAbs[0]) < bendLimitPixels + lowDist: #if goal is within bending limits plus the radius of the center region
            action = "u"
            if doVisualize:

                print(f"Action: Up - rotationDistance:{np.degrees(rotationDistance):.1f} within bending axis and goalBendAngle: {abs(goalBendAngleAbs[0]):.1f} < bendLimits:{bendLimitPixels+lowDist:.1f} with down bendDirection:{bendDirection:.1f}")
        elif bendDirection > 0 and abs(goalBendAngleAbs[1]) < bendLimitPixels + lowDist: #if goal is within bending limits plus the radius of the center region
            action = "d"
            if doVisualize:
                print(f"Action: Down - rotationDistance:{np.degrees(rotationDistance):.1f}½ and goalBendAngle: {abs(goalBendAngleAbs[1]):.1f} < bendLimits:{bendLimitPixels+lowDist:.1f} with up bendDirection:{bendDirection:.1f}")
   
        else: #Bending is outside limits - go backwards
            action = "b"

            if doVisualize:
                print(f"Action: Backwards - rotationDistance:{np.degrees(rotationDistance):.1f}  within bending axis and goalBendAngle: {abs(goalBendAngleAbs[0]):.1f} > bendLimits:{bendLimitPixels+lowDist:.1f}")
    else:
        if rotationDirection < 0:
            action = "l"
            accumulatedScore -= 0.1
            print(f"Action: Left - rotationDistance:{np.degrees(rotationDistance):.1f} outside bending axis and rotationDirection:{rotationDirection:.1f} < 0")

        elif rotationDirection > 0:
            action = "r"
            accumulatedScore += 0.1
            print(f"Action: Right - rotationDistance:{np.degrees(rotationDistance):.1f} outside bending axis and rotationDirection:{rotationDirection:.1f} > 0")
        else:
            action = "b" #if no rotation, go backwards - this will only happen if both goals are outside bending limits

            print(f"Action: Backwards - rotationDistance:{np.degrees(rotationDistance):.1f} outside bending axis and rotationDirection:{rotationDirection:.1f} = 0")

    accumulatedScore *= 0.9

    if doVisualize:
        pass
        #print(f"Guess: {action}, distance: {centerGoalDist}, rotation: {np.degrees(rotationDirection*rotationDistance):.2f}, bend: {np.degrees(bendDirection*bendDistance):.2f}")
    return action


accumulatedScores =  [0,0,0,0]

def guessBranch2(state, goal, imageSize, doVisualize = False, maxDist = 5000, limitCount = 0, bendMultiplier = 1, bbox = None, doKeepScores = False, newTarget = False):

    global accumulatedScore, accumulatedScores
    if newTarget: #reset accumulated scores
        accumulatedScores = [0,0,0,0]

    rotationDegrees = state["rotationReal_deg"]
    bendDegrees = state["bendReal_deg"]
    extensionMM = state["extensionReal_mm"]


    rotationRadians = np.radians(rotationDegrees)

    imageSize = np.array(imageSize)

    bendOffset = int(bendDegrees*bendMultiplier) #angle -> pixels

    rotationCenter = (imageSize[0]//2, imageSize[1]//2-bendOffset)
    actualImageCenter = (imageSize[0]//2, imageSize[1]//2)


    goal = np.array(goal)

    




    rotationLimits = np.array((-170, 170))
    rotationLimits = np.radians(rotationLimits)


    lowDist = 50
    highDist = 300

    goalVector = goal - rotationCenter #vector from approximate point of rotation to goal
    goalDist = np.linalg.norm(goalVector)

    centerGoalVector = goal - actualImageCenter #vector from center of image to goal
    centerGoalDist = np.linalg.norm(centerGoalVector)

    #if bbox is not None:
#
    #    minx, miny, maxx, maxy = bbox
#
#
    #    if minx < actualImageCenter[0] < maxx and miny < actualImageCenter[1] < maxy:
    #        centerGoalDist = 0
    #    else:
#
#
    #        dx = max(minx-actualImageCenter[0], 0, actualImageCenter[0]-maxx)
    #        dy = max(miny-actualImageCenter[1], 0, actualImageCenter[1]-maxy)
#
    #        centerGoalDist = np.sqrt(dx**2 + dy**2)
#
#
    #    #print(f"CenterGoalDist: {centerGoalDist}, bbox: {bbox}, imageCenter: {actualImageCenter}")
        


        





    if goalDist > maxDist and limitCount < 10:
        return ""


    highRotationLimitVector = np.array([np.cos(rotationLimits[1]), np.sin(rotationLimits[1])])
    goalOppositeVector = -goalVector


    goalAngle = np.arctan2(goalVector[1], goalVector[0])+np.pi/2
    goalAngleOpposite = np.arctan2(goalOppositeVector[1], goalOppositeVector[0])+np.pi/2

    goalAnglesRelative = [goalAngle, goalAngleOpposite, goalAngle, goalAngleOpposite]
    goalAnglesRelative = [(angle+np.pi) % (2*np.pi) - np.pi for angle in goalAnglesRelative]

    goalAnglesAbs = [angle+np.radians(rotationDegrees) for angle in goalAnglesRelative]
    #use circle angles from -pi to pi
    #goalAnglesAbs = [((angle+np.pi) % (2*np.pi)) - np.pi for angle in goalAnglesAbs]


    bendLimits = 170
    bendLimitPixels = bendLimits*bendMultiplier #angle -> pixels

    goalBendAngleAbs = [bendDegrees - (goalDist/bendMultiplier), bendDegrees + (goalDist/bendMultiplier), bendDegrees - (goalDist/bendMultiplier), bendDegrees + (goalDist/bendMultiplier)] #pixel -> angle

    if doVisualize:
        print(f"Bend degrees: {bendDegrees}, goalDist: {goalDist}, goalBendAngleAbs: {goalBendAngleAbs}")

    #print bend stats
    if doVisualize:
        pass
        #print(f"Goal: {goalDist}, bend: {bendDegrees}, goalBend: {goalBendAngleAbs}")

    #remove those that are outside of the rotation limits
    newGoalAnglesAbs = [None, None, None, None]
    newGoalAnglesRel = [None, None, None, None]

    for index in range(len(goalAnglesAbs)): #check both bend angles
        angle = goalAnglesAbs[index]
        if index >= 2:
            angle = angle+np.pi
        if rotationLimits[0] <= angle <= rotationLimits[1]:# or rotationLimits[0] <= angle+2*np.pi <= rotationLimits[1] or rotationLimits[0] <= angle-2*np.pi <= rotationLimits[1]: #check if bend angle is within rotation limits
            if abs(goalBendAngleAbs[index]) < bendLimitPixels+lowDist*2.5: #check if bend angle is within bending limits plus the radius of the center region

                newGoalAnglesAbs[index] = angle
                newGoalAnglesRel[index] = goalAnglesRelative[index]
            else:
                accumulatedScores[index] -= 0.3
        else:
            accumulatedScores[index] -= 0.3
    goalAnglesAbs = newGoalAnglesAbs
    goalAnglesRel = newGoalAnglesRel



    bestScore = -999999



    

    if doVisualize:
        pass
        #print("")



    scores = [-999999, -999999, -999999, -999999]
    directions = ["" , "", "", ""]
    diffs = [0, 0, 0, 0]
    for index in range(len(goalAnglesAbs)):
        if goalAnglesAbs[index] is None:
            continue
        else:
            angle = goalAnglesAbs[index]
            angleRel = goalAnglesRel[index]


            currentRotation = rotationRadians
            tartgetRotation = goalAnglesAbs[index]

            diff = tartgetRotation - currentRotation
            diffs[index] = diff

            angleToNearestLimit = min(abs(angle-rotationLimits[0]), abs(angle-rotationLimits[1]))
            angleToCurrent = abs(angleRel)

            score = np.sqrt(angleToNearestLimit)-(angleToCurrent)

            dir = "r" if diff > 0 else "l"
            directions[index] = dir

            accScore = accumulatedScores[index] if doKeepScores else 0
            combinedScore = score+accScore


            if combinedScore > bestScore:
                bestScore = combinedScore

            scores[index] = combinedScore
            

            if doVisualize:
                pass

                #accScore = accumulatedScore if dir == "r" else -accumulatedScore
                target = "Goal" if index%2 == 0 else "Opposite"
                direction = "Upwards" if index < 2 else "Downwards"
                print(f"Target: {target}, Direction: {direction}, angleToCurrent: {np.degrees(angleToCurrent):.2f}, angleToLimit: {np.degrees(angleToNearestLimit):.2f}, combinedScore: {combinedScore:.2f}, score: {score:.2f}, accumulatedScore: {accScore:.2f}")

    bendDirection = np.sign(goalVector[1])
    bendDistance = np.abs(goalVector[1])

    directionsNames = ["Right" if dir == "r" else "Left" if dir == "l" else "None" for dir in directions]

    #convert angles to degrees
    goalAnglesAbs = [np.degrees(angle) if angle is not None else None for angle in goalAnglesAbs]
    goalAnglesRel = [np.degrees(angle) if angle is not None else None for angle in goalAnglesRel]
    diffs = [np.degrees(angle) if angle is not None else None for angle in diffs]

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

    if doVisualize:
        print(f"GoalBendAngleAbs: {goalBendAngleAbs}")

    centralRotationLimit = 25


    bendRatioComparedToMax = abs(bendDegrees)/bendLimits
    

    bestScoreIndex = np.argmax(scores)
    

    if centerGoalDist < lowDist or (-lowDist*1.5<centerGoalVector[0]<lowDist*1.5 and -(1+(bendRatioComparedToMax*2))*lowDist<centerGoalVector[1]<lowDist*(1+(bendRatioComparedToMax*2))):
        action = "f"

        if doVisualize:
            #explain why we are going forwards
            print(f"Action: Forward, centerGoalDist:{centerGoalDist:.1f} < lowDist:{lowDist:.1f}")
    elif centerGoalDist > highDist:
        action = "b"
        if doVisualize:
            print(f"Action: Backwards, centerGoalDist:{centerGoalDist:.1f} > highDist:{highDist:.1f}")

    #First check if using the upwards or downwards bending target
    elif scores[bestScoreIndex]>-1000: #upwards bending target
        #check if we are within the rotation limits
        #if abs(goalAnglesRel[0]) < centralRotationLimit:
        index = bestScoreIndex
        upwards = True if bestScoreIndex % 2 else False

        isWithinY = goalVector[1] < lowDist if upwards else goalVector[1] > -lowDist

        target = "Goal" if index%2 == 0 else "Opposite"
        direction = "Upwards" if index < 2 else "Downwards"
        actionToName = {"r":"Right", "l":"Left", "u":"Upwards", "d":"Downwards", "b":"Backwards", "f":"Forwards"}

        accumulatedScores[index] += 0.1
        if -lowDist< goalVector[0] < lowDist and not abs(diffs[index]) > 45:
            
            if bendDirection < 0:
                action = "u"

            else:
                action = "d"
        else:
            
            action = directions[bestScoreIndex]
        if doVisualize:
            print(f"Action: {actionToName[action]} - target: {target} - direction: {direction} - beacuse of goalVector: {goalVector}, isWithinY: {isWithinY}, bendDirection: {bendDirection}")
        
    else: #should not happen, both up and down targets are out of bounds
        action = "b" #go backwards
    
    accumulatedScores = [score*0.95 for score in accumulatedScores]

    if doVisualize:
        pass
        #print(f"Guess: {action}, distance: {centerGoalDist}, rotation: {np.degrees(rotationDirection*rotationDistance):.2f}, bend: {np.degrees(bendDirection*bendDistance):.2f}")
    return action
