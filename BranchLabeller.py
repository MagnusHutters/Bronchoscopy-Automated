

import cv2
import numpy as np
import time
from DataHandling.Episode import EpisodeManager, Episode
import keyboard

import FindBranchesCV as FindBranches

mouse_x = 0
mouse_y = 0
mouse_clicked = False

def mouse_callback(event, x, y, flags, param):
    #print(f"Mouse event: {event}, x: {x}, y: {y}, flags: {flags}, param: {param}")
    global mouse_x, mouse_y, mouse_clicked
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_clicked = True

def serializeContour(contour):

    listContour = []
    for point in contour:
        listContour.append((int(point[0][0]), int(point[0][1])))
    
    return listContour

def deserializeContour(contour):

    newContour = []
    for point in contour:
        newContour.append([[point[0], point[1]]])

    
    return np.array(newContour, dtype=np.int32)


def contsructBranchDataFromCandidates(candidateBranches):


    branchData = []

    for branch in candidateBranches:

        
        contour, area, circumference, roundness, center, lowestPoint, size, depth,index, numChildren, numSiblings, childrenIndices, siblingIndices, parentIndex = branch.getStats()

        #print(f"==================================================================================================")
        #print (f"Pre-Serialization: {contour}")
        contour = serializeContour(contour)
        #print (f"Post-Serialization: {contour}")
        #dcontour =deserializeContour(contour)

        #print (f"Post-Deserialization: {dcontour}")

        #print(f"")
        #print(f"")
        #print(f"")


        branchData.append({
            "Contour": contour,
            "Area": area,
            "Circumference": circumference,
            "Roundness": roundness,
            "Coords": center,
            "LowestPoint": lowestPoint,
            "Size": size,
            "Depth": depth,
            "Index": index,
            "NumChildren": numChildren,
            "NumSiblings": numSiblings,
            "ChildrenIndices": childrenIndices,
            "SiblingIndices": siblingIndices,
            "ParentIndex": parentIndex,
            "enabled": False
        })

    return branchData

    

def extractContoursFromBranchData(branchData):

    contours = []

    for branch in branchData:
        contour = deserializeContour(branch["Contour"])
        contours.append(contour)

    return contours


def enableFromPrevious(currentBranchData, previousBranchData):

    if len(previousBranchData) == 0:
        return currentBranchData

    for i, oldBranch in enumerate(previousBranchData):#check through the old branch data

        if oldBranch["enabled"]: #if the branch was enabled in the previous frame, check if it is close to a branch in the current frame
            oldContour = deserializeContour(oldBranch["Contour"])

            newContours = extractContoursFromBranchData(currentBranchData)

            index = FindBranches.matchContour(oldContour, newContours)

            if index != -1:
                currentBranchData[index]["enabled"] = True

    return currentBranchData

        


def constructBranchData(image, previousImage = None, previousBranchData= [], currentBranchData = []):

    candidateBranches = FindBranches.thresholdTree(image)

    
    newBranchData = contsructBranchDataFromCandidates(candidateBranches)

    





    if len(currentBranchData) > 0:
        

        currentBranchData = enableFromPrevious(newBranchData, currentBranchData)

        pass



    #if there are previousBranches, match them to the current branches and update the data
    elif len(previousBranchData) > 0:
        currentBranchData = newBranchData


        currentBranchData = enableFromPrevious(newBranchData, previousBranchData)

    else:
        currentBranchData = newBranchData

    return currentBranchData
        
        




def findSelectedBranch(branchData, x, y):
    selectedIndex = -1
    bestDepth = 100000 # depth 0 is best depth

    for i, branch in enumerate(branchData):
        #print(f"Checking branch: {i}, at coords: {(x, y)}")
        contour = deserializeContour(branch["Contour"])

        if cv2.pointPolygonTest(contour, (x, y), False) > 0:
            #print(f"Point in contour: {i}")

            depth = branch["Depth"]
            if depth < bestDepth:
                bestDepth = depth
                selectedIndex = i
            
            
            

    return selectedIndex




def drawBranches(image, branchData, selectedIndex):


    for i, branch in enumerate(branchData):

        if branch["enabled"]:
            color = (0, 255, 0) #green
        else:
            color = (0, 0, 255) #red

        contour = deserializeContour(branch["Contour"])
        cv2.drawContours(image, [contour], -1, color, 1)

        center = branch["Coords"]
        cv2.circle(image, center, 3, color, -1)


        if i == selectedIndex: #fill contour with transparent color
            overlay = image.copy()

            newColor = (255, 255, 255)
            # in between colors
            newColor = (newColor[0] * 0.5 + color[0] * 0.5, newColor[1] * 0.5 + color[1] * 0.5, newColor[2] * 0.5 + color[2] * 0.5)
            newColor = (int(newColor[0]), int(newColor[1]), int(newColor[2]))


            cv2.drawContours(overlay, [contour], -1, (newColor), -1)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)







def main():
    global mouse_x, mouse_y, mouse_clicked
    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="Database/")


    episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()

    currentIndex = 0

    cv2.namedWindow('Labeller')
    cv2.setMouseCallback('Labeller', mouse_callback)

    oldKey = 0
    newFrame=True

    while True:

        #print(f"Current Index: {currentIndex}")
        episode = episodeManager.getCurrentEpisode()

        


        previousIndex = max(currentIndex-1, 0)

        frame = episode[currentIndex]
        previousFrame = episode[previousIndex]

        drawImage = frame.image.copy()


        currentBranchData = frame.data.get("Branches", [])
        previousBranchData = previousFrame.data.get("Branches", [])

        #print(f"currentBranchData: {currentBranchData}")
        #print(f"previousBranchData: {previousBranchData}")

        


        if newFrame:
            currentBranchData = constructBranchData(frame.image, previousFrame.image, previousBranchData, currentBranchData)
            frame.data["Branches"] = currentBranchData
            newFrame = False

        #print(f"currentBranchData: {currentBranchData}")

        selected = findSelectedBranch(currentBranchData, mouse_x, mouse_y)
        
        #print(f"Selected: {selected}")

        if selected != -1:
            if mouse_clicked:
                currentBranchData[selected]["enabled"] = not currentBranchData[selected]["enabled"]
                mouse_clicked = False

        drawBranches(drawImage, currentBranchData, selected)



        







        episode[currentIndex] = frame



        #handle inputs here

        cv2.imshow("Labeller", drawImage)
        #cv2.updateWindow('Labeller')

        key = cv2.waitKeyEx(1)



        if key == ord('q'):
            break
        elif key == 27:
            break
        # d or right arrow
        elif key == ord('d') or key == 83:
            currentIndex += 1
            if currentIndex >= len(episode):
                currentIndex = len(episode) - 1
            newFrame = True
        # a or left arrow
        elif key == ord('a') or key == 81:
            currentIndex -= 1
            if currentIndex < 0:
                currentIndex = 0

            newFrame = True
        elif key == 3014656: #delete
            #remove branch data from current frame

            episode._data[currentIndex]["Branches"] = []

            #if shift is held, remove all data from future frames
            if keyboard.is_pressed('shift'):
                for i in range(currentIndex+1, len(episode)):
                    episode._data[i]["Branches"] = []

            newFrame = True

        elif key == 13: #enter
            #save the episode
            episodeManager.nextEpisode()
            currentIndex = 0
        elif key == 8: #backspace
            #delete the current episode
            episodeManager.previousEpisode()
            currentIndex = 0

        if oldKey != key:
            print(f"Key: {key}")

        #time.sleep(0.1)

        oldKey = key

    
    episodeManager.endEpisode()
    cv2.destroyAllWindows()


        






if __name__ == '__main__':
    
    main()

    