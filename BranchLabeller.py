

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
    return contour.tolist()

def deserializeContour(contour):
    return np.array(contour)


def contsructBranchDataFromCandidates(candidateBranches):


    branchData = []

    for branch in candidateBranches:

        contour = serializeContour(branch.contour)
        depth = branch.depth
        coords = branch.coords
        area = branch.area
        roundness = branch.roundness
        size = branch.size




        branchData.append({
            "Contour": contour,
            "Depth": depth,
            "Coords": coords,
            "Area": area,
            "Roundness": roundness,
            "Size": size,
            "enabled": False
        })

    return branchData

    

def extractContoursFromBranchData(branchData):

    contours = []

    for branch in branchData:
        contour = deserializeContour(branch["Contour"])
        contours.append(contour)

    return contours

def constructBranchData(image, previousImage = None, previousBranchData= [], currentBranchData = []):

    candidateBranches = FindBranches.findBranches(image, contourDepth=0.3,doDraw=False)


    newBranchData = contsructBranchDataFromCandidates(candidateBranches)

    





    if len(currentBranchData) > 0:
        
        newContours = extractContoursFromBranchData(newBranchData)
        currentContours = extractContoursFromBranchData(currentBranchData)

        matches = FindBranches.matchBranches(newContours, currentContours)

        for match in matches:
            newBranchData[match[0]]["enabled"] = currentBranchData[match[1]]["enabled"]

        currentBranchData = newBranchData



    #if there are previousBranches, match them to the current branches and update the data
    elif len(previousBranchData) > 0:
        currentBranchData = newBranchData

        currentContours = extractContoursFromBranchData(newBranchData)
        previousContours = extractContoursFromBranchData(previousBranchData)

        matches = FindBranches.matchBranches(currentContours, previousContours)

        for match in matches:
            currentBranchData[match[0]]["enabled"] = previousBranchData[match[1]]["enabled"]

    else:
        currentBranchData = newBranchData

    return currentBranchData
        
        




def findSelectedBranch(branchData, x, y):
    selectedIndex = -1
    minDistance = float("inf")

    for i, branch in enumerate(branchData):
        #print(f"Checking branch: {i}, at coords: {(x, y)}")
        contour = deserializeContour(branch["Contour"])

        if cv2.pointPolygonTest(contour, (x, y), False) > 0:
            #print(f"Point in contour: {i}")
            
            center = branch["Coords"]
            distance = np.linalg.norm(np.array(center) - np.array([x, y]))

            if distance < minDistance:
                selectedIndex = i
                minDistance = distance

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
    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelledPost/", loadLocation="DatabaseLabelled/")


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

            episode.data[currentIndex]["Branches"] = []

            #if shift is held, remove all data from future frames
            if keyboard.is_pressed('shift'):
                for i in range(currentIndex+1, len(episode)):
                    episode.data[i]["Branches"] = []

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

    