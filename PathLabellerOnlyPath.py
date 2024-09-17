




import numpy as np
from branchModelTracker import BranchModelTracker, Detection
from DataHandling.Episode import EpisodeManager, Episode

from Input import Input




def labelEpisode(episode, modelPath):
    
    
    #def detectionsToDict(detections):
#
    #    detectionsDict = {}
#
    #    for id in detections.keys():
#
    #        detectionsDict[id] = detections[id].toDict()
    #    return detectionsDict
    #
    #create branch tracker


    

    startIndexes = {}
    endIndexes = {}



    for i in range(len(episode)):
        frame = episode[i]
        detections = frame.data["paths"]

        for pathId in detections.keys():
            if pathId not in startIndexes.keys():
                startIndexes[pathId] = i
            
            
            detection, _, _ = Detection.fromDict(detections[pathId])

            if detection.inView:
                endIndexes[pathId] = i


    print(f"Start indexes {startIndexes}")
    print(f"End indexes {endIndexes}")

    paths = ["" for i in range(len(episode))]
    usedPaths = []

    for index in range(len(episode)):


        frame = episode.get_frame(index, getImage = False)
        detections = frame.data["paths"]


        currentKeys = detections.keys()

        #find key with highest start index
        highestEndIndex = -1
        highestEndIndexKey = ""
        for key in currentKeys:

            if endIndexes.get(key, -1) > highestEndIndex:
                highestEndIndex = endIndexes[key]
                highestEndIndexKey = key


        #set the path to the highestEndIndexKey

        paths[index] = highestEndIndexKey

        if highestEndIndexKey not in usedPaths:
            usedPaths.append(highestEndIndexKey)

    newUsedPaths = []

    print(f"Used paths {usedPaths}")
    print(f"Filtering...")

    for index in range(len(episode)):
        
        

        currentKey = paths[index]

        keyOrderIndex = usedPaths.index(currentKey)

        if keyOrderIndex == 0 or keyOrderIndex == len(usedPaths) - 1:
            pass
        else:

            #check if the path can be skipped

            previousKey = usedPaths[keyOrderIndex - 1]
            previousKeyEndFrame = endIndexes[previousKey]

            nextKey = ""
            nextKeyStartFrame = 9999999
            overlap = False

            for i in range(keyOrderIndex + 1, len(usedPaths)-1):

                _nextKey = usedPaths[i]
                _nextKeyStartFrame = startIndexes[_nextKey]

                if _nextKeyStartFrame < previousKeyEndFrame:
                    nextKey = _nextKey
                    nextKeyStartFrame = _nextKeyStartFrame
                    overlap = True
                else:
                    break

            if overlap:
                if index >= nextKeyStartFrame:
                    currentKey = nextKey
                elif index < previousKeyEndFrame:
                    currentKey = previousKey
                else:
                    raise Exception("Should not happen")
                    pass
            else:
                pass
                #path could not be skipped

        if currentKey not in newUsedPaths:
            newUsedPaths.append(currentKey)
        paths[index] = currentKey

    print(f"New used paths {newUsedPaths}")


    for index in range(len(episode)):
        frame = episode.get_frame(index, getImage = False)
        frame.data["pathId"] = paths[index]
        episode.set_frame(index, frame, setImage = False)


            
                
                


            
            

    

    #see if paths can be skipped
    



    

    



    print("Processing paths")








def main():


    episodeManager = EpisodeManager(mode = "labelling", loadLocation= "DatabaseLabelled", saveLocation= "DatabaseLabelled")




    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()

        print(f"Epsiode {episodeManager.currentIndex}")



        labelEpisode(episode, "C:/Users/magnu/OneDrive/Misc/BronchoYolo/yolov5/runs/train/branchTraining8-XL/weights/best.pt")


    episodeManager.endEpisode()




if __name__ == "__main__":
    main()