




import numpy as np
from branchModelTracker import BranchModelTracker
from DataHandling.Episode import EpisodeManager, Episode

from Input import Input




def labelEpisode(episode, modelPath):
    
    def processEndFrame(frameIndex):
        #get the detections of the frame



        #then go bach through the frames, until all the detections are gone, and label them as the last detection left


        ids = detections[frameIndex].keys()

        #remove ids that are not in view, that is only those where detections[frameIndex][id].inView == True

        ids = [id for id in ids if detections[frameIndex][id].inView == True]



        lastId=-1


        print(f"Processing the path ending at frame {frameIndex}")

        while len(ids) > 0:

            print(f"\rAt frame {frameIndex} with {len(ids)} paths left", end="")

            
            frameIndex -= 1
            if frameIndex < 0:
                return 0, ids[0]

            newIds = detections[frameIndex].keys()

            #remove any ids that are not in the new frame ids

            remainingIds = [id for id in ids if id in newIds]

            
            if len(remainingIds) == 0:
                lastId = ids[0]
                print("")
                return frameIndex+1, lastId

                


            ids = remainingIds
            
            

        print("")



        return frameIndex, lastId
        #================================
        #================================
        #================================

    def detectionsToDict(detections):

        detectionsDict = {}

        for id in detections.keys():

            detectionsDict[id] = detections[id].toDict()
        return detectionsDict
    
    def setPath(pathId, startFrame, endFrame):

        print(f"Setting path {pathId} from frame {startFrame} to {endFrame}")
        for i in range(startFrame, endFrame+1):
            episode[i].data["pathId"] = pathId



            episode[i].data["paths"] = detectionsToDict(detections[i])

            #ensure that pathId is in the paths dictionary

            if pathId not in episode[i].data["paths"].keys():
                raise Exception(f"Path {pathId} not in paths dictionary")
            



        #================================
        #================================
        #================================

    #create branch tracker

    
    branchTracker = BranchModelTracker(modelPath)

    #track the branches of whole episode

    detections = []


    print(f"Episode length: {len(episode)}")

    print("Tracking branches")

    for i in range(len(episode)):


        #print on same line

        print(f"\rTracking frame {i}/{len(episode)}           ", end="")

        frame = episode[i]

        #track the branches of the frame

        points, frameDetections = branchTracker.predict(frame.image)

        detections.append(frameDetections)



        #fix action(input)

        input = Input.fromDict(frame.action)
        frame.action = input.toDict()
        episode[i] = frame

    print("")


    

    currentFrame = len(episode) - 1

    print("Processing paths")

    while currentFrame >= 0:
        startFrame, pathId  = processEndFrame(currentFrame)
        if pathId == -1:
            print(f"No path found for frame {startFrame} to {currentFrame}")
            for i in range(currentFrame, startFrame-1, -1):
                print(f"Setting frame {i} to path from frame {i+1}")
                episode[i].data["pathId"] = episode[i+1].data["pathId"]
                episode[i].data["paths"] = episode[i+1].data["paths"]
            
        else:

            print(f"Path {pathId} from frame {startFrame} to {currentFrame}")


            setPath(pathId, startFrame, currentFrame)

        currentFrame = startFrame-1


    #ensure whole episode is labelled
    for i in range(len(episode)):
        if "pathId" not in episode[i].data.keys() or "paths" not in episode[i].data.keys():
            raise Exception(f"Frame {i} not labelled")






def main():


    episodeManager = EpisodeManager(mode = "labelling", loadLocation= "Database", saveLocation= "DatabaseLabelled")




    while episodeManager.hasNextEpisode():
        episode = episodeManager.nextEpisode()

        print(f"Epsiode {episodeManager.currentIndex}")



        labelEpisode(episode, "C:/Users/magnu/OneDrive/Misc/BronchoYolo/yolov5/runs/train/branchTraining8-XL/weights/best.pt")


    episodeManager.endEpisode()




if __name__ == "__main__":
    main()