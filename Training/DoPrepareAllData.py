


import os
import cv2
import json



from EpisodeLoader import *


databasePath = "Database"
BronchoPath = "Training\Data\BronchoData"


from SETTINGS import *











def DoPrepareAllData():
    

    #get all epsiode in Database folder
    episodes = [f for f in os.listdir(databasePath) if f.endswith('.zip')]

    for episode in episodes:
        

        episodeName = episode.split(".")[0]

        images, inputs, states, paths, predictions = prepEpisode(os.path.join(databasePath, episode))

        episodeFolder = os.path.join(BronchoPath, episodeName)
        saveEpisode(episodeFolder, images, inputs, states, paths, predictions)

        
if __name__ == '__main__':
    DoPrepareAllData()
        

        