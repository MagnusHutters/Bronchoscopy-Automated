
import os
import shutil
import time
import json

from DataHandling.Options import *
from DataHandling.Frame import Frame




class Episode:





    def __init__(self):
        self.frames = []
        self.base_dir = TEMPDIR
        self.dir_path = self._create_temp_dir()
        self.length = 0

    def _create_temp_dir(self):
        """Creates a temporary directory with a unique name."""
        n = 0
        while True:
            dir_path = os.path.join(self.base_dir, f'temp{n}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                return dir_path
            n += 1



    def addFrame(self, image, data={}):

        frame = Frame.fromImage(self.length,self.dir_path, image, data)
        self.frames.append(frame)
        self.length += 1

    

    def saveEpisode(self, name=None):
        """Saves the episode to the given path."""
        
        #if data is None: set name to the current time in yy-mm-dd-hh-mm-ss format
        if name is None:
            name = time.strftime("%y-%m-%d-%H-%M-%S")

        dataStruct = {}


        frames = {}

        for frame in self.frames:
            frames[frame.index] = frame.getJsonStruct()
            #frames.append(frame.getJsonStruct())
        
        dataStruct["frames"] = frames

        jsonPath = f"{self.dir_path}/data.json"

        with open(jsonPath, 'w') as f:
            json.dump(dataStruct, f, indent=4)


        #Compress the directory using snappy and save it to the database
        

        pathName = f"{DATABASEDIR}/{name}_0"
        #if the file already exists, add a number to the name
        n = 0
        while os.path.exists(f"{pathName}.zip"):
            n += 1
            pathName = f"{DATABASEDIR}/{name}_{n}"


        shutil.make_archive(pathName, 'zip', self.dir_path)
        

    def __del__(self):
        """Destructor to clean up the temporary directory."""
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)

    @classmethod
    def loadEpisode(cls, episodePath):
        episode = cls()
        #episode.loadFrames(episodePath)
        return episode







