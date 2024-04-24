


import time

import numpy as np


from Input import *
from Interface import *

class Controller:
    def __init__(self):
        
        
        self.closed=False
        try:
            
            self.interface = Interf
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








import os
import shutil
import time
import json

from DataHandling.Options import *
from DataHandling.Frame import Frame

from PIL import Image
import multiprocessing


class Episode:





    def __init__(self):
        self.frames = []
        self.base_dir = TEMPDIR
        self.dir_path = self._create_temp_dir()
        self.length = 0
        
        self.maxSavesInProgress=4
        self.pool = multiprocessing.Pool(processes=self.maxSavesInProgress)
        self.pooledProcesses=True
        self.saving=[]
        self.unsavedFramesIndex=[]

    def _create_temp_dir(self):
        """Creates a temporary directory with a unique name."""
        n = 0
        while True:
            dir_path = os.path.join(self.base_dir, f'temp{n}')
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                return dir_path
            n += 1


    def manage(self):

        #check if done saving

        #for backwards


        for i in range(len(self.saving)-1,-1,-1):
            if self.saving[i].isDoneSaving():
                self.saving[i].unloadFromRam()
                self.saving.pop(i)


        print(f"Saving: {len(self.saving)}")        
        if not self.pooledProcesses:

            #for number of lower than max saves in progress


            for i in range(self.maxSavesInProgress-len(self.saving)):
                if len(self.unsavedFramesIndex)==0:
                    break
                index=self.unsavedFramesIndex.pop(0)
                self.frames[index].ensureSaved()
                self.saving.append(self.frames[index])
        else:
            for i in range(len(self.unsavedFramesIndex)):
                if len(self.unsavedFramesIndex)==0:
                    break
                index=self.unsavedFramesIndex.pop(0)
                self.frames[index].ensureSaved(pool=self.pool)
                self.saving.append(self.frames[index])


    

    def addFrame(self, image, data={}):

        pil_image = Image.fromarray(image)
        frame = Frame.fromImage(self.length,self.dir_path, pil_image, data)
        self.frames.append(frame)

        print(f"Frame {self.length} added")
        self.length += 1

        #add index to Unsaved frame index
        self.unsavedFramesIndex.append(self.length-1)       
        
        #self.saveData()

        self.manage()
    
    def saveEpisode(self, name=None):
        """Saves the episode to the given path."""


        #if data is None: set name to the current time in yy-mm-dd-hh-mm-ss format
        if name is None:
            name = time.strftime("%y-%m-%d-%H-%M-%S")

        print(f"Saving episode {name}")
        print(f"Saving {len(self.frames)} frames")

        dataStruct = {}

        #ensure all frames are saved
        for frame in self.frames:
            frame.ensureSaved()


        #ensure all frames are done saving
        for frame in self.frames:

            while not frame.isDoneSaving():
                time.sleep(0.1)
            print(f"Frame {frame.index} is done saving")

        
        
        #saving data to json
        self.saveData()


        pathName = f"{DATABASEDIR}/{name}_{0}"

        print(f"Compressing to {pathName}.zip")
        
        
        shutil.make_archive(pathName, 'zip', self.dir_path)
        #achive in new process
        
        #self.pool.apply_async(shutil.make_archive, (pathName, 'zip', self.dir_path))
        #print(f"Compressed to {pathName}.zip")
        

    def saveData(self):
        
        
        data = {}
        frames = {}
        
        for frame in self.frames:
            frames[frame.index] = frame.getJsonStruct()
            #frames.append(frame.getJsonStruct())

        data["frames"] = frames
        
        
        
        
        
        
        
        jsonPath = f"{self.dir_path}/data.json"
        with open(jsonPath, 'w') as f:
            json.dump(data, f, indent=4)
            








    def __del__(self):
        
        """Destructor to clean up the temporary directory."""
        #join all processes in self.pool
        
        self.pool.close()
        self.pool.join()
        print(f"Multiprocessing pool has shut down")
        
        
        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        print("Temp folder deleted")

    @classmethod
    def loadEpisode(cls, episodePath):
        episode = cls()
        #episode.loadFrames(episodePath)
        return episode







ace()
        except:
            self.close()
            raise ValueError("Interface failed to initialize")
        
        
    def __del__(self): 
        self.close()  
        
        
    def doStep(self, image):
        
        return Input(0,0,0), 0, 0
        
    def update(self):
        
        image = self.interface.getImage()
        
        
        
        input, doStart, doStop = self.doStep(image)
        
        self.interface.updateInput(input, doStart, doStop)
        
    def run(self, interval =0.1):
        
        self.interval = interval
        
        while True:
            start_time = time.time()  # Record start time

            self.update()  # Execute the task

            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            #print(elapsed_time)
            sleep_time = self.interval - elapsed_time  # Calculate remaining time to sleep

            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep for the remaining time of the interval
            else:
                # Processing took longer than the interval
                print("Warning: Processing time exceeded the interval.")
                
                
    def close(self):
        if(not self.closed):
            self.interface.close()
            self.closed=True
        
        