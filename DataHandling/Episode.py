
import os
import json
import tempfile
import shutil
import numpy as np
import time
import cv2
import multiprocessing

from PIL import Image



def saveToDiskThreadSafe(image, path):
    #signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    
    
    #convert - switch the color channels from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    
    image = Image.fromarray(image)
    image.save(path)

class Frame:
    def __init__(self, image, state, action, data, topImage=None):
        # Ensure all attributes are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        

        self.image = image
        self.state = state
        self.action = action
        self.data = data
        self.topImage = topImage

    def __repr__(self):
        return f"Frame(image={self.image}, state={self.state}, action={self.action}, data={self.data}, topImage={self.topImage})"

    def copy(self):
        return Frame(
            image=self.image.copy(),
            state=self.state.copy(),
            action=self.action.copy(),
            data=self.data.copy(),
            
            topImage=self.topImage.copy()
        )

class Episode:
    



    def __init__(self, doAskForLabel=True, doAskForSave=True):
        

        self.doAskForLabel = doAskForLabel
        self.doAskForSave = doAskForSave
        
        self.saveEnabled = True
        #Create a new temporary directory for the episode

        self.path = tempfile.mkdtemp()
        self.dataPath = os.path.join(self.path, "_episodeData.json")

        self.episodeProbertyLabels = {"Section": None, "Branch": None}


        #Setup the episode's properties
        self.isSaved = False
        self.isLabeled = False
        self.label = None
        
        self.pool = multiprocessing.Pool(processes=8)


        #data
        self.images = []
        self.states = []
        self.actions = []
        self.data = []
        self.topImages = []
        
        self.doSave=True

        self._index = 0
        
        
        
        self.timeStart = time.time()
        self.timeEnd = None



    def get_frame(self, index):
        #print(f"Getting frame at index {index}")
        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of range")
        return Frame(
            image=self.images[index],
            state=self.states[index],
            action=self.actions[index],
            data=self.data[index],
            topImage=self.topImages[index]
        )

    #overload the [] operator to construct and get the frame at the given index
    def __getitem__(self, index):
        return self.get_frame(index)

    #overload the [] operator to set the frame at the given index
    def __setitem__(self, index, frame):

        imageHasChanged = not np.array_equal(self.images[index], frame.image)

        self.images[index] = frame.image
        self.states[index] = frame.state
        self.actions[index] = frame.action
        self.data[index] = frame.data
        
        if frame.topImage is not None:
            self.topImages[index] = frame.topImage
            self.saveImage(frame.topImage, index, prefix="top")

        if imageHasChanged:
            self.saveImage(frame.image, index, prefix="broncho")   
            
            
            




    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.images):
            result = self[self._index]
            self._index += 1
            
            #print(f"Returning frame {result} at index {self._index - 1}" )
            return result, self._index - 1
        else:
            raise StopIteration

    def __len__(self):
        return len(self.images)




    def saveImage(self, image, index, prefix=""):
        
        imagePath = self.getImagePath(index, prefix=prefix)
        
        #print(f"Saving image of shape {image.size} to {imagePath}")
        
        
        #save the image to the disk in seperate process
        
        self.pool.apply_async(saveToDiskThreadSafe, args=(image, imagePath))
        
        
    

    def _addFrame(self, frame):
        self.images.append(frame.image.copy())
        self.states.append(frame.state.copy())
        self.actions.append(frame.action.copy())
        self.data.append(frame.data.copy())
        self.topImages.append(frame.topImage.copy())

        Index = len(self.images) - 1
        self.saveImage(frame.image,Index, prefix="broncho")
        self.saveImage(frame.topImage,Index, prefix="top")


    def append(self, frameOrImage, state=None, action=None, data=None):
        #print(f"Appending frame with image of shape {frameOrImage.shape}")
        
        frame=None
        
        if state is None: #if only one argument is given, it is a frame
            #check if the frame is a Frame object
            if not isinstance(frameOrImage, Frame):
                raise ValueError("If only one argument is given, it must be a Frame object.")
            frame=frameOrImage
        else:
            frame = Frame(image=frameOrImage, state=state, action=action, data=data)
        
        
        
        #populate the frame with meta data
        
        frame.data["index"] = len(self.images)
        
        
        currentTimestamp = time.time()
        
        
        frame.data["Timestamp"] = f"{(currentTimestamp - self.timeStart):.3f}"
        
        
        frame.data["Raw_Timestamp"] = currentTimestamp
        frame.data["timeReadable"] = time.strftime("%Y-%m-%d %H:%M:%S")+f".{str(currentTimestamp%1)[2:5]}"
        
        lastTimestamp = self.data[-1]["Raw_Timestamp"] if len(self.data) > 0 else None
        frame.data["timeDelta"] = currentTimestamp - lastTimestamp if lastTimestamp is not None else 0
        
        
        
           
        self._addFrame(frame)
            

    def getImagePath(self, index,prefix): #returns the path of the image at the given index, usign zfill to pad the index with zeros
        return os.path.join(self.path, self.getName(index,prefix))


    def getName(self,index,prefix):
        return f"{prefix}{str(index).zfill(5)}.png"

    def __del__(self):
        try:
            shutil.rmtree(self.path)
            print(f"Temporary folder at {self.path} deleted.")
        except Exception as e:
            print(f"Error deleting temporary folder: {e}")



    def askForLabel(self):
        
        
        if self.doAskForSave:
            while True:
                saveInput="y"
                if not self.doSave:
                    saveInput = input("Do you want to save the episode? (y/n): ")
                if saveInput.lower() == "n":
                    return
                elif saveInput.lower() == "y":
                    self.doSave = True
                    
                    if not self.isLabeled:
                        for key in self.episodeProbertyLabels.keys():
                            value = input(f"Enter the {key}: ")
                            self.episodeProbertyLabels[key] = value
                            
                        self.isLabeled = True
                        
                    
                    return
                    
                else:
                    print("Invalid input. Please enter 'y' or 'n'")
            
                
            


    def save(self, directory=None):
        
        #ask for the label
        doSave = self.askForLabel()
        if self.doSave is False:
            return
        

        #construct the dictionary to save

        episodeData = {}

        #construct header data
        header = {}
        
        
        self.timeEnd = self.data[-1]["Raw_Timestamp"]
        
        header["timeStart"] = self.timeStart
        header["timeEnd"] = self.timeEnd
        header["duration"] = self.timeEnd - self.timeStart
        header["timeStartReadable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timeStart))+f".{str(self.timeStart%1)[2:5]}"
        header["timeEndReadable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timeEnd))+f".{str(self.timeEnd%1)[2:5]}"
        
        
        
        for key, value in self.episodeProbertyLabels.items():
            header[key] = value
        
        episodeData["header"] = header

        #save the frames
        frames = {}
        
        
        for frame, index in self:
            
            
            #print(f"Saving frame {frame} at index {index}")
            frameData = {}
            frameData["state"] = frame.state
            frameData["action"] = frame.action
            frameData["data"] = frame.data
            frameData["imagePath"] = self.getName(index, prefix="broncho")
            frameData["topImagePath"] = self.getName(index, prefix="top")
            
            frames[index] = frameData
            
            
        episodeData["frames"] = frames
        
        #save the episode data to a json file
        
        with open(self.dataPath, 'w') as f:
            json.dump(episodeData, f, indent=4)
            
            
        #save the episode to a zip file
        
        name = time.strftime("20%y-%m-%d - %H:%M:%S")
        
        #if a directory is given, save the episode there
        
        
        #print the objects in the directory
        
        
        if directory is not None:
            name = os.path.join(directory, name)
            
            print(f"Saving episode to {name}")
            shutil.make_archive(name, 'zip', self.path)



    

    def clear(self):
        #clear the episode's data and empty the temporary folder
        self.images = []
        self.states = []
        self.actions = []
        self.data = []


        try:
            shutil.rmtree(self.path)
            print(f"Temporary folder at {self.path} deleted.")
        except Exception as e:
            print(f"Error deleting temporary folder: {e}")




class EpisodeManager:

    def __init__(self):
        self.currentEpisode = None

        self.episodeDatabasePath = "Database/"


        allEpisodePaths = self.findAllEpisodes(self.episodeDatabasePath)


    def findAllEpisodes(self, episodeDatabasePath): #returns a list of all episodes in the database. Each episode is a zip file
        episodes = []
        #Search each folder and file in the database
        for root, dirs, files in os.walk(episodeDatabasePath):
            for file in files:
                if file.endswith(".zip"):
                    episodes.append(os.path.join(root, file))
        return episodes
    




    def newEpisode(self, force=False):
        if self.currentEpisode is not None:
            if force:
                self.endEpisode(discard=True)
            else:
                return
        
        self.currentEpisode = Episode()
        return
        

        
                
    def endEpisode(self, discard=False):
        if self.currentEpisode is None:
            return
        
        #save the episode to the database
        if discard is True:
            self.currentEpisode.clear()
            self.currentEpisode = None
            return
        
        

        if self.currentEpisode.isSaved is False:
            
            self.currentEpisode.save(self.episodeDatabasePath)

        self.currentEpisode.clear()
        self.currentEpisode = None
        
        


    def nextEpisode(self):
        self.endEpisode()
        self.newEpisode()



    def hasEpisode(self):
        return self.currentEpisode is not None
    
    
    #pass append on to current episode
    def append(self, *args, **kwargs):
        #print(f"Appending to episode with args {args} and kwargs {kwargs}")
        if self.currentEpisode is not None:
            self.currentEpisode.append(*args, **kwargs)
        else:
            print("No current episode")

    
        

            
            




        

