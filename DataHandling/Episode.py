
import os
import json
import tempfile
import shutil
import numpy as np

from PIL import Image

class Frame:
    def __init__(self, image, state, action, data):
        # Ensure all attributes are numpy arrays
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.image = image
        self.state = state
        self.action = action
        self.data = data

    def __repr__(self):
        return f"Frame(image={self.image}, state={self.state}, action={self.action}, data={self.data})"

    def copy(self):
        return Frame(
            image=self.image.copy(),
            state=self.state.copy(),
            action=self.action.copy(),
            data=self.data.copy()
        )

class Episode:
    



    def __init__(self, doAskForLabel=True, doAskForSave=True):
        

        self.doAskForLabel = doAskForLabel
        self.doAskForSave = doAskForSave
        #Create a new temporary directory for the episode

        self.path = tempfile.mkdtemp()
        self.dataPath = os.path.join(self.path, "episodeData.json")

        self.episodeProbertyLabels = {"Section": None, "Branch": None}


        #Setup the episode's properties
        self.isSaved = False
        self.isLabeled = False
        self.label = None


        #data
        self.images = []
        self.states = []
        self.actions = []
        self.data = []

        self._index = 0



    def get_frame(self, index):
        if index < 0 or index >= len(self.images):
            raise IndexError("Index out of range")
        return Frame(
            image=self.images[index],
            state=self.states[index],
            action=self.actions[index],
            data=self.data[index]
        )

    #overload the [] operator to construct and get the frame at the given index
    def __getitem__(self, index):
        self.get_frame(index)

    #overload the [] operator to set the frame at the given index
    def __setitem__(self, index, frame):

        imageHasChanged = not np.array_equal(self.images[index], frame.image)

        self.images[index] = frame.image
        self.states[index] = frame.state
        self.actions[index] = frame.action
        self.data[index] = frame.data

        if imageHasChanged:
            self.saveImage(index)   




    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.images):
            result = self[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def __len__(self):
        return len(self.images)

    def saveImage(self, index):
        image = self.images[index]
        image = Image.fromarray(image)
        imagePath = self.getImagePath(index)
        image.save(imagePath)

    def _addFrame(self, frame):
        self.images.append(frame.image.copy())
        self.states.append(frame.state.copy())
        self.actions.append(frame.action.copy())
        self.data.append(frame.data.copy())

        Index = len(self.images) - 1
        self.saveImage(Index)


    def append(self, frameOrImage, state=None, action=None, data=None):
        if state is None: #if only one argument is given, it is a frame
            #check if the frame is a Frame object
            if not isinstance(frameOrImage, Frame):
                raise ValueError("If only one argument is given, it must be a Frame object.")
            self._addFrame(frameOrImage)
        else:
            self._addFrame(Frame(frameOrImage, state, action, data))
            

    def getImagePath(self, index): #returns the path of the image at the given index, usign zfill to pad the index with zeros
        return os.path.join(self.path, f"{str(index).zfill(5)}.png")



    def __del__(self):
        try:
            shutil.rmtree(self.path)
            print(f"Temporary folder at {self.path} deleted.")
        except Exception as e:
            print(f"Error deleting temporary folder: {e}")



    def askForLabel(self):
        #ask the user for the label of the episode
        print("Please label the episode.")
        for key, value in self.episodeProbertyLabels.items():
            self.episodeProbertyLabels[key] = input(f"{key}: ")

        self.isLabeled = True


    def save(self, directory=None):
        

        #construct the dictionary to save

        episodeData = {}

        #construct header data
        header = {}
        header["timestamp"] = time.time()
        for key, value in self.episodeProbertyLabels.items():
            header[key] = value
        
        episodeData["header"] = header

        #save the frames
        frames = {}


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

        self.episodeDatabasePath = "/Database/"


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
            if force is False:
                self.endEpisode()
            else:
                self.endEpisode(discard=True)
        
        self.currentEpisode = Episode()
        return self.currentEpisode
        

        
                
    def endEpisode(self, discard=False):
        #save the episode to the database
        if discard is True:
            self.currentEpisode.clear()
            self.currentEpisode = None
            return
        
        if self.currentEpisode.isLabeled is False and self.currentEpisode.doAskForLabel:
            self.currentEpisode.askForLabel()

        if self.currentEpisode.isSaved is False:
            if self.currentEpisode.doAskForSave:
                answer = input("Do you want to save the episode? (y/n): ")
                if answer.lower() == 'y':
                    self.currentEpisode.save(self.episodeDatabasePath)
                else:
                    print("The episode was not saved.")
                    return
            else:
                self.currentEpisode.save(self.episodeDatabasePath)

        self.currentEpisode.clear()
        self.currentEpisode = None
        
        


    def nextEpisode(self):
        self.endEpisode()
        self.newEpisode()

        

            
            




        

