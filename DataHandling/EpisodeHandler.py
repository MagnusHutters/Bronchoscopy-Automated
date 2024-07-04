
import os
import json
import tempfile
import shutil
import numpy as np

from PIL import Image




class Episode:

    class Frame:
        def __init__(self, image, state, action, ladatabel):

            #ensure the image is a numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            self.image = image
            self.state = state
            self.action = action
            self.data = data



    def __init__(self, doAskForLabel=True, doAskForSave=True):
        
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


    #overload the [] operator to construct and get the frame at the given index
    def __getitem__(self, index):
        return self.Frame(self.images[index], self.states[index], self.actions[index], self.data[index])
    
    #overload the [] operator to set the frame at the given index
    def __setitem__(self, index, frame):
        self.images[index] = frame.image
        self.states[index] = frame.state
        self.actions[index] = frame.action
        self.data[index] = frame.data



    def _addFrame(self, frame):
        self.images.append(frame.image)
        self.states.append(frame.state)
        self.actions.append(frame.action)
        self.data.append(frame.data)

        #save image - from numpy format to png
        
        
        image = Image.fromarray(image)
        imagePath = self.getImagePath(len(self.images)-1)
        image.save(imagePath)


    def append(self, frameOrImage, state=None, action=None, data=None):
        if state is None: #if only one argument is given, it is a frame
            #check if the frame is a Frame object
            if not isinstance(frameOrImage, self.Frame):
                raise ValueError("If only one argument is given, it must be a Frame object.")
            self._addFrame(frameOrImage)
        else:
            self._addFrame(self.Frame(frameOrImage, state, action, data))
            

    def getImagePath(self, index): #returns the path of the image at the given index, usign zfill to pad the index with zeros
        return os.path.join(self.path, f"{str(index).zfill(5)}.png")



    def __del__(self):
        try:
            shutil.rmtree(self.path)
            print(f"Temporary folder at {self.path} deleted.")
        except Exception as e:
            print(f"Error deleting temporary folder: {e}")