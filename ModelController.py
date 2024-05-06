

import pygame
import pygame.camera
from pygame.locals import *

#import tensorflow as tf

import time
from Controller import*
from TFLiteModel import TFLiteModel
from GUI import *
from PathTrackerInterfaceEmbedded import PathTrackerInterfaceEmbedded


class ModelController(Controller):
    
    def __init__(self):
        super().__init__()
        
        self.gui = GUI()
        self.pathInterface= PathTrackerInterfaceEmbedded("pathModel.tflite")
        #self.input_shape = self.pathInterface.getInputShape()
        
        #load tf-lite model


        # Load the TFLite model and allocate tensors
        self.model = TFLiteModel("BronchoModel.tflite")
        
        
        
        
        
        
        
       
        
    def doStep(self, image):
        
        doStart = False
        doStop = False
        
        image = np.array(image)
        image = image.astype(np.float32)
        image = image/255.0
        
        _, doExit, objects = self.pathInterface.predictAndTrack(valImage,image)
        
        index, doExit = self.gui.update(image, objects)
        
        
        paths = []
        for key, item in objects.items():
            
            existance=1
            chosen=0
            x=item[0]
            y=item[1]
            if key is index:
                chosen=1
                
            paths.append([x,y,existance,chosen])
            
        #fill paths up till 4
        for i in range(len(paths),4):
            paths.append([0,0,0,0])
            
        state = self.interface.currentState
        
        input=Input(0,0,0)
        print(f"index: {index}")
        print(f"Keys: {objects.keys()}")
        #if new index is in obejcts keys
        if index in objects.keys():
            #prediction with 3 inputs: image, paths, state
            #prediction = self.bronchoModel.predict([image, paths, state])
            
            state = np.array([[state]])
            valImage = np.array([valImage])
            paths = np.array([paths])
            
            #to float32
            state = state.astype(np.float32)
            valImage = valImage.astype(np.float32)
            paths = paths.astype(np.float32)
            
            
            prediction = self.model.predict(state, valImage, paths)
            print(prediction)
            
            input = Input(*(prediction[0]))
        
        
        
            
        
            
        
            
            
            
        
        
        
        
        
        
        return input, doStart, doStop
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()