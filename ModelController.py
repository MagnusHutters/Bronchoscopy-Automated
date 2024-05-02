

import pygame
import pygame.camera
from pygame.locals import *

#import tensorflow as tf

import time
from Controller import*
from TFLiteModel import TFLiteModel
from GUI import *


class ModelController(Controller):
    
    def __init__(self):
        
        self.gui = GUI()
        self.pathInterface= PathTrackerInterface("Training/model.keras")
        self.input_shape = self.pathInterface.getInputShape()
        
        #load tf-lite model


        # Load the TFLite model and allocate tensors
        self.model = TFLiteModel("")
        
        
        
        
        
        
        
       
        
    def doStep(self, image):
        
        doStart = False
        doStop = False
        
        newIndex, doExit, objects = self.pathInterface.predictAndTrack(image,image)
        
        
        paths = []
        for key, item in objects.items():
            
            existance=1
            chosen=0
            x=item[0]
            y=item[1]
            if key is newIndex:
                chosen=1
                
            paths.append([x,y,existance,chosen])
            
        #fill paths up till 4
        for i in range(len(paths),4):
            paths.append([0,0,0,0])
            
        state = self.interface.currentState
        
        
        
        #if new index is in obejcts keys
        if newIndex in objects.keys():
            #prediction with 3 inputs: image, paths, state
            prediction = self.bronchoModel.predict([image, paths, state])
            print(prediction)
        
        
        
            
        
            
        
            
            
            
        
        
        
        
        input=[0,0,0]
        
        return input, doStart, doStop
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()