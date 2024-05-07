

import pygame
import pygame.camera
from pygame.locals import *

#import tensorflow as tf

import time
from Controller import*
from TFLiteModel import TFLiteModel
from GUI import *

from Input import Input
from Training.ImageMod import preprocess_image
from PathTrackerInterfaceCV import PathTrackerInterface


class ModelController(Controller):
    
    def __init__(self):
        
        #super init
        super().__init__()
        
        self.gui = GUI()
        self.pathInterface= PathTrackerInterface("Training/model.keras")
        #self.input_shape = self.pathInterface.getInputShape()
        
        #load tf-lite model


        # Load the TFLite model and allocate tensors
        self.model = TFLiteModel("BronchoModel.tflite")
        
        
        
        
        
        
        
       
        
    def doStep(self, image):
        debug = False
        
        
        doStart = False
        doStop = False
        
        
        _, doExit, objects = self.pathInterface.predictAndTrack(image,image)
        
        currentKey=self.gui.update(image,objects)
        
        
        paths = []
        for key, item in objects.items():
            
            existance=1
            chosen=0
            x=item[0]
            y=item[1]
            if key is currentKey:
                chosen=1
                
            paths.append([x,y,existance,chosen])
            
        #fill paths up till 4
        for i in range(len(paths),4):
            paths.append([0,0,0,0])
           
        
         
        state = self.interface.currentState
        
        
        image=preprocess_image(image, mode="basic")
        
        #convert to numpy array in float32 bit format
        state = np.array([[state]], dtype=np.float32)
        image = np.array([image], dtype=np.float32)
        paths = np.array([paths], dtype=np.float32)
        
        
        #if new index is in obejcts keys
        if currentKey in objects.keys():
            
            print("Predicting")
            #prediction with 3 inputs: image, paths, state
            prediction = self.model.predict(state,image, paths)
            #print(prediction)
        
        
        
        
            
        
            
            
            
        
        
        
        
        input=Input(0,0,0)
        
        return input, doStart, doStop
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()