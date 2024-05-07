

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
        
        
        state = self.interface.currentState
        
        currentKey, doExit, joystick, manual=self.gui.update(image,objects,state)
        #print(f"Current Key: {currentKey}, keys: {objects.keys()}")
        
        
        
        input=Input(0,0,0)
        
        if manual:
            input=Input(joystick.rotation,joystick.bend,joystick.forwards)
            
        else:
            
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
            
            
            
            
            
            
            image=preprocess_image(image, mode="basic")
            
            #convert to numpy array in float32 bit format
            state = np.array([[state]], dtype=np.float32)
            image = np.array([image], dtype=np.float32)
            paths = np.array([paths], dtype=np.float32)
            
            
            #if new index is in obejcts keys
            if currentKey in objects.keys() and joystick.forwards>0.5:
                
                #print("Predicting")
                #prediction with 3 inputs: image, paths, state
                prediction = self.model.predict(state,image, paths)
                prediction=prediction[0]
                print(f"Prediction: {prediction}                ")
                input=Input(*prediction)
                
                #print prediction on same line
        
            elif joystick.forwards < -0.5:
                
                # 1 if state[1] is negative and -1 if state[1] is positive
                toNeutral = 1 if state[0][0][1] < 0 else -1
                
                input=Input(0,0,-1)
        
        
        
            
        
            
            
            
        
        
        
        
        
        
        return input, doStart, doStop
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()