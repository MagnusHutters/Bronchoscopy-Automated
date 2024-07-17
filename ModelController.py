

import pygame
import pygame.camera
from pygame.locals import *

#import tensorflow as tf


from Timer import Timer
import time
from Controller import*
#from TFLiteModel import TFLiteModel
from GUI import *

from Input import Input
#from Training.ImageMod import preprocess_image
#from PathTrackerInterfaceCV import PathTrackerInterface


class ModelController(Controller):
    
    def __init__(self):
        
        #super init
        super().__init__()
        
        self.gui = GUI()
        #self.pathInterface= PathTrackerInterface("Training/model.keras")
        #self.input_shape = self.pathInterface.getInputShape()
        
        #load tf-lite model


        # Load the TFLite model and allocate tensors
        self.model = None #TFLiteModel("BronchoModel.tflite")
        
        self.override_active=False
        
        
        
        
        
       
        
    def doStep(self, image, topImage=None):
        
        Timer.point("doStepStart")
        debug = False
        
        
        doStart = False
        doStop = False
        
        
        #_, doExit, objects = self.pathInterface.predictAndTrack(image,image)
        
        
        
        objects=[]
        
        state = self.interface.currentState
        
        
        
        recording = False
        currentFrame=0
        if self.interface.episodeManager.hasEpisode():
            recording = True
            currentFrame = len(self.interface.episodeManager.currentEpisode)
            
        
        
        
        
        
        Timer.point("beforeGUIUpdate")
        currentKey, doExit, joystick, manual=self.gui.update(image,objects,state, recording, currentFrame, topImage)
        Timer.point("afterGUIUpdate")
        #print(f"Current Key: {currentKey}, keys: {objects.keys()}")
        
        
        doStart = joystick.start
        doStop = joystick.select



        
        input=Input()


        ## Track last non-zero joystick.rotation
        #if joystick.rotation != 0:
        #    self.last_nonzero_rotation = joystick.rotation
        #
        ## Check for joystick.r2 change from 0 to 1 and activate extension override
        #if joystick.r2 == 1 and not self.override_active:
        #    self.override_active = True
        #    self.override_end_time = time.time() + 1  # Set override for 1 second
        #    self.override_type = 'extension'
        #
        ## Check for joystick.l2 change from 0 to 1 and activate rotation override
        #if joystick.l2 == 1 and not self.override_active:
        #    self.override_active = True
        #    self.override_end_time = time.time() + 1  # Set override for 1 second
        #    self.override_type = 'rotation'
        

        if False: #self.override_active:
            if self.override_type == 'extension':
                #input = Input(0, 0, 1)
                pass
            elif self.override_type == 'rotation':
                rotation_direction = 1 if self.last_nonzero_rotation > 0 else -1
                #input = Input(rotation_direction, 0, 0)
            
            if time.time() >= self.override_end_time:
                self.override_active = False
                self.override_type = None
        else:

            if manual:
                input=Input.fromJoystick(joystick)

                if joystick.l2:
                    self.interface.broncho.home() # Reset the broncho to the home position
                #print(f"Manual: {input}")
                
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
                
                
                
                
                
                
                #image=preprocess_image(image, mode="basic")
                
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
            
        
        
            
        
            
            
            
        
        
        
        if doExit:
            self.running = False
            self.close()
            
        Timer.point("doStepEnd")
        
        return input, doStart, doStop
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()