

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
        
        
        self.override_end_time = 0
        self.override_active = False
        self.override_type = None
        self.last_nonzero_rotation = 1
        
        self.save_directory = "Override_images"
        self.override_timestamp = None
        
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        
        
        
        
        
    def save_image(self, image, override_type, timing):
        prefix = "exp6"
        
        timestamp=0
        if timing =="before":
            self.override_timestamp = int(time.time() * 1000)  # Milliseconds timestamp for uniqueness
        timestamp = self.override_timestamp
        
        
        
        dir = f"{self.save_directory}/{prefix}_{override_type}_{timestamp}"
        
        #if dir does not exist, create it
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        
        # Convert the image to a format that can be saved (if needed)
        filename = f"{dir}/{timing}.png"
        cv2.imwrite(filename, image)
        print(f"Saved image: {filename}")
        
        
       
        
    def doStep(self, image):
        debug = False
        
        
        doStart = False
        doStop = False
        
        
        _, doExit, objects = self.pathInterface.predictAndTrack(image,image)
        
        
        state = self.interface.currentState
        
        
        recording = False
        currentFrame=0
        if self.interface.currentEpisode is not None:
            recording = True
            currentFrame = self.interface.currentEpisode.length
        
        
        currentKey, doExit, joystick, manual, screenImage=self.gui.update(image,objects,state, recording,currentFrame)
        #print(f"Current Key: {currentKey}, keys: {objects.keys()}")
        
        
        doStart = joystick.start
        doStop=joystick.stop



        
        input=Input(0,0,0)


        # Track last non-zero joystick.rotation
        if joystick.rotation != 0:
            self.last_nonzero_rotation = joystick.rotation
        
        # Check for joystick.r2 change from 0 to 1 and activate extension override
        if joystick.r2 == 1 and not self.override_active:
            self.override_active = True
            self.override_end_time = time.time() + 1  # Set override for 1 second
            self.override_type = 'extension'
            self.save_image(image, self.override_type, "before")
        
        # Check for joystick.l2 change from 0 to 1 and activate rotation override
        if joystick.l2 == 1 and not self.override_active:
            self.override_active = True
            self.override_end_time = time.time() + 1  # Set override for 1 second
            self.override_type = 'rotation'
            self.save_image(image, self.override_type, "before")
        

        if self.override_active:
            if self.override_type == 'extension':
                input = Input(0, 0, 1)
            elif self.override_type == 'rotation':
                rotation_direction = 1 if self.last_nonzero_rotation > 0 else -1
                input = Input(rotation_direction, 0, 0)
            
            if time.time() >= self.override_end_time:
                self.override_active = False
                self.save_image(image, self.override_type, "post")
                self.override_type = None
        else:

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
            
        
        
            
        
            
            
            
        
        
        
        
        
        
        return input, doStart, doStop, None
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()