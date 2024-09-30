

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
from branchModelTracker import BranchModelTracker
from BronchoBehaviourModelImplicit import BronchoBehaviourModelImplicit


#from pathLabelNewActions import guessBranch
from VisualServoing import doVisualServoing


class ModelController(Controller):
    
    def __init__(self):

        cv2.namedWindow("Visual servoing", cv2.WINDOW_NORMAL)
        
        #super init
        super().__init__()
        
        self.gui = GUI()
        #self.pathInterface= PathTrackerInterface("Training/model.keras")
        #self.input_shape = self.pathInterface.getInputShape()
        

        print("Loading branch model")

        self.branchModelTracker = BranchModelTracker(\
            "C:/Users/magnu/OneDrive/Misc/Ny mappe/Bronchoscopy-Automated/BronchoYolo/yolov5/runs/train/branchTraining11-X/weights/best.pt",\
            featureScale=1)
        
        #behavioir model
        

        print("Loading behaviour model")
        self.model = BronchoBehaviourModelImplicit(model_path="C:/Users/magnu/OneDrive/Misc/Ny mappe/Bronchoscopy-Automated/runs/implictTraining_63/epoch/8/epochModel.pth")
        #C:\Users\magnu\OneDrive\Misc\Ny mappe\Bronchoscopy-Automated\runs\implictTraining_63\epoch\8\epochModel.pth
        
        self.override_active=False
        #self.manual=True

        self.mode = 1 #0: Manual #1: visual servoing, 2: behaviour model
        self.oldMode = 0


        self.currentKey = -1
        self.oldKey = -1

        self.oldVisualKey = -1
        
        
        
    
        
        
        
       
        
    def doStep(self, image, topImage=None):
        
        Timer.point("doStepStart")
        debug = False
        
        
        doStart = False
        doStop = False

        if self.mode != self.oldMode:
            print(f"Mode changed to {self.mode}")
            self.oldMode = self.mode
            self.branchModelTracker.reset()
        
        
        #_, doExit, objects = self.pathInterface.predictAndTrack(image,image)
        
        activeTracking = False if self.mode==0 else True
        branchPoints, branchPredictions = self.branchModelTracker.predict(image, active = activeTracking, currentKey = self.currentKey)

        #print(f"BranchPoints: {branchPoints}")
        
        
        #objects=[]
        
        state = self.interface.currentState
        
        rotationDegrees = state["rotationReal_deg"]
        bendDegrees = state["bendReal_deg"]
        extensionMM = state["extensionReal_mm"]
        
        
        recording = False
        currentFrame=0
        if self.interface.episodeManager.hasEpisode():
            recording = True
            currentFrame = len(self.interface.episodeManager.currentEpisode)
            
            
            
        
        
        
        
        
        Timer.point("beforeGUIUpdate")
        currentKey, doExit, joystick, mode, screenImage=self.gui.update(image,(branchPoints, branchPredictions),state, recording, currentFrame, topImage)

        self.interface.updateScreenImage(screenImage)
        self.oldKey = self.currentKey
        self.currentKey = currentKey
        self.mode = mode
        Timer.point("afterGUIUpdate")
        #print(f"Current Key: {currentKey}, keys: {objects.keys()}")
        

        newKey = self.currentKey != self.oldKey
        
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
        

        if joystick.l2: #reset the broncho to the home position
            self.interface.broncho.home()
            self.branchModelTracker.reset()

        if mode==0:
            input=Input.fromJoystick(joystick)

             # Reset the broncho to the home position
            #print(f"Manual: {input}")
            
        elif mode==1: #visual servoing

            state = self.interface.currentState

            detectionDict = {}
            if currentKey in branchPoints.keys() and joystick.forwards>0.5:
                goalAbs = branchPoints.get(currentKey, (0,0))
                goalDetection = branchPredictions.get(currentKey, {})

                detectionDict = goalDetection.toDict()


                imageSize = (image.shape[1], image.shape[0])
                center = (imageSize[0]//2, imageSize[1]//2)

                goal = (goalAbs[0] - center[0], goalAbs[1] - center[1])

                #currentJoints = None #not used - also redundant with state

                doVisualize = False
                maxDist = 99999
                bendMultiplier = 1.5

                #print(f"State: {state}")
                #print(f"Goal: {goal}")½½
                #print(f"imageSize: {imageSize}")
                #print(f"GoalAbs: {goalAbs}")

                newVisualKey = currentKey != self.oldVisualKey
                
                print(f"Rotation: {rotationDegrees}, Bend: {bendDegrees}, Extension: {extensionMM}")
                action = doVisualServoing(image, state, detectionDict, imageSize, doVisualize, maxDist, bendMultiplier, resetAcumulator = newVisualKey)
                input = Input.fromChar(action)
                

                self.oldVisualKey = currentKey
            
            elif joystick.forwards < -0.5:
                
                # 1 if state[1] is negative and -1 if state[1] is positive
                #toNeutral = 1 if state[0][0][1] < 0 else -1
                
                input=Input.fromChar("b")

            else:
                input = Input()



            #uses this function
            #action = guessBranch(state, goal, imageSize, currentJoints, doVisualize = False, maxDist = 5000, limitCount = 0):

            



        elif mode==2: #behaviour model
            

            
            
            
            
            
            
            
            #image=preprocess_image(image, mode="basic")
            
            #convert to numpy array in float32 bit format
            #state = np.array([[state]], dtype=np.float32)
            #image = np.array([image], dtype=np.float32)
            #paths = np.array([paths], dtype=np.float32)
            
            
            #if new index is in obejcts keys
            #if currentKey in objects.keys() and joystick.forwards>0.5:
                
                #print("Predicting")
                #prediction with 3 inputs: image, paths, state
            #    prediction = self.model.predict(state,image, paths)
            #    prediction=prediction[0]
            #    print(f"Prediction: {prediction}                ")
            #    input=Input(*prediction)
                
                #print prediction on same line
            if joystick.forwards > 0.5:
                if currentKey in branchPredictions.keys():
            
                    action = self.model.predict(image, state, branchPredictions, currentKey)
                else:
                    action = -1

                input = Input.fromActionValue(action)

            elif joystick.forwards < -0.5:
                
                # 1 if state[1] is negative and -1 if state[1] is positive
                #toNeutral = 1 if state[0][0][1] < 0 else -1
                
                input=Input.fromChar("b")

            else:
                input = Input()
        
    
    
        
    
            
            
            
        
        
        
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