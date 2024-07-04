

from Bronchoscope import *
from Camera import *
from DataHandling.EpisodeOld import *



class Interface:
    
    
    def __init__(self):
        
        self.broncho=Bronchoscope()
        self.camera=Camera()
        
        self.currentEpisode=None
        self.currentState=(0,0) #self.broncho.getState()
        self.currentInput=None
        
        
        #self.startEpisode()
        
        
    
    
    def getImage(self):
        
        
        self.currentFrame = self.camera.get_frame()
        return self.currentFrame
    
    def updateInput(self, input, doStart, doStop):
        
        self.currentInput=input
        
        
        
        self.currentState=(0,0) #self.broncho.getState()
        self.broncho.rotate(input.rotation)
        
        self.broncho.changeBend(input.bend)
        
        self.broncho.move(input.extend)   
        
        
        if(doStart):
            self.startEpisode()
            print("Starting Episode")
        if(doStop):
            self.endEpisode()
            print("Ending Episode")
        
        self.doStoreCurrentFrame()
        
        
    def doStoreCurrentFrame(self):
        
        
        frameDict=self.currentInput.toDict()
        frameDict['currentBend']=self.currentState[0]
        frameDict['currentRot']=self.currentState[1]
        
        
        if(self.currentEpisode is not None):
            
            self.currentEpisode.addFrame(self.currentFrame,frameDict)
        
        
        
    def startEpisode(self):
        
        self.currentEpisode=Episode()
        
        
    def endEpisode(self):
        if self.currentEpisode != None:
            self.currentEpisode.saveEpisode()
            
        #delete current episode
        
        self.currentEpisode=None
    
    def nextEpisode(self):
        
        self.endEpisode()
        self.startEpisode()
        
        
    def close(self):
        print("Closing Interface")
        self.camera.release()
        self.endEpisode()
        
        self.broncho.close()
        
        print("Interface closed")
    
    
    