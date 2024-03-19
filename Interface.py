

from Bronchoscope import *
from Camera import *
from DataHandling.Episode import *



class Interface:
    
    
    def __init__(self):
        
        self.broncho=Bronchoscope()
        self.camera=Camera()
        self.currentEpisode=None
        
        
        self.startEpisode()
        
        
    
    
    def getImage(self):
        
        
        self.currentFrame = self.camera.get_frame()
        return self.currentFrame
    
    def updateInput(self, input):
        
        self.currentInput=input
        self.broncho.rotate(input.rotation)
        
        self.broncho.changeBend(input.bend)
        
        self.broncho.move(input.extend)   
        
        
        self.doStoreCurrentFrame()
        
        
    def doStoreCurrentFrame(self):
        
        state=self.broncho.getState()
        
        self.currentEpisode.addFrame(self.currentFrame,self.currentInput.toDict())
        pass
        
        
    def startEpisode(self):
        
        self.currentEpisode=Episode()
        
        
    def endEpisode(self):
        if self.currentEpisode != None:
            self.currentEpisode.saveEpisode()
    
    def nextEpisode(self):
        
        self.endEpisode()
        self.startEpisode()
        
        
    def close(self):
        print("Closing Interface")
        self.endEpisode()
        self.camera.release()
        
        self.broncho.close()
        
        print("Interface closed")
    
    
    