

from Bronchoscope import *
from Camera import Camera
from CameraTop import CameraTop
from DataHandling.Episode import *
from Timer import Timer



class Interface:
    
    
    def __init__(self):
        
        self.broncho=Bronchoscope(port='/dev/broncho', baudrate=115200)
        self.broncho.start()
        self.bronchoCamera=Camera('/dev/broncho_camera')
        
        self.topCamera=CameraTop(exposure=50)
        
        self.currentEpisode=None
        self.currentState= self.broncho.getDict()
        self.currentInput=None
        
        self.episodeManager=EpisodeManager()
        
        self.oldDoStart=False
        self.oldDoStop=False
        #self.startEpisode()
        
        
    
    
    def getImage(self):
        
        
        
        
        self.currentImage = self.bronchoCamera.get_frame()
        Timer.point("gotBrocnhoImage")
        self.topImage = self.topCamera.get_frame()
        Timer.point("gotTopImage")
        
        
        return self.currentImage, self.topImage
    
    def updateInput(self, input, doStart, doStop):
        
        self.currentInput=input
        
        
        
        #self.currentState=(0,0) #self.broncho.getState()
        
        self.currentState= self.broncho.getDict()
        self.broncho.send(input)
        
        
        if(doStart):
            print("Starting Episode")
        if(doStop):
            print("Ending Episode")
            
            
        self.handleEpisode(doStart, doStop)
        
        self.doStoreCurrentFrame()
        
        
        
    def handleEpisode(self, doStart, doStop):
        if doStart and not self.oldDoStart:
            self.episodeManager.newEpisode()
            
        if doStop and not self.oldDoStop:
            self.episodeManager.endEpisode()
            
        self.oldDoStart=doStart
        self.oldDoStop=doStop
            
        
    def doStoreCurrentFrame(self):
        
        
        
        
        if(self.episodeManager.hasEpisode()):
            
            #frameDict=self.currentInput.toDict()
        
            stateDict = self.currentState
            inputDict = self.currentInput.toDict()
            miscDict = {}
        
        
        
            newFrame = Frame(self.currentImage, state=stateDict, action=inputDict, data=miscDict, topImage=self.topImage)
        
            
            self.episodeManager.append(newFrame)
        
        
        
    
    
        
        
    def close(self):
        print("Closing Interface")
        #self.camera.release()
        self.episodeManager.endEpisode()
        
        self.bronchoCamera.release()
        
        
        
        self.broncho.close()
        
        print("Interface closed")
    
    
    