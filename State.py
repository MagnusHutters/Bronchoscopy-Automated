






class State:
    # rotation, bend, extendHistory, rotationHistory
    
    def __init__(self, currentRotation, currentBend, extendHistory, rotationHistory):
            
        self.currentRotation=currentRotation
        self.currentBend=currentBend
        self.extendHistory=extendHistory
        self.rotationHistory=rotationHistory
        
    @classmethod
    def fromDict(cls, data):
        
        currentRotation=data["currentRotation"]
        currentBend=data["currentBend"]
        extendHistory=data["extendHistory"]
        rotationHistory=data["rotationHistory"]
        
        return cls(currentRotation, currentBend, extendHistory, rotationHistory)
    
    def toDics(self):
        data={}
        data["currentRotation"]=self.currentRotation
        data["currentBend"]=self.currentBend
        data["extendHistory"]=self.extendHistory
        data["rotationHistory"]=self.rotationHistory
        return data
        
        
        