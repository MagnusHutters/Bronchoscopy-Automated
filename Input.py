



class Input:
    
    def __init__(self, rotation, bend, extend):
        
        
        
        #clip at -1 to 1
        rotation = max(-1, min(1, rotation))
        bend = max(-1, min(1, bend))
        extend = max(-1, min(1, extend))
        
        self.rotation=rotation
        self.bend=bend
        self.extend=extend
        
        
        
        
        
        
    
    @classmethod
    def fromDict(cls, data):
        
        rotation=data["rotation"]
        bend=data["bend"]
        extend=data["extend"]
        
        return cls(rotation,bend,extend)
    
    
    
    def toDict(self):
        data={}
        data["rotation"]=self.rotation
        data["bend"]    =self.bend
        data["extend"]  =self.extend
        return data
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    