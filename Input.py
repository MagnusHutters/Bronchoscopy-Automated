



class Input:
    
    def __init__(self, rotation, bend, extend):
        
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
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    