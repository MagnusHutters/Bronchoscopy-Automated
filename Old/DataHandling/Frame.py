

import signal
from PIL import Image

from multiprocessing import Process

MULTIPROCESS_SAVE = True


def saveToDiskThreadSafe(images, paths):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for i in range(len(images)):
        images[i].save(paths[i])
    
    
 
class Frame:
    def __init__(self, index, dirPath, imagePaths, data, images=[], storedOnDisk=True):
        self.index = index
        self.imagePaths = imagePaths
        self.images = images
        self.data = data
        
        
        self.storedOnDisk=storedOnDisk
        self.storedInRam=not storedOnDisk
        self.p =None
        self.result=None
        self.pooledProcess=False

        
        
    
        
    def isDoneSaving(self):
        if self.pooledProcess and self.result is not None:
            if self.result.ready():
                self.storedOnDisk = True
                return True
            else:
                return False
        if self.p is not None and not self.pooledProcess:
            if self.p.is_alive():
                return False
            else:
                self.p.join()
                self.p=None
                self.storedOnDisk = True
                return True    
        
        if(self.storedOnDisk):
            return True
        else:
            print(f"WARNING checking if done saving, but saving has not been initialized, start saving...")
            self.saveToDisk()
    
    def saveToDisk(self, pool = None):
        if(MULTIPROCESS_SAVE):
            if pool is not None:
                self.result = pool.apply_async(saveToDiskThreadSafe, (self.images, self.imagePaths))
                self.pooledProcess=True
                
            else:
                self.p = Process(target=saveToDiskThreadSafe, args=(self.images, self.imagePaths))
                self.p.start()
                self.pooledProcess=False
        
        else:
            for i in range(len(self.images)):
                self.images[i].save(self.imagePaths[i])
                self.storedOnDisk = True
         
    def ensureSaved(self, pool=None):
        if(not self.storedOnDisk):
            self.saveToDisk(pool)  
            
    def loadFromDisk(self):
        if(self.storedOnDisk):
            if(not self.storedInRam):
                for i in range(len(self.imagePaths)):
                    self.images.append(Image.open(self.imagePaths[i]))
                self.storedInRam = True
        else:
            raise Exception("Trying to load from disk, but frame is not stored on disk")
        
        
    def unloadFromRam(self):
        if(self.storedInRam):
            #Save if not on disk, then unload
            if(not self.storedOnDisk):
                self.saveToDisk()
            
            self.images = []
            self.storedInRam = False
        
    def getJsonStruct(self):
        return {
            "index": self.index,
            "imagePaths": self.imagePaths,
            "data": self.data
        }


    @classmethod
    def fromJsonFrame(cls, jsonFrame, episodePath):
        
        imagePaths = jsonFrame["imagePaths"]
        for i in range(len(imagePaths)):
            imagePaths[i] = episodePath + "/" + imagePaths[i]

        frame = cls(jsonFrame["index"], imagePaths, jsonFrame["data"], storedOnDisk=True)
        return frame
        


    @classmethod
    def fromImages(cls, index,dirPath, images, data):
        imagePaths = []
        for i in range(len(images)):
            imageName = f"frame{index}_{i}.png"
            imagePath = f"{dirPath}/{imageName}"

            #images[i].save(imagePath)

            imagePaths.append(imagePath)

        return cls(index,dirPath, imagePaths, data, images=images, storedOnDisk=False)

    
    @classmethod
    def fromImage(cls, index,dirPath, image, data):

        images=[image]


        return cls.fromImages(index, dirPath, images, data)
