




class Frame:
    def __init__(self, index, dirPath, imagePaths, data):
        self.index = index
        self.imagePaths = imagePaths
        self.data = data

        
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

        frame = cls(jsonFrame["index"], imagePaths, jsonFrame["data"])
        return frame
        


    @classmethod
    def fromImages(cls, index,dirPath, images, data):
        imagePaths = []
        for i in range(len(images)):
            imageName = f"frame{index}_{i}.png"
            imagePath = f"{dirPath}/{imageName}"

            images[i].save(imagePath)

            imagePaths.append(imagePath)

        return cls(index,dirPath, imagePaths, data)

    
    @classmethod
    def fromImage(cls, index,dirPath, image, data):

        images=[image]


        return cls.fromImages(index, dirPath, images, data)
