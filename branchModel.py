import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape
import numpy as np

from DataHandling.Episode import EpisodeManager, Episode
import FindBranchesCV as FindBranches
import BranchLabeller as BranchLabeller


def createModel():
    

    imageInput = Input(shape=(50, 50, 3))
    detailsInput = Input(shape=(30))


    conv1 = Conv2D(8, kernel_size=(3, 3), activation='relu')(imageInput)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(16, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(32, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    

    flattenCNN = Flatten()(pool3)

    detailsDense = Dense(32, activation='relu')(detailsInput)
    detailsDense2 = Dense(16, activation='relu')(detailsDense)

    concatenatedArrays = concatenate([flattenCNN, detailsDense2])

    dense1 = Dense(32, activation='relu')(concatenatedArrays)
    dense2 = Dense(16, activation='relu')(dense1)
    output = Dense(1, activation='sigmoid')(dense2)


    model = tf.keras.Model(inputs=[imageInput, detailsInput], outputs=output)


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model



def prepareData(path):
    episodeManager = EpisodeManager(mode = "Read", loadLocation="DatabaseLabelled/")


    inputImages = []
    inputDetails = []
    outputs = []


    originalImageSize = (400,400)
    modelImageSize = (50,50)


    hasMoreBranches = True

    while episodeManager.hasNextEpisode() and hasMoreBranches:
        episodeManager.nextEpisode()
        episode = episodeManager.getCurrentEpisode()

        for index in range(len(episode)):

            frame = episode[index]

            print(f"index: {len(outputs)}, Episode: {episodeManager.currentIndex}, Frame: {index}                   ", end="\r")

            #print(f"Frame: {frame}")

            #check if the frame dont have branches

            #type of object
            #print(f"Type of frame: {type(frame)}")


            #print(f"Frame data: {frame.data}")

            if not frame.data.get("Branches"):
                hasMoreBranches = False
                break

            branches = frame.data.get("Branches")


            branchImages, branchDetails, branchOutputs = extractModelInput(branches, modelImageSize, originalImageSize)

            inputImages.extend(branchImages)
            inputDetails.extend(branchDetails)
            outputs.extend(branchOutputs)


        #break

    print(f"Converting to numpy arrays...")
    inputImages = np.array(inputImages)
    inputDetails = np.array(inputDetails)
    outputs = np.array(outputs)



    return inputImages, inputDetails, outputs

def extractModelInput(branches, modelImageSize, originalImageSize):

    
    scalingFactor = modelImageSize[0]/originalImageSize[0]
    branchImages = []
    branchDetails = []
    branchOutputs = []

    for branch in branches:
                #print status update on same line
        #print(f"Index {len(outputs)}, Episode: {episodeManager.currentIndex}, Frame: {index}, Branch: {branches.index(branch)}                   ", end="\r")


        childrenIndices = branch.get("ChildrenIndices", [])
        parendIndex = branch.get("ParentIndex", -1)

        selfContour = branch.get("Contour")
        selfContour = BranchLabeller.deserializeContour(selfContour)
        selfContours = selfContour


        roundness = branch.get("Roundness")
        depth = branch.get("Depth")
        size = branch.get("Size")
        x, y = branch.get("LowestPoint")
        numChildren = branch.get("NumChildren")
        numSiblings = branch.get("NumSiblings")
        area = branch.get("Area")
        circumference = branch.get("Circumference")



        parentRoundness = 0
        parentDepth = 0
        parentSize = 0
        parentArea = 0
        parentCircumference = 0



        parentContours=[]
        if parendIndex >= 0 and parendIndex < len(branches):
            parent = branches[parendIndex]
            parentContour = parent.get("Contour")
            parentContour = BranchLabeller.deserializeContour(parentContour)
            parentContours.append(parentContour)


            parentRoundness = parent.get("Roundness")
            parentDepth = parent.get("Depth")
            parentSize = parent.get("Size")
            parentArea = parent.get("Area")
            parentCircumference = parent.get("Circumference")

                    
                
                #take the average of the childrens details
                
                    




        childRoundness = []
        childDepth = []
        childSize = []
        childArea = []
        childCircumference = []
                
        childContours = []

        for childIndex in childrenIndices:
            child = branches[childIndex]

            childContour = child.get("Contour")
            childContour = BranchLabeller.deserializeContour(childContour)

            childContours.append(childContour)

            childRoundness.append(child.get("Roundness"))
            childDepth.append(child.get("Depth"))
            childSize.append(child.get("Size"))
            childArea.append(child.get("Area"))
            childCircumference.append(child.get("Circumference"))

        childRoundness = np.mean(childRoundness)
        childDepth = np.mean(childDepth)
        childSize = np.mean(childSize)
        childArea = np.mean(childArea)
        childCircumference = np.mean(childCircumference)


        details = [roundness, depth, size, x, y, numChildren, numSiblings, area, circumference, \
                           parentRoundness, parentDepth, parentSize, parentArea, parentCircumference, \
                            childRoundness, childDepth, childSize, childArea, childCircumference]
                
                #fill the details with zeros if the length is less than 30
        details = details + [0] * (30 - len(details))

        details = np.array(details, dtype=np.float32)



                #contours to np array
        selfContours = np.array(selfContours)
        parentContours = np.array(parentContours)
                #childContours = np.array(childContours)

                #resize the contours

        selfContours = selfContours * scalingFactor
        parentContours = parentContours * scalingFactor

        for i in range(len(childContours)):
            childContours[i] = childContours[i] * scalingFactor
            childContours[i] = childContours[i].astype(np.int32)
                #convert to int
        selfContours = selfContours.astype(np.int32)
        parentContours = parentContours.astype(np.int32)
                #childContours = childContours.astype(np.int32)

                #reconstruct threshold images from contours using cv2.drawContours
        selfImage = np.zeros(modelImageSize)
        parentImage = np.zeros(modelImageSize)
        childImage = np.zeros(modelImageSize)



                #print(f"Self contours: {selfContours}")
        cv2.drawContours(selfImage, selfContours, -1, 1, -1)
        cv2.drawContours(parentImage, parentContours, -1, 1, -1)
        cv2.drawContours(childImage, childContours, -1, 1, -1)

                #stack the images in different channels
        image = np.stack([selfImage, parentImage, childImage], axis=-1)
        image = image.astype(np.float32)

        enabled = branch.get("enabled", False)

        output = 1 if enabled else 0

        branchImages.append(image)
        branchDetails.append(details)
        branchOutputs.append(output)
    return branchImages,branchDetails,branchOutputs




class BranchModel:
    def __init__(self, path):
        self.model = tf.keras.models.load_model(path)


    def predict(self, image):

        originalImageSize = (400,400)
        modelImageSize = (50,50)



        candidates = FindBranches.thresholdTree(image)

        branchData = BranchLabeller.contsructBranchDataFromCandidates(candidates)



        branchImages, branchDetails, _ = extractModelInput(branchData, modelImageSize, originalImageSize)

        branchImages = np.array(branchImages)
        branchDetails = np.array(branchDetails)

        predictions = self.model.predict([branchImages, branchDetails])

        return predictions


                
def main():
    model = createModel()

    inputImages, inputDetails, output = prepareData("DatabaseLabelled/")

    print(f"Input images: {inputImages.shape}")

    validationSplit = 0.2
    splitIndex = int(len(inputImages) * (1 - validationSplit))


    indices = np.arange(len(inputImages))
    np.random.shuffle(indices)

    inputImages = inputImages[indices]
    inputDetails = inputDetails[indices]
    output = output[indices]


    inputImages, inputImagesValidation = inputImages[:splitIndex], inputImages[splitIndex:]
    inputDetails, inputDetailsValidation = inputDetails[:splitIndex], inputDetails[splitIndex:]
    output, outputValidation = output[:splitIndex], output[splitIndex:]

    #shuffle
    



    model.fit([inputImages, inputDetails], output, epochs=10, batch_size=32, validation_data=([inputImagesValidation, inputDetailsValidation], outputValidation))

    model.save("BranchModel.h5")




if __name__ == '__main__':
    main()


    



#createModel()