

import os
import shutil
import tempfile
import json
import cv2
import numpy as np
import tensorflow as tf




from PathTracker import PathTracker
from SETTINGS import *





class PathModelInterface:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.model = None
        self.loadModel()
        self.pathTracker = None
        self.resetPathTracker()


    def loadModel(self):
        self.model = tf.keras.models.load_model(self.modelPath)

    def resetPathTracker(self):
        self.pathTracker = PathTracker()

    def predict(self, image, isNormalized=False, doTracking=True):

        #ensure the image is in the correct format
        image = np.array([image])
        
        #from settings
        requiredShape = IMAGE_SIZE
        #if not the correct shape, resize
        if image.shape[1:] != requiredShape:
            image = cv2.resize(image, requiredShape)
        
        #normalize the image
        if not isNormalized:
            image = image / 255.0

        #predict
        prediction = self.model.predict(image)[0]

        if not doTracking:
            return prediction

        trackedPredictions = self.pathTracker.update(prediction)

        return trackedPredictions






        