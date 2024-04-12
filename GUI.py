



import pygame
import pygame.camera
from pygame.locals import *


import numpy as np
import cv2



import time

from Training.PathModelInterface import PathModelInterface

class GUI:
    def __init__(self, size=(640, 480)):
        pygame.init()

            
            
        self.js = pygame.joystick.Joystick(0)


        self.size = size

        self.screen = pygame.display.set_mode(self.size)
        
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)

        self.pathInterface = PathModelInterface("Training/model.keras")


    def update(self, image):


        prediction = self.pathInterface.predict(image, doTracking=True)
        print(prediction)
        

        #draw image on screen
        self.screen.blit(image, (0, 0))

        for key, value in prediction.items():
            x, y = value
            x = int(x)
            y = int(y)
            pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 5)
        pygame.display.flip()









def main():
    gui = GUI()

    #load images
    imageFolder = "Training\Data\PathData\24-03-19-15-59-18_0"

    

        

        




        
