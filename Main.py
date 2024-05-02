
import pygame
import pygame.camera
from pygame.locals import *
import time

import numpy as np

import cv2



#from PygameController import *
from ModelController import *







#controller = PygameController()
controller = ModelController()



#try except keyboard interupt

try:
    controller.run()
except KeyboardInterrupt:
    print("Controlled Keyboard interupt")
    print(f"Shutting down")
    controller.close()
    quit()