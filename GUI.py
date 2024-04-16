



import pygame
import pygame.camera
from pygame.locals import *


import numpy as np
import cv2



import time

from Training.PathTrackerInterface import PathTrackerInterface
from Training.BasicPaths import load_images_single_episode, load_images

class GUI:
    def __init__(self, size=None):
        
        self.hasWindow = False
        self.size = size

        if size is not None:
            self.create_window(size)

        
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)

        self.current_index = -1

        #self.pathInterface = PathModelInterface("Training/model.keras")


    def create_window(self, size):
        pygame.init()
        self.size = size
        self.screen = pygame.display.set_mode(size)
        
        self.hasWindow = True
        

        


    def select_point(self, points, current_index, key_press):
        # Early exit if no points are available
        if not points:
            return -1  # Return -1 to indicate no points are selected

        #move points range from -1 to 1 to 0 to 1
        points=points.copy()

        for key, value in points.items():
            x, y = value
            x = (x + 1) / 2
            y = (y + 1) / 2
            #points[key] = (x, y)

        # Key mappings and their orthogonal directions for lateral calculations
        directions = {
            0: np.array([0, -1]),    # Up
            1: np.array([1, 0]),    # Right
            2: np.array([0, 1]),   # Down
            3: np.array([-1, 0])    # Left
        }
        orthogonal = {
            0: np.array([1, 0]),    # Lateral for Up/Down
            1: np.array([0, 1]),    # Lateral for Right/Left
            2: np.array([1, 0]),    # Lateral for Up/Down
            3: np.array([0, 1])     # Lateral for Right/Left
        }
        directionNames = {
            0: "Up",
            1: "Right",
            2: "Down",
            3: "Left"
        }


        # Weights for directional and lateral distances
        directional_weight = 0.7
        lateral_weight = 0.3
        tolerance = 0.01  # Allows slight backwardness in the direction
        
        # Validate current index
        if current_index not in points and current_index != -1:
            current_index = -1  # Default to no selection if the current index is invalid
        
        # Handling no key press or unselection
        if key_press == -1:
            return current_index
        if key_press == -2:
            return -1
        
        # Calculate the centroid for the "unselected" state node
        centroid = np.mean(list(points.values()), axis=0) if points else np.array([0.5, 0.5])
        
        # Determine the current position
        current_pos = centroid if current_index == -1 else np.array(points[current_index])
        
        # Initialize best index to current index to maintain selection if no better point is found
        best_index = current_index
        best_score = float('inf')
        if key_press in directions:
            direction = directions[key_press]
            lateral_dir = orthogonal[key_press]
            
            for idx, point in points.items():
                point_vec = np.array(point)
                point_diff = point_vec - current_pos
                directional_distance = np.dot(point_diff, direction)
                lateral_distance = np.dot(point_diff, lateral_dir)
                
                # Calculate a weighted score for the point
                if directional_distance > -tolerance and idx is not current_index:
                    #if lateral distance is too much greater than the directional distance, ignore the point
                    if np.abs(lateral_distance) > directional_distance*2:
                        continue

                    score = (directional_weight * directional_distance +
                            lateral_weight * np.abs(lateral_distance))
                    
                    print(f"Point {idx} - Directional: {directional_distance}, Lateral: {lateral_distance}, Score: {score}")
                    # Select the point with the lowest score (considering lower lateral distance, higher forward distance)
                    if score < best_score:
                        best_score = score
                        best_index = idx
        

        #print information about the selected point
        print(f"Points are:")
        for key, value in points.items():
            print(f"\t{key}:\t {value}")
        print(f"Current position: \n\t0:\t {current_pos}")
        print(f"You pressed: \n\t {key_press}:\t {directionNames[key_press]}")
        print(f"In the direction \n\t {directions[key_press]}")
        #and selected
        print(f"Selected: \n\t {best_index}:\t {points[best_index]}")
        



        return best_index


    




    def update(self, originalImage, objects):

        #create window if it does not exist using the size of the image
        if not self.hasWindow:
            self.create_window(originalImage.shape[:2])




        selectEvent = -1

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                #exit the program
                return True
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selectEvent = 0
                if event.key == K_RIGHT:
                    selectEvent = 1
                if event.key == K_DOWN:
                    selectEvent = 2
                if event.key == K_LEFT:
                    selectEvent = 3
                if event.key == K_SPACE:
                    selectEvent = -2
            

        self.current_index = self.select_point(objects, self.current_index, selectEvent)

        #Exit if doExit is True
        
        


        #prediction = self.pathInterface.predict(image,originalImage, doTracking=True)
        #print(prediction)
        
        #print(f"DRAWING IMAGE {image.shape} {originalImage.shape} {len(objects)}")

        #convert image to pygame format

        originalImage=originalImage*255
        originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        #flip x and y axis by transposing the image
        originalImage = np.transpose(originalImage, (1, 0, 2))


        originalImage = pygame.surfarray.make_surface(originalImage)

        #draw image on screen
        self.screen.blit(originalImage, (0, 0))

        




        for key, value in objects.items():
            x, y = value
            
            x = int(((x + 1) / 2) * self.size[0])
            y = int(((y + 1) / 2) * self.size[1])

            if key == self.current_index:
                pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 5)
            else:
                pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 5)
        pygame.display.flip()

        
            
        return self.current_index, False
            
        #time.sleep(0.1)









def main():
    



    pathInterface= PathTrackerInterface("Training/model.keras")
    input_shape = pathInterface.getInputShape()

        #val_images, realImageSize, originalImages = load_images("Training/Data/PathData", input_shape, saveOriginalImages=True)
    val_images, realImageSize, originalImages = load_images_single_episode("Training/Data/PathData/24-03-19-15-59-18_0", input_shape, saveOriginalImages=True)

    pathInterface.realImageSize = realImageSize

    #load images
    imageFolder = "Training/Data/PathData"

    #images, realSize, originals = load_images(imageFolder, (128, 128), saveOriginalImages=True)

    gui = GUI()

    index = 0

    while True:
        #image = images[imageIndex]
        #gui.update(image,originals[imageIndex])
        #imageIndex += 1
        #if imageIndex >= len(images):
        #    imageIndex = 0


        val_image = val_images[index]

        displayImage = originalImages[index].copy()    
        newIndex, doExit, objects = pathInterface.predictAndTrack(val_image,displayImage)

        #Update index
        index = index+1
        index = index % len(val_images) # make sure index is within bounds

        #print(f"Index: {index}")



        currentObjectIndex, doExit = gui.update(displayImage, objects)
        if doExit:
            break
        
        
        #time.sleep(0.1)


if __name__ == "__main__":
    main()




    

        

        




        
