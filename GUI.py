



import pygame
import pygame.camera
from pygame.locals import *


import numpy as np
import cv2



import time

#from Training.PathTrackerInterfaceCV import PathTrackerInterface
#from Training.BasicPaths import load_images_single_episode, load_images



class JoystickData:
    def __init__(self, forwards, rotate,bend,l1,r1,l2,r2):
        self.forwards=forwards
        self.rotation=rotate
        self.bend=bend
        self.l1=l1
        self.r1=r1
        self.l2=l2
        self.r2=r2
        self.dir=[self.rotation,self.bend]
        
        
    #from joystick
    @staticmethod
    def fromJoystick(js):
        num_axes = js.get_numaxes()
        axes = [js.get_axis(i) for i in range(num_axes)]
        
        
        num_buttons = js.get_numbuttons()
        buttons=[js.get_button(i) for i in range(num_buttons)]    
        
        forwards=buttons[2]-buttons[0]
        
        hat=js.get_hat(0)
        rotate=axes[0]+hat[0]
        bend = axes[1]+hat[1]
        
        #clip rotate and bend to [-1,1]
        rotate = np.clip(rotate, -1, 1)
        bend = np.clip(bend, -1, 1)
        
        l1=buttons[4]
        r1=buttons[5]
        l2=buttons[6]
        r2=buttons[7]
        
        
        
        
        return JoystickData(forwards, rotate,bend,l1,r1,l2,r2)

class GUI:
    def __init__(self, size=None):
        
        self.hasWindow = False
        self.size = size

        if size is not None:
            self.create_window(size)


        self.manual=True
        
        pygame.font.init()
        self.font = pygame.font.Font(None, 18)

        self.current_index = -1

        #self.pathInterface = PathModelInterface("Training/model.keras")
        
        #init joystick
        pygame.joystick.init()
        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        


    def create_window(self, size):
        pygame.init()
        self.size = size
        self.screen = pygame.display.set_mode(size)
        
        self.hasWindow = True
        

    def get_key_press_from_dir(self, dir): #dir is a x,y vector
        if dir[0] > 0.5:
            return 2
        if dir[0] < -0.5:
            return 0
        if dir[1] > 0.5:
            return 1
        if dir[1] < -0.5:
            return 3
        return -1
        


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
        if best_index>=0:
            print(f"Selected: \n\t {best_index}:\t {points[best_index]}")
        else: 
            print(f"No points selected")



        return best_index


    

    def drawBar(self, value):
        print(f"Drawing bar with value: {value}")
            
        value=value*self.size[1]    
            
        # Calculate bar dimensions
        bar_width = 20
        bar_height = abs(value/2)

        # Create a surface for the bar
        bar_surface = pygame.Surface((bar_width, bar_height)).convert_alpha()
        #bar_surface.fill(pygame.TRANSPARENT)

        # Determine the color based on the sign of the value
        if value >= 0:
            color = (0, 255, 0)  # Green for positive values
        else:
            color = (255, 0, 0)  # Red for negative values

        # Fill the bar with the color
        pygame.draw.rect(bar_surface, color, (0, 0, bar_width, bar_height))

        # Position the bar in the middle of the screen
        bar_rect = bar_surface.get_rect()
        bar_rect.centery = 200+(value/4)
        bar_rect.right = 400

        # Blit the bar onto the screen
        self.screen.blit(bar_surface, bar_rect)




    def refreshScreen(self, image):
        
        
        if not self.hasWindow:
            self.create_window(image.shape[:2])
            
        image=np.array(image, dtype=np.uint8)
        
        #blur
        image=cv2.GaussianBlur(image, (5, 5), 0)
        
        
        
        
        
        #originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        #flip x and y axis by transposing the image
        #image = np.transpose(image, (1, 0, 2))


        surface = pygame.surfarray.make_surface(image)

        #draw image on screen
        self.screen.blit(surface, (0, 0))

    def update(self, originalImage, objects, state):


        self.refreshScreen(originalImage)



        #create window if it does not exist using the size of the image
        

        pygame.event.pump() #update the event queue



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
            
        
        
        
        joystick = JoystickData.fromJoystick(self.js)
    
        manualSwitch=joystick.r1-joystick.l1
        if manualSwitch < -0.5:
            self.manual = True
            print("Manual")
        elif manualSwitch >0.5:
            self.manual = False
            print("Auto")
    
        if(self.manual):
            pass
            
        else:
            
    
            #Select path point to choose
            
            self.current_index = self.select_point(objects, self.current_index, self.get_key_press_from_dir(joystick.dir))
            


            #draw path points
            for key, value in objects.items():
                y, x = value
                
                x = int(((x + 1) / 2) * self.size[0])
                y = int(((y + 1) / 2) * self.size[1])

                if key == self.current_index:
                    pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 5)
                else:
                    pygame.draw.circle(self.screen, (255, 0, 0), (x, y), 5)
            

        
            
        self.drawBar(state[0])
        
        
        pygame.display.flip()
        
        return self.current_index, False, joystick, self.manual
    
            
        #time.sleep(0.1)








'''
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
'''



    

        

        




        
