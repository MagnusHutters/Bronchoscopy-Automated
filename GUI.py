



import pygame
import pygame.camera
from pygame.locals import *


import numpy as np
import cv2
from Timer import Timer



from DataHandling.Episode import EpisodeManager, Episode

import time



import warnings


from branchModelTracker import BranchModelTracker



import time

#from Training.PathTrackerInterfaceCV import PathTrackerInterface
#from Training.BasicPaths import load_images_single_episode, load_images



class JoystickData:
    def __init__(self, forwards, rotate,bend,l1,r1,l2,r2, start, select):
        self.forwards=forwards #from -1 to 1
        self.rotation=rotate #from -1 to 1
        self.bend=bend #from -1 to 1
        self.l1=l1
        self.r1=r1
        self.l2=l2
        self.r2=r2
        self.dir=[self.rotation,self.bend]
        self.start=start
        self.select=select
        
    #from joystick
    @staticmethod
    def fromJoystick(js):
        if js is None:
            return JoystickData(0,0,0,0,0,0,0,0,0)
        
        
        num_axes = js.get_numaxes()
        axes = [js.get_axis(i) for i in range(num_axes)]
        
        
        num_buttons = js.get_numbuttons()
        buttons=[js.get_button(i) for i in range(num_buttons)]    
        
        forwards=buttons[2]-buttons[0]
        
        hat=js.get_hat(0)
        rotate=axes[0]+hat[0]
        bend = -axes[1]+hat[1]
        
        
        
        #clip rotate and bend to [-1,1]
        rotate = np.clip(rotate, -1, 1)
        bend = np.clip(bend, -1, 1)
        
        l1=buttons[4]
        r1=buttons[5]
        l2=buttons[6]
        r2=buttons[7]
        
        start = buttons[9]
        select = buttons[8]
        
        #print(f"Joystick: {start}, {select}")
        
        
        
        
        return JoystickData(forwards, rotate,bend,l1,r1,l2,r2, start, select)

class GUI:
    def __init__(self, size=None):
        
        self.hasWindow = False
        
        size = (1600,800)
        
        self.create_window(size)


        self.manual=True
        self.mode=0 #0=manual, 1=visual servoing, 2=behavioural cloning
        
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
            return 0
        if dir[0] < -0.5:
            return 2
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
            0: np.array([1, 0]),    # Right
            1: np.array([0, -1]),    # Up
            2: np.array([-1, 0]),   # Left
            3: np.array([0, 1])    # Down
        }
        orthogonal = {
            0: np.array([0, 1]),    # Lateral for Up/Down
            1: np.array([1, 0]),    # Lateral for Right/Left
            2: np.array([0, 1]),    # Lateral for Up/Down
            3: np.array([1, 0])     # Lateral for Right/Left
        }
        directionNames = {
            0: "Right",
            1: "Up",
            2: "Left",
            3: "Down"
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
        #print(f"Drawing bar with value: {value}")
            
        
        
        oldDomain = [-180, 180]
        
        newDomain = [-400, 400]
        
        value = (value - oldDomain[0]) * (newDomain[1] - newDomain[0]) / (oldDomain[1] - oldDomain[0]) + newDomain[0]
            
            
            
            
            
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
        bar_rect.centery = 400+(value/4)
        bar_rect.right = 800

        # Blit the bar onto the screen
        self.screen.blit(bar_surface, bar_rect)




    def refreshScreen(self, image, topImage=None):
        
        
        #if not self.hasWindow:
            #self.create_window(image.shape[:2])
            
        #reset screen
        self.screen.fill((0, 0, 0))
            
            
        image=np.array(image, dtype=np.uint8)
        #resize image to 800x800
        image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_AREA)
        
        
        #blur
        
        #correct colors
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (1, 0, 2))
        
        #image=cv2.GaussianBlur(image, (5, 5), 0)
        
        
        
        
        
        #originalImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2RGB)
        #flip x and y axis by transposing the image
        #image = np.transpose(image, (1, 0, 2))


        surface = pygame.surfarray.make_surface(image)

        #draw image on screen
        self.screen.blit(surface, (0, 0))
        
        
        
        
        topImage=np.array(topImage, dtype=np.uint8)
        topImage = cv2.resize(topImage, (800, 800), interpolation=cv2.INTER_AREA)
        
        topImage = cv2.cvtColor(topImage, cv2.COLOR_BGR2RGB)
        topImage = np.transpose(topImage, (1, 0, 2))
        
        
        topSurface = pygame.surfarray.make_surface(topImage)
        self.screen.blit(topSurface, (800, 0))



    def doHandleEvents(self):
        pygame.event.pump() #update the event queue

        doQuit = False

        selectEvent = -1

        for event in pygame.event.get():
            if event.type == QUIT:
                
                #exit the program
                doQuit = True
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
                if event.key == K_ESCAPE:
                    doQuit = True
                    
                    
        #joystick = None      
        joystick = JoystickData.fromJoystick(self.js)
        
        
        if joystick.l1:
            self.mode = 0 #manual
        elif joystick.r1:
            self.mode = 1 #visual servoing
        elif joystick.r2:
            self.mode = 2 #behavioural cloning
    
        
        
        return doQuit, selectEvent, joystick, self.mode


    def drawRecording(self, recording, currentFrame):
        if recording:
            pygame.draw.circle(self.screen, (255,0,0), (12,12), 6)
            
            #display current frame number on screen
            
            text = self.font.render(f'Frame: {currentFrame}', True, (255, 255, 255))
            self.screen.blit(text, (24,6))

    def update(self, originalImage, objects, state, recording=False, currentFrame=0, topImage=None):


        objects, detections = objects



        self.refreshScreen(originalImage, topImage)
        Timer.point("screenRefreshed")


        #create window if it does not exist using the size of the image
        

        doQuit, selectEvent, joystick, mode = self.doHandleEvents()
        #doQuit=False
              
        Timer.point("eventsHandled")
        
        if(mode==0):
            pass
            
        else:
            
    
            #Select path point to choose

            #print(f"Joystick: {joystick.dir}")
            
            self.current_index = self.select_point(objects, self.current_index, self.get_key_press_from_dir(joystick.dir))

            #draw path points
            for key, value in objects.items():
                x, y = value
                x=x
                y=y
                
                color = (255, 0, 0)
                if key == self.current_index:
                    color = (0, 255, 0)
                    #pygame.draw.circle(self.screen, (0, 255, 0), (x, y), 5)

                pygame.draw.circle(self.screen, color, (x*2, y*2), 5)


                if not detections[key].inView:


                    pX = detections[key].polygon.centroid.x
                    pY = detections[key].polygon.centroid.y

                    #line from point to centroid
                    pygame.draw.line(self.screen, color, (x*2, y*2), (pX*2, pY*2), 2)
                    #draw text with distance at point

                    distance = np.sqrt((x-pX)**2 + (y-pY)**2)
                    text = self.font.render(f'{distance:.2f}', True, color)
                    self.screen.blit(text, (x*2+5, y*2+5))


                polygon = detections[key].polygon #shapely polygon
                polygon = np.array(polygon.exterior.coords)

                for i in range(len(polygon)):
                    x1, y1 = polygon[i]
                    x2, y2 = polygon[(i+1)%len(polygon)]
                    pygame.draw.line(self.screen, color, (x1*2, y1*2), (x2*2, y2*2), 2)

            
        self.drawBar(state["bendReal_deg"])
        Timer.point("drawnBar")
        
        
        self.drawRecording(recording, currentFrame)
        Timer.point("drawnRecording")
        
        
        pygame.display.flip()
        
        Timer.point("displayed")
        
        
        if doQuit:
            pygame.quit()
        
        return self.current_index, doQuit, joystick, self.mode
    
            
        #time.sleep(0.1)









def main():

    episodeManager = EpisodeManager(mode = "Labelling", saveLocation="DatabaseLabelled/", loadLocation="DatabaseLabelled/")

    episodeManager.currentIndex = 20
    episodeManager.nextEpisode()
    #episodeManager.nextEpisode()

    episode = episodeManager.getCurrentEpisode()


    branchModelTracker = BranchModelTracker("BronchoYolo/yolov5/runs/train/branchTraining3/weights/best.pt")
    
    gui = GUI()

    for index in range(len(episode)):
        

        if index < 110:
            continue

        frame = episode[index]




        #print(f"Processing frame {index}")
        #print(frame.image.shape)
        #findBranches(frame.image, doDraw=True)

        #watershed(frame.image)
        #watershed(frame.image)



        timeStamp1 = time.time()
        #contours = thresholdTree(frame.image, 1)
        timeStamp2 = time.time()

        #time taken in seconds
        #print(f"Time taken: {timeStamp2 - timeStamp1}")

        drawImage = frame.image.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            detections = branchModelTracker.predict(frame.image)

            gui.update(drawImage, detections, {"bendReal_deg": 0}, recording=False, currentFrame=index, topImage=frame.topImage)
        
        cv2.imshow("Threshold Contours", drawImage)

        #create_and_display_contour_tree(frame.image)
        
        key = cv2.waitKey(1)
        # If the user presses 'q', exit the loop
        if key == ord('q'):
            break
        # if esc is pressed, exit the loop
        elif key == 27:
            break


        

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


    

        

        




        
