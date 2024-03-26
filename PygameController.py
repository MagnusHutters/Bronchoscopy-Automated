

import pygame
import pygame.camera
from pygame.locals import *

import time
from Controller import*




class PygameController(Controller):
    
    def __init__(self):
        
        
        super().__init__()
        
        pygame.init()

        pygame.joystick.init()
        
        if pygame.joystick.get_count()==0:
            print("No joystick!")
            quit()
            
            
        self.js = pygame.joystick.Joystick(0)


        self.size = self.interface.camera.get_size()

        self.screen = pygame.display.set_mode(self.size)

        self.js.init()
        
        
        
        
    def drawBar(self, value):
        print(f"Drawing bar with value: {value}")
            
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
        
    def doStep(self, image):
        pygame.event.pump()        
        num_axes = self.js.get_numaxes()
        print(num_axes)
        #time.sleep(1)
        #rot = self.js.get_axis(2)
        #bend = self.js.get_axis(3)
        move=-self.js.get_axis(1)
        
        print(self.size)
        
        
        #num_buttons = self.js.get_numbuttons()
        #buttons=[self.js.get_button(i) for i in range(num_buttons)]
        #print(buttons)
        
        rot     =-(self.js.get_button(3)-self.js.get_button(1))
        bend    =(self.js.get_button(2)-self.js.get_button(0))
        #movef = self.js.get_button(0)
        #moveb = self.js.get_button(2)
        
        #toMove=moveb-movef
        
        print(f"rot: {rot}, bend: {bend}, move: {move}")
        
        
        
        currentBend = self.interface.broncho.getBendRel()
        
        
        
        
        input=Input(rot,bend,move)
        
        #if (bend < -0.5):
        #    if (m_bend > 800):
        #        m_bend -= 10
        #        set_servo_pulse(bend_pin, m_bend)
        #        print("bend up")
        #if (bend > 0.5):
        #    if (m_bend < 1600):
        #        m_bend += 10
        #        set_servo_pulse(bend_pin, m_bend)
        #        time.sleep(0.5)
        #        print("bend down")
        
        # check joystick button -> forward or backward
        
        
        
        
        #frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(image)
        
        self.screen.blit(frame, (0,0))
        
        self.drawBar(currentBend)
        
        
        
        # Display the resulting frame
        #cv2.imshow('Frame', frame)
        
        print("Showing frame")
        #broncho.rotate(rot)
        
        #broncho.changeBend(bend)
        
        #broncho.move(toMove)   
        
         
        pygame.display.flip()  
        
        
        return input  
    
    def close(self):
        #pygame shutdown
        pygame.quit()
        print("Pygame closed")
        
        super().close()