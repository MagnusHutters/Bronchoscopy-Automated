
import time
import numpy as np
from Input import *
import traceback


from Timer import Timer

from Interface import *
class Controller:
    def __init__(self):
        self.closed=False
        self.interface = Interface()
        
        
    def __del__(self): 
        self.close()  
        

    def doStep(self, image):

        #return Input(0,0,0)
        return Input(0,0,0), 0, 0

    def update(self):

        
        
        Timer.point("updateStart")
        image, topImage = self.interface.getImage()
        Timer.point("gotImage")


        #input = self.doStep(image)
        input, doStart, doStop = self.doStep(image, topImage)
        if self.closed:
            return

        #self.interface.updateInput(input)
        #print(f"Input: {input.toChar()}")
        
        
        self.interface.updateInput(input, doStart, doStop)

    def run(self, interval =0.05):



        self.interval = interval

        start_time = time.time()
        end_time = time.time()
        
        try:    
            while not self.closed:
                
                
                #start_time = time.time()  # Record start time
                self.update()  # Execute the task
                
                
                
                end_time = time.time()  # Record end time
                elapsed_time = end_time - start_time  # Calculate elapsed time


                #print(elapsed_time)
                sleep_time = self.interval - elapsed_time  # Calculate remaining time to sleep
                start_time = end_time + sleep_time  # Update start time for next iteration
                
                Timer.point("BeforeSleep")
                if sleep_time > 0:
                    time.sleep(sleep_time)  # Sleep for the remaining time of the interval
                else:
                    # Processing took longer than the interval
                    #print("Warning: Processing time exceeded the interval.")
                    pass
                
                Timer.point("End")
                report, num=Timer.reset()
                Timer.point("Start")
                #print(report)
                #print(num)


                #os.system('cls' if os.name == 'nt' else 'clear')

                # Print the entire frame at once
                #print(report, end='', flush=True)


        except Exception as e:
            print(f"Error in main loop: {e}")
            self.close()
            #traceback.print_exc()
            
            raise e
        
        
        
                
                
    def close(self):
        
        if(not self.closed):
            #check if the interface exists
            if(hasattr(self, 'interface')):
                self.interface.close()
            self.closed=True
        




