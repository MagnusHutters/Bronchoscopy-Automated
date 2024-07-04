
import time
import numpy as np
from Input import *
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

        
        image = self.interface.getImage()



        #input = self.doStep(image)
        input, doStart, doStop = self.doStep(image)

        #self.interface.updateInput(input)
        self.interface.updateInput(input, doStart, doStop)

    def run(self, interval =0.1):

        self.interval = interval
        
        while True:
            start_time = time.time()  # Record start time
            self.update()  # Execute the task
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            #print(elapsed_time)
            sleep_time = self.interval - elapsed_time  # Calculate remaining time to sleep
            if sleep_time > 0:
                time.sleep(sleep_time)  # Sleep for the remaining time of the interval
            else:
                # Processing took longer than the interval
                #print("Warning: Processing time exceeded the interval.")
                pass
                
                
    def close(self):
        #if self.closed exists
        try:
            if(not self.closed):
                self.interface.close()
                self.closed=True
        except Exception as e:
            print("Error closing controller: ", e)




