## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2




class CameraTop:
    
    
    def __init__(self, exposure=100):
        
        self.exposure = exposure
        
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
        
    
        self.device = self.pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))



        self.color_sensor = self.device.first_color_sensor()
        self.color_sensor.set_option(rs.option.exposure, self.exposure)

        self.found_rgb = False
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                self.found_rgb = True
                break
        if not self.found_rgb:
            #throw exception
            raise ConnectionError("No RGB camera found")
            

        #self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
        self.pipeline.start(self.config)


    def get_frame(self):
        
        while(1):
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            #cv2.imshow('RealSense', color_image)
            
            #make the image square
            color_image = color_image[:, 80:560]
            
            
            
            
            return color_image
        
    def __del__(self):
        self.pipeline.stop()
        
        
    def get_size(self):
        
        frame = self.get_frame()
        
        return frame.shape[:2]
     
        
        
        
if __name__ == "__main__":
    topCamera = CameraTop()
    
    while True:
        topCamera.get_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
    del topCamera
    print("Done")
    
