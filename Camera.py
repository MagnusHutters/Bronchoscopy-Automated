import cv2
import numpy as np

class Camera:
    def __init__(self, camera):
        
        self.camera = camera
        print(f"Camera: {camera}")
        # Initialize the video capture with the given camera index
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, -40) 
        #self.cap.set(cv2.CAP_PROP_EXPOSURE, 0.25)
        
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        #self.cameraOpened=True
        if not self.cap.isOpened():
            #self.cameraOpened=False
            raise ValueError("Unable to open video source")
            

    def get_frame(self):
        if(self.cap is None): #if cap is none return None
            return None
        # Capture frame-by-frame
        #if not self.cameraOpened:
        #    return None
        
        ret, frame = self.cap.read()
        
        
        #darken the image
        
        
        #display the captured image
        #cv2.imshow(str(self.camera), frame)
        #cv2.waitKey(1)
        
        
        #frame = np.rot90(frame)
        if not ret:
            raise RuntimeError("Failed to capture image")
        return frame

    def get_size(self):
        
        frame = self.get_frame()
        
        return frame.shape[:2]
     
    def __del__(self):
        self.release()
           

    def release(self):
        # Release the capture when everything is done
        self.cap.release()
        