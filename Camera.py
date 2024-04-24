import cv2
import numpy as np

class Camera:
    def __init__(self, camera_index=0):
        # Initialize the video capture with the given camera index
        self.cap = cv2.VideoCapture(camera_index)
        #self.cameraOpened=True
        if not self.cap.isOpened():
            #self.cameraOpened=False
            raise ValueError("Unable to open video source", camera_index)

    def get_frame(self):
        # Capture frame-by-frame
        #if not self.cameraOpened:
        #    return None
        
        ret, frame = self.cap.read()
        
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
        