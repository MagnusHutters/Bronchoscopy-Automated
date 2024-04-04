import cv2
import datetime
import numpy as np
import os

class VideoRecorder:
    def __init__(self, name, folder="", fps=20.0, frame_size=(640, 480)):
        self.name = name
        self.fps = fps
        self.frame_size = frame_size
        # Add current date to filename and specify MP4 format
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        self.filename = f"{name}_{date_str}.mp4"
        if folder != "":
            #ensure folder exists
            
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.filename = f"{folder}/{self.filename}"
        
        # Define the codec and create VideoWriter object
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.filename, self.fourcc, self.fps, self.frame_size)

    def write(self, frame):
        
        self.out.write((frame*255).astype(np.uint8))

    def release(self):
        if self.out.isOpened():
            self.out.release()
    def __del__(self):
        # Make sure to release the video writer
        self.release()