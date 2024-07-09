import cv2
import numpy as np
import multiprocessing
import time

class Camera:
    def __init__(self, camera):
        self.camera = camera
        print(f"Camera: {camera}")
        self.frame = None
        self.frame_lock = multiprocessing.Lock()
        self.stop_event = multiprocessing.Event()
        self.process = multiprocessing.Process(target=self._capture_frames)
        self.process.start()

    def _capture_frames(self):
        cap = cv2.VideoCapture(self.camera)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        if not cap.isOpened():
            raise ValueError("Unable to open video source")

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.frame = frame
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage

        cap.release()

    def get_frame(self):
        frame=None
        while frame is None:
            with self.frame_lock:
                if self.frame is not None:
                    frame = self.frame.copy()
            
            print("frame is none")
        return frame

    def get_size(self):
        frame = self.get_frame()
        if frame is None:
            return None
        return frame.shape[:2]

    def release(self):
        self.stop_event.set()
        self.process.join()

    def __del__(self):
        self.release()