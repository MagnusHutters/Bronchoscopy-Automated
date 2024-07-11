import cv2
from pygrabber.dshow_graph import FilterGraph

# Function to list all connected cameras
def list_cameras():
    graph = FilterGraph()
    devices = graph.get_input_devices()
    for i, device in enumerate(devices):
        print(f"Camera {i}: {device}")

# Function to open a camera by name
def open_camera_by_name(camera_name):
    graph = FilterGraph()
    devices = graph.get_input_devices()
    
    for i, device in enumerate(devices):
        if camera_name in device:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
    return None

# List all connected cameras
list_cameras()