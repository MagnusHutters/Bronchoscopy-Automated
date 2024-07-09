import subprocess
import cv2

# Replace these with your actual idVendor and idProduct values
TARGET_CAMERAS = [
    {"idVendor": "8086", "idProduct": "0b5c"},
    # Add more if you have more cameras to filter
]

def get_video_devices():
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE, text=True)
    output = result.stdout
    devices = {}
    lines = output.split('\n')
    current_device = None

    for line in lines:
        if line.strip() == '':
            continue
        if '\t' not in line:
            current_device = line.split(' (')[0]
            devices[current_device] = []
        else:
            device_path = line.strip()
            devices[current_device].append(device_path)

    return devices

def get_device_info(device):
    result = subprocess.run(['udevadm', 'info', '--name', device, '--query', 'all'], stdout=subprocess.PIPE, text=True)
    return result.stdout

def matches_target_camera(device_info):
    
    
    for target in TARGET_CAMERAS:
        if f'ID_VENDOR_ID={target["idVendor"]}' in device_info and f'ID_MODEL_ID={target["idProduct"]}' in device_info:
            
            print("=============================================================================")
            print(f"Found target camera: {target}")
            
            print(device_info)
            
            print("=============================================================================")
            print("")
            
            
            
            return True
    return False

def find_target_cameras(devices):
    target_cameras = []
    for device_name, paths in devices.items():
        for path in paths:
            device_info = get_device_info(path)
            
            
            if matches_target_camera(device_info):
                #print(device_info)
                target_cameras.append(path)
    return target_cameras

devices = get_video_devices()
target_camera_paths = find_target_cameras(devices)


def get_device_info(device):
    result = subprocess.run(['v4l2-ctl', '--device', device, '--all'], stdout=subprocess.PIPE, text=True)
    return result.stdout





if target_camera_paths:
    print("=============================================================================")
    print("=============================================================================")
    print("=============================================================================")
    print("=============================================================================")
    print("=============================================================================")
    for path in target_camera_paths:
        info = get_device_info(path)
        
        print("=============================================================================")
        print(f"Device info for {path}")
        
        print(info)
        
        print("=============================================================================")
        print("")
        
        
        if "0x04200001" in info:
            print("Found camera")
            print(path)

    
    
    
    
    print(f"Found target cameras at: {', '.join(target_camera_paths)}")
    
    
    
    exit()
    caps = [cv2.VideoCapture(path) for path in target_camera_paths]




    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Could not open camera at {target_camera_paths[i]}.")
            exit()

    while True:
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from {target_camera_paths[i]}")
                continue

            cv2.imshow(f'Stream {i}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
else:
    print("No target cameras found.")