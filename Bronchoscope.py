


import serial
import threading
import queue
import time


from Input import Input

class Bronchoscope(threading.Thread):
    
    
    
    
    
    joint_limits = [170,170,250]# mm, degree, degree
    vellimits = [25,25,100]# deg/s, deg/s, mm
    
    
    
    
    
    
    jointBendingConvert = [956,1554] # valores pwd, representa [-170,170]
    jointRotationConvert = [544,2400] # valores pwd, representa [-170,170]
    jointTranslationConvert = [0,3600] # valores step, 250 mm
    
    
    jointStepLimits = [[956,1554], [544,2400], [0,3600]]
    
    bendingInitial = ((jointStepLimits[0][1]-jointStepLimits[0][0])/2)+jointStepLimits[0][0]
    rotationInitial = ((jointStepLimits[1][1]-jointStepLimits[1][0])/2)+jointStepLimits[1][0]
    translationInitial = jointStepLimits[2][0]
    
    jointStepInitial = [bendingInitial, rotationInitial, translationInitial]
    jointStepMaxChange = [64,64,64]
    
    
    axisToJoint = {
        0: "b",
        1: "r",
        2: "t"
    }
    
    
    valid_commands = ['f', 'b', 'l', 'r', 'u', 'd', 'i', 'j']
    def __init__(self, port, baudrate=9600, start_position=(0, 0, 0), min_limits=(0, 0, 0), max_limits=(100, 100, 100), increment=(1, 1, 1)):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.start_position = start_position
        self.min_limits = min_limits
        self.max_limits = max_limits
        self.increment = increment
        self.current_position = list(start_position)
        self.command_queue = queue.Queue()
        self.serial = None
        self.running = False



    def home(self):
        # Home the robot
        
        self.query_joint_positions(doForce=True)
        
                    
        
        
        
        
        
    
        

    def run(self):
        self.running = True
        self.serial = serial.Serial(self.port, self.baudrate, timeout=0.05)
        self.home()
        
        time.sleep(0.5)
        self.query_joint_positions(doForce=True)
        
        while self.running:
            try:
                # Process commands from the queue
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self.handle_command(command)
                
                time.sleep(0.01)  # Adjust delay between loops as needed
            except Exception as e:
                print(f"Error in main loop: {e}")
        
        command="i"
        self.serial.write(command.encode() + b'\n')
        
        self.serial.close()
        print("Serial connection closed.")
    
    def handle_command(self, command):
        
        #command must be of type Input
        if not isinstance(command, Input):
            raise ValueError("Command must be of type Input")
        if command.hasInput:
            axis = command.axis
            change = command.change
            
            #cap change from -max_change to max_change
            change = max(-self.jointStepMaxChange[axis], min(self.jointStepMaxChange[axis], change))
            
            #print(f"Axis: {axis}, Change: {change}")
            
            
            current_position = self.current_position[axis]
            new_position = current_position + change
            #cap new position from min to max
            new_position = max(self.jointStepLimits[axis][0], min(self.jointStepLimits[axis][1], new_position))
            
            
            #print(f"Axis: {axis}, Change: {change}, Current: {current_position}, New: {new_position}")
            
            command = self.axisToJoint[axis] + str(new_position)
            
            self.send_serial_command(command)
            
            
            # Wait for and process response
            try:
                response = self.serial.readline().decode().strip()
                
                
                #print(f"Received response: {response}")
                try:
                    position = int(response)
                except ValueError as e: #some error ocurred. flush serial buffer and return
                    print(f"Error parsing position response: {e}")
                    #flush serial buffer
                    self.serial.flushInput()
                    return
                # Update current_position for the corresponding joint
                
                #print(f"Axis: {axis}, New position: {position}")
                self.current_position[axis] = position
            except serial.SerialException as e:
                print(f"Serial error while reading response: {e}")    
            
            
            
    

        
        
    
    def send_serial_command(self, command):
        
        #flush input buffer
        self.serial.flushInput()
        
        #print(f"Sending command: {command}")
        try:
            
            self.serial.write(command.encode() + b'\n')
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
    
    
    def query_joint_positions(self, doForce=False):

        success = False
        while not success:
            #flush input buffer
            self.serial.flushInput()
            
            self.send_serial_command('j')  # Send 'j' command to query positions
            try:
                responses = []
                for _ in range(3):
                    response = self.serial.readline().decode().strip()
                    position = int(response)
                    responses.append(position)
                
                self.current_position = responses
                print(f"Initialized current positions: {self.current_position}")
                success = True
            
            except ValueError as e:
                print(f"Failed to get joint positions")
                if doForce:
                    print("Retrying...")
                success = False
            except serial.SerialException as e:
                print(f"Serial error while reading response: {e}")
                if doForce:
                    print("Retrying...")
            if not doForce:
                break
            else:
                
                time.sleep(0.1)
    
    def send(self, command):
        #print(command)
        if not isinstance(command, Input):
            raise ValueError("Command must be of type Input")
        
        
        if command.hasInput:
            self.command_queue.put(command)
    
    
    
    
    def getJoints(self):
        #sself.ser.write(b'j\n')
        self.jointvalues = []
        #data = self.ser.readline().decode('utf-8').strip()
        #self.jointvalues.append(int(data))
        self.jointvalues.append(((int(self.current_position[0]) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]) #((value-min) / (max-min) * jointrange) - halfjointrange
        #data = self.ser.readline().decode('utf-8').strip()
        #self.jointvalues.append(int(data))
        self.jointvalues.append(((int(self.current_position[1]) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]) #((value-min) / (max-min) * jointrange) - halfjointrange
        #data = self.ser.readline().decode('utf-8').strip()
        #self.jointvalues.append(int(data))
        self.jointvalues.append(((int(self.current_position[2]) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2]) #((value-min) / (max-min) * jointrange) - halfjointrange

        return self.jointvalues.copy()
    
    def get_state(self):
        return tuple(self.current_position)
    def getDict(self):
        joints = self.getJoints()
        
        return {
            "bend_steps": self.current_position[0],
            "rotation_steps": self.current_position[1],
            "extension_steps": self.current_position[2],
            
            "bendReal_deg": joints[0],
            "rotationReal_deg": joints[1],
            "extensionReal_mm": joints[2]
        }
    
    def stop(self):
        self.running = False
        
    def close(self):
        if self.is_alive():
            self.stop()
        
        
            self.join()