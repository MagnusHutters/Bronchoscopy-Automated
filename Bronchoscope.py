


class Bronchoscope(threading.Thread):
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

    def run(self):
        self.running = True
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"Connected to serial port {self.port}")
        except serial.SerialException as e:
            print(f"Failed to connect to serial port {self.port}: {e}")
            return
        
        while self.running:
            try:
                # Process commands from the queue
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    self.handle_command(command)
                
                time.sleep(0.1)  # Adjust delay between loops as needed
            except Exception as e:
                print(f"Error in main loop: {e}")
        
        self.serial.close()
        print("Serial connection closed.")
    
    def handle_command(self, command):
        self.send_serial_command(command)
        # Wait for and process response
        try:
            response = self.serial.readline().decode().strip()
            position = int(response)
            # Update current_position for the corresponding joint
            if command == 'u' or command == 'd':
                self.current_position[0] = position
            elif command == 'l' or command == 'r':
                self.current_position[1] = position
            elif command == 'f' or command == 'b':
                self.current_position[2] = position
        except ValueError as e:
            print(f"Error parsing position response: {e}")
        except serial.SerialException as e:
            print(f"Serial error while reading response: {e}")
    
    def send_serial_command(self, command):
        try:
            self.serial.write(command.encode() + b'\n')
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
    
    def send_command(self, command):
        if isinstance(command, Input):
            command = command.toChar()  # Get command character from Input object
        self.command_queue.put(command)
    
    def get_state(self):
        return tuple(self.current_position)
    
    def stop(self):
        self.running = False