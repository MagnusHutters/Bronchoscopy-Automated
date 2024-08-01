import serial
import numpy as np
import time

class brochoRobClass(object):

    def __init__(self, usbPort = 'COM4', baundrate = 115200, timestep = 0.05, jointlimits = [170,170,250], vellimits = [25,25,100]):

        self.ser = serial.Serial(usbPort, baundrate, timeout=1)
        time.sleep(0.5)

        # initialization - home position
        self.ser.write(b'i\n')
        time.sleep(0.5)

        self.jointvalues = []
        
        self.timestep = timestep

        self.joint_limits = jointlimits # mm, degree, degree
        self.vel_limits = vellimits # deg/s, deg/s, mm
        #self.accelaration = accelaration # mm/s2, deg/s2, deg/s2

        self.jointBendingConvert = [956,1554] # valores pwd, representa [-170,170]
        self.jointRotationConvert = [544,2400] # valores pwd, representa [-170,170]
        self.jointTranslationConvert = [0,3600] # valores step, 250 mm

        self.getJoints() #[Bending, Rotation, Translation]

        self.jointvelControl = 0.25

    # Read current value of joints
    def getJoints(self):
        self.ser.write(b'j\n')
        self.jointvalues = []
        while True: # READ initial robot position (we should add a time limit here)
            if self.ser.in_waiting > 0:
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]) #((value-min) / (max-min) * jointrange) - halfjointrange
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]) #((value-min) / (max-min) * jointrange) - halfjointrange
                data = self.ser.readline().decode('utf-8').strip()
                #self.jointvalues.append(int(data))
                self.jointvalues.append(((int(data) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2]) #((value-min) / (max-min) * jointrange) - halfjointrange
                break
        return self.jointvalues.copy()

    # Set the joint positions. At this stage, the arduino side is controlling the joint value incrementally (+ 1)
    def setJointsIncremental(self, destinationJoints):
        try:
            # 1 - Bending Joint
            if (destinationJoints[0] != -1) :
                if(destinationJoints[0] == 1):
                    self.ser.write(b'u\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[0] = int(data)
                            break
                    time.sleep(self.timestep)
                elif destinationJoints[0] == -1:
                    self.ser.write(b'd\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[0] = int(data)
                            break
                    time.sleep(self.timestep)

            # 2 - Rotation joint
            if(destinationJoints[1] != 0): 
                if(destinationJoints[1] == 1):
                    self.ser.write(b'l\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[1] = int(data)
                            break
                    time.sleep(self.timestep)
                elif(destinationJoints[1] == -1):
                    self.ser.write(b'r\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[1] = int(data)
                            break
                    time.sleep(self.timestep)

            #3 - Translational joint
            if(destinationJoints[2] != 0): #rotation joint
                if(destinationJoints[2] == 1):
                    self.ser.write(b'f\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[2] = int(data)
                            break
                    time.sleep(self.timestep)
                elif(destinationJoints[2] == -1):
                    self.ser.write(b'b\n') # send rotation to the servos (+2 arduino side)
                    while True:
                        if self.ser.in_waiting > 0:
                            data = self.ser.readline().decode('utf-8').strip()
                            self.jointvalues[2] = int(data)
                            break
                    time.sleep(self.timestep)
            return True
        except Exception as e:
            raise e

    # Set the joint positions, defining the desired position
    def setJoints(self, destinationJoints):
        try:
            # 1 - Bending Joint
            if(destinationJoints[0]  != self.jointvalues[0]):

                cmd = 'b' + str(int(((destinationJoints[0] + self.joint_limits[0]) / (self.joint_limits[0] * 2)) * (self.jointBendingConvert[1] - self.jointBendingConvert[0]) + self.jointBendingConvert[0])) + '\n'
                #cmd = 'b' + str(destinationJoints[0]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[0] = ((int(data) - self.jointBendingConvert[0]) / (self.jointBendingConvert[1] - self.jointBendingConvert[0])) * (self.joint_limits[0]*2) - self.joint_limits[0]
                        break
                #time.sleep(self.timestep)

            # 2 - Rotation joint
            if(destinationJoints[1]  != self.jointvalues[1]): 

                cmd = 'r' + str(int(((destinationJoints[1] + self.joint_limits[1]) / (self.joint_limits[1] * 2)) * (self.jointRotationConvert[1] - self.jointRotationConvert[0]) + self.jointRotationConvert[0])) + '\n'
                #cmd = 'r' + str(destinationJoints[1]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[1] = ((int(data) - self.jointRotationConvert[0]) / (self.jointRotationConvert[1] - self.jointRotationConvert[0])) * (self.joint_limits[1]*2) - self.joint_limits[1]
                        break
                #time.sleep(self.timestep)

            #3 - Translational joint
            if(destinationJoints[2]  != self.jointvalues[2]): #rotation joint

                cmd = 't' + str(int(((destinationJoints[2]) / (self.joint_limits[2])) * (self.jointTranslationConvert[1] - self.jointTranslationConvert[0]) + self.jointTranslationConvert[0])) + '\n'
                #cmd = 't' + str(destinationJoints[2]) + '\n'
                self.ser.write(cmd.encode()) # send desired joint value to the servo                
                while True:
                    if self.ser.in_waiting > 0:
                        data = self.ser.readline().decode('utf-8').strip()
                        self.jointvalues[2] = round(((int(data) - self.jointTranslationConvert[0]) / (self.jointTranslationConvert[1] - self.jointTranslationConvert[0])) * self.joint_limits[2],1)
                        break
                #time.sleep(self.timestep)
                
            return True
        except Exception as e:
            raise e