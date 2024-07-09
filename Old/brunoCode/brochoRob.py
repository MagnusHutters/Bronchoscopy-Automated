
import pygame
#import pygame.camera
#from pygame.locals import *
import time
import numpy as np
import cv2
import serial

# values
m_rot = 50
m_bend = 50
m_move = 50

# serial
ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(0.5)
# initialization
ser.write(b'i')
time.sleep(0.5)


# joystick 
pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count()==0:
    print("No joystick!")
    quit()
#else:
#    print("joystick detect!")

js = pygame.joystick.Joystick(0)
js.init()

print("System Ready!")

''' start control loop '''
try:
    while True:
        pygame.event.pump()        
        num_axes = js.get_numaxes()

        rot = js.get_axis(0)
        bend = js.get_axis(1)
        
        if (rot > 0.5):
            
            if (m_rot >= 2):
                ser.write(b'r') # -2 on arduino side
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_rot = int(data)
                        break
                print("rotate clockwise")

            '''
                m_rot -= 2
                cmd = 'r' + str(m_rot)
                ser.write(cmd.encode())
                time.sleep(0.2)
                print("rotate clockwise")
            '''
            
        if (rot < -0.5):            
            if (m_rot < 180):
                ser.write(b'l') # 2 on arduino side
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_rot = int(data)
                        break
                print("rotate counter-clockwise")
                '''        
                m_rot += 2
                cmd = 'r' + str(m_rot)
                ser.write(cmd.encode())
                time.sleep(0.2)
                print("rotate counter-clockwise")
                '''
            
        if (bend < -0.5):
            if (m_bend >= 42):
                ser.write(b'd')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_bend = int(data)
                        break
                print("bend down")
                '''
                m_bend -= 2
                cmd = 'b' + str(m_bend)
                ser.write(cmd.encode())
                time.sleep(0.2)
                print("bend down")
                '''
        if (bend > 0.5):
            if (m_bend < 96):
                ser.write(b'u')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_bend = int(data)
                        break
                print("bend up")
                '''
                m_bend += 2
                cmd = 'b' + str(m_bend)
                ser.write(cmd.encode())
                time.sleep(0.2)
                print("bend up")
                '''
        
        # check joystick button -> forward or backward
        num_buttons = js.get_numbuttons()
        movef = js.get_button(0)
        moveb = js.get_button(2)
        if (movef > 0.5):
            # servo
            if (m_move < 3600):
                ser.write(b'f')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_move = int(data)
                        #time.sleep(0.01)
                        break
                print("Move forward")
                '''
                m_move += 50
                cmd = 'm' + str(m_move)
                ser.write(cmd.encode())
                time.sleep(0.2)
                #pi.i2c_write_device(i2c_dac, [64, (m_move >> 4) & 0xFF, (m_move & 15) << 4])
                print("Move forward")
                '''
            
            # stepper move
            #pi.write(fwd_dir, 1)
            #pi.write(fwd_stp, 1)
            #time.sleep(0.06)
            #pi.write(fwd_stp, 0)
            #time.sleep(0.5)
            '''
            for tmp in range(steps_fwd):
                pi.write(fwd_stp, 0)
                time.sleep(0.001)
                pi.write(fwd_stp, 1)
                time.sleep(0.001)
            '''
        if (moveb > 0.5):
            # stepper move
            #pi.write(fwd_dir, 0)
            #pi.write(fwd_stp, 1)
            #time.sleep(0.06)
            #pi.write(fwd_stp, 0)
            #time.sleep(0.5)
            '''
            for tmp in range(steps_fwd):
                pi.write(fwd_stp, 0)
                time.sleep(0.001)
                pi.write(fwd_stp, 1)
                time.sleep(0.001)
            '''
            
            # servo
            if (m_move >= 50):
                ser.write(b'b')
                time.sleep(0.015)
                while True:
                    if ser.in_waiting > 0:
                        data = ser.readline().decode('utf-8').strip()
                        m_move = int(data)
                        #time.sleep(0.01)
                        break
                print("Move backward")
                '''
                m_move -= 50
                cmd = 'm' + str(m_move)
                ser.write(cmd.encode())
                time.sleep(0.2)
                #pi.i2c_write_device(i2c_dac, [64, (m_move >> 4) & 0xFF, (m_move & 15) << 4])
                print("Move backward")
                '''
        time.sleep(0.01)
        
except KeyboardInterrupt:
    print("Closing...")
    ser.close()
    #set_servo_pulse(bend_pin, 0)
    #set_servo_pulse(rot_pin, 0)
    #pi.i2c_close(i2c_dac)
    #pi.stop()