

import pygame
import pigpio
import time
import threading
import subprocess


from State import State

def linear_transform(value, old_min, old_max, new_min=-1, new_max=1):
    """
    Linearly transforms a value from one range to another.

    Parameters:
        value (float): The original value to transform.
        old_min (float): The minimum value of the original range.
        old_max (float): The maximum value of the original range.
        new_min (float): The minimum value of the new range.
        new_max (float): The maximum value of the new range.

    Returns:
        float: The transformed value in the new range.
    """
    return new_min + ((value - old_min) / (old_max - old_min)) * (new_max - new_min)


class Bronchoscope:
    
    def __init__(self):
        self._try_start_pigpiod()
        self.pi = pigpio.pi()
        
        
        
        self.pi.set_mode(21, pigpio.OUTPUT)
        self.pi.write(21, 1)
        
        self.bend_pin = 13
        self.rot_pin = 12

        # stepper for fwd
        self.fwd_dir = 23
        self.fwd_stp = 24
        self.pi.set_mode(self.fwd_dir, pigpio.OUTPUT)
        self.pi.set_mode(self.fwd_stp, pigpio.OUTPUT)
        
        
        #self.pi.set_PWM_frequency(self.fwd_stp, 1000)
        
        self.stepper_in1 = 4
        self.stepper_in2 = 17
        self.stepper_in3 = 27
        self.stepper_in4 = 22
        
        
        self.m_rot = 600
        
        
        
        
        
        self.stepOffset=1500
        self.minRotation=-900
        self.maxRotation=900
        
        self.linearExtendOffset=200
        
        self.linearExtendMultiplier=1
        self.linearMinExtend=200
        self.linearMaxExtend=3900
        
        self.minExtend=0
        self.maxExtend=6000
        self.currentExtend=0
        self.targetExtend=0
        
        self.toMove=0
        
        
        self.historyLength=50
        self.rotationHistory=[]
        self.extendHistory=[]
        
        #fill with zeros
        for i in range(self.historyLength):
            self.rotationHistory.append(0)
            self.extendHistory.append(0)
            
        
        
        
        
        
        self.m_bend = 1200
        self.m_move = 3000
        self.m_rot2=0
        self.current_rot_step=0
        self.target_rot_step=0
        self.relative_current_rot_step=0
        self.relative_target_rot_step=0


        self.seq_cw = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
            ]

        self.seq_ccw = [
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0]
            ]

        for pin in [self.stepper_in1, self.stepper_in2, self.stepper_in3, self.stepper_in4 ]:
            self.pi.set_mode(pin, pigpio.OUTPUT)
            
            
        
        self.MCP_ADDR = 0x60
        self.i2c_dac = self.pi.i2c_open(1, self.MCP_ADDR)

        self.pi.i2c_write_device(self.i2c_dac, [64, (self.m_move >> 4) & 0xFF, (self.m_move & 15) << 4])

        
        self.doRunThread=True
        self.rotationThread=threading.Thread(target=self.rotaionLoop)
        self.rotationThread.start()
        self.extensionThread=threading.Thread(target=self.extensionLoop)
        self.extensionThread.start()
        
        
        #self.target_rot_step=0
        #self.current_rot_step=0
        
        #time.sleep(10)
        #self.target_rot_step=0
        #self.current_rot_step=-900
        
        
        
        
        self.steps_fwd = 64*4
        
        
        
    def set_step(self, w):
        self.pi.write(self.stepper_in1, w[0])
        self.pi.write(self.stepper_in2, w[1])
        self.pi.write(self.stepper_in3, w[2])
        self.pi.write(self.stepper_in4, w[3])
        
        
        
    
    def rot_stepper(self, steps, delay):
        for _ in range(abs(steps)):
            if steps > 0:
                for step in self.seq_cw:
                    self.set_step(step)
                    time.sleep(delay)
            else:
                for step in self.seq_ccw:
                    self.set_step(step)
                    time.sleep(delay)





    def stepsToDegrees(self,step):
        
        angle = (step*360)/4800
        
        return angle

    def degreesToSteps(self, angle):
        
        
        step=int((angle*4800)/360)
        
        return step



    def getBendRel(self):
        return self.m_bend-1200

    def move(self, toMove):
        
        
        self.targetExtend=self.currentExtend+(toMove*10)
        self.toMove=toMove
        
        
        self.extendHistory.append(toMove)
        #if lenght is over historyLenght remove first element
        if len(self.extendHistory)>self.historyLength:
            self.extendHistory.pop(0)
        
        # servo
        
        #print(toMove)
        #print(self.currentExtend)
        
        #for tmp in range(int(self.steps_fwd * abs(toMove))):
        #    self.pi.write(self.fwd_stp, 0)
        #    time.sleep(0.001)
        #    self.pi.write(self.fwd_stp, 1)
        #    time.sleep(0.001)
        

    def rotate(self,rotation):
        
        rot=(rotation*10)
        
        
        self.m_rot+=int(rot)
        if self.m_rot<600: self.m_rot=600
        if self.m_rot>2400: self.m_rot=2400
        
        
        self.relative_target_rot_step=self.relative_target_rot_step+rot
        self.target_rot_step=self.target_rot_step+rot
        
        
        self.rotationHistory.append(rotation)
        #if lenght is over historyLenght remove first element
        if len(self.rotationHistory)>self.historyLength:
            self.rotationHistory.pop(0)
        
        #self.set_servo_pulse(self.rot_pin, self.m_rot)
        
        #print((self.stepsToDegrees(self.current_rot_step)))
        
        
        
    def set_servo_pulse(self, pin, pulse):
        self.pi.set_servo_pulsewidth(pin, pulse)    
    
    def _try_start_pigpiod(self):
        try:
            subprocess.run(['sudo', 'pigpiod'], check=True)
        except:
            pass
        
        
    def setBend(self, bend):
        
        if bend > 1600: bend = 1600
        if bend < 800: bend=800
        #print(bend)
        
        self.set_servo_pulse(self.bend_pin, bend)
    
    def changeBend(self, bendChange):
        
        
        self.m_bend+=bendChange*3
        if self.m_bend > 1600: self.m_bend = 1600
        if self.m_bend < 800: self.m_bend=800
        
        self.setBend(self.m_bend)
        
    
    def stop(self):
        self.doRunThread=False    
    
    
    
    
    def getState(self):
        
        normBend=linear_transform(self.m_bend, 800, 1600)
        normRot=linear_transform(self.current_rot_step, -900, 900)
        
        
        
        return (normBend,normRot)
        
        
        
        
    
    def extensionLoop(self):
        while(self.doRunThread):
            
            if (self.targetExtend > self.maxExtend): self.targetExtend=self.maxExtend
            if (self.targetExtend < self.minExtend): self.targetExtend=self.minExtend
            toMove=0
            if(self.targetExtend>self.currentExtend): toMove=1
            if(self.targetExtend<self.currentExtend): toMove=-1
            
            self.currentExtend+=toMove
            
            #print(f"{toMove}, {self.targetExtend}, {self.currentExtend}")
            
            linear_move=(self.currentExtend*self.linearExtendMultiplier*10)+self.linearExtendOffset
            #print(linear_move)
            if(linear_move>=self.linearMaxExtend): linear_move=self.linearMaxExtend
            if(linear_move<=self.linearMinExtend): linear_move=self.linearMinExtend
            
            linear_move=int(linear_move)
                
            #self.pi.i2c_write_device(self.i2c_dac, [64, (linear_move >> 4) & 0xFF, (linear_move & 15) << 4])
                
                # stepper move
                
            toMove=round(self.toMove)
                
                
            #dir=1
            #if(toMove<0): dir=0
            
            #self.pi.write(self.fwd_dir, dir)
            #print(toMove)
            #if(toMove is not 0):
            #    self.pi.set_PWM_dutycycle(self.fwd_stp, 1)
                
            
            
            dir=1
            if(toMove<0): dir=0
            sleepTime=0.010
            #print(f"toMove: {toMove}")
            self.pi.write(self.fwd_dir, dir)
            if(toMove is not 0):
                self.pi.write(self.fwd_stp, 1)
                time.sleep(sleepTime)
                self.pi.write(self.fwd_stp, 0)
                
            else:
                time.sleep(sleepTime)
                
            time.sleep(0.04)
                
            
        
    
    def rotaionLoop(self):
        
        while(self.doRunThread):
            
            
            if(self.target_rot_step>self.maxRotation): self.target_rot_step=self.maxRotation
            if(self.target_rot_step<self.minRotation): self.target_rot_step=self.minRotation
            
            toMove=0
            if(self.target_rot_step<self.current_rot_step): toMove=-1
            if(self.target_rot_step>self.current_rot_step): toMove=1
            
            
            
            
            relativeToMove=0
            if(self.relative_target_rot_step<self.relative_current_rot_step): relativeToMove=-1
            if(self.relative_target_rot_step>self.relative_current_rot_step): relativeToMove=1
            
            
            self.current_rot_step+=toMove
            self.relative_current_rot_step+=relativeToMove
            
            step = self.seq_ccw[(self.relative_current_rot_step)%4]
            
            
            
            self.set_step(step)
            if(toMove!=0):
                pass
                self.set_servo_pulse(self.rot_pin, self.current_rot_step+self.stepOffset)
            
            #print(self.current_rot_step)
            
            
            
            time.sleep(0.005)
            
            
    def close(self):
        self.stop()
        self.rotationThread.join()
        self.extensionThread.join()
        
        #close pigpio devices
        
        self.pi.i2c_close(self.i2c_dac)
        self.pi.set_mode(self.fwd_dir, pigpio.INPUT)
        self.pi.set_mode(self.fwd_stp, pigpio.INPUT)
        self.pi.set_mode(self.stepper_in1, pigpio.INPUT)
        self.pi.set_mode(self.stepper_in2, pigpio.INPUT)
        self.pi.set_mode(self.stepper_in3, pigpio.INPUT)
        self.pi.set_mode(self.stepper_in4, pigpio.INPUT)
        self.pi.stop()
