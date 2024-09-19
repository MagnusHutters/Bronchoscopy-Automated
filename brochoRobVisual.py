
import numpy as np
import time

class brochoRobClass(object):

    def __init__(self, jointlimits = [170,170,250], vellimits = [25,25,100]):


        time.sleep(0.5)

        self.jointvalues = []
        

        self.joint_limits = jointlimits # mm, degree, degree
        self.vel_limits = vellimits # deg/s, deg/s, mm
        #self.accelaration = accelaration # mm/s2, deg/s2, deg/s2

        self.jointBendingConvert = [850,1700] # valores pwd, representa [-170,170]
        self.jointRotationConvert = [544,2400] # valores pwd, representa [-170,170]
        self.jointTranslationConvert = [0,3600*64] # valores step, 250 mm //AQUI 64, o valor do microstepper que estamos a usar agora

        self.jointvelControl = 0.25

    # Visual servoing control of the robot based on the desired image size
    def visualservoingcontrol(self, imgTarget, m_transStep, imgSize = [400, 400], m_currJoints = [0, 0, 0]):
        
        m_middleImage = np.divide(imgSize,2)
        m_jointsVelRel = [0, 0, 0]


        #Working (arc length from current position to the desired one) Normalize between [0 : 1], 1 corrspondes to a angle diff of 90 in the limit of the image (pi/2 * raio(100 pixels))
        m_jointsVelRel[1] = (np.arcsin((imgTarget[0] - m_middleImage[0])/(np.linalg.norm(imgTarget - m_middleImage))) * np.linalg.norm(imgTarget - m_middleImage)) \
        / (np.pi / 2 * 200) # calcula o perimetro que falta percorrer para alinhar com o eixo. quanto menor o perimetro, menor a velocidade


        # Nao esta a entrar aqui
        if((imgTarget[0] <= 0 and imgTarget[1] >= 0) or \
            (imgTarget[0] > 0 and imgTarget[1] < 0)):
            m_jointsVelRel[1] = - (m_jointsVelRel[1])

        # Sequential control, define limits where each joint will work

        if(np.linalg.norm(imgTarget - m_middleImage) < 20 or np.abs(m_jointsVelRel[1]) < 0.1):
                m_jointsVelRel[1] = 0
                m_jointsVelRel[0] = (imgTarget[1] - m_middleImage[1]) / (m_middleImage[1])
                

        #extend or retract the bronchoscope
        dist = np.linalg.norm(imgTarget - m_middleImage)
        if dist < 75: # extend
            m_jointsVelRel[2] = 1
            m_jointsVelRel[1] = 0
            m_jointsVelRel[0] = 0
        elif dist > 400: # retract
            m_jointsVelRel[2] = -1
            m_jointsVelRel[1] = 0
            m_jointsVelRel[0] = 0



        # Compute next joint values
        m_nextJoints = []
        for i in range(3):
            m_nextJoints.append(m_currJoints[i] + m_jointsVelRel[i] * self.vel_limits[i] * self.jointvelControl)

        

        # to test
        #m_jointsVelRel[2] = m_transStep * 0.005
        #m_nextJoints[2] = m_transStep * 0.005

        # Joint lmits
        '''
        # Theta 1 - the rotation of the bronchoscope
        # if the next joint passes a limit, the robot rotate around to get the desired value to the joint
        if(np.abs(m_nextJoints[0]) > np.radians(self.joint_limits[0])):
            m_nextJoints[1] = m_currJoints[1] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta

            if(m_nextJoints[0] > np.radians(self.joint_limits[0])):
                m_nextJoints[0] = np.minimum(np.maximum(m_currJoints[0] + (- np.radians(180) + (m_jointsVelRel[0] * self.vel_limits[0])), np.radians(-self.joint_limits[0])), np.radians(self.joint_limits[0]))
            else:
                m_nextJoints[0] = np.minimum(np.maximum(m_currJoints[0] + (np.radians(180) + (m_jointsVelRel[0] * self.vel_limits[0])), np.radians(-self.joint_limits[0])), np.radians(self.joint_limits[0]))

        # Check if theta 2 is ok
        else:
            # Theta 2 - rotate the theta 1 do change the direction of movement of theta 2
            if(np.abs(m_nextJoints[1]) > np.radians(self.joint_limits[1])):
                m_nextJoints[1] = m_currJoints[1] # nao rodar a junta2, apenas roda a 3 para sair do limite de junta
                if(m_nextJoints[0] >= 0):
                    m_nextJoints[0] = np.maximum(m_nextJoints[0] - np.radians(180), - np.radians(self.joint_limits[0]))
                else:
                    m_nextJoints[0] = np.minimum(m_nextJoints[0] + np.radians(180), np.radians(self.joint_limits[0]))'''
        
        ##self.setJoints(m_nextJoints)
        
        return m_jointsVelRel,m_nextJoints