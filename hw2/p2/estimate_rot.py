import numpy as np
from scipy import io
from quaternion import Quaternion
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')

    accel = imu['vals'][0:3,:] # order Ax, Ay, Az. Ax and Ay direction is flipped
    gyro = imu['vals'][3:6,:] # order: Wz, Wx, Wy
    T = np.shape(imu['ts'])[1]

    # your code goes here



    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw


def rotation_angles(matrix):
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    yaw = np.arctan(r21 / r11) # alpha #z
    pitch = np.arctan(-r31 / np.sqrt(r32**2 + r33**2)) #beta #y
    roll = np.arctan(r32 / r33) #sigma #x
    # Calculate the roll, pitch, and yaw angles
    # pitch = np.arcsin(-R[2,0])
    # roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
    # yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))

    return roll,pitch,yaw


def accel_2_euler(accel):
    pitch = np.hstack([math.asin(accel[0,t]/np.sqrt(accel[0,t]**2+accel[1,t]**2+accel[2,t]**2)) for t in range(np.shape(accel)[1])])
    roll = np.hstack([math.atan2(accel[1,t], accel[2,t]) for t in range(np.shape(accel)[1])])
    return np.vstack((roll, pitch))

