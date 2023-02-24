import numpy as np
from scipy import io
from quaternion import Quaternion
import math
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

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
    ts_imu = imu['ts'].reshape(-1,)
    angle_names = ['roll', 'pitch', 'yaw']

    # your code goes here

    # Observation data processing
    accel_bias, accel_sensitivity = accel_calib_params(accel)
    accel = accel_calibration(accel, accel_bias, accel_sensitivity)

    gyro_bias, gyro_sensitivity = gyro_calib_params(gyro)
    gyro = gyro_calibration(gyro, gyro_bias, gyro_sensitivity)

    # for i in range(3):
    #     plt.plot(np.arange(T), accel[i,:], label = 'accel '+ angle_names[i])
    # plt.title('Plot IMU accel raw data')
    # plt.legend()    
    # plt.show()

    # Initialization: choose the values of the initial covariance of the state, dynamics noise and measurement noise
    


    # Step 1:  Propagate the dynamics


    # Step 2: Obtain observation from accel and gyro



    # Step 3: 



    # roll, pitch, yaw are numpy arrays of length T
    # return roll,pitch,yaw
















# def rotation_angles(matrix):
#     r11, r12, r13 = matrix[0]
#     r21, r22, r23 = matrix[1]
#     r31, r32, r33 = matrix[2]

#     yaw = np.arctan2(r21, r11) # alpha #z
#     pitch = np.arctan2(-r31, np.sqrt(r32**2 + r33**2)) #beta #y
#     roll = np.arctan2(r32, r33) #sigma #x
#     # Calculate the roll, pitch, and yaw angles
#     # pitch = np.arcsin(-R[2,0])
#     # roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
#     # yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))

#     return roll,pitch,yaw


def accel_2_euler(accel):
    pitch = np.hstack([np.arcsin(accel[0,t]/ 9.81) for t in range(np.shape(accel)[1])])
    roll = np.hstack([math.atan2(accel[1,t], accel[2,t]) for t in range(np.shape(accel)[1])])
    return np.vstack((roll, pitch))


# def R_matrix_rotation(roll, pitch, yaw):
#     Rx = np.array([[1, 0, 0],
#                 [0, np.cos(roll), -np.sin(roll)],
#                 [0, np.sin(roll), np.cos(roll)]])

#     Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
#                 [0, 1, 0],
#                 [-np.sin(pitch), 0, np.cos(pitch) ]])

#     Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
#                 [np.sin(yaw), np.cos(yaw), 0],
#                 [0, 0, 1]])
#     return Rx @ Ry @ Rz


def accel_calibration(accel, bias, sensitivity):
    ''' Input: raw accel shape(3,T), bias shape(3,), sensitivity shape(3,)
    This function return calibrated accel data (unit: m/s^2) shape(3,T)
    '''
    accel_imu = (accel - np.expand_dims(bias, axis=1)) * 3300 / (1023 * np.expand_dims(sensitivity, axis=1))
    # Do we need to flip Ax, Ay here?

    return accel_imu

def accel_calib_params(accel):
    N_calib = 200
    g = 9.81
    sensitivity_imu = 330 # found this value from IMU specification document
    sensitivity = np.ones(3) * sensitivity_imu/g
    bias_x = np.mean(accel[0,:N_calib])
    bias_y = np.mean(accel[1,:N_calib])
    bias_z = np.mean(accel[2,:N_calib]) - g * (1023 * sensitivity[2]) / 3300
    bias = np.array([bias_x, bias_y, bias_z])
    return bias, sensitivity


def gyro_calibration(gyro, bias, sensitivity):
    ''' Input: raw gyro shape(3,T), bias shape(3,), sensitivity shape(3,)
    Output: calibrated angular velo data (unit: rad/s) shape(3,T)
    '''
    gyro_imu = (gyro - np.expand_dims(bias, axis=1)) * 3300 / (1023 * np.expand_dims(sensitivity, axis=1))

    # reorder angular velocity data
    gyro_imu = np.roll(gyro_imu, -1, axis=0) 
    return gyro_imu

def gyro_calib_params(gyro):
    N_calib = 200
    bias_x = np.mean(gyro[0,:N_calib])
    bias_y = np.mean(gyro[1,:N_calib])
    bias_z = np.mean(gyro[2,:N_calib])
    bias = np.array([bias_x, bias_y, bias_z])

    sensitivity = np.ones(3) * 200 # for now I choose a 200 suggested in the manual

    return bias, sensitivity


estimate_rot()
