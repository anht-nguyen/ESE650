from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.linalg import sqrtm
from quaternion import Quaternion
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter



def estimate_rot(data_num=1):
    '''
    roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw
    '''
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

    ## Step 0: Initialization: choose the values of the initial covariance of the state, dynamics noise and measurement noise
    n = 6
    # state covariance initialized
    cov0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # 6-dimension vector, similar dimension with process noise vector w_k =(w_{quaternion}, w_{angular velo})

    cov_k_k = np.diag(cov0) #6x6 matrix

    # initialize mu state
    mu0 = np.array([[0.5, 0., -0.6, 1, 1, 1]]).T # mean vector of orientation data and angular velo
    Q_mu = Quaternion()
    Q_mu.from_axis_angle(mu0[:3].reshape(-1))
    mu_k_k = (Q_mu.q, mu0[3], mu0[4], mu0[5]) # a 7-d vector (first 4 for quaternion, remaining 3 for angular velo)
    # Q_bar = Quaternion(scalar = mu_k_k[0][0], vec = mu_k_k[0][1:4] )

    # process noise/dynamic noise
    R = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    R = np.diag(R)
    # measurement noise:
    Q = ... 

    
    
    # T = 2

    for t in range(T-1):
        dt = ts_imu[t+1] - ts_imu[t]

        ## Step 1:  Propagate the dynamics
        # first compute the square root
        sqrt_cov = sqrtm(cov_k_k + R*dt)
        sqrt_cov_cols = np.concatenate((sqrt_cov * np.sqrt(n), sqrt_cov * (-np.sqrt(n))), axis = 1)
        Q_bar = Quaternion(scalar = mu_k_k[0][0], vec = mu_k_k[0][1:4] )
        sigma_q = generate_sigma_q(sqrt_cov_cols, Q_bar)
        # print("\nsigma_q: ", sigma_q)
        Q_bar, sigma_cov_q = propagating_q(sigma_q, Q_bar)
        sigma_mu_w, sigma_cov_w = propagating_w(mu_k_k, sqrt_cov_cols)

        # Predicted mu and covariance of state
        mu_k1_k = (Q_bar.q, sigma_mu_w[0], sigma_mu_w[1], sigma_mu_w[2])
        cov_k1_k = np.zeros((6,6))
        cov_k1_k[:3, :3] = sigma_cov_q
        cov_k1_k[3:, 3:] = sigma_cov_w
        print(mu_k1_k)


        # So far so good!

        # Step 2: Obtain observation from accel and gyro
            # step 2.1: 
        # new sigma points for udpated state distribution N(mu_k1_k, cov_k1_k)
        sqrt_cov = sqrtm(cov_k1_k + R*dt)
        sqrt_cov_cols = np.concatenate((sqrt_cov * np.sqrt(n), sqrt_cov * (-np.sqrt(n))), axis = 1)
        updated_Q_bar = Quaternion(scalar = mu_k1_k[0][0], vec = mu_k1_k[0][1:4] )
        updated_sigma_q = generate_sigma_q(sqrt_cov_cols, updated_Q_bar)

        # Calculate mean of new transformed sigma points (PC eq 3.33)
        y_hat = ...

        # Step 3:

        # Step 4: 


        # End iteration by the following
        # mu_k_k = mu_k1_k1
        # cov_k_k = cov_k1_k1














# Methods for measurement calibration
'''
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
'''

def accel_2_euler(accel):
    pitch = np.hstack([np.arcsin(accel[0,t]/ 9.81) for t in range(np.shape(accel)[1])])
    roll = np.hstack([math.atan2(accel[1,t], accel[2,t]) for t in range(np.shape(accel)[1])])
    return np.vstack((roll, pitch))

'''
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
'''

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


## UKF methods (for modularity)
def generate_sigma_q(sqrt_cov_cols, Q_bar):
    '''
    Input: 
    - mu_k|k
    - sqrt_cov_cols: computed from cov_k_k
    Output: sigma_q (list of 12 quaternions)
    '''
    n=6
    sigma_q = []

    for i in range(2*n):
        Q_sigma = Quaternion()
        Q_sigma.from_axis_angle(sqrt_cov_cols[:3, i])
        quat = Q_bar * Q_sigma
        sigma_q.append(quat.q)
    return sigma_q

def propagating_q(sigma_q, Q_bar):
    ''' Perform gradient descent
    Input: 
    - sigma_q: list of 12 arrays of size 1,4 (orientation quaternion)
    - Q_bar: [quaternion instance] - estimation of mean (q_bar)_t
    Output: 
    - Q_bar: (q_bar)_t+1 - new estimation of mean q_bar
    - sigma_cov_q: (3,3 matrix) 
    '''
    n = 6
    error_norm = []
    for iter in range(100):
        error = np.zeros((3, 2*n))
        for i in range(2*n):
            quat = Quaternion(scalar=sigma_q[i][0], vec=sigma_q[i][1:4])
            Q_error = quat * Q_bar.inv()
            Q_error.normalize()
            error[:,i] = Q_error.axis_angle()
        error_bar = np.sum(error, axis = 1)/ (2*n)
        error_norm.append(np.linalg.norm(error_bar))
        Q_error_bar = Quaternion()
        Q_error_bar.from_axis_angle(error_bar)
        Q_bar = Q_error_bar * Q_bar
    print('error_norm: ', error_norm[iter])    

    sigma_cov_q = np.zeros((3,3))
    for i in range(2*n):
        sigma_cov_q += (error[:,i] - error_bar).reshape(3,1) @ (error[:,i] - error_bar).reshape(1,3) /(2*n)

    return Q_bar, sigma_cov_q

def propagating_w(mu_k_k, sqrt_cov_cols):
    ''' 
    Inputs: 
    - mu_k|k: previous mean
    - sqrt_cov_cols: computed from cov_k_k (previous cov)
    Output: 
    - sigma_mu_w: 
    - sigma_cov_w: covariance estimate of state cov_k1_k (3x3 matrix)
    '''
    n=6
    weight = 1/(2*n)
    sigma_w = np.array([np.array(mu_k_k[1:4]).reshape(-1) + sqrt_cov_cols[3:, 0] for i in range(2*n)]).T 
    # mu
    sigma_mu_w = np.array([ np.sum(weight * (mu_k_k[j+1] + sigma_w[j, :])) for j in range(3)]) 
    # cov
    sigma_cov_w = np.zeros((3,3))
    for i in range(12):
        diff = ((np.array(mu_k_k[1:4]).reshape(-1) + sigma_w[:,i]) - sigma_mu_w).reshape(3,1)
        sigma_cov_w += weight * (diff @ diff.T)
    
    return sigma_mu_w, sigma_cov_w




estimate_rot()
