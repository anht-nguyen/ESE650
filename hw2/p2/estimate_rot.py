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



def estimate_rot(data_num=3):
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
    # plt.title('Plot IMU accel calibrated data')
    # plt.legend()    
    # plt.show()

    # for i in range(3):
    #     plt.plot(np.arange(T), gyro[i,:], label = 'gyro '+ angle_names[i])
    # plt.title('Plot IMU gyro calibrated data')
    # plt.legend()    
    # plt.show()    

    ## Step 0: Initialization: choose the values of the initial covariance of the state, dynamics noise and measurement noise
    n = 6
    # state covariance initialized
    cov0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) # 6-dimension vector, similar dimension with process noise vector w_k =(w_{quaternion}, w_{angular velo})

    cov_k_k = np.diag(cov0) #6x6 matrix

    # initialize mu state
    mu0 = np.array([[0.5, 0.1, 9.8, 0.1, .1, .1]]).T # mean vector of orientation data and angular velo
    Q_mu = Quaternion()
    Q_mu.from_axis_angle(mu0[:3].reshape(-1))
    mu_k_k = (Q_mu.q, mu0[3], mu0[4], mu0[5]) # a 7-d vector (first 4 for quaternion, remaining 3 for angular velo)
    # Q_bar = Quaternion(scalar = mu_k_k[0][0], vec = mu_k_k[0][1:4] )
    
    
    # process noise/dynamic noise
    R = np.array([1, 1, 1, 1, 1, 1]) 
    diag_R = np.diag(R)
    # measurement noise:
    Q = np.array([1, 1, 1, 1, 1, 1]) 
    diag_Q = np.diag(Q)
    
    roll_ukf = []
    pitch_ukf = []
    yaw_ukf = []

    mu_q_plot = []
    cov_q_plot = []
    mu_w_plot = []
    cov_w_plot = []

    # T = 1000
    for t in range(T-1):
        print('=== ', t, ' ===')
        dt = ts_imu[t+1] - ts_imu[t]

        ## Step 1:  Propagate the dynamics
        # first compute the square root
        cov_k_k = cov_k_k + diag_R*dt
        cov_k_k = np.diag(np.diag(cov_k_k))
        print(cov_k_k)
        sqrt_cov = sqrtm(cov_k_k)#.round(5))
        print(sqrt_cov)
        sqrt_cov_cols = np.concatenate((sqrt_cov * np.sqrt(n), sqrt_cov * (-np.sqrt(n))), axis = 1)
        # print(sqrt_cov_cols)
        Q_bar = Quaternion( mu_k_k[0][0],  mu_k_k[0][1:4] )
        sigma_q = generate_sigma_q(sqrt_cov_cols, Q_bar)
        # print("\nsigma_q: ", sigma_q)
        Q_bar, error = propagating_q(sigma_q, Q_bar)
        sigma_w, sigma_mu_w = propagating_w(mu_k_k, sqrt_cov_cols)
        # print(sigma_mu_w)

        # Predicted mu and covariance of state
        mu_k1_k = (Q_bar.q, sigma_mu_w[0], sigma_mu_w[1], sigma_mu_w[2])
        cov_k1_k = a_priori_cov(error, sigma_w, sigma_mu_w)
        cov_k1_k = cov_k1_k.round(5)
        # cov_k1_k = np.zeros((6,6))
        # cov_k1_k[:3, :3] = sigma_cov_q
        # cov_k1_k[3:, 3:] = sigma_cov_w
        print("Step 1 Flag")

        # Step 2: Update with measurements from accel and gyro
            # step 2.1: 
        # new sigma points for udpated state distribution N(mu_k1_k, cov_k1_k)
        cov_k1_k = np.diag(np.diag(cov_k1_k))
        print(cov_k1_k)
        sqrt_cov = sqrtm(cov_k1_k)# + diag_R*dt)
        print(sqrt_cov)
        sqrt_cov_cols = np.concatenate((sqrt_cov * np.sqrt(n), sqrt_cov * (-np.sqrt(n))), axis = 1)
        updated_Q_bar = Quaternion(scalar = mu_k1_k[0][0], vec = mu_k1_k[0][1:4] )
        updated_sigma_q = generate_sigma_q(sqrt_cov_cols, updated_Q_bar)
        updated_sigma_w, _ = propagating_w(mu_k1_k, sqrt_cov_cols)
            # here: sigma_q is list of 12 quaternions Q_sigma.q and sigma_w is array size (3,12) of angular velos part of sigma points

            # EK sec 2.3:
        # measurements models H1, H2
        y_sigma_rot, y_sigma_acc = measurement_transform(updated_sigma_q, updated_sigma_w) 
        y_sigma = [np.append(y_sigma_acc[i], y_sigma_rot[i]) for i in range(2*n)]
        
        # Calculate mean of new transformed sigma points (PC eq 3.33)
        y_hat = np.zeros(6)
        for i in range(2*n):
            y_hat += y_sigma[i] / (2*n)
        # Compute the covariances
        Cov_yy = np.zeros((6,6))
        Cov_xy = np.zeros((6,6))
        for i in range(2*n):
            Cov_yy += (y_sigma[i] - y_hat).reshape(6,1) @ (y_sigma[i] - y_hat).reshape(1,6)

            diff_w = updated_sigma_w[:,i] - mu_k1_k[1:4]
            quat = Quaternion(scalar=updated_sigma_q[i][0], vec=updated_sigma_q[i][1:4])
            diff_q = quat * Q_bar.inv() #Q_bar is quaternion component in mu_k1_k
            W_i_prime = np.append(diff_q.vec(), diff_w) #vector size(6,)
            Cov_xy += W_i_prime.reshape(6,1) @ (y_sigma[i] - y_hat).reshape(1,6) / (2*n)
        Cov_yy = diag_Q + Cov_yy / (2*n)

        # Innovation
        innov = np.append(accel[:,t], gyro[:,t]) - y_hat #vector size(6,)

        # Kalman gain
        K = Cov_xy @ np.linalg.inv(Cov_yy) #matrix size(6,6)

        # Update
        K_term = K @ innov
        quat_K = Quaternion()
        quat_K.from_axis_angle(K_term[:3])
        # K_term = (quat_K.q, K_product[3], K_product[4], K_product[5])
        Q_mu_k1 = Q_bar * quat_K 
        mu_k1_k1 = (Q_mu_k1.q, mu_k1_k[1] + K_term[3], mu_k1_k[2] + K_term[4],  mu_k1_k[3] +K_term[5])
        cov_k1_k1 = cov_k1_k - K @ Cov_yy @ K.T

        # print("the end")
        # print(mu_k1_k1)
        # print(cov_k1_k1)

        euler_angles = Q_mu_k1.euler_angles()
        roll_ukf.append(euler_angles[0])
        pitch_ukf.append(euler_angles[1])
        yaw_ukf.append(euler_angles[2])
        mu_q_plot.append(Q_mu_k1.q)
        cov_q_plot.append(np.diag(cov_k1_k1)[:3])
        mu_w_plot.append(np.array(mu_k1_k1[1:4]))
        cov_w_plot.append(np.diag(cov_k1_k1)[3:])

        # End iteration by the following update
        mu_k_k = mu_k1_k1
        cov_k_k = cov_k1_k1
    
    roll_ukf = np.array(roll_ukf)
    pitch_ukf = np.array(pitch_ukf)
    yaw_ukf = np.array(yaw_ukf) 
    # plt.subplot(3,1,1)
    # plt.plot(np.arange(T-1), roll_ukf.reshape(-1))
    # plt.title('roll')

    # plt.subplot(3,1,2)
    # plt.plot(np.arange(T-1), pitch_ukf.reshape(-1))
    # plt.title('pitch')

    # plt.subplot(3,1,3)
    # plt.plot(np.arange(T-1), yaw_ukf.reshape(-1))
    # plt.title('yaw')
    # plt.show()

    mu_q_plot = np.array(mu_q_plot)
    cov_q_plot = np.array(cov_q_plot)
    mu_w_plot = np.array(mu_w_plot)
    cov_w_plot = np.array(cov_w_plot)
    axis_names = ['x', 'y', 'z']
    
    plt.subplot(2,2,1)
    for i in range(4):
        plt.plot(np.arange(T-1), mu_q_plot[:,i], label='q'+str(i))
    plt.title('Orientation (Quaternion)')
    plt.ylabel('Mean')
    plt.legend()
    plt.subplot(2,2,2)
    for i in range(3):
        plt.plot(np.arange(T-1), cov_q_plot[:,i], label='A'+axis_names[i])
    plt.ylabel('Covariance')
    plt.legend()
    # plt.show()

    plt.subplot(2,2,3)
    for i in range(3):
        plt.plot(np.arange(T-1), mu_w_plot[:,i], label='W'+axis_names[i])
    plt.title('Angular velocities')
    plt.ylabel('Mean')
    plt.legend()
    plt.subplot(2,2,4)
    for i in range(3):
        plt.plot(np.arange(T-1), cov_w_plot[:,i], label='W'+axis_names[i])
    plt.ylabel('Covariance')
    plt.legend()
    plt.show()

    return roll_ukf, pitch_ukf, yaw_ukf



# ================================================================
# ================================================================
## UKF methods (for modularity)
def generate_sigma_q(sqrt_cov_cols, Q_bar):
    '''
    Input: 
    - mu_k|k
    - sqrt_cov_cols: computed from cov_k_k
    Output: sigma_q (list of 12 vector size(4,) - quaternion values)
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
    # print('error_norm: ', error_norm[iter])    

    # sigma_cov_q = np.zeros((3,3))
    # for i in range(2*n):
    #     # sigma_cov_q += (error[:,i] - error_bar).reshape(3,1) @ (error[:,i] - error_bar).reshape(1,3) /(2*n)
    #     sigma_cov_q += (error[:,i]).reshape(3,1) @ (error[:,i]).reshape(1,3) /(2*n)
    return Q_bar, error# sigma_cov_q, error

def propagating_w(mu_k_k, sqrt_cov_cols):
    ''' 
    Inputs: 
    - mu_k|k: previous mean
    - sqrt_cov_cols: computed from cov_k_k (previous cov)
    - R: process noise - vector size(6,)
    Output: 
    - sigma_w: array size (3,12) of angular velos part of sigma points
    - sigma_mu_w: mean estimate of angular velos in sigma points (angular velo part of mu_{k+1|k})
    - sigma_cov_w: covariance estimate of state cov_k1_k (3x3 matrix)
    '''
    n=6
    weight = 1/(2*n)
    sigma_w = np.array([np.array(mu_k_k[1:4]).reshape(-1) + sqrt_cov_cols[3:, 0] for i in range(2*n)]).T 
    # mu
    # sigma_mu_w = np.array([ np.sum(weight * (R[-j] + sigma_w[j, :])) for j in range(3)])
    sigma_mu_w = np.array([ np.sum(weight * sigma_w[j, :]) for j in range(3)]) 
    # cov
    # sigma_cov_w = np.zeros((3,3))
    # for i in range(2*n):
    #     # diff = ((R[3:] + sigma_w[:,i]) - sigma_mu_w).reshape(3,1)
    #     diff = ( sigma_w[:,i] - sigma_mu_w).reshape(3,1)
    #     sigma_cov_w += weight * (diff @ diff.T)
    return sigma_w, sigma_mu_w#, sigma_cov_w

def a_priori_cov(error, sigma_w, sigma_mu_w):
    n=6
    sigma_cov = np.zeros((6,6))
    for i in range(2*n):
        Wi_prime = np.append(error[:,i], sigma_w[:,i] - sigma_mu_w)
        # print(Wi_prime)
        sigma_cov += Wi_prime.reshape(6,1) @ Wi_prime.reshape(1,6) /(2*n)
    return sigma_cov
    

def measurement_transform(x_q, x_w):
    '''
    Input: 
    - x_q: list of 12 vectors size (4,)
    - x_w: array size (3,12)
    - Q: measurement noise
    Output:
    - y_rot, y_acc: each is list of 12 of vector size (3,) - transformed sigma points
    '''

    n = 6
    g = Quaternion(0, [0,0,9.81])
    y_rot = []
    y_acc = []
    for i in range(2*n):
        y_rot.append(x_w[:,i])# + Q[3:])
        
        quat = Quaternion(scalar=x_q[i][0], vec=x_q[i][1:4])
        g_prime = quat.inv() * g * quat
        y_acc.append(g_prime.vec())# + Q[:3])
    return y_rot, y_acc


def process_model(x_q_k, x_w_k, R, dt):
    '''
    Input:
    - x_q_k: vector size(4,)
    - x_w_k: vector size(3,)
    - R: vector size(6,)
    - dt: time interval
    Output:
    - x_q_k1:
    - x_w_k1:
    '''
    x_w_k1 = x_w_k + R[3:]
    
    quat_k = Quaternion(scalar=x_q_k[0], vec=x_q_k[1:4])
    alpha_delta = np.linalg.norm(x_w_k) * dt
    eta_delta = x_w_k / np.linalg.norm(x_w_k)
    quat_delta = Quaternion(scalar=math.cos(alpha_delta/2),
                            vec = eta_delta * math.sin(alpha_delta/2) )
    quat_noise = Quaternion()
    quat_noise.from_axis_angle(R[:3])
    quat_k1 = quat_k * quat_noise * quat_delta 
    x_q_k1 = quat_k1.q

    return x_q_k1, x_w_k1



# ===================================
# Methods for measurement calibration


def accel_2_euler(accel):
    pitch = np.hstack([np.arcsin(accel[0,t]/ 9.81) for t in range(np.shape(accel)[1])])
    roll = np.hstack([math.atan2(accel[1,t], accel[2,t]) for t in range(np.shape(accel)[1])])
    return np.vstack((roll, pitch))

def accel_calibration(accel, bias, sensitivity):
    ''' Input: raw accel shape(3,T), bias shape(3,), sensitivity shape(3,)
    This function return calibrated accel data (unit: m/s^2) shape(3,T)
    '''
    accel_imu = (accel - np.expand_dims(bias, axis=1)) * 3300 / (1023 * np.expand_dims(sensitivity, axis=1))
    # Do we need to flip Ax, Ay here?
    accel_imu = np.vstack([accel_imu[0,:]*(-1), accel_imu[1,:]*(-1), accel_imu[2,:]])

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


roll, pitch, yaw = estimate_rot()



