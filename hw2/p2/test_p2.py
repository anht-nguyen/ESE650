from scipy import io
import numpy as np
from estimate_rot import rotation_angles, accel_2_euler
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


data_num = 1
imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
accel = imu['vals'][0:3,:] # order: pitch, roll, yaw
gyro = imu['vals'][3:6,:]
T = np.shape(imu['ts'])[1]
ts_imu = imu['ts'].reshape(-1,)
angle_names = ['roll', 'pitch', 'yaw']

accel_adjusted = np.vstack([accel[0,:]*(-1), accel[1,:]*(-1), accel[2,:]])

euler_accel = accel_2_euler(accel_adjusted) 

# plt.plot(ts_imu, )

plt.subplot(2,2,1)
for i in range(2):
    plt.plot(ts_imu, accel_adjusted[i,:], label = 'accel '+ angle_names[i])
plt.title('Plot IMU accel data')
plt.legend()

plt.subplot(2,2,2)
for i in range(2):
    plt.plot(ts_imu, euler_accel[i,:], label = angle_names[i])
plt.title('Plot roll, pitch from accel')
plt.legend()

gyro_adjusted = np.vstack([gyro[1,:], gyro[2,:], gyro[0,:]])
plt.subplot(2,2,3)
for i in range(3):
    plt.plot(ts_imu, gyro_adjusted[i,:], label = 'gyro '+ angle_names[i])
plt.title('Plot IMU gyro data')
plt.legend()



vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
T_vicon = np.shape(vicon['ts'])[1]
ts_vicon = vicon['ts'].reshape(-1,)



print('DONE')



