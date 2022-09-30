#! /usr/bin/env python3

# ROS libraries
import rospy

# DESPINA : add the correct IMU and FORCE rosmsgs
from sensor_msgs.msg import Imu      # IMU msg
from unitree_legged_msgs.msg import HighState # force msg Unitree

from std_msgs.msg import Float32 # Contact probability msg
# Generic libraries
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

FOOT_ID = 1 

class pce:
    def __init__(self,force_topic_name,imu_topic_name,params_filename,batch_size,eval_samples):
        self.force_topic= force_topic_name
        self.imu_topic  = imu_topic_name
        self.batch_size = batch_size
        self.eval_samples = eval_samples
        self.mytimer = 0

        # Get params from file and distribute them to variables # DESPINA : change params to go to the right variable
        self.params  = genfromtxt(params_filename,delimiter=',')        # Bias and std, for each axis
        self.ax_bias = self.params[0]
        self.ay_bias = self.params[1]
        self.az_bias = self.params[2]
        self.wx_bias = self.params[3]
        self.wy_bias = self.params[4]
        self.wz_bias = self.params[5]

        self.fz_bias = self.params[6]

        self.ax_std = self.params[7] 
        self.ay_std = self.params[8]
        self.az_std = self.params[9]
        self.wx_std = self.params[10]
        self.wy_std = self.params[11]
        self.wz_std = self.params[12]

        print(self.ax_bias, self.wz_std)

        self.data_batch = np.empty((self.batch_size,7)) # Batch of data to made predictions on


        rospy.init_node('ProbabilisticContactEstimation', anonymous=True)

        self.rate = rospy.Rate(250) 
        self.initialize_batch()

        pub = rospy.Publisher('/contact_probability', Float32, queue_size=10)

        stable_probability = Float32()

        while not rospy.is_shutdown():
            self.mytimer = self.mytimer + 2
            # DESPINA: Fix msg , also maybe add condition to check if msg actually arrived
            force_msg = rospy.wait_for_message(self.force_topic, HighState , timeout=None)
            imu_msg   = rospy.wait_for_message(self.imu_topic, Imu , timeout=None)

            stable_probability.data = self.stable_contact_detection(force_msg,imu_msg)
            # if self.mytimer > 3500:
            #     stable_probability.data = 0.3
            print(stable_probability.data,"\n")
            pub.publish(stable_probability)
            self.rate.sleep()


    '''
    Description:
    Initializes the first batch of data to avoid seg faults
    '''
    def initialize_batch(self):
        
        print("In initialize_batch START")

        for i in range(self.batch_size):
            force_msg = rospy.wait_for_message(self.force_topic, HighState, timeout=None)
            imu_msg   = rospy.wait_for_message(self.imu_topic, Imu, timeout=None)
            
            self.data_batch[i,0] =  force_msg.footForce[FOOT_ID]   
            self.data_batch[i,1] =  imu_msg.linear_acceleration.x
            self.data_batch[i,2] =  imu_msg.linear_acceleration.y 
            self.data_batch[i,3] =  imu_msg.linear_acceleration.z
            self.data_batch[i,4] =  imu_msg.angular_velocity.x
            self.data_batch[i,5] =  imu_msg.angular_velocity.y
            self.data_batch[i,6] =  imu_msg.angular_velocity.z
            self.rate.sleep()

        print("In initialize_batch END")


    '''
    Description:
    Calculates the probability in an interval [start_value, end_value] by integrating the PDF
    -- Input --
    * start_value (float): small limit of interval
    * end_value (float): large limit of interval
    * eval_points (int): How many points should be used to approximate the integral
    * kd (KernelDensity): A kernel density object by sklearn

    -- Output --
    * The probability of the PDF in the interval, rounded to 4 digits
    '''
    def get_axis_probability(self,start_value, end_value, kd):
        # Number of evaluation points 
        N = self.eval_samples                                      
        step = (end_value - start_value) / (N - 1)  # Step size

        x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
        kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
        probability = np.sum(kd_vals * step)   # Approximate the integral of the PDF

        return probability.round(4)



    '''
    Description:
    Employs KDE to estimate the PDF over a batch of data
    -- Input --
    * force_msg (HighState/footForce:): Force message
    * imu_msg (sensor_msgs/Imu) : Linear acceleration and angular velocity message
    -- Output --
    * Probability of stable contact
    '''

    def stable_contact_detection(self,force_msg,imu_msg):
        # DESPINA TODO: assign msg values to f and IMU measurements
        f  = force_msg.footForce[FOOT_ID] + self.fz_bias # TODO select foot

        ax = imu_msg.linear_acceleration.x - self.ax_bias
        ay = imu_msg.linear_acceleration.y - self.ay_bias
        az = imu_msg.linear_acceleration.z - self.az_bias

        wx = imu_msg.angular_velocity.x - self.wx_bias
        wy = imu_msg.angular_velocity.y - self.wy_bias
        wz = imu_msg.angular_velocity.z - self.wz_bias


        # Parameters (These are estimated experimentally ONCE during normal gait with stable contact)
        thres_ax = 0.6    # In g-scale of IMU 
        thres_ay = 0.5
        thres_az = 0.6
        thres_wx = 60
        thres_wy = 35
        thres_wz = 70

        # print(self.data_batch.shape)
        self.data_batch = np.delete(self.data_batch,0,axis = 0)                             # Delete first element
               
        self.data_batch = np.vstack( [self.data_batch, np.array([f,ax,ay,az,wx,wy,wz]) ] ) # Append the new measurement
        # print(self.data_batch.shape)
        ax_batch = self.data_batch[:,1]
        ay_batch = self.data_batch[:,2]
        az_batch = self.data_batch[:,3]
        wx_batch = self.data_batch[:,4]
        wy_batch = self.data_batch[:,5]
        wz_batch = self.data_batch[:,6]




        kde_ax = KernelDensity(bandwidth=self.ax_std, kernel='gaussian').fit(ax_batch.reshape((self.batch_size,1)))
        kde_ay = KernelDensity(bandwidth=self.ay_std, kernel='gaussian').fit(ay_batch.reshape((self.batch_size,1)))
        kde_az = KernelDensity(bandwidth=self.az_std, kernel='gaussian').fit(az_batch.reshape((self.batch_size,1))) 
        kde_wx = KernelDensity(bandwidth=self.wx_std, kernel='gaussian').fit(wx_batch.reshape((self.batch_size,1))) 
        kde_wy = KernelDensity(bandwidth=self.wy_std, kernel='gaussian').fit(wy_batch.reshape((self.batch_size,1))) 
        kde_wz = KernelDensity(bandwidth=self.wz_std, kernel='gaussian').fit(wz_batch.reshape((self.batch_size,1))) 

        
        prob_ax = self.get_axis_probability(-thres_ax,thres_ax,kde_ax)
        prob_ay = self.get_axis_probability(-thres_ay,thres_ay,kde_ay)
        prob_az = self.get_axis_probability(-thres_az,thres_az,kde_az)


        
        prob_wx = self.get_axis_probability(-thres_wx,thres_wx,kde_wx) 
        prob_wy = self.get_axis_probability(-thres_wy,thres_wy,kde_wy)
        prob_wz = self.get_axis_probability(-thres_wz,thres_wz,kde_wz)

        return prob_ax*prob_ay*prob_az*prob_wx*prob_wy*prob_wz




if __name__ == "__main__":
    try:
        # DESPINA: add the topic names for force + imu, the params filename with bias and stds
        force_topic_name = "/high_state"
        imu_topic_name = "/imu"
        params_filename = 'imuBias.txt'

        batch_size = 50
        eval_samples = 500
        pce(force_topic_name,imu_topic_name,params_filename,batch_size,eval_samples)


    except rospy.ROSInterruptException:
        print("Fail with ROS.")
