#! /usr/bin/env python3

import numpy as np
from numpy import genfromtxt
import math
from scipy import signal as sp 
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


# FORCES AND BASE IMU ---> 500 hz
# Foot imu ---> 260


# Indices for features (file structure)
id_F = 0
id_foot_ax,id_foot_ay,id_foot_az,id_foot_wx,id_foot_wy,id_foot_wz =  1,2,3,4,5,6

# Standard deviation for 1000hz (Rotella et al.)
sigma_a = 0.02467
sigma_w = 1
sigma_F = 2


'''
1) Reads the data from "filename".
2) Creates the ground truth labels with Coulomb friction model 
3) Adds noise (0 mean gaussian distribution)
4) Uses median filter to remove raisim random spikes in measurements
'''
def prepare_data():

    # Read Dataset
    data_filename = "../data/dataset_spike_wd40_extra.csv"

    data = np.genfromtxt(data_filename,delimiter=",")
    

    f = data[:,0] 

    foot_ax = data[:,id_foot_ax]
    foot_ay = data[:,id_foot_ay]
    foot_az = data[:,id_foot_az]

    foot_wx = data[:,id_foot_wx]
    foot_wy = data[:,id_foot_wy]
    foot_wz = data[:,id_foot_wz]

    for i in range(foot_ax.shape[0]):
        if (np.isnan(foot_ax[i])):
            break_point = i
            break
    foot_ax = foot_ax[0:break_point]
    foot_ay = foot_ay[0:break_point]
    foot_az = foot_az[0:break_point]
    foot_wx = foot_wx[0:break_point]
    foot_wy = foot_wy[0:break_point]
    foot_wz = foot_wz[0:break_point]


    return f, foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz





def get_axis_probability(start_value, end_value, eval_points, kd):
    
    # Number of evaluation points 
    N = eval_points                                      
    step = (end_value - start_value) / (N - 1)  # Step size

    x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF


    return probability.round(4)


def stable_contact_detection(f,ax,ay,az,wx,wy,wz):
    # Parameters
    batch_size = 50
    stride = 1
    
    thres_ax = 0.9
    thres_ay = 0.5  
    thres_az = 0.3

    thres_wx = 56
    thres_wy = 50
    thres_wz = 100


    
    eval_samples = 500
    N = ax.shape[0] # Number of samples

    probs_ax = np.empty((N,))
    probs_ay = np.empty((N,))
    probs_az = np.empty((N,))
    
    probs_wx = np.empty((N,))
    probs_wy = np.empty((N,))
    probs_wz = np.empty((N,))



    # First batch
    ax_batch = ax[0:batch_size]
    ay_batch = ay[0:batch_size]
    az_batch = az[0:batch_size]

    wx_batch = wx[0:batch_size]
    wy_batch = wy[0:batch_size]
    wz_batch = wz[0:batch_size]

    kd_ax = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ax_batch.reshape((len(ax_batch),1)))
    kd_ay = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ay_batch.reshape((len(ay_batch),1)))
    kd_az = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(az_batch.reshape((len(az_batch),1))) 

    # Vertical

    kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
    kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1)))  
    kd_wz = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wz_batch.reshape((len(wz_batch),1)))


    # Tangential
    prob_ax_i = get_axis_probability(-thres_ax,thres_ax,eval_samples,kd_ax)
    prob_ay_i = get_axis_probability(-thres_ay,thres_ay,eval_samples,kd_ay)
    prob_az_i = get_axis_probability(-thres_az,thres_az,eval_samples,kd_az)



    prob_wx_i = get_axis_probability(-thres_wx,thres_wx,eval_samples,kd_wx)
    prob_wy_i = get_axis_probability(-thres_wy,thres_wy,eval_samples,kd_wy)
    prob_wz_i = get_axis_probability(-thres_wz,thres_wz,eval_samples,kd_wz)



    probs_ax[0:batch_size] =  prob_ax_i
    probs_ay[0:batch_size] =  prob_ay_i
    probs_az[0:batch_size] =  prob_az_i


    probs_wx[0:batch_size] =  prob_wx_i
    probs_wy[0:batch_size] =  prob_wy_i
    probs_wz[0:batch_size] =  prob_wz_i

    # stable_prob_vertical[0:batch_size] =  prob_az*prob_wx*prob_wy



    for i in range(batch_size,ax.shape[0],stride):
        ax_batch = ax[(i-batch_size):i]
        ay_batch = ay[(i-batch_size):i]
        az_batch = az[(i-batch_size):i]

        wz_batch = wz[(i-batch_size):i]

        # TANGENTIAL
        kd_ax = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ax_batch.reshape((len(ax_batch),1)))
        kd_ay = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ay_batch.reshape((len(ay_batch),1)))
        kd_az = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(az_batch.reshape((len(az_batch),1))) 
        

        kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
        kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1))) 
        kd_wz = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wz_batch.reshape((len(wz_batch),1))) 

        
        prob_ax_i = get_axis_probability(-thres_ax,thres_ax,eval_samples,kd_ax)
        prob_ay_i = get_axis_probability(-thres_ay,thres_ay,eval_samples,kd_ay)
        prob_az_i = get_axis_probability(-thres_az,thres_az,eval_samples,kd_az)



        prob_wx_i = get_axis_probability(-thres_wx,thres_wx,eval_samples,kd_wx)
        prob_wy_i = get_axis_probability(-thres_wy,thres_wy,eval_samples,kd_wy)
        prob_wz_i = get_axis_probability(-thres_wz,thres_wz,eval_samples,kd_wz)

        probs_ax[i] =  prob_ax_i
        probs_ay[i] =  prob_ay_i
        probs_az[i] =  prob_az_i

        probs_wx[i] =  prob_wx_i
        probs_wy[i] =  prob_wy_i
        probs_wz[i] =  prob_wz_i


    plot_probs(f,probs_ax,probs_ay,probs_az,probs_wx,probs_wy,probs_wz)


    return
def plot_probs(f,probs_ax,probs_ay,probs_az,probs_wx,probs_wy,probs_wz):
    time = np.arange(probs_ax.shape[0])
    time_f = np.arange(0,f.shape[0]*0.002,0.002)

    fig, axs = plt.subplots(8)  
    axs[0].plot(time_f,f)
    axs[1].scatter(time,probs_ax,c='g',s=5)  
    axs[2].scatter(time,probs_ay,c='g',s=5)  
    axs[3].scatter(time,probs_az,c='g',s=5)  
    axs[4].scatter(time,probs_wx,c='g',s=5)  
    axs[5].scatter(time,probs_wy,c='g',s=5)  
    axs[6].scatter(time,probs_wz,c='g',s=5)  
    axs[7].scatter(time,probs_ax*probs_ay*probs_az*probs_wx*probs_wy*probs_wz,c='g',s=5)  
    plt.show()
    return
    


if __name__ == "__main__":


    f,foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz= prepare_data() 

    bias_ax = 0
    bias_az = -0.29
    bias_ay = 0.95
    foot_ay += bias_ay
    foot_az += bias_az
    foot_ax += bias_ax;






    time_f = np.arange(0,f.shape[0]*0.002,0.002)
    time_a = np.arange(0,foot_az.shape[0]*0.004,0.004)
    
    fig, axs = plt.subplots(7)
    axs[0].plot(time_f,f)
    axs[1].plot(time_a,foot_ax)
    axs[2].plot(time_a,foot_ay)
    axs[3].plot(time_a,foot_az)
    axs[4].plot(time_a,foot_wx)
    axs[5].plot(time_a,foot_wy)
    axs[6].plot(time_a,foot_wz)



    stable_contact_detection(f,foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz)
    
    
