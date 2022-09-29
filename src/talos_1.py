#! /usr/bin/env python3
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import genfromtxt
import math
from scipy import signal as sp 
import matplotlib.pyplot as plt
import random



# Indices for features (file structure)
id_Fx,id_Fy,id_Fz,id_Tx,id_Ty,id_Tz = 0,1,2,3,4,5
id_ax,id_ay,id_az,id_wx,id_wy,id_wz = 6,7,8,9,10,11


# Standard deviation for 1000hz (Rotella et al.)
sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2

contact_threshold_label = 50 # How much larger mu*Fz must be from Ftan, for the contact to be considered stable
friction_coef = 0.1

'''
1) Reads the data from "filename".
2) Creates the ground truth labels with Coulomb friction model 
3) Adds noise (0 mean gaussian distribution)
4) Uses median filter to remove raisim random spikes in measurements
'''


def prepare_data():

    # Read Dataset
    data_filename = "../data/realTALOS1.csv"

    data = np.genfromtxt(data_filename,delimiter=",")


    # Fx = data[:,0]
    # Fy = data[:,1]
    # Fz = data[:,2]

    # labels = np.empty((Fx.shape[0],))
    # for i in range(Fx.shape[0]):
    #     if Fz[i] > 1 and np.sqrt(Fx[i]**2 + Fy[i]**2) -friction_coef*Fz[i] <= -contact_threshold_label:
    #         labels[i] = 1
    #     else: 
    #         labels[i] = 0


    # Add gaussian noise
    # for i in range (data.shape[0]):
    #     for j in range (3):
    #         data[i,j] += np.random.normal(0,sigma_F)
    
    # for i in range (data.shape[0]):
    #     for j in range (6,9):
    #         data[i,j] += np.random.normal(0,sigma_a)
    
    # for i in range (data.shape[0]):
    #     for j in range (9,12):
    #         data[i,j] += np.random.normal(0,sigma_w)
    

    # # Median filter to get rid of the random spikes (raisim problem)
    # median_window = 7
    # data[:,0] = sp.medfilt(data[:,0], median_window)
    # data[:,1] = sp.medfilt(data[:,1], median_window)
    # data[:,2] = sp.medfilt(data[:,2], median_window)
    # data[:,3] = sp.medfilt(data[:,3], median_window)
    # data[:,4] = sp.medfilt(data[:,4], median_window)
    # data[:,5] = sp.medfilt(data[:,5], median_window)
    # data[:,6] = sp.medfilt(data[:,6], median_window)
    # data[:,7] = sp.medfilt(data[:,7], median_window)
    # data[:,8] = sp.medfilt(data[:,8], median_window)
    # data[:,9] = sp.medfilt(data[:,9], median_window)
    # data[:,10]= sp.medfilt(data[:,10], median_window)
    # data[:,11]= sp.medfilt(data[:,11], median_window)


    return data[:,0],data[:,1],data[:,2],data[:,3],data[:,4],data[:,5],data[:,6],data[:,7],data[:,8],









def get_axis_probability(start_value, end_value, eval_points, kd):
    
    # Number of evaluation points 
    N = eval_points                                      
    step = (end_value - start_value) / (N - 1)  # Step size

    x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)  # Approximate the integral of the PDF
    return probability.round(4)






def stable_contact_detection(ax,ay,az,wx,wy,wz):
    # Parameters
    batch_size = 25
    stride = 1
    thres_a = 0.5
    thres_w = 0.08
    eval_samples = 500
    N = ax.shape[0] # Number of samples

    stable_prob_tangential= np.empty((N,))
    stable_prob_vertical  = np.empty((N,))

    stable_ax = np.empty((N,))
    stable_ay = np.empty((N,))
    stable_az = np.empty((N,))
    stable_wx = np.empty((N,))
    stable_wy = np.empty((N,))
    stable_wz = np.empty((N,))

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
    kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
    kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1))) 
    kd_wz = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wz_batch.reshape((len(wz_batch),1))) 

    prob_ax = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ax)
    prob_ay = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ay)
    prob_az = get_axis_probability(-thres_a,thres_a,eval_samples,kd_az)
    prob_wx = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wx)
    prob_wy = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wy)
    prob_wz = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wz)

    stable_ax[0:batch_size] = prob_ax
    stable_ay[0:batch_size] = prob_ay
    stable_az[0:batch_size] = prob_az
    stable_wx[0:batch_size] = prob_wx
    stable_wy[0:batch_size] = prob_wy
    stable_wz[0:batch_size] = prob_wz

    # stable_prob_tangential[0:batch_size] =  prob_ax*prob_ay*prob_wz
    # stable_prob_vertical[0:batch_size] =  prob_az*prob_wx*prob_wy



    for i in range(batch_size,ax.shape[0],stride):
        ax_batch = ax[(i-batch_size):i]
        ay_batch = ay[(i-batch_size):i]
        wz_batch = wz[(i-batch_size):i]
        az_batch = az[(i-batch_size):i]
        wx_batch = wx[(i-batch_size):i]
        wy_batch = wy[(i-batch_size):i]

        # TANGENTIAL
        kd_ax = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ax_batch.reshape((len(ax_batch),1)))
        kd_ay = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ay_batch.reshape((len(ay_batch),1)))
        kd_wz = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wz_batch.reshape((len(wz_batch),1))) 
        prob_ax = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ax)
        prob_ay = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ay)
        prob_wz = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wz)
        # stable_prob_tangential[i] =  prob_ax*prob_ay*prob_wz


        # VERTICAL
        kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
        kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1))) 
        kd_az = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(az_batch.reshape((len(az_batch),1))) 
        prob_wx = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wx)
        prob_wy = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wy)
        prob_az = get_axis_probability(-thres_a,thres_a,eval_samples,kd_az)

        # stable_prob_vertical[i] =  prob_az*prob_wx*prob_wy


        stable_ax[(i-batch_size):i] = prob_ax
        stable_ay[(i-batch_size):i] = prob_ay
        stable_az[(i-batch_size):i] = prob_az
        stable_wx[(i-batch_size):i] = prob_wx
        stable_wy[(i-batch_size):i] = prob_wy
        stable_wz[(i-batch_size):i] = prob_wz


    return stable_ax, stable_ay, stable_az, stable_wx, stable_wy, stable_wz
    


if __name__ == "__main__":


    fx,fy,fz,ax,ay,az,wx,wy,wz = prepare_data() 

    bias_ax, bias_ay, bias_az = -0.98, -1.26, -9.81
    ax += bias_ax
    ay += bias_ay
    az += bias_az


    time = np.arange(fx.shape[0])
    # fig,axs = plt.subplots(7)
    # axs[0].plot(time,fz)
    # axs[1].plot(time,ax)
    # axs[2].plot(time,ay)
    # axs[3].plot(time,az)
    # axs[4].plot(time,wx)
    # axs[5].plot(time,wy)
    # axs[6].plot(time,wz)


    stable_ax, stable_ay, stable_az, stable_wx, stable_wy, stable_wz = stable_contact_detection(ax,ay,az,wx,wy,wz)

    time_f = np.arange(340)
    time_p = np.arange(0,690*0.01,0.01)
    fig,axs = plt.subplots(2)
    axs[0].plot(time_f,fz[920:1260])
    total = stable_ax*stable_ay*stable_az*stable_wx
    axs[1].scatter(time_p,total[1810:2500],c='g',s=5)

    # axs[2].scatter(time,stable_ax*stable_ay*stable_az,c='g',s=5)
    # axs[3].scatter(time,stable_ax*stable_az,c='g',s=5)
    axs[0].set_ylabel(r'$F_z(N)$ ',fontsize=20)

    axs[1].set_ylabel(r'$P_{tot}(stable)$',fontsize=20)
    axs[1].set_xlabel(r'Time (s)',fontsize=16)


    # time = np.arange(fx.shape[0])
    # fig,axs2 = plt.subplots(7)
    # axs2[0].plot(time,fz)
    # axs2[1].scatter(time,stable_ax,c='g',s=5)
    # axs2[2].scatter(time,stable_ay,c='g',s=5)
    # axs2[3].scatter(time,stable_az,c='g',s=5)
    # axs2[4].scatter(time,stable_wx,c='g',s=5)
    # axs2[5].scatter(time,stable_wy,c='g',s=5)
    # axs2[6].scatter(time,stable_wz,c='g',s=5)

    # axs[1].scatter(time,stable_ax*stable_ay*stable_az*stable_wx*stable_wy*stable_wz,c ='g',s =5)

    fig.subplots_adjust(wspace=0, hspace=0)

    plt.show()

    
