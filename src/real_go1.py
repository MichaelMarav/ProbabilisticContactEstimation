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
id_F1,id_F2,id_F3,id_F4 = 0,1,2,3
id_base_ax,id_base_ay,id_base_az,id_base_wx,id_base_wy,id_base_wz =  4,5,6,7,8,9
id_foot_ax,id_foot_ay,id_foot_az,id_foot_wx,id_foot_wy,id_foot_wz =  10,11,12,13,14,15

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
    data_filename = "../data/dataset_spike_normal_surface.csv"

    data = np.genfromtxt(data_filename,delimiter=",")
    
    
    indx_to_remove = []
    for j in range(data.shape[1]):
        for i in range (data.shape[0]):
            if np.isnan(data[i,j]):
                indx_to_remove.append(i)
    data = np.delete(data,indx_to_remove,axis = 0)

    # BASE
    f1 = data[:,id_F1] 
    f2 = data[:,id_F2] 
    f3 = data[:,id_F3] 
    f4 = data[:,id_F4] 
    base_ax = data[:,id_base_ax]
    base_ay = data[:,id_base_ay]
    base_az = data[:,id_base_az]
    base_wx = data[:,id_base_wx]
    base_wy = data[:,id_base_wy]
    base_wz = data[:,id_base_wz]



    foot_ax = data[:,id_foot_ax]
    foot_ay = data[:,id_foot_ay]
    foot_az = data[:,id_foot_az]

    foot_wx = data[:,id_foot_wx]
    foot_wy = data[:,id_foot_wy]
    foot_wz = data[:,id_foot_wz]


    # Median filter to get rid of the random spikes (raisim problem)
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


    return f1,f2,f3,f4,base_ax,base_ay,base_az,base_wx,base_wy,base_wz,foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz






'''
Input: Takes a list with the features
Output: np array means with all the mean of gaussians for batch size with stride
'''
def mle_means(data):
    batch_size = 50
    stride = 1
    means = np.empty((data[0].shape[0],len(data)))
    sum_x = 0

    # Compute mean for first 50 samples
    for f in range(means.shape[1]):
        for i in range(batch_size):
            sum_x += data[f][i]
        means[0:batch_size,f] = sum_x/batch_size
    

    # Compute the rest
    for f in range(means.shape[1]):
        feature_i = data[f]    

        for i in range(batch_size,means.shape[0],stride):
            means[i,f] = sum(feature_i[(i-batch_size):i])/batch_size
    return means




'''
***Input***:
mu: mean of normal distribution
sigma: Std
x: the point we want to compute cdf
***Output***:
the cdf at x (a float number)
''' 
def normal_cdf(mu,sigma,x):
    a = (x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1 + math.erf(a))




'''
*** Input ***: 
1) A list (6,num_samples) with all features (that we care) 
2) An np.array (num_samples,num_features) with parameters (means) for the gaussian distribution for every sample

*** Output ***:
Computes the probability of the contact to be stable.
prob: np.array (num_samples) -> Contains the probabilities of stable contact
'''
def contact_probability(means):
    prob = np.empty((means.shape[0],))
    sigma_thresh = 2 # How close to 0 I want the value to be (symmetrical)
    for i in range(means.shape[0]):
        Pax = normal_cdf(means[i,0],sigma_a,sigma_thresh*sigma_a) - normal_cdf(means[i,0],sigma_a,-sigma_thresh*sigma_a)
        Pay = normal_cdf(means[i,1],sigma_a,sigma_thresh*sigma_a) - normal_cdf(means[i,1],sigma_a,-sigma_thresh*sigma_a)
        Pwz = normal_cdf(means[i,2],sigma_w,sigma_thresh*sigma_w) - normal_cdf(means[i,2],sigma_w,-sigma_thresh*sigma_w)
        # Fix this       
        PFz = 1 - normal_cdf(means[i,5],sigma_F, 0) #-sigma_thresh*sigma_F)
        prob[i] = Pax*Pay*Pwz#*PFz
    return prob


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
    batch_size = 50
    stride = 1


    thres_ax = 0.9
    thres_ay = 0.5  
    thres_az = 0.3

    thres_wx = 0.04
    thres_wy = 0.04
    thres_wz = 40

    eval_samples = 50
    N = ax.shape[0] # Number of samples

    probs_x = np.empty((N,))
    probs_y = np.empty((N,))
    probs_z = np.empty((N,))



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

    prob_ax = get_axis_probability(-thres_ax,thres_ax,eval_samples,kd_ax)
    prob_ay = get_axis_probability(-thres_ay,thres_ay,eval_samples,kd_ay)
    prob_az = get_axis_probability(-thres_az,thres_az,eval_samples,kd_az)
    prob_wx = get_axis_probability(-thres_wx,thres_wx,eval_samples,kd_wx)
    prob_wy = get_axis_probability(-thres_wy,thres_wy,eval_samples,kd_wy)
    prob_wz = get_axis_probability(-thres_wz,thres_wz,eval_samples,kd_wz)



    probs_x[0:batch_size] =  prob_ax
    probs_y[0:batch_size] =  prob_ay
    probs_z[0:batch_size] =  prob_wz

    # stable_prob_vertical[0:batch_size] =  prob_az*prob_wx*prob_wy



    for i in range(batch_size,ax.shape[0],stride):
        ax_batch = ax[i:(i+batch_size)]
        ay_batch = ay[i:(i+batch_size)]
        wz_batch = wz[i:(i+batch_size)]

        az_batch = az[i:(i+batch_size)]
        wx_batch = wx[i:(i+batch_size)]
        wy_batch = wy[i:(i+batch_size)]

        # TANGENTIAL
        kd_ax = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ax_batch.reshape((len(ax_batch),1)))
        kd_ay = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(ay_batch.reshape((len(ay_batch),1)))
        kd_wz = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wz_batch.reshape((len(wz_batch),1))) 
        
        prob_ax = get_axis_probability(-thres_ax,thres_ax,eval_samples,kd_ax)
        prob_ay = get_axis_probability(-thres_ay,thres_ay,eval_samples,kd_ay)
        prob_wz = get_axis_probability(-thres_wz,thres_wz,eval_samples,kd_wz)


        probs_x[i] =  prob_ax
        probs_y[i] =  prob_ay
        probs_z[i] =  prob_wz
        #stable_prob_tangential[i] =  prob_ax*prob_ay*prob_wz


        # VERTICAL
        # kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
        # kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1))) 
        # kd_az = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(az_batch.reshape((len(az_batch),1))) 

        # prob_wx = get_axis_probability(-thres_wx,thres_wx,eval_samples,kd_wx)
        # prob_wy = get_axis_probability(-thres_wy,thres_wy,eval_samples,kd_wy)
        # prob_az = get_axis_probability(-thres_az,thres_az,eval_samples,kd_az)

        # stable_prob_vertical[i] =  prob_az*prob_wx*prob_wy
    stable_prob_vertical = 0
    return probs_x,probs_y,probs_z


if __name__ == "__main__":


    f1,f2,f3,f4,base_ax,base_ay,base_az,base_wx,base_wy,base_wz,foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz= prepare_data() 

    # f3_down = np.empty((f3.shape[0]//2,))
    # p = 0
    # # Downsample force
    # for i in range(0,f3.shape[0]-1,2):
    #     f3_down[p] = f3[i]
    #     p += 1


    
    # Biases
    # bias_ay = 1
    # bias_wz = 35


    # foot_ay += bias_ay
    # foot_wz += 35
    probs_x,probs_y,probs_z = stable_contact_detection(foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz)
    print(f3.shape,foot_ax.shape, probs_x.shape)


    time_f = np.arange(f3.shape[0])

    time_a = np.arange(probs_x.shape[0])

    fig, axs = plt.subplots(4)
    axs[0].plot(time_f,f3)
    axs[1].scatter(time_a,probs_x,c='g',s=5)  
    axs[2].scatter(time_a,probs_y,c='g',s=5)  
    axs[3].scatter(time_a,probs_x*probs_y,c='g',s=5)  

    plt.show()
    # fig, axs = plt.subplots(7)
    # axs[0].plot(time,f3, c='g')
    # axs[1].plot(time_a,foot_ax, c='g')
    # axs[2].plot(time_a,foot_ay, c='g')
    # axs[3].plot(time_a,foot_az, c='g')
    # axs[4].plot(time_a,foot_wx, c='g')
    # axs[5].plot(time_a,foot_wy, c='g')
    # axs[6].plot(time_a,foot_wz, c='g')

    # plt.show()
    
    # data = [ax,ay,wz,fx,fy,fz]


    # means = mle_means(data)

    # probs = contact_probability(means)

    # friction_estimation(data,probs)


    
    # PLOT DATA
    # plot_data(data,probs)
    

