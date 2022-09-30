#! /usr/bin/env python3

import numpy as np
from numpy import genfromtxt
import math
from scipy import signal as sp 
import matplotlib.pyplot as plt

# Indices for features (file structure)
id_Fx,id_Fy,id_Fz,id_Tx,id_Ty,id_Tz = 0,1,2,3,4,5
id_ax,id_ay,id_az,id_wx,id_wy,id_wz = 6,7,8,9,10,11


# Standard deviation for 1000hz (Rotella et al.)
sigma_a = 2
sigma_w = 0.01
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
    filename = "../data/realTALOS1.csv"

    data = np.genfromtxt(filename,delimiter=",")

    Fx = data[:,0]
    Fy = data[:,1]
    Fz = data[:,2]
    
    ax = data[:,3]
    ay = data[:,4]
    az = data[:,5]
    
    wx = data[:,6]
    wy = data[:,7]
    wz = data[:,8]

    return Fx,Fy,Fz,ax,ay,az,wx,wy,wz




'''
Input: Takes a list with the features
Output: np array means with all the mean of gaussians for batch size with stride
'''
def mle_means(data):
    batch_size = 20
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
    sigma_thresh = 3 # How close to 0 I want the value to be (symmetrical)
    for i in range(means.shape[0]):
        Pax = normal_cdf(means[i,0],sigma_a,sigma_thresh*sigma_a) - normal_cdf(means[i,0],sigma_a,-sigma_thresh*sigma_a)
        Pay = normal_cdf(means[i,1],sigma_a,sigma_thresh*sigma_a) - normal_cdf(means[i,1],sigma_a,-sigma_thresh*sigma_a)
        Pwz = normal_cdf(means[i,2],sigma_w,sigma_thresh*sigma_w) - normal_cdf(means[i,2],sigma_w,-sigma_thresh*sigma_w)
        # Fix this       
        PFz = 1 - normal_cdf(means[i,5],sigma_F, 0) #-sigma_thresh*sigma_F)
        prob[i] = Pax*Pay*Pwz#*PFz
    return prob



if __name__ == "__main__":


    fx,fy,fz,ax,ay,az,wx,wy,wz = prepare_data() 
    data = [ax,ay,wz,fx,fy,fz]


    means = mle_means(data)

    probs = contact_probability(means)

    Ftan = np.sqrt(data[3][:]**2+data[4][:]**2)

    time = np.arange(probs.shape[0])

    # time_a = np.arange(ax.shape[0])
    # time_F = np.arange(fx.shape[0])
    # fig, axs = plt.subplots(4)
    # axs[0] = axs[0].plot(time_a,ax)
    # axs[1] = axs[1].plot(time_a,ay)
    # axs[2] = axs[2].plot(time_a,wz)
    # axs[3] = axs[3].plot(time_F,fz)
    # plt.show()
    fig, axs = plt.subplots(2)
    axs[0].plot(time,probs, c='g')#,s=5)

    time_f = np.arange(fz.shape[0])
    axs[1].plot(time_f,fz, c= 'b')

    plt.show()
    