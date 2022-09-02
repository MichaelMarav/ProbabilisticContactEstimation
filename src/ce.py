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
sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2


'''
1) Reads the data from "filename". 
2) Adds noise (0 mean gaussian distribution)
3) Uses median filter to remove raisim random spikes in measurements
'''
def prepare_data():

    # Read Dataset
    data_filename = "../data/atlas_1000hz_01ground.csv"

    data = np.genfromtxt(data_filename,delimiter=",")

    # Add gaussian noise
    for i in range (data.shape[0]):
        for j in range (3):
            data[i,j] += np.random.normal(0,sigma_F)
    
    for i in range (data.shape[0]):
        for j in range (6,9):
            data[i,j] += np.random.normal(0,sigma_a)
    
    for i in range (data.shape[0]):
        for j in range (9,12):
            data[i,j] += np.random.normal(0,sigma_w)
    

    # Median filter to get rid of the random spikes (raisim problem)
    data[:,0] = sp.medfilt(data[:,0], 5)
    data[:,1] = sp.medfilt(data[:,1], 5)
    data[:,2] = sp.medfilt(data[:,2], 5)
    data[:,3] = sp.medfilt(data[:,3], 5)
    data[:,4] = sp.medfilt(data[:,4], 5)
    data[:,5] = sp.medfilt(data[:,5], 5)
    data[:,6] = sp.medfilt(data[:,6], 5)
    data[:,7] = sp.medfilt(data[:,7], 5)
    data[:,8] = sp.medfilt(data[:,8], 5)
    data[:,9] = sp.medfilt(data[:,9], 5)
    data[:,10]= sp.medfilt(data[:,10], 5)
    data[:,11]= sp.medfilt(data[:,11], 5)


    return data[:,6], data[:,7], data[:,11], data[:,0], data[:,1], data[:,2]


def export_labels():
    data_filename = "../data/atlas_1000hz_01ground.csv"

    data = np.genfromtxt(data_filename,delimiter=",")
    labels = data[:,-1]
    for i in range(labels.shape[0]):
        if labels[i] == 2 or labels[i] == 1:
            labels[i] = 0
        else:
            labels[i] = 1


    return labels



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
        Pwz = normal_cdf(means[i,2],sigma_w,sigma_thresh*sigma_a) - normal_cdf(means[i,2],sigma_w,-sigma_thresh*sigma_w)
        # Fix this       
        # PFz = normal_cdf(means[i,5],sigma_F,sigma_thresh*sigma_F) - normal_cdf(means[i,5],sigma_F,-sigma_thresh*sigma_F)
        prob[i] = Pax*Pay*Pwz
    return prob



if __name__ == "__main__":

    ax, ay, wz, fx, fy, fz = prepare_data() 
    data = [ax,ay,wz,fx,fy,fz]

    labels = export_labels()
    means = mle_means(data)

    probs = contact_probability(means)

    Ftan = np.sqrt(data[3][:]**2+data[4][:]**2)

    time = np.arange(probs.shape[0])

    fig, axs = plt.subplots(2)
    axs[0].scatter(time,probs, c='g',s=5)
    axs[0].scatter(time,labels,c='r',s=5) # Ground truth


    axs[1].plot(time,Ftan,c ='r')
    axs[1].plot(time,0.4*data[5][:], c= 'b')




    plt.show()