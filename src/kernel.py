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
2) Creates the ground truth labels with Coulomb friction model 
3) Adds noise (0 mean gaussian distribution)
4) Uses median filter to remove raisim random spikes in measurements
'''


def prepare_data():

    # Read Dataset
    data_filename = "../data/atlas_1000hz_01ground.csv"

    data = np.genfromtxt(data_filename,delimiter=",")


    Fx = data[:,0]
    Fy = data[:,1]
    Fz = data[:,2]

    labels = np.empty((Fx.shape[0],))
    for i in range(Fx.shape[0]):
        if Fz[i] > 1 and np.sqrt(Fx[i]**2 + Fy[i]**2) -friction_coef*Fz[i] <= -contact_threshold_label:
            labels[i] = 1
        else: 
            labels[i] = 0


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
    median_window = 7
    data[:,0] = sp.medfilt(data[:,0], median_window)
    data[:,1] = sp.medfilt(data[:,1], median_window)
    data[:,2] = sp.medfilt(data[:,2], median_window)
    data[:,3] = sp.medfilt(data[:,3], median_window)
    data[:,4] = sp.medfilt(data[:,4], median_window)
    data[:,5] = sp.medfilt(data[:,5], median_window)
    data[:,6] = sp.medfilt(data[:,6], median_window)
    data[:,7] = sp.medfilt(data[:,7], median_window)
    data[:,8] = sp.medfilt(data[:,8], median_window)
    data[:,9] = sp.medfilt(data[:,9], median_window)
    data[:,10]= sp.medfilt(data[:,10], median_window)
    data[:,11]= sp.medfilt(data[:,11], median_window)


    return data[:,6], data[:,7], data[:,11], data[:,0], data[:,1], data[:,2] , labels 


def gauss_kernel(x):
    return (1./np.sqrt(2*math.pi))*math.exp(-0.5*x**2)




def kde(x,h,xi):
    batch_size = 25
    sum_kernels = 0 
    h = 2
    for i in range(batch_size):
        sum_kernels += gauss_kernel((x-xi[i])/h)
    sum_kernels = sum_kernels*(1./(batch_size*h))



def make_data():
    data = np.empty((100,))
    m1 = 1
    m2 = 7
    s1 = 2
    s2 = 2

    for i in range(50):
        data[i] = np.random.normal(m1,s1)
    for i in range(50,100):
        data[i] = np.random.normal(m2,s2)

    return data


if __name__ == "__main__":


    ax, ay, wz, fx, fy, fz, labels = prepare_data() 
    # data = [ax,ay,wz,fx,fy,fz]
    start,end = 9700,9850
    ax = ax[start:end]


    
    kde = KernelDensity(bandwidth=sigma_a, kernel='gaussian')
    ax = ax.reshape((len(ax),1))
    kde.fit(ax)

    values = np.arange(-0.2,1,0.01)
    values = values.reshape((len(values), 1))
    
    probabilities = kde.score_samples(values)
    probabilities = np.exp(probabilities)

    plt.hist(ax,bins=30)
    plt.plot(values[:],probabilities)
    plt.show()
    