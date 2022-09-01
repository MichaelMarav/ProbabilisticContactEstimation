#! /usr/bin/env python3

import numpy as np
from numpy import genfromtxt
import math
from scipy import signal as sp 



id_Fx,id_Fy,id_Fz,id_Tx,id_Ty,id_Tz = 0,1,2,3,4,5
id_ax,id_ay,id_az,id_wx,id_wy,id_wz = 6,7,8,9,10,11

sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2


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
 

def mle_means(data):
    batch_size = 50
    stride = 1
    means = np.empty((data[0].shape[0],len(data)))
    sum_x = 0
     
    for i in range(batch_size,means.shape[0],stride):
        for j in range(means.shape[1]):
            means[i] 
        batch = data
        pass


if __name__ == "__main__":

    ax, ay, wz, fx, fy, fz = prepare_data()
    data = [ax,ay,wz,fx,fy,fz]
    mle_means(data)