#! /usr/bin/env python3 

import numpy as np 
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import signal as sp 

filename = "../data/atlas_1000hz_01gr_003road.csv" 

Fx,Fy,Fz,Tx,Ty,Tz = 0,1,2,3,4,5
ax,ay,az,wx,wy,wz = 6,7,8,9,10,11

sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2


subset_size = 100

def cdf(mu,sigma,x):
    a = (x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1 + math.erf(a))


def add_noise(data):

    for i in range (data.shape[0]):
        for j in range (3):
            data[i,j] += np.random.normal(0,sigma_F)
    
    for i in range (data.shape[0]):
        for j in range (6,9):
            data[i,j] += np.random.normal(0,sigma_a)
    
    for i in range (data.shape[0]):
        for j in range (9,12):
            data[i,j] += np.random.normal(0,sigma_w)
    
    return data





def  calculate_prob(ax,ay,wz,Fz):
    prob = np.empty((ax.shape[0],)) # Try to predict not only for means but for every sample
    sigma_thresh = 2
    for i in range(ax.shape[0]):
        Pax = cdf(ax[i],sigma_a, sigma_thresh*sigma_a) - cdf(ax[i],sigma_a,-sigma_thresh*sigma_a)
        Pay = cdf(ay[i],sigma_a, sigma_thresh*sigma_a) - cdf(ay[i],sigma_a,-sigma_thresh*sigma_a)
        Pwz = cdf(wz[i],sigma_w, sigma_thresh*sigma_w) - cdf(wz[i],sigma_w,-sigma_thresh*sigma_w)
        PFz = cdf(Fz[i],sigma_F, sigma_thresh*sigma_F) - cdf(Fz[i],sigma_F,-sigma_thresh*sigma_F)
        print("Probabilities: ",Pax,Pay,Pwz,PFz)
        prob[i] = Pax*Pay*Pwz#*PFz
    return prob


def mle_means(v):
    means = np.empty((v.shape[0]//subset_size,))
    sum_x = 0 
    for i in range(1,v.shape[0]):
        if (i+1)%subset_size == 0:
            sum_x += v[i]
            means[(i+1)//subset_size - 1] = sum_x/subset_size
            sum_x = 0
        else:
            sum_x += v[i] 
    return means






if __name__ == "__main__":

    data = genfromtxt(filename,delimiter = ",")
    del_list = np.arange(-59,0,1)
    data = np.delete(data,del_list, axis = 0)
    data = add_noise(data)
    time = np.arange(data.shape[0])

    # Use median to filter out spikes
    data_ax = sp.medfilt(data[:,ax], 5)
    data_ay = sp.medfilt(data[:,ay], 5)
    data_az = sp.medfilt(data[:,az], 5)
    data_wx = sp.medfilt(data[:,wx], 5)
    data_wy = sp.medfilt(data[:,wy], 5)
    data_wz = sp.medfilt(data[:,wz], 5)
    data_Fx = sp.medfilt(data[:,Fx], 5) 
    data_Fy = sp.medfilt(data[:,Fy], 5) 
    data_Fz = sp.medfilt(data[:,Fz], 5) 

    
    means_ax = mle_means(data_ax)
    means_ay = mle_means(data_ay)
    means_az = mle_means(data_az)
    means_wx = mle_means(data_wx)
    means_wy = mle_means(data_wy)
    means_wz = mle_means(data_wz)
    means_Fx = mle_means(data_Fx)
    means_Fy = mle_means(data_Fy)
    means_Fz = mle_means(data_Fz)
    sub_time = np.arange(data.shape[0]//subset_size)

    # square = np.sqrt(means_Fx**2 + means_Fy**2)

    square = np.sqrt(data_Fx**2 + data_Fy**2)

    # PLOT MEANS 
    # fig, axs = plt.subplots(2)

    # axs[0].plot(sub_time,0.1*means_Fz,c ='r')
    # axs[0].plot(sub_time,square,c='b')
    
    # axs[1].scatter(sub_time,means_ax, c = 'r',s=5)

    # axs[1].axvline(x=39,c = 'r')
    # axs[1].axvline(x=41,c = 'r')
    # axs[0].axvline(x=39,c = 'r')
    # axs[0].axvline(x=41,c = 'r')



    # plt.show()



    # PLOTS
    
    # fig, axs = plt.subplots(2)

    # p = 0
    # col = ['c','r','g','b','y']
    # data = add_noise(data)

    # for i in range(150):
    #     start = i*100
    #     stop  = start + 100
    #     color = col[p]
    #     p += 1
    #     axs[0].scatter(time[start:stop],data_ax[start:stop],c = color,s=5)
    #     if p == 5:
    #         p = 0
    # axs[1].plot(time,0.1*data[:,2],c = 'r')
    # Ftan = np.sqrt(data[:,0]**2 +data[:,1]**2)
    # axs[1].plot(time,Ftan,c ='b')

    # axs[1].axvline(x=4007,c = 'r')
    # axs[1].axvline(x=4195,c = 'r')
    # axs[0].axvline(x=4007,c = 'r')
    # axs[0].axvline(x=4195,c = 'r')


    # axs[0].axvline(x=4600,c = 'y')
    # axs[0].axvline(x=6400,c = 'y')
    # axs[1].axvline(x=4600,c = 'y')
    # axs[1].axvline(x=6400,c = 'y')

    # plt.show()

    probs = calculate_prob(means_ax,means_ay,means_wz,means_Fz)
    time_sub = np.arange(probs.shape[0])
    time = np.arange(data_Fz.shape[0])
    fig, axs = plt.subplots(2)
    axs[0].plot(time,0.1*data_Fz,c  = 'r')
    axs[0].plot(time,square,c  = 'b')
    axs[1].plot(time_sub,probs,c  = 'r')#, s=5)
    plt.show()