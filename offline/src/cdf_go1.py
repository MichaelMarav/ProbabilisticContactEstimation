#! /usr/bin/env python3 

import numpy as np 
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import signal as sp 

filename = "../data/vel_500hz.csv" 
refresh_rate = 500 # Hz
Fx,Fy,Fz,Tx,Ty,Tz = 0,1,2,3,4,5
ax,ay,az,wx,wy,wz = 6,7,8,9,10,11

sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2


subset_size = 50

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
# def contact(data):
#     data_ax = sp.medfilt(data[:,ax], 5)
#     data_ay = sp.medfilt(data[:,ay], 5)
#     data_wz = sp.medfilt(data[:,wz], 5)
#     data_Fz = sp.medfilt(data[:,Fz], 5)

    # cut_off = 5

    # Prob = np.empty((data_ax.shape[0],))

    # for i in range(data.shape[0]):
    #     if data_Fz[i] < cut_off*sigma_F and data_Fz[i] > -cut_off*sigma_F:
    #         if data_ax[i] < cut_off*sigma_a and data_ax[i] > -cut_off*sigma_a:    
    #             if data_ay[i] < cut_off*sigma_a and data_ay[i] > -cut_off*sigma_a:
    #                 if data_wz[i] < cut_off*sigma_w and data_wz[i] > -cut_off*sigma_w:
    #                     sigma_thresh = 1
                        
                        
    #                     Pax = cdf(data_ax[i],sigma_a,data_ax[i] + sigma_thresh*sigma_a) - cdf(data_ax[i],sigma_a,data_ax[i]-sigma_thresh*sigma_a)
    #                     Pay = cdf(data_ay[i],sigma_a,data_ay[i] + sigma_thresh*sigma_a) - cdf(data_ay[i],sigma_a,data_ay[i]-sigma_thresh*sigma_a)
    #                     Pwz = cdf(data_wz[i],sigma_w,data_wz[i] + sigma_thresh*sigma_w) - cdf(data_wz[i],sigma_w,data_wz[i]-sigma_thresh*sigma_w)
    #                     PFz = cdf(data_Fz[i],sigma_F,data_Fz[i] + sigma_thresh*sigma_F) - cdf(data_Fz[i],sigma_F,data_Fz[i]-sigma_thresh*sigma_F)

    #                     Prob[i] = Pax*Pay*Pwz*PFz
    #                 else:
    #                     print("--> Wz fail")
    #                     Prob[i] = -1
    #             else:
    #                 print("--> ay fail")
    #                 Prob[i] = -1
    #         else:
    #             print("--> ax fail")(data[i,0]-data[i-1,0])
    #             Prob[i] = -1
    #     else:
    #         print("--> Fz fail")
    #         Prob[i] = -1

    # return Prob

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



def convert_vel2acc(data):
    data_imu = np.empty((data.shape[0]-1,data.shape[1]-1))
    for i in range(1,data.shape[0]):
        for j in range(1,4):
            data_imu[i-1,j] = (data[i,j]-data[i-1,j])*refresh_rate
    for i in range(1,data.shape[0]):
        data_imu[i-1,3] = data[i,3]
        data_imu[i-1,4] = data[i,4]
        data_imu[i-1,5] = data[i,5]
    return data_imu


if __name__ == "__main__":

    data = genfromtxt(filename,delimiter = ",")
    del_list = np.arange(-55,0,1)
    data = np.delete(data,del_list, axis = 0)

    data_imu = convert_vel2acc(data)
    time = np.arange(data_imu.shape[0])
    fig , ax = plt.subplots(6)
    ax[0].scatter(time,data_imu[:,0],s=5)
    ax[1].scatter(time,data_imu[:,1],s=5)
    ax[2].scatter(time,data_imu[:,2],s=5)
    ax[3].scatter(time,data_imu[:,3],s=5)
    ax[4].scatter(time,data_imu[:,4],s=5)
    ax[5].scatter(time,data_imu[:,5],s=5)

    plt.show()