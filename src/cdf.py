#! /usr/bin/env python3 

import numpy as np 
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import signal as sp 

filename = "../data/atlas_1000hz_01ground_3steps.csv" 

Fx,Fy,Fz,Tx,Ty,Tz = 0,1,2,3,4,5
ax,ay,az,wx,wy,wz = 6,7,8,9,10,11

mu = 0
sigma_a = 0.02467
sigma_w = 0.01653
sigma_F = 2

def cdf(mu,sigma,x):
    a = (x-mu)/(sigma*math.sqrt(2))
    return 0.5*(1 + math.erf(a))


def add_noise(data):

    for i in range (data.shape[0]):
        for j in range (3):
            data[i,j] += np.random.normal(mu,sigma_F)
    
    for i in range (data.shape[0]):
        for j in range (6,9):
            data[i,j] += np.random.normal(mu,sigma_a)
    
    for i in range (data.shape[0]):
        for j in range (9,12):
            data[i,j] += np.random.normal(mu,sigma_w)
    
    return data


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
    #             print("--> ax fail")
    #             Prob[i] = -1
    #     else:
    #         print("--> Fz fail")
    #         Prob[i] = -1

    # return Prob


if __name__ == "__main__":

    data = genfromtxt(filename,delimiter = ",")
    time = np.arange(data.shape[0])

    data_ax = sp.medfilt(data[:,ax], 5)
    data_ay = sp.medfilt(data[:,ay], 5)
    data_wz = sp.medfilt(data[:,wz], 5)
    data_Fz = sp.medfilt(data[:,Fz], 5)

    p = 0
    col = ['c','r','g','b','y']

    for i in range(150):
        start = i*100
        stop  = start + 100
        color = col[p]
        p += 1
        plt.scatter(time[start:stop],data_ax[start:stop],c = color,s=5)
        if p == 5:
            p = 0
        

    # prob = contact(data)
    # fig, axs = plt.subplots(2)
    # axs[0].scatter(time,data_ax,c = 'r',s = 5)
    # axs[1].scatter(time,data[:,6],c = 'b',s = 5)
    
    # plt.plot(time,data_ax,c = 'r')

    # data = add_noise(data)

    # plt.plot(time,data[:,az],c='b')
    # P(a < x < b)
    # a = -3*sigma
    # b = 3*sigma
    # f1 = cdf(mu,sigma,a)
    # f2 = cdf(mu,sigma,b)

    # print(f2-f1)
    plt.show()