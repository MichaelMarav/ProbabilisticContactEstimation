#! /usr/bin/env python3
from curses import start_color
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import genfromtxt
import math
from scipy import signal as sp 
import time 
import scipy.stats as stats
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import random

plt.rcParams['text.usetex'] = True

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


    return data[:,6], data[:,7],data[:,8],data[:,9],data[:,10], data[:,11], data[:,0], data[:,1], data[:,2] , labels 



def get_axis_probability(start_value, end_value, eval_points, kd):
    
    # Number of evaluation points 
    N = eval_points                                      
    step = (end_value - start_value) / (N - 1)  # Step size

    x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
    kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
    probability = np.sum(kd_vals * step)   # Approximate the integral of the PDF
    return probability.round(4)


def stable_contact_detection(ax,ay,az,wx,wy,wz):
    # Parameters
    batch_size = 50
    stride = 1
    thres_a = 0.4
    thres_w = 0.04
    eval_samples = 50
    N = ax.shape[0] # Number of samples

    stable_prob_tangential= np.empty((N,))
    stable_prob_vertical  = np.empty((N,))



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
    stable_prob_tangential[0:batch_size] =  prob_ax*prob_ay*prob_wz
    stable_prob_vertical[0:batch_size] =  prob_az*prob_wx*prob_wy



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
        prob_ax = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ax)
        prob_ay = get_axis_probability(-thres_a,thres_a,eval_samples,kd_ay)
        prob_wz = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wz)
        stable_prob_tangential[i] =  prob_ax*prob_ay*prob_wz


        # VERTICAL
        kd_wx = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wx_batch.reshape((len(wx_batch),1))) 
        kd_wy = KernelDensity(bandwidth=sigma_w, kernel='gaussian').fit(wy_batch.reshape((len(wy_batch),1))) 
        kd_az = KernelDensity(bandwidth=sigma_a, kernel='gaussian').fit(az_batch.reshape((len(az_batch),1))) 
        prob_wx = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wx)
        prob_wy = get_axis_probability(-thres_w,thres_w,eval_samples,kd_wy)
        prob_az = get_axis_probability(-thres_a,thres_a,eval_samples,kd_az)

        stable_prob_vertical[i] =  prob_az*prob_wx*prob_wy

    return stable_prob_tangential,stable_prob_vertical

def plot_paper(ax):
    ax_batch = ax[1280:1380]
    ax_batch[0] = -1
    
    kde = KernelDensity(bandwidth= sigma_a, kernel='gaussian')
    kde.fit(ax_batch[:,None])
    x_d = np.arange(-1.2,0.75,0.001)

    thres_a = 0.1
    log_prob = kde.score_samples(x_d[:,None])

    # fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
    plt.fill_between(x_d,np.exp(log_prob),alpha = 0.5)
    plt.plot(ax_batch,np.full_like(ax_batch,-0.01),'|k',markeredgewidth = 1, label ="samples", c= 'b')

    plt.xlabel(r"Acceleration-x ($\frac{m}{s^2}$)", fontsize = 16)
    
    plt.ylabel(r"PDF", fontsize = 16)
    plt.axvline(x =-0.1 , c= 'r', label = "limits")
    plt.axvline(x = 0.1 , c = 'r')
    plt.legend(fontsize = 16)
    # plt.xlim(-0.5,0.7)
    # ax.title("Probability Density Function")


    plt.show()
    


def get_labels(fx,fy,fz):
    labels = np.empty(fz.shape[0])
    max_fz= np.max(fz)
    threhsold = friction_coef*max_fz

    for i in range(fz.shape[0]):
        if i < 6220:
            if fz[i] > 0 and np.sqrt(fx[i]**2+fy[i]**2)-0.1*fz[i] < -50:
                labels[i] = 1
            else:
                labels[i] = 0 
        else:
            if fz[i] > 0 and np.sqrt(fx[i]**2+fy[i]**2)-0.03*fz[i] < -10:
                labels[i] = 1
            else:
                labels[i] = 0 

    return labels

def compute_rmse(prob,labels):
    sum_rms = 0 
    for i in range(prob.shape[0]):
        sum_rms += (labels[i]-prob[i])**2
    sum_rms = np.sqrt(sum_rms/prob.shape[0])
    print(sum_rms)

def make_histogram(prob,labels):
    difs = np.empty(prob.shape[0])
    for i in range(prob.shape[0]):
        difs[i] = abs(prob[i]-labels[i])

    return difs
if __name__ == "__main__":

    plt.rcParams.update({'font.size': 12})


    ax, ay,az,wx,wy, wz, fx, fy, fz, labels = prepare_data() 
    ax = ax[21:]
    ay = ay[21:]
    az = az[21:]
    wx = wx[21:]
    wy = wy[21:]
    wz = wz[21:]
    fx = fx[21:]
    fy = fy[21:]
    fz = fz[21:]

    plot_paper(ax)

    labels = get_labels(fx,fy,fz)

    rotella = np.genfromtxt("../data/rotella_probs.csv",delimiter=",")
    total   = np.genfromtxt("../data/my_probs.csv",delimiter=",")

    difs_rot  = make_histogram(rotella,labels)

    # plot_paper(ax)

    # probs_xy,probs_z = stable_contact_detection(ax,ay,az,wx,wy,wz)

    # total = probs_xy*probs_z
    # np.savetxt("my_probs.csv",total, delimiter=",")

    difs_our = make_histogram(total,labels)

    # compute_rmse(total,labels)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(difs_our,bins = 12)
    ax2.hist(difs_rot,bins = 12)
    ax1.set_ylim([0, 17000])
    ax2.set_ylim([0, 17000])
    fig.subplots_adjust(wspace=0.1, hspace=0)

    ax1.set_xlabel("Proposed method: Difference from ground truth",fontsize=16)
    ax2.set_xlabel("Method [10]: Difference from ground truth",fontsize=16)

    # ax2.spines["bottom"].set_linewidth(6)
    # ax2.tick_params(axis='both', which='major', labelsize=10)
    # ax1.tick_params(axis='both', which='major', labelsize=10)
    # ax1.xticks(fontsize=12)
    # ax2.xticks(fontsize=12)
    ax1.set_ylabel("Frequency",fontsize=16)

    plt.show()  

    '''
    time = np.arange(wz.shape[0])
    fig, axs = plt.subplots(3)

    axs[0].plot(time,fz)
    axs[0].set_ylabel(r'$F_z(N)$ ',fontsize=20)
    # axs[0].axvspan(6520, 14189, facecolor='purple', alpha=0.2,label = r"$\mu_s = 0.03$")


    # axs[1].scatter(time,probs_xy,c='g',s=5)
    # axs[1].set_ylabel(r'$P_{xy}(stable)$',fontsize=20)

    # axs[2].scatter(time,probs_z,c='g',s=5)
    # axs[2].set_ylabel(r'$P_{z}(stable)$',fontsize=20)


    axs[1].scatter(time,probs_xy*probs_z,c='g',s=5, label = "predicted")
    axs[1].set_ylabel(r'$P_{tot}(stable)$',fontsize=20)
    axs[1].scatter(time,labels,c = 'r', s= 2, label ="ground truth")

    axs[2].scatter(time,rotella,c= 'b', s = 5,label = "FCM predictions")
    axs[2].scatter(time,labels,c = 'r', s= 2, label ="ground truth")
    axs[2].set_ylabel(r'$P_{tot}(stable)$',fontsize=20)

    axs[2].set_xlabel(r'Time (ms)',fontsize=16)

    # axs[1].axvline(x = 3884, c= 'r')
    # axs[1].axvline(x = 5340, c= 'r')
    # axs[1].axvline(x = 7810, c= 'r')
    # axs[1].axvline(x = 9410, c= 'r')
    # axs[1].axvline(x = 11830, c= 'r')
    # axs[1].axvline(x = 13310, c= 'r')
    
    # axs[0].axvspan(6520, 14189, facecolor='purple', alpha=0.2)
    # axs[1].axvspan(6520, 14189, facecolor='purple', alpha=0.2,label = r"$\mu_s = 0.03$")


    # plt.plot(time,friction_coef*fz)
    # plt.plot(time,np.sqrt(fx**2+fy**2))
    # plt.show()

    plt.legend(fontsize=16)
    # plt.subplots_adjust(left=0.1,       
    #                     bottom=0.1, 
    #                     right=0.9, 
    #                     top=0.9, 
    #                     wspace=0.4, 
    #                     hspace=0.4)

    # axs[4].set_title("Linear acceleration x")  
    # axs[4].plot(time,ax)

    # axs[5].set_title("Linear acceleration y")  
    # axs[5].plot(time,ay)
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.show() 

    '''
