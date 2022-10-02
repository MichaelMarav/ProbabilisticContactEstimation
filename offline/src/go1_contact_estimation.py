#! /usr/bin/env python3

'''
 * PCE - Probabilistic Contact Estimator
 *
 * Copyright 2022-2023  Michael Maravgakis and Despina-Ekaterini Argyropoulos, Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	 nor the names of its contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import time


data_filenames = ["../data/GO1_normal_surface.csv","../data/GO1_matress.csv","../data/GO1_slippery_surface.csv"]

class pce:
    '''
    Description:
    Initializes the contact estimator object
    Parameters: 
    * batch_size (int): The size of the batch of data that the contact probability will be estimated
    * eval_samples (int): The number of points to use in order to estimate 
    * [blank]_std (float): The standard deviation of the error of the measurements
    '''

    def __init__(self):
        self.filename   = data_filenames[2]
        self.batch_size = 50
        self.eval_samples = 500

        # Standard deviations 
        self.ax_std = 0.02467
        self.ay_std = 0.02467
        self.az_std = 0.02467
        self.wx_std = 1
        self.wy_std = 1
        self.wz_std = 1


        self.data_batch = np.empty((self.batch_size,6)) # Batch of data to made predictions on


    def read_data(self):
        data = np.genfromtxt(self.filename,delimiter=",")
        f = data[:,0]
        foot_ax = data[:,1]
        foot_ay = data[:,2]
        foot_az = data[:,3]

        foot_wx = data[:,4]
        foot_wy = data[:,5]
        foot_wz = data[:,6]

        # Because the force measurement frame rate is different than 
        # the IMU refresh rate
        for i in range(foot_ax.shape[0]):
            if (np.isnan(foot_ax[i])):
                break_point = i
                break
        foot_ax = foot_ax[0:break_point]
        foot_ay = foot_ay[0:break_point]
        foot_az = foot_az[0:break_point]
        foot_wx = foot_wx[0:break_point]
        foot_wy = foot_wy[0:break_point]
        foot_wz = foot_wz[0:break_point]

        bias_ax = 0.2
        bias_az = -0.32
        bias_ay = 1

        foot_ax += bias_ax
        foot_ay += bias_ay
        foot_az += bias_az

        return f, foot_ax,foot_ay,foot_az,foot_wx,foot_wy,foot_wz



    '''
    Description:
    Calculates the probability in an interval [start_value, end_value] by integrating the PDF
    -- Input --
    * start_value (float): small limit of interval
    * end_value (float): large limit of interval
    * eval_points (int): How many points should be used to approximate the integral
    * kd (KernelDensity): A kernel density object by sklearn

    -- Output --
    * The probability of the PDF in the interval, rounded to 4 digits
    '''
    def get_axis_probability(self,start_value, end_value, kd):
        # Number of evaluation points 
        N = self.eval_samples                                      
        step = (end_value - start_value) / (N - 1)  # Step size

        x = np.linspace(start_value, end_value, N)[:, np.newaxis]  # Generate values in the range
        kd_vals = np.exp(kd.score_samples(x))  # Get PDF values for each x
        probability = np.sum(kd_vals * step)   # Approximate the integral of the PDF

        return probability.round(4)



    '''
    Description:
    Employs KDE to estimate the PDF over a batch of data
    -- Input --
    * force_msg (HighState/footForce:): Force message
    * imu_msg (sensor_msgs/Imu) : Linear acceleration and angular velocity message
    -- Output --
    * Probability of stable contact
    '''

    def stable_contact_detection(self,ax,ay,az,wx,wy,wz):
        
        N_samples = ax.shape[0]
        # Parameters (These are estimated experimentally ONCE during normal gait with stable contact)
        thres_ax = 0.9    # In g-scale of IMU 
        thres_ay = 0.5
        thres_az = 0.3
        thres_wx = 56
        thres_wy = 50
        thres_wz = 100
        
        probs = np.empty((N_samples,))


        print("[INFO]: Beginning Contact Estimation ")

        start = time.time() # Start time

        # First batch
        ax_batch = ax[0:self.batch_size]
        ay_batch = ay[0:self.batch_size]
        az_batch = az[0:self.batch_size]

        wx_batch = wx[0:self.batch_size]
        wy_batch = wy[0:self.batch_size]
        wz_batch = wz[0:self.batch_size]

        kde_ax = KernelDensity(bandwidth=self.ax_std, kernel='gaussian').fit(ax_batch.reshape((self.batch_size,1)))
        kde_ay = KernelDensity(bandwidth=self.ay_std, kernel='gaussian').fit(ay_batch.reshape((self.batch_size,1)))
        kde_az = KernelDensity(bandwidth=self.az_std, kernel='gaussian').fit(az_batch.reshape((self.batch_size,1))) 
    
        kde_wx = KernelDensity(bandwidth=self.wx_std, kernel='gaussian').fit(wx_batch.reshape((self.batch_size,1))) 
        kde_wy = KernelDensity(bandwidth=self.wy_std, kernel='gaussian').fit(wy_batch.reshape((self.batch_size,1))) 
        kde_wz = KernelDensity(bandwidth=self.wz_std, kernel='gaussian').fit(wz_batch.reshape((self.batch_size,1))) 

        
        prob_ax = self.get_axis_probability(-thres_ax,thres_ax,kde_ax)
        prob_ay = self.get_axis_probability(-thres_ay,thres_ay,kde_ay)
        prob_az = self.get_axis_probability(-thres_az,thres_az,kde_az)

        prob_wx = self.get_axis_probability(-thres_wx,thres_wx,kde_wx) 
        prob_wy = self.get_axis_probability(-thres_wy,thres_wy,kde_wy)
        prob_wz = self.get_axis_probability(-thres_wz,thres_wz,kde_wz)

        probs[0:self.batch_size] = prob_ax*prob_ay*prob_az*prob_wx*prob_wy*prob_wz # First batch contact probabilities

        for i in range(self.batch_size,N_samples):
            ax_batch = ax[(i-self.batch_size):i]
            ay_batch = ay[(i-self.batch_size):i]
            az_batch = az[(i-self.batch_size):i]
            wx_batch = wx[(i-self.batch_size):i]
            wy_batch = wy[(i-self.batch_size):i]
            wz_batch = wz[(i-self.batch_size):i]

            kde_ax = KernelDensity(bandwidth=self.ax_std, kernel='gaussian').fit(ax_batch.reshape((self.batch_size,1)))
            kde_ay = KernelDensity(bandwidth=self.ay_std, kernel='gaussian').fit(ay_batch.reshape((self.batch_size,1)))
            kde_az = KernelDensity(bandwidth=self.az_std, kernel='gaussian').fit(az_batch.reshape((self.batch_size,1))) 
        
            kde_wx = KernelDensity(bandwidth=self.wx_std, kernel='gaussian').fit(wx_batch.reshape((self.batch_size,1))) 
            kde_wy = KernelDensity(bandwidth=self.wy_std, kernel='gaussian').fit(wy_batch.reshape((self.batch_size,1))) 
            kde_wz = KernelDensity(bandwidth=self.wz_std, kernel='gaussian').fit(wz_batch.reshape((self.batch_size,1))) 

            
            prob_ax = self.get_axis_probability(-thres_ax,thres_ax,kde_ax)
            prob_ay = self.get_axis_probability(-thres_ay,thres_ay,kde_ay)
            prob_az = self.get_axis_probability(-thres_az,thres_az,kde_az)

            prob_wx = self.get_axis_probability(-thres_wx,thres_wx,kde_wx) 
            prob_wy = self.get_axis_probability(-thres_wy,thres_wy,kde_wy)
            prob_wz = self.get_axis_probability(-thres_wz,thres_wz,kde_wz)

            probs[i] = prob_ax*prob_ay*prob_az*prob_wx*prob_wy*prob_wz
    
        end = time.time()

        elapsed_time = end - start

        print("[INFO]: Contact Estimation Complete! ")
        print("        1) Number of estimated samples = " ,N_samples ,"Total time elapsed (s) = " ,elapsed_time)
        print("        2) Refresh rate: ", N_samples/elapsed_time)


        return probs


    def plot_results(self,probs,fz):

        time_imu = np.arange(0,probs.shape[0]*0.004,0.004) 
        time_f   = np.arange(0,fz.shape[0]*0.002,0.002) 


        fig, axs = plt.subplots(2)  
        axs[0].plot(time_f,fz)
        axs[0].set_ylabel(r'$F_z(N)$ ',fontsize=20)


        axs[1].scatter(time_imu,probs,c='g',s=5)  
        axs[1].set_ylabel(r'$P_{tot}(stable)$',fontsize=20)
        axs[1].set_xlabel(r'Time (s)',fontsize=16)
        # axs[0].axvspan(3.326, 4.673, facecolor='purple', alpha=0.2)#,label = r"$\mu_s = 0.03$")
        # axs[1].axvspan(3.525, 4.960, facecolor='purple', alpha=0.2)
        
        axs[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.show()
        return

    

if __name__ == "__main__":


  
    contact_estimator = pce()

    fz,ax,ay,az,wx,wy,wz = contact_estimator.read_data()
    print("[INFO]: Data loaded successfully")


    probs = contact_estimator.stable_contact_detection(ax,ay,az,wx,wy,wz)
    
    contact_estimator.plot_results(probs,fz)

