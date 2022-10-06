# Probabilistic Contact Estimation

This ROS package contains:

* A **real-time contact detection module** that operates in real-time(~500hz) and uses only IMU mounted on the foot of the robot. It has been tested on a quadrupedal GO1, a real TALOS humanoid robot and a simulated ATLAS humanoid. 
* **Datasets**  (*./offline/data/*) from an ATLAS simulated humanoid in RAISIM and a real GO1 quadruped on various terrains
* An offline contact probability estimator module to test on custom datasets.

# Setup
The only dependency for the offline module is *sklearn*. Run:
```
$ pip install -U scikit-learn
```
Then define the data filename you want to use at *offline/src/filename.py* and:
```
$ ./atlas_contact_estimation.py
             or 
$ ./go1_contact_estimation.py
```
Please note in case you are using another robot, you  will need to fine-tune the threshold parameters inside the file by extracting them from a normal gait pattern (no slip). These thresholds are robot/control specific.(More details about how to extract them will be added soon)

# Datasets
### ATLAS
All files at ./offline/data/ATLAS_ have the same structure: \
Fx | Fy |  Fz | Tx | Ty | Tz | ax | ay | az | wx | wy | wz | label

The refresh rate of the meausurements is 1000 hz

Description
* ATLAS_01ground.csv : 0.1 static friction coef., ATLAS walking around. 
* ATLAS_01ground_3steps.csv: 0.1 static friction, 3 steps of ATLAS
* ATLAS_01ground_003slip.csv: 0.1 ground static friction, 0.03 static friction on extremely slippery surface.

### GO1
All files at ./offline/data/ATLAS_ have the same structure: \
| Fz | ax | ay | az | wx | wy | wz | 

The refresh rate is 250 hz for the IMU and 500 hz for the Fz 

* *GO1_matress.csv*: Walking on soft terrain (a matress)
* *GO1_normal_surface.csv*: Walking on a normal surface.(these data are unsynchronized)
* *GO1_slippery_surface.csv*: Walking on a low friction surface. A greased smooth surface, extremely slippery.


#  Real-time Probabilistic Contact State Estimation 

Using Unitree's Go1 quadrupedal robot and low-end IMU sensor (Arduino RP2040 integrated IMU) mounted on the foot.

## Description

An analytical package description of the real experiment with the Go1 robotic dog of Unitree. This package is developed to extra support the theoretical analysis and the simulated experiments of the submitted paper to ICRA 2022 with title "Probabilistic Contact State Estimation for Legged Robots using Inertial Information". 


# Dependencies
## Unitree's Go1 legged robot:

*  [unitree_ros_to_real](https://github.com/unitreerobotics/unitree_ros_to_real)
* [unitree\_ros](https://github.com/unitreerobotics/unitree_ros)
* [unitree_legged_sdk](https://github.com/unitreerobotics/unitree_legged_sdk)
## Arduino Nano RP2040's IMU:
* [rosserial_python](http://wiki.ros.org/rosserial_python)

##  System 
*  Ubuntu 20.04
* ROS Noetic

# Installing

* Follow the instructions of the above listed depended packages.
* Under your 'workspace/src', git clone ProbabilisticContactEstimation package.
```
git clone https://github.com/MichaelMarav/ProbabilisticContactEstimation
```
# Executing program

## Terminal 1
Connect to the real Go1 robot
```
sudo ProbabilisticContactEstimation/BashScripts/ipconfig.sh
```
```
roslaunch unitree_legged_real real.launch
```
## Terminal 2
Arduino pubs IMU data at ```\imu```
```
sudo ProbabilisticContactEstimation/BashScripts/imu.sh
```

## Terminal 3
Initialize IMU bias info 
* Set your path of "/../ws/src/ProbabilisticContactEstimation/src/exe/imuBias.txt", at line 137 of 'init_imu_force.cpp' .
```
rosrun ProbabilisticContactEstimation init_imu_force
```
```
rosrun ProbabilisticContactEstimation slip_recovery
```
# Experiment setup

Mount IMU on a leg (or multiple legs) of Go1 Unitree's. Set the communication between Arduino Nano PR2040 and your Laptop through USB - micro cable. 