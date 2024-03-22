# Probabilistic Contact Estimation

## CPP version 

* A **real-time contact detection module** 
* **Datasets**  (*./offline/data/*) from an ATLAS simulated humanoid in RAISIM and a real GO1 quadruped on various terrains
* An offline contact probability estimator module to test on custom datasets.

The published manuscript can be found at: 
https://ieeexplore.ieee.org/document/10161485
# Setup

Please note in case you are using another robot, you  will need to fine-tune the threshold parameters inside the file by extracting them from a normal gait pattern (no slip). These thresholds are robot/control specific.(More details about how to extract them will be added soon)

# Datasets
### **ATLAS bipedal**
All files at ./offline/data/ATLAS_ have the same structure: \
Fx | Fy |  Fz | Tx | Ty | Tz | ax | ay | az | wx | wy | wz | label

The refresh rate of the meausurements is 1000 hz

Description
* ATLAS_01ground.csv : 0.1 static friction coef., ATLAS walking around. 
* ATLAS_01ground_3steps.csv: 0.1 static friction, 3 steps of ATLAS
* ATLAS_01ground_003slip.csv: 0.1 ground static friction, 0.03 static friction on extremely slippery surface.

### **GO1 quadrupedal**
All files at ./offline/data/GO1_ have the same structure: \
| Fz | ax | ay | az | wx | wy | wz | 

The refresh rate is 250 hz for the IMU and 500 hz for the Fz 

* *GO1_matress.csv*: Walking on soft terrain (a matress)
* *GO1_normal_surface.csv*: Walking on a normal surface.(these data are unsynchronized)
* *GO1_slippery_surface.csv*: Walking on a low friction surface. A greased smooth surface, extremely slippery.


#  Real-time Probabilistic Contact State Estimation 



## Description

An analytical package description of the real experiment with the Go1 robotic dog of Unitree. This package is developed to extra support the theoretical analysis and the simulated experiments of the submitted paper to ICRA 2022 with title "Probabilistic Contact State Estimation for Legged Robots using Inertial Information". 


# Dependencies
## Multidimensional kernel density estimator
*  [kdepp](https://github.com/duncanmcn/kdepp.git)

##  System 
*  Ubuntu 20.04

# Installing

* Under your 'workspace', git clone cpp_main branch of ProbabilisticContactEstimation package.
```
git clone --branch cpp_main https://github.com/MichaelMarav/ProbabilisticContactEstimation
```
* In place of ProbabilisticContactEstimation/kdepp, git clone kdepp package
```
cd ProbabilisticContactEstimation 
rm -rf kdepp
git clone https://github.com/duncanmcn/kdepp.git
```
* Build ProbabilisticContactEstimation
```
mkdir build
cd build
cmake ..
make -j16
```

# Executing program 

## Example
```
cd ProbabilisticContactEstimation/build/
./pce_cpp
```
## Code explanasion 

### Parameters of IMU sensor
* Set line 74 of main.cpp:
```
FIRST_TIME_CALC_PARAMS = true;
```
* First time to compute bias and std: IMU Stationary.
```
  for (some time)
      $store_data();
  $compute_mean_std();
```
* Replace the printed value to 'include/pce_params.h'
###  Main flow
* Set line 74 of main.cpp:
```
FIRST_TIME_CALC_PARAMS = false;
```
* Initialize with IMU msgs 
```
for time=batch_size*dt
  $init_things()
```
* Compute probability for each new msg
```
while (condition)
  $update: imu_msg, Fz
  $one_loop(imu_msg) or one_loop(Fz, imu_msg)
  $save_probability() //in csv
```
* Plot probability for tuning ax, ay, az, wx, wy, wz thresshold defined in 'include/pce_params.h'
```
cd bags
python3 plot_probs.py
```
# Citation
If you are using this work please use the following citation:
```
@INPROCEEDINGS{10161485,

  author={Maravgakis, Michael and Argiropoulos, Despina-Ekaterini and Piperakis, Stylianos and Trahanias, Panos},

  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)}, 

  title={Probabilistic Contact State Estimation for Legged Robots using Inertial Information}, 

  year={2023},

  volume={},

  number={},

  pages={12163-12169},

  doi={10.1109/ICRA48891.2023.10161485}}

```
