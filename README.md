# Probabilistic Contact Estimation

This ROS package contains:

* A **real-time contact detection module** that operates in real-time(~500hz). It has been tested on a quadrupedal GO1 and a TALOS humanoid robot. There is a need for IMU mounted on the foot and it predicts the stable contact probability.
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

# Instruction for operating this in real time will be added soon