# Probabilistic Contact Estimation

In this package we provide, real-time (~500hz) probabilistic contact estimation module for legged robots. We also provide dataset from simulated ATLAS robot in RAISIM. (file structure: Fx Fy Fz Tx Ty Tz ax ay az wx wy wz label). \\

We also provide real data from GO1 quadrupedal robot with an IMU mounted on one foot. 
(file structure: Fz ax ay az wx wy wz)

For running the offline experiments: 

```
$ cd offline/src/
$ ./atlas_contact_estimation.py or go1_contact_estimation
```

Change the filename inside the .py to view results from different datasets. 
(Further description and tutorial + results will be uploaded soon) 