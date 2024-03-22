#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt


data = np.genfromtxt("./pce_probs.csv", delimiter=" ", skip_header=1)

# times
t_real = data[:,0]

# CoM current pos
p0 = data[:,1]

plt.figure()
plt.plot(t_real,p0,Label="Stable Contact Probability")

plt.legend()
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Probability  - time")

plt.show()
plt.waitforbuttonpress(0) # this will wait for indefinite timeplt.close(fig)
plt.close('all')
