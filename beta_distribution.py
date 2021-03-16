#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

X = np.zeros(1000)
for i in np.arange(1000): 
	X[i] = np.random.beta(0.5, 0.5)

H = plt.hist(X, bins=20)
plt.title('Histogram')
plt.show()


