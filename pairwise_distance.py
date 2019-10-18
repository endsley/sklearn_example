#!/usr/bin/env python

import numpy as np
import sklearn.metrics



X = np.array([[1,1],[0,0]])
D = sklearn.metrics.pairwise.pairwise_distances(X)
print(D)


#[[0.         1.41421356]
# [1.41421356 0.        ]]

import pdb; pdb.set_trace()
