#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import NearestNeighbors
# Return the id of k nearest sample points


samples = [[0, 0, 2], [1, 0, 1], [0, 0, 1], [1,1,1]]

neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples)
result = neigh.kneighbors([[1, 1, 1.3]], 2, return_distance=False)

print(result)


