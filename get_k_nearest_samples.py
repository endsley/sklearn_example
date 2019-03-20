#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
# Return the id of k nearest sample points


samples = [[0, 0, 2], [1, 0, 1], [0, 0, 1], [1,1,1]]

neigh = NearestNeighbors(2, 0.4)
neigh.fit(samples)
[distances, indices] = neigh.kneighbors([[1, 1, 1.3]], 2, return_distance=True)



print(distances)
print(indices)


d3 = euclidean_distances(np.array([[1,1,1]]), np.array([[1, 1, 1.3]]))
print(d3)

