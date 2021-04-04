#!/usr/bin/env python

import sklearn.metrics
import numpy as np
import time

#	K(x, y) = exp(-gamma ||x-y||^2)
#	sigma = sqrt( 1/(2*gamma) )
#	gamma = 1/(2*sigma^2)

#	X here defines a set of functions used to approximate another function

#X = np.array([[1,2],[2,3],[1,1],[1,3]], dtype='f')	 #	rows are samples
#xᵢ = np.array([[1,2]], dtype='f')	 #	rows are samples
#rbk = sklearn.metrics.pairwise.rbf_kernel(X, xᵢ, gamma=0.5)
#print(rbk)

#	[[ 1.          0.36787945  0.60653067  0.60653067]
#	 [ 0.36787945  1.          0.082085    0.60653067]
#	 [ 0.60653067  0.082085    1.          0.13533528]
#	 [ 0.60653067  0.60653067  0.13533528  1.        ]]

X = np.random.randn(80000, 1000)
Y = np.random.randn(600, 1000)
start = time.time()
K = sklearn.metrics.pairwise.rbf_kernel(X, Y, gamma=0.5)
end = time.time()
print(end - start)
import pdb; pdb.set_trace()
