#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels


X = np.array([[ 1., -2.,  2.], [ -2.,  1.,  3.], [ 4.,  1., -2.]])
K = pairwise_kernels(X, metric='linear')


transformer = KernelCenterer().fit(K)
centered_K = transformer.transform(K)
print(centered_K)



H = np.eye(3) - (1.0/3)*np.ones((3,3))
centered_K = H.dot(K).dot(H)
print(centered_K)
