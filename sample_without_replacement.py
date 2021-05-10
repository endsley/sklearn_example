#!/usr/bin/env python

import sklearn.utils.random

#	out of a 100 samples, pick randomly 10
A = sklearn.utils.random.sample_without_replacement(100,10)	
print(A)
