#!/usr/bin/env python

import numpy as np


# labels unsorted into sorted label of same class next to eachother
def rearrange_sample_to_same_class(X,Y):
	l = np.unique(Y)
	newX = np.empty((0, X.shape[1]))
	newY = np.empty((0))

	for i in l:
		indices = np.where(Y == i)[0]
		newX = np.vstack((newX, X[indices, :]))
		newY = np.hstack((newY, Y[indices]))

	return [newX, newY]

if __name__ == "__main__":
	#data_name = 'cancer'
	data_name = 'cifar10'
	
	X = np.loadtxt('data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			

	[X,Y] = rearrange_sample_to_same_class(X,Y)
	np.savetxt(data_name + '.csv', X, delimiter=',', fmt='%.4f') 
	np.savetxt(data_name + '_label.csv', Y, delimiter=',', fmt='%d') 
