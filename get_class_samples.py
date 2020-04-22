#!/usr/bin/env python

import numpy as np

X = np.array([[1,0],[2,0] ,[3,0] ,[1.1, 0.1] ,[2.1, 0.1] ,[3.1, 0.1]])
Y = np.array([ 0 ,0 ,0 ,1 ,1 ,1])

##	obtain samples from a class
#class_id = 0
#indices = np.where(Y == class_id)[0]
#Ẋ = X[indices, :]
#
##	set those samples to 0
#X[indices] = 0
#print(X)


#	obtain samples from a other classes and set them to 0
class_id = 0
indices = np.where(Y != class_id)[0]
X[indices, :] = 0
print(X)
