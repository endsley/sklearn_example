#!/usr/bin/env python


import numpy as np
from sklearn.preprocessing import OneHotEncoder


def one_hot_encoding(Y):
	Y = np.reshape(Y,(len(Y),1))
	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(Y)
	return Yₒ

Y = np.array([0,0,1,1,2,2,3,3])
Yₒ = one_hot_encoding(Y)

print(Yₒ)

